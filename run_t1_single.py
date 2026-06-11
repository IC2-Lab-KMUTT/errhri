"""T1 single-model-over-everything vs the calibrated stack — settle whether one tuned model on a
denoised concatenation beats the per-stream ensemble (current best 0.685 official video macro-F1).

Honest protocol:
  * concat ALL tabular modalities into ONE matrix (FeatureBank handles merge + per-subject norm +
    leak_clean) — this is exactly the "single XGBoost over all features" the user proposed;
  * NOISE REMOVAL is done INSIDE each train fold only (no val leakage): near-zero variance drop ->
    redundant-pair pruning (|corr|>0.95) -> univariate top-k by mutual information (k swept);
  * train ONE model on the surviving columns, predict the held-out fold;
  * official eval.py on every config so numbers are directly comparable to run_t1_maxfuse (0.685).

Models compared: xgb(all, no selection), xgb(selected, k-sweep), HistGBM(selected), L1-logreg(all).
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.models import make_model
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ALL_MODS = ["au", "au_graph", "gaze", "pose", "blend", "audio", "faceemb"]

mods = []
for m in ALL_MODS:
    if (CACHE_DIR / f"{m}_t{track}.csv").exists():
        mods.append(m)
    else:
        print(f"  [skip] no cache for {m}")
print(f"concatenating modalities: {mods}")

bank = FeatureBank(track, mods).load()
X, y, g = bank.matrix(select="all", leak_clean=True, normalize=True)
cols = np.array(bank.columns)
nf = bank.n_frames
print(f"=== T{track}: X = {X.shape} (clips x concatenated features), {len(np.unique(g))} subjects ===", flush=True)


def _spw(yy):
    return (yy == 0).sum() / max(1, (yy == 1).sum())


def select_train(Xtr, ytr, k):
    """Fit selection on TRAIN ONLY. Returns a column index array. Steps: drop near-constant ->
    prune redundant pairs (|corr|>0.95, keep the higher-MI one) -> univariate top-k by MI."""
    keep = np.arange(Xtr.shape[1])
    var = Xtr.var(0)
    keep = keep[var[keep] > 1e-8]
    if len(keep) == 0:
        return keep
    # redundant-pair pruning on the surviving block
    Xk = Xtr[:, keep]
    C = np.corrcoef(Xk, rowvar=False)
    C = np.nan_to_num(C)
    mi_all = mutual_info_classif(Xk, ytr, random_state=0)
    drop = set()
    n = len(keep)
    order = np.argsort(-mi_all)  # process most informative first; drop its lower-MI duplicates
    for i in order:
        if i in drop:
            continue
        dup = np.where(np.abs(C[i]) > 0.95)[0]
        for j in dup:
            if j != i and j not in drop and mi_all[j] <= mi_all[i]:
                drop.add(j)
    surv = np.array([i for i in range(n) if i not in drop])
    keep = keep[surv]; mi = mi_all[surv]
    if k is not None and k < len(keep):
        top = np.argsort(-mi)[:k]
        keep = keep[top]
    return keep


def official(pred, prob, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str)))
    for (pid, vid), yt, yp, pr, n in zip(keys, y, pred, prob, nf):
        n = int(max(n, ws))
        for f in range(1, n + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        n_win = max((n - ws) // slide + 1, 1)
        for w in range(n_win):
            sub_rows.append((pid, vid, w, int(yp), float(1 - pr), float(pr)))
    d = tempfile.mkdtemp(prefix="t1single_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True)
    print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-400:])


def make_xgb_factory(ytr):
    return make_model("xgb", scale_pos_weight=_spw(ytr))


def make_hgb_factory(_ytr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return lambda: HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=3,
        l2_regularization=1.0, class_weight="balanced", random_state=0)


def make_l1_factory(_ytr):
    from sklearn.linear_model import LogisticRegression
    return lambda: LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                      max_iter=2000, class_weight="balanced")


def run_single(make_factory, k, tag):
    """One model over the (optionally selected) concatenated matrix, subject-grouped 5-fold.
    Selection + threshold are both fit on the TRAIN fold only."""
    fid = subject_folds(g, 5)
    prob = np.zeros(len(y)); pred = np.zeros(len(y), int)
    nfeat = []
    for tr, va in iter_folds(fid):
        sel = np.arange(X.shape[1]) if k == "ALL" else select_train(X[tr], y[tr], k)
        nfeat.append(len(sel))
        mdl = make_factory(y[tr])()
        mdl.fit(X[tr][:, sel], y[tr])
        ptr = mdl.predict_proba(X[tr][:, sel])[:, 1]
        prob[va] = mdl.predict_proba(X[va][:, sel])[:, 1]
        thr = M.tune_threshold(y[tr], ptr)
        pred[va] = (prob[va] >= thr).astype(int)
    sp = M.primary(track, y, pred, prob)
    print(f"  [{tag}] clip-primary {sp:.3f}  (mean {np.mean(nfeat):.0f} feats/fold)", flush=True)
    official(pred, prob, tag)
    return sp


print("\n--- single model over ALL features (the proposal) ---")
run_single(make_xgb_factory, "ALL", "xgb_ALLfeat_noselect")
for k in (300, 150, 80, 40):
    run_single(make_xgb_factory, k, f"xgb_select_k{k}")
run_single(make_hgb_factory, 150, "histgbm_select_k150")
run_single(make_l1_factory, "ALL", "logregL1_ALLfeat")
print("\nReference: per-stream calibrated stack (run_t1_maxfuse) = 0.685 official video macro-F1")
