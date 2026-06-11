"""T1 proper models round 2 — TabM DONE RIGHT + FT-Transformer tuned. Round 1 fed TabM raw 4849 feats
(0.616); its real recipe is MI-feature-selection + PiecewiseLinearEmbeddings (target-aware bins). Here:
  F2) TabM + PLE, MI-top-200, target-aware bins (n_bins=24), k=32 internal heads
  F3) TabM + PLE, MI-top-100
  E2) FT-Transformer, MI-top-256, default
  E3) FT-Transformer, MI-top-128, regularized small (2 blocks, d=128, higher dropout) for small-N
Same harness: subject-grouped 5-fold + inner early-stop + temperature calibration + official eval.
Reference: stack 0.685 | FT-top128 (round1) 0.633 | TabM-raw 0.616.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

torch.set_num_threads(8)
track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ALL_MODS = ["au", "au_graph", "gaze", "pose", "blend", "audio", "faceemb"]
mods = [m for m in ALL_MODS if (CACHE_DIR / f"{m}_t{track}.csv").exists()]

cols = []
y = g = nf = keys = None
mats = []
for m in mods:
    bank = FeatureBank(track, [m]).load()
    Xm, y, g = bank.matrix(select="all", leak_clean=True, normalize=True)
    if keys is None:
        keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str))); nf = bank.n_frames
    mats.append(Xm.astype(np.float32))
X = np.hstack(mats)
print(f"=== T{track}: X = {X.shape} ===", flush=True)


def official(pred, prob, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, pr, n in zip(keys, y, pred, prob, nf):
        n = int(max(n, ws))
        for f in range(1, n + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((n - ws) // slide + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - pr), float(pr)))
    d = tempfile.mkdtemp(prefix="t1p2_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True); print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-300:])


def temperature(logit_tr, ytr):
    t = torch.zeros(1, requires_grad=True)
    lt = torch.tensor(logit_tr, dtype=torch.float32); yy = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=60); lossf = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = lossf(lt / t.exp(), yy); loss.backward(); return loss
    opt.step(closure); return float(t.exp().detach())


def train_torch(build, forward_logits, Xtr, ytr, Xva, gtr, max_ep=200, patience=25, lr=2e-3, wd=3e-4, bs=256):
    ufid = subject_folds(gtr, 4)
    in_tr = np.where(ufid != 0)[0]; in_va = np.where(ufid == 0)[0]
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    net = build()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(spw)))
    Xt = torch.tensor(Xtr[in_tr]); yt = torch.tensor(ytr[in_tr], dtype=torch.float32)
    Xv = torch.tensor(Xtr[in_va]); yv = ytr[in_va]
    best_auc, best_state, bad = -1, None, 0
    for ep in range(max_ep):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(forward_logits(net, Xt[idx]), yt[idx]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(forward_logits(net, Xv)).numpy()
        auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        lg_tr = forward_logits(net, torch.tensor(Xtr)).numpy()
        lg_va = forward_logits(net, torch.tensor(Xva)).numpy()
    T = temperature(lg_tr, ytr)
    return 1 / (1 + np.exp(-lg_tr / T)), 1 / (1 + np.exp(-lg_va / T))


def evaluate(build, forward_logits, tag, select_fn, **kw):
    fid = subject_folds(g, 5); prob = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        sel = select_fn(X[tr], y[tr])
        Xtr_s, Xva_s = X[tr][:, sel], X[va][:, sel]
        thunk = (lambda Xs=Xtr_s, ys=y[tr]: build(Xs.shape[1], Xs, ys))
        ptr, pva = train_torch(thunk, forward_logits, Xtr_s, y[tr], Xva_s, g[tr], **kw)
        prob[va] = pva
        thr = M.tune_threshold(y[tr], ptr); pred[va] = (pva >= thr).astype(int)
    print(f"  [{tag}] clip-primary {M.primary(track, y, pred, prob):.3f}", flush=True)
    official(pred, prob, tag)


def mi_top(k):
    def f(Xtr, ytr):
        nonconst = np.where(Xtr.std(0) > 1e-8)[0]  # PLE bins require >=2 distinct values per col
        mi = mutual_info_classif(Xtr[:, nonconst], ytr, random_state=0)
        return nonconst[np.argsort(-mi)[:k]]
    return f


# ---- TabM + PiecewiseLinearEmbeddings (target-aware bins) --------------------
def build_tabm_ple(nfeat, Xtr, ytr, n_bins=24, d_emb=16, k=32):
    import rtdl_num_embeddings as E
    from tabm import TabM
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    bins = E.compute_bins(Xt, n_bins=min(n_bins, max(2, len(Xtr) // 20)),
                          y=torch.tensor(ytr), regression=False, tree_kwargs={"min_samples_leaf": 32})
    emb = E.PiecewiseLinearEmbeddings(bins, d_embedding=d_emb, activation=False, version="B")
    return TabM(n_num_features=nfeat, cat_cardinalities=[], d_out=1, num_embeddings=emb,
                n_blocks=3, d_block=512, dropout=0.1, arch_type="tabm", k=k, start_scaling_init="random-signs")


def tabm_logits(net, x):
    return net(x).mean(1).squeeze(1)


# ---- FT-Transformer ---------------------------------------------------------
def build_ft_default(nfeat, Xtr, ytr):
    from rtdl_revisiting_models import FTTransformer
    return FTTransformer(n_cont_features=nfeat, cat_cardinalities=[], d_out=1, **FTTransformer.get_default_kwargs())


def build_ft_small(nfeat, Xtr, ytr):
    from rtdl_revisiting_models import FTTransformer
    return FTTransformer(n_cont_features=nfeat, cat_cardinalities=[], d_out=1, n_blocks=2, d_block=128,
                         attention_n_heads=8, attention_dropout=0.3, ffn_d_hidden_multiplier=4 / 3,
                         ffn_dropout=0.2, residual_dropout=0.0)


def ft_logits(net, x):
    return net(x, None).squeeze(1)


print("\n--- TabM done right + FT tuned ---", flush=True)
evaluate(build_tabm_ple, tabm_logits, "F2_TabM_PLE_top200", mi_top(200), max_ep=200, patience=25, lr=2e-3, wd=3e-4)
evaluate(build_tabm_ple, tabm_logits, "F3_TabM_PLE_top100", mi_top(100), max_ep=200, patience=25, lr=2e-3, wd=3e-4)
evaluate(build_ft_default, ft_logits, "E2_FT_top256", mi_top(256), max_ep=200, patience=25, lr=1e-3, wd=1e-5)
evaluate(build_ft_small, ft_logits, "E3_FT_small_top128", mi_top(128), max_ep=250, patience=30, lr=1e-3, wd=1e-4)
print("\nReference: stack 0.685 | FT-top128 0.633 | midfusion 0.632 | TabM-raw 0.616", flush=True)
