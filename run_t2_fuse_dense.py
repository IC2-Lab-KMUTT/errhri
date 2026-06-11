"""T2 (Bad Idea, AUC) — fuse the dense temporal OOF (temporal_dense_oof_t2.csv, TD_) into the T2
tabular stack. Metric = AUC (balanced data), so: NO audio (noise on T2), greedy selection by clip-level
AUC, isotonic-calibrated logistic stack, official video-level AUC report. Tests whether the dense
temporal streams add anything to T2 (expected: no — T2 isn't a reaction-spike task).
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

track = 2
ZOO = {  # no audio (noise on T2)
    "au_xgb":  Stream(("au",), model="xgb"),
    "aug_xgb": Stream(("au_graph",), model="xgb"),
    "gaze":    Stream(("gaze",), model="xgb"),
    "pose":    Stream(("pose",), model="xgb"),
    "blend":   Stream(("blend",), model="xgb"),
    "embed":   Stream(("embed",), model="xgb"),
}

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

oofs = {}
for name, s in ZOO.items():
    try:
        o = _oof_by_key(track, s, 5)
    except FileNotFoundError:
        o = None
    if o is not None:
        oofs[name] = o; print(f"  [{name}] OOF done", flush=True)

tdf = pd.read_csv(CACHE_DIR / f"temporal_dense_oof_t{track}.csv")
tdf["participant"] = tdf.participant.astype(str); tdf["video"] = tdf.video.astype(str)
for col in [c for c in tdf.columns if c.startswith("oof_")]:
    name = col.replace("oof_", "TD_")
    oofs[name] = {(p, v): float(val) for p, v, val in zip(tdf.participant, tdf.video, tdf[col])}
    print(f"  [{name}] loaded", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T2: {len(common)} clips, {len(P)} streams ({list(P)}) ===", flush=True)


def fuse(names, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    fid = subject_folds(gv, 5); pf = np.zeros(len(yv))
    for tr, va in iter_folds(fid):
        ctr, cva = [], []
        for n in names:
            iso = IsotonicRegression(out_of_bounds="clip").fit(P[n][tr], yv[tr])
            ctr.append(iso.transform(P[n][tr])); cva.append(iso.transform(P[n][va]))
        Atr, Ava = np.column_stack(ctr), np.column_stack(cva)
        pf[va] = Ava[:, 0] if len(names) == 1 else \
            LogisticRegression(max_iter=2000, C=C).fit(Atr, yv[tr]).predict_proba(Ava)[:, 1]
    return pf


def auc(pf):
    return roc_auc_score(yv, pf) if len(np.unique(yv)) > 1 else 0.5


def official(pf, tag):
    pred = (pf >= 0.5).astype(int)
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, prob, nf in zip(common, yv, pred, pf, nv):
        nf = int(max(nf, WS[track]))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((nf - WS[track]) // SLIDE[track] + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - prob), float(prob)))
    d = tempfile.mkdtemp(prefix="t2fd_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(FPS[track]), "--window_size", str(WS[track]), "--slide", str(SLIDE[track])],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True); print(out.stdout)


print("solo clip-AUC:")
for n in P:
    print(f"  {n:16} {auc(P[n]):.3f}")

# greedy forward selection on clip AUC
remaining = list(P); chosen = []; best = -1
while remaining:
    scored = [(auc(fuse(chosen + [s])), s) for s in remaining]
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:16} -> AUC {best:.3f}", flush=True)
print(f"\ngreedy chosen: {chosen} (clip-AUC {best:.3f})")

tab = [k for k in ZOO if k in P]
official(fuse(chosen), f"T2_greedy {chosen}")
official(fuse(tab), "T2_control_tabular_no_temporal")
print("\nReference: T2 best ~0.601 AUC", flush=True)
