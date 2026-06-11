"""T1 Direction-A — fuse the temporal+baseline-contrast OOF (from run_t1_temporal) INTO the calibrated
stack that scored 0.685. The temporal streams are a different inductive bias (ordered dynamics relative
to neutral) and gru_attn_bc hit AUC 0.724 solo, so they should be additive. Isotonic-calibrated greedy
forward selection over the 6 stack streams + 4 temporal streams; official eval. Reference: 0.685.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

track = 1
ZOO = {
    "au_xgb":  Stream(("au",), model="xgb"),
    "aug_xgb": Stream(("au_graph",), model="xgb"),
    "gaze":    Stream(("gaze",), model="xgb"),
    "pose":    Stream(("pose",), model="xgb"),
    "blend":   Stream(("blend",), model="xgb"),
    "audio":   Stream(("audio",), model="xgb"),
}

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

oofs = {}
for name, s in ZOO.items():
    o = _oof_by_key(track, s, 5)
    if o is not None:
        oofs[name] = o; print(f"  [{name}] OOF done", flush=True)

# temporal OOF streams from run_t1_temporal
tdf = pd.read_csv(CACHE_DIR / f"temporal_oof_t{track}.csv")
tdf["participant"] = tdf.participant.astype(str); tdf["video"] = tdf.video.astype(str)
for col in [c for c in tdf.columns if c.startswith("oof_")]:
    name = col.replace("oof_", "T_")
    oofs[name] = {(p, v): float(val) for p, v, val in zip(tdf.participant, tdf.video, tdf[col])}
    print(f"  [{name}] temporal OOF loaded", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T1: {len(common)} clips, {len(P)} streams ({list(P)}) ===", flush=True)


def fuse(sub, calibrate=True, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    names = list(sub); fid = subject_folds(gv, 5); pf = np.zeros(len(yv))
    for tr, va in iter_folds(fid):
        ctr, cva = [], []
        for n in names:
            p = sub[n]
            if calibrate:
                iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], yv[tr])
                ctr.append(iso.transform(p[tr])); cva.append(iso.transform(p[va]))
            else:
                ctr.append(p[tr]); cva.append(p[va])
        Atr, Ava = np.column_stack(ctr), np.column_stack(cva)
        if len(names) == 1:
            pf[va] = Ava[:, 0]
        else:
            pf[va] = LogisticRegression(max_iter=2000, C=C).fit(Atr, yv[tr]).predict_proba(Ava)[:, 1]
    return pf


def clip_primary(pf):
    fid = subject_folds(gv, 5); pred = np.zeros(len(yv), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(yv[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    return M.primary(track, yv, pred, pf)


def official(pf, tag):
    fid = subject_folds(gv, 5); pred = np.zeros(len(yv), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(yv[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, prob, nf in zip(common, yv, pred, pf, nv):
        nf = int(max(nf, ws))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((nf - ws) // slide + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - prob), float(prob)))
    d = tempfile.mkdtemp(prefix="t1ft_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(FPS[track]), "--window_size", str(WS[track]), "--slide", str(SLIDE[track])],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True); print(out.stdout)


print("solo clip-primary:")
for n in P:
    print(f"  {n:14} {clip_primary(P[n]):.3f}")

# greedy forward selection on clip-primary
remaining = list(P); chosen = []; best = -1
while remaining:
    scored = [(clip_primary(fuse({k: P[k] for k in chosen + [s]})), s) for s in remaining]
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:14} -> {best:.3f}", flush=True)
print(f"\ngreedy chosen: {chosen} (clip-primary {best:.3f})")

official(fuse({k: P[k] for k in chosen}), f"A_greedy {chosen}")
# also: the prior 6-stream stack WITHOUT temporal, as the 0.685 control
stack6 = [k for k in ["au_xgb","blend","gaze","aug_xgb","pose","audio"] if k in P]
official(fuse({k: P[k] for k in stack6}), "control_stack6_no_temporal")
print("\nReference: stack6 = 0.685 | best temporal solo (gru_attn_bc) AUC 0.724 / mF1 0.605", flush=True)
