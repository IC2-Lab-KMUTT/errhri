"""T1 — fuse philix's honest HSEmotion-windowed streams (lgbm_ws25 AUC 0.851, gru) INTO our
calibrated stack. Their OOF is honest LOPO; we add it as extra orthogonal streams (their MediaPipe+
HSEmotion features vs our OpenGraphAU/MediaPipe). Join on (participant, QID) — strip our _N take suffix.
Isotonic-calibrated greedy forward selection on official video macro-F1. Reference: our 0.696, their
honest ws25 solo 0.728.
"""
import sys, os, re, subprocess, tempfile
import numpy as np, pandas as pd
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

track = 1
ZOO = {"au_xgb": Stream(("au",), model="xgb"), "aug_xgb": Stream(("au_graph",), model="xgb"),
       "gaze": Stream(("gaze",), model="xgb"), "pose": Stream(("pose",), model="xgb"),
       "blend": Stream(("blend",), model="xgb"), "audio": Stream(("audio",), model="xgb")}

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

def qid(v):  # strip _N take suffix -> stimulus id
    return re.sub(r"_.*", "", str(v))

oofs = {}
for name, s in ZOO.items():
    o = _oof_by_key(track, s, 5)
    if o is not None:
        oofs[name] = o; print(f"  [{name}] OOF done", flush=True)

# our traj-temporal OOF (the streams that made 0.696)
tdf = pd.read_csv(CACHE_DIR / f"temporal_oof_t{track}.csv")
tdf["participant"] = tdf.participant.astype(str); tdf["video"] = tdf.video.astype(str)
for col in [c for c in tdf.columns if c.startswith("oof_")]:
    oofs["T_" + col.replace("oof_", "")] = {(p, v): float(val)
        for p, v, val in zip(tdf.participant, tdf.video, tdf[col])}

# philix OOF, keyed by (participant, QID) -> remap to OUR (participant, video)
pz = np.load("/tmp/p_track1_oof.npz", allow_pickle=True)
p_pid = pz["pids"].astype(str); p_vid = pz["vids"].astype(str)
ourkey_by_pq = {}
for (p, v) in keys:
    ourkey_by_pq[(p, qid(v))] = (p, v)
for stream in ("lgbm_ws25", "gru"):
    sc = pz[stream]
    o = {}
    for pp, vv, s in zip(p_pid, p_vid, sc):
        ok = ourkey_by_pq.get((pp, qid(vv)))
        if ok is not None:
            o[ok] = float(s)
    oofs["P_" + stream] = o
    print(f"  [P_{stream}] joined {len(o)}/{len(keys)} clips", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T1: {len(common)} clips, {len(P)} streams ({list(P)}) ===", flush=True)


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


def clip_primary(pf):
    fid = subject_folds(gv, 5); pred = np.zeros(len(yv), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(yv[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    return M.primary(track, yv, pred, pf)


def official(pf, tag):
    fid = subject_folds(gv, 5); pred = np.zeros(len(yv), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(yv[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, prob, nf in zip(common, yv, pred, pf, nv):
        nf = int(max(nf, WS[track]))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((nf - WS[track]) // SLIDE[track] + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - prob), float(prob)))
    d = tempfile.mkdtemp(prefix="t1px_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(FPS[track]), "--window_size", str(WS[track]), "--slide", str(SLIDE[track])],
                         capture_output=True, text=True)
    vid = out.stdout.split("n=1319")[-1] if "n=1319" in out.stdout else out.stdout
    f1 = [l for l in vid.splitlines() if "F1 macro" in l]
    print(f"  {tag:48} video {f1[0].split(':')[-1].strip() if f1 else '?'}", flush=True)


print("solo clip-primary:")
for n in P:
    print(f"  {n:16} {clip_primary(P[n]):.3f}")

remaining = list(P); chosen = []; best = -1
while remaining:
    scored = [(clip_primary(fuse(chosen + [s])), s) for s in remaining]
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:16} -> {best:.3f}", flush=True)
print(f"\ngreedy chosen: {chosen} (clip-primary {best:.3f})", flush=True)

official(fuse(chosen), f"FUSED+philix greedy")
official(fuse([k for k in ["au_xgb","blend","gaze","aug_xgb","pose","T_gru_attn_bc","T_mil_lse_bc","T_mil_lse"] if k in P]),
         "our_prior_winner (0.696 ctrl)")
official(fuse(["P_lgbm_ws25"]), "philix_ws25_solo (honest)")
print("\nReference: our 0.696 | philix honest ws25 solo 0.728 | leaked 0.7433", flush=True)
