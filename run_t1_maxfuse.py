"""Phase 1 — squeeze T1 through the OFFICIAL evaluator with the full rich stream zoo + calibrated
greedy fusion. The old `au` cache carries AU intensities + GAZE + HEAD POSE (leak-cleaned), so it is
the breadth complement to dense `au_graph`. We greedily select on the official video macro-F1.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE

track = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Stream zoo: name -> Stream (tabular caches only here; all dense/rich)
ZOO = {
    "au_xgb":      Stream(("au",), model="xgb"),       # AU intensity + gaze + head pose (sparse 10f)
    "au_rf":       Stream(("au",), model="rf"),
    "aug_xgb":     Stream(("au_graph",), model="xgb"), # dense 41-AU probs (48f)
    "aug_rf":      Stream(("au_graph",), model="rf"),
    "faceemb":     Stream(("faceemb",), model="xgb"),  # DINOv2 dense face dynamics
    "pose":        Stream(("pose",), model="xgb"),     # dense head/body pose
    "gaze":        Stream(("gaze",), model="xgb"),     # dense eye-gaze + 6DoF head pose (NEW)
    "blend":       Stream(("blend",), model="xgb"),    # mediapipe blendshapes
    "audio":       Stream(("audio",), model="xgb"),    # prosody
}

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

oofs = {}
for name, s in ZOO.items():
    try:
        o = _oof_by_key(track, s, 5)
    except FileNotFoundError:
        print(f"  [{name}] cache missing — skipped"); continue
    if o is None:
        print(f"  [{name}] no signal cols — skipped"); continue
    oofs[name] = o; print(f"  [{name}] OOF done", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T{track}: {len(common)} clips, {len(P)} streams ===", flush=True)


def fuse(streams_subset, calibrate=True, C=1.0):
    """isotonic-calibrated logistic stack inside subject-grouped meta-CV -> OOF fused probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    names = list(streams_subset); fid = subject_folds(gv, 5); pf = np.zeros(len(yv))
    for tr, va in iter_folds(fid):
        cols_tr, cols_va = [], []
        for n in names:
            p = streams_subset[n]
            if calibrate:
                iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], yv[tr])
                cols_tr.append(iso.transform(p[tr])); cols_va.append(iso.transform(p[va]))
            else:
                cols_tr.append(p[tr]); cols_va.append(p[va])
        Atr, Ava = np.column_stack(cols_tr), np.column_stack(cols_va)
        if len(names) == 1:
            pf[va] = Ava[:, 0]
        else:
            pf[va] = LogisticRegression(max_iter=2000, C=C).fit(Atr, yv[tr]).predict_proba(Ava)[:, 1]
    return pf


def clip_primary(pf):
    fid = subject_folds(gv, 5); pred = np.zeros(len(yv), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(yv[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    return M.primary(track, yv, pred, pf), pred


def official(pf, tag):
    """Run real eval.py on a fused prob vector (clip pred replicated across windows)."""
    fid = subject_folds(gv, 5); pred = np.zeros(len(yv), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(yv[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, prob, nf in zip(common, yv, pred, pf, nv):
        nf = int(max(nf, ws))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        n_win = max((nf - ws) // slide + 1, 1)
        for w in range(n_win):
            sub_rows.append((pid, vid, w, int(yp), float(1 - prob), float(prob)))
    d = tempfile.mkdtemp(prefix="t1mf_")
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


# solo clip-primary
print("solo clip-primary:")
solo = {}
for n in P:
    sp, _ = clip_primary(P[n]); solo[n] = sp; print(f"  {n:10} {sp:.3f}")

# greedy forward selection on clip-primary (fast proxy for official)
remaining = list(P); chosen = []; best = -1
while remaining:
    scored = []
    for s in remaining:
        sub = {k: P[k] for k in chosen + [s]}
        pf = fuse(sub, calibrate=True)
        scored.append((clip_primary(pf)[0], s))
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:10} -> {best:.3f}", flush=True)
print(f"\ngreedy chosen: {chosen} (clip-primary {best:.3f})")

# official eval on key configs
official(fuse({k: P[k] for k in chosen}, calibrate=True), f"greedy_cal {chosen}")
allset = fuse(P, calibrate=True)
official(allset, "ALL streams (cal stack)")
core = {k: P[k] for k in ["au_xgb","au_rf","blend"] if k in P}
official(fuse(core, calibrate=True), "baseline core (au+rf+blend) cal")
rich = {k: P[k] for k in ["au_xgb","au_rf","aug_xgb","faceemb","pose","blend"] if k in P}
official(fuse(rich, calibrate=True), "rich (au+aug+faceemb+pose+blend) cal")
