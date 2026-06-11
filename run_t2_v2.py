"""T2 push to >=0.60 AUC. Two levers vs run_t2_boost:
  (1) SEED-ENSEMBLE the trajectory GRU (4 seeds of the best h64 config, averaged) -> lower-variance,
      stronger temporal stream;
  (2) RE-ADD prosody: audio was dropped because select='signal' stripped all its cols on T2; use
      select='all' (still leak-cleaned) so the orthogonal speech-prosody channel is back.
Fuse gru_ens + gaze + blend + audio + rocket, isotonic-calibrated greedy, report official AUC.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, _oof_seq, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE

track = 2
ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

# (1) GRU seed ensemble on the landmark trajectory
seed_oofs = []
for sd in (0, 1, 2, 3):
    o = _oof_seq(track, 5, "gru", "traj", hidden=64, epochs=40, seed=sd)
    seed_oofs.append(o); print(f"  [gru seed {sd}] done", flush=True)
gru_keys = set(seed_oofs[0])
gru_ens = {k: float(np.mean([o[k] for o in seed_oofs])) for k in gru_keys}

oofs = {"gru_ens": gru_ens}
# (2) decorrelated streams, audio with select='all' to keep prosody
specs = {
    "rocket_tj": lambda: _oof_seq(track, 5, "rocket", "traj", n_kernels=4000),
    "gaze":  lambda: _oof_by_key(track, Stream(("gaze",), model="xgb"), 5),
    "blend": lambda: _oof_by_key(track, Stream(("blend",), model="xgb"), 5),
    "audio": lambda: _oof_by_key(track, Stream(("audio",), model="xgb", select="all"), 5),
    "pose":  lambda: _oof_by_key(track, Stream(("pose",), model="xgb"), 5),
}
for nm, build in specs.items():
    try:
        o = build()
    except Exception as e:
        print(f"  [{nm}] ERR {type(e).__name__}: {str(e)[:70]}", flush=True); continue
    if o is None:
        print(f"  [{nm}] no signal — skipped", flush=True); continue
    oofs[nm] = o; print(f"  [{nm}] done", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T2 v2: {len(common)} clips, {len(P)} streams ===")
print("solo AUC:")
for n in P:
    print(f"  {n:10} {roc_auc_score(yv, P[n]):.3f}")


def fuse(sub, calibrate=True, method="stack", C=1.0):
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
        elif method == "wmean":
            w = np.array([max(roc_auc_score(yv[tr], sub[n][tr]) - 0.5, 0) for n in names])
            w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
            pf[va] = Ava @ w
        else:
            pf[va] = LogisticRegression(max_iter=2000, C=C).fit(Atr, yv[tr]).predict_proba(Ava)[:, 1]
    return pf


def official(pf, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, prob, nf in zip(common, yv, pf, nv):
        nf = int(max(nf, ws))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        n_win = max((nf - ws) // slide + 1, 1)
        yp = int(prob >= 0.5)
        for w in range(n_win):
            sub_rows.append((pid, vid, w, yp, float(1 - prob), float(prob)))
    d = tempfile.mkdtemp(prefix="t2v2_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########")
    print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-300:])


official(P["gru_ens"], "gru_ens solo")
remaining = list(P); chosen = []; best = -1
while remaining:
    scored = []
    for s in remaining:
        sub = {k: P[k] for k in chosen + [s]}
        scored.append((roc_auc_score(yv, fuse(sub, calibrate=True)), s))
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:10} -> AUC {best:.3f}", flush=True)
print(f"\ngreedy chosen: {chosen} (OOF AUC {best:.3f})")
official(fuse({k: P[k] for k in chosen}, calibrate=True), f"greedy_cal {chosen}")
official(fuse(P, calibrate=True, method="wmean"), "ALL wmean cal")
