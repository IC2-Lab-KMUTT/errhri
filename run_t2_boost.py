"""T2 booster — target >=0.60 AUC. T2's signal is the landmark-trajectory temporal model, NOT facial
AUs (au_graph solo was below chance). So: tune the trajectory GRU + ROCKET, then fuse with prosody and
any decorrelated dense streams (gaze/pose), calibrated, optimizing AUC. Reports official eval.py AUC.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, _oof_seq, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE

track = 2
ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

# temporal trajectory models (the T2 signal) + decorrelated tabular streams
GRU_GRID = [
    ("gru_tj_h64",  dict(hidden=64,  epochs=40)),
    ("gru_tj_h96b", dict(hidden=96,  epochs=60, bidir=True)),
    ("gru_tj_h128b",dict(hidden=128, epochs=70, bidir=True, dropout=0.3)),
]
builders = {}
for nm, kw in GRU_GRID:
    builders[nm] = (lambda kw=kw: _oof_seq(track, 5, "gru", "traj", **kw))
builders["rocket_tj"] = lambda: _oof_seq(track, 5, "rocket", "traj", n_kernels=4000)
builders["audio"]   = lambda: _oof_by_key(track, Stream(("audio",), model="xgb"), 5)
builders["pose"]    = lambda: _oof_by_key(track, Stream(("pose",), model="xgb"), 5)
builders["gaze"]    = lambda: _oof_by_key(track, Stream(("gaze",), model="xgb"), 5)
builders["faceemb"] = lambda: _oof_by_key(track, Stream(("faceemb",), model="xgb"), 5)
builders["blend"]   = lambda: _oof_by_key(track, Stream(("blend",), model="xgb"), 5)

oofs = {}
for nm, build in builders.items():
    try:
        o = build()
    except FileNotFoundError:
        print(f"  [{nm}] cache missing — skipped", flush=True); continue
    except Exception as e:
        print(f"  [{nm}] ERR {type(e).__name__}: {str(e)[:80]}", flush=True); continue
    if o is None:
        print(f"  [{nm}] no signal — skipped", flush=True); continue
    oofs[nm] = o; print(f"  [{nm}] OOF done", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T2: {len(common)} clips, {len(P)} streams ===", flush=True)
print("solo AUC:")
for n in P:
    print(f"  {n:12} {roc_auc_score(yv, P[n]):.3f}")


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
    d = tempfile.mkdtemp(prefix="t2b_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########")
    print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-400:])


# best single GRU
gru_names = [n for n in P if n.startswith("gru_tj")]
if gru_names:
    bestg = max(gru_names, key=lambda n: roc_auc_score(yv, P[n]))
    print(f"\nbest GRU: {bestg} ({roc_auc_score(yv, P[bestg]):.3f})")
    official(P[bestg], f"best_gru {bestg}")

# greedy on OOF AUC (calibrated stack)
remaining = list(P); chosen = []; best = -1
while remaining:
    scored = []
    for s in remaining:
        sub = {k: P[k] for k in chosen + [s]}
        pf = fuse(sub, calibrate=True)
        scored.append((roc_auc_score(yv, pf), s))
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:12} -> AUC {best:.3f}", flush=True)
print(f"\ngreedy chosen: {chosen} (OOF AUC {best:.3f})")
official(fuse({k: P[k] for k in chosen}, calibrate=True), f"greedy_cal {chosen}")
official(fuse(P, calibrate=True, method="wmean"), "ALL wmean")
