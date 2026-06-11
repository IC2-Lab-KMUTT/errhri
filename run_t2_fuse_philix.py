"""T2 (AUC) — fuse philix's honest T2 OOF streams (lgbm_tail*, gru_a/b) into our T2 stack.
Honest: greedy forward selection by clip AUC, isotonic-calibrated logistic stack (monotonic → AUC-safe),
no audio. Join on (participant, strip_suffix(video)). Reference: our ~0.601, philix pz-fused 0.6315.
"""
import sys, os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from errhri_features.config import CACHE_DIR

track = 2
ZOO = {"au_xgb": Stream(("au",), model="xgb"), "aug_xgb": Stream(("au_graph",), model="xgb"),
       "gaze": Stream(("gaze",), model="xgb"), "pose": Stream(("pose",), model="xgb"),
       "blend": Stream(("blend",), model="xgb"), "embed": Stream(("embed",), model="xgb")}

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups))
def qid(v): return re.sub(r"_\d+$", "", str(v))   # strip only trailing _<take>; works T1 & T2

oofs = {}
for name, s in ZOO.items():
    try:
        o = _oof_by_key(track, s, 5)
    except FileNotFoundError:
        o = None
    if o is not None:
        oofs[name] = o; print(f"  [{name}] OOF done", flush=True)

pz = np.load("/tmp/p_track2_oof.npz", allow_pickle=True)
p_pid = pz["pids"].astype(str); p_vid = pz["vids"].astype(str)
ourkey = {(p, qid(v)): (p, v) for (p, v) in keys}
for stream in ("lgbm_tail60", "lgbm_tail90", "gru_a", "gru_b"):
    o = {}
    for pp, vv, s in zip(p_pid, p_vid, pz[stream]):
        ok = ourkey.get((pp, qid(vv)))
        if ok is not None:
            o[ok] = float(s)
    oofs["P_" + stream] = o
    print(f"  [P_{stream}] joined {len(o)}/{len(keys)}", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
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

def auc(pf): return roc_auc_score(yv, pf) if len(np.unique(yv)) > 1 else 0.5

print("solo clip-AUC:")
for n in P:
    print(f"  {n:16} {auc(P[n]):.3f}")

remaining = list(P); chosen = []; best = -1
while remaining:
    scored = [(auc(fuse(chosen + [s])), s) for s in remaining]
    scored.sort(reverse=True)
    if scored[0][0] <= best + 1e-4:
        break
    best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    print(f"  greedy + {scored[0][1]:16} -> AUC {best:.4f}", flush=True)

ours = [k for k in ZOO if k in P]
phil = [k for k in P if k.startswith("P_")]
print(f"\ngreedy chosen: {chosen} (clip-AUC {best:.4f})")
print(f"our-only AUC:      {auc(fuse(ours)):.4f}")
print(f"philix-only AUC:   {auc(fuse(phil)):.4f}")
print("Reference: our ~0.601 | philix pz-fused (mild leak) 0.6315", flush=True)
