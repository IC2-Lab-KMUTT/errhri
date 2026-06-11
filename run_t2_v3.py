"""T2 v3 — final push to >=0.60 AUC. Lessons from v1/v2: config-diverse GRUs (h64 + h128b) beat a
seed-ensemble of one config; gaze + blend add; audio/rocket/pose don't. So v3 keeps exactly v1's
winning four streams and sweeps the FUSION itself: stack-C grid x {cal, nocal} x {stack, wmean,
rank-mean}, plus drop-one ablations. Selection metric = OOF AUC; every candidate also gets the
official eval so we report the ranked number.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, _oof_seq, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE

track = 2
ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))

oofs = {}
oofs["gru_h64"] = _oof_seq(track, 5, "gru", "traj", hidden=64, epochs=40)
print("  [gru_h64] done", flush=True)
oofs["gru_h128b"] = _oof_seq(track, 5, "gru", "traj", hidden=128, epochs=70, bidir=True, dropout=0.3)
print("  [gru_h128b] done", flush=True)
oofs["gaze"] = _oof_by_key(track, Stream(("gaze",), model="xgb"), 5)
print("  [gaze] done", flush=True)
oofs["blend"] = _oof_by_key(track, Stream(("blend",), model="xgb"), 5)
print("  [blend] done", flush=True)

common = [k for k in keys if all(k in o for o in oofs.values())]
P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])
print(f"\n=== T2 v3: {len(common)} clips ===")
for n in P:
    print(f"  {n:10} solo AUC {roc_auc_score(yv, P[n]):.3f}")


def fuse(sub, method="stack", calibrate=True, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    names = list(sub); fid = subject_folds(gv, 5); pf = np.zeros(len(yv))
    for tr, va in iter_folds(fid):
        ctr, cva = [], []
        for n in names:
            p = sub[n]
            if method == "rank":
                # rank-normalize within fold split (uses only ordering, robust to calibration)
                rtr = rankdata(p[tr]) / len(tr); rva = rankdata(p[va]) / len(va)
                ctr.append(rtr); cva.append(rva); continue
            if calibrate:
                iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], yv[tr])
                ctr.append(iso.transform(p[tr])); cva.append(iso.transform(p[va]))
            else:
                ctr.append(p[tr]); cva.append(p[va])
        Atr, Ava = np.column_stack(ctr), np.column_stack(cva)
        if len(names) == 1:
            pf[va] = Ava[:, 0]
        elif method in ("wmean", "rank"):
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
    d = tempfile.mkdtemp(prefix="t2v3_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    vid_auc = None
    lines = out.stdout.splitlines()
    blocks = [i for i, l in enumerate(lines) if "AUC-ROC" in l]
    if len(blocks) >= 2:
        vid_auc = lines[blocks[1]].split(":")[-1].strip()
    print(f"  OFFICIAL[{tag}] video AUC = {vid_auc}", flush=True)
    return vid_auc


# fusion sweep on the fixed 4-stream set
cands = {}
for C in (0.1, 0.3, 1.0, 3.0):
    for cal in (True, False):
        cands[f"stack_C{C}{'_cal' if cal else ''}"] = fuse(P, "stack", cal, C)
cands["wmean_cal"] = fuse(P, "wmean", True)
cands["wmean_nocal"] = fuse(P, "wmean", False)
cands["rank_wmean"] = fuse(P, "rank")
# drop-one ablations on the best simple method
for drop in P:
    sub = {k: v for k, v in P.items() if k != drop}
    cands[f"drop_{drop}"] = fuse(sub, "stack", True, 1.0)
# pairwise GRU + one
cands["grus_only"] = fuse({k: P[k] for k in ("gru_h64", "gru_h128b")}, "stack", True, 1.0)

print("\nOOF AUC of all candidates:")
ranked = sorted(cands.items(), key=lambda kv: -roc_auc_score(yv, kv[1]))
for nm, pf in ranked:
    print(f"  {nm:18} {roc_auc_score(yv, pf):.4f}")

print("\nOfficial eval on top-5:")
for nm, pf in ranked[:5]:
    official(pf, nm)
official(cands.get("stack_C1.0_cal", ranked[0][1]), "v1-equivalent stack_C1.0_cal")
