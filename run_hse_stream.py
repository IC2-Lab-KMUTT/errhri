"""Build an independent HSEmotion stream from philix's per-frame features (our copy),
aggregate per-clip with reaction-aware stats + baseline-contrast, per-fold PCA, XGBoost
in OUR subject-grouped CV -> OOF, then fuse into our calibrated greedy stack.
Usage: run_hse_stream.py <track>
"""
import sys, os, re, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.recipes import _oof_by_key, Stream
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

track = int(sys.argv[1])
NPZ = f"/home/ic2/research/errhri/philix_feats/t{track}_feats.npz"
PCA_K, BASE_K = 96, 8

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))
def qid(v): return re.sub(r"_\d+$", "", str(v))

z = np.load(NPZ, allow_pickle=True)
pid = z["participant_ids"].astype(str); vid = z["official_video_ids"].astype(str)
fnum = z["frame_nums"].astype(int)
HSE = z["features"][:, int(z["mp_dim"][0]):].astype(np.float32)   # HSE block only
print(f"T{track}: {HSE.shape[0]} frames x {HSE.shape[1]} HSE dims", flush=True)

# group frame-row indices per our clip key
ourkey = {(p, qid(v)): (p, v) for (p, v) in keys}
rows_by_clip = {}
for i, (p, v) in enumerate(zip(pid, vid)):
    ok = ourkey.get((p, qid(v)))
    if ok is not None:
        rows_by_clip.setdefault(ok, []).append(i)
for k in rows_by_clip:
    idx = np.array(rows_by_clip[k]); rows_by_clip[k] = idx[np.argsort(fnum[idx])]
common = [k for k in keys if k in rows_by_clip]
print(f"matched {len(common)}/{len(keys)} clips", flush=True)

yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
nv = np.array([nk[k] for k in common])

def agg_clip(A):  # A: (T, PCA_K) -> reaction-aware per-clip vector
    if len(A) == 0: A = np.zeros((1, PCA_K), np.float32)
    mean, std = A.mean(0), A.std(0)
    mx, mn = A.max(0), A.min(0)
    top3 = np.sort(A, 0)[-3:].mean(0) if len(A) >= 3 else mean
    last = A[-1]
    d = np.abs(np.diff(A, axis=0)) if len(A) > 1 else np.zeros((1, PCA_K), np.float32)
    velm, velx = d.mean(0), d.max(0)
    bc = A - A[:BASE_K].mean(0)          # baseline-contrast vs neutral early frames
    bcmax = np.abs(bc).max(0)
    return np.concatenate([mean, std, mx, mn, top3, last, velm, velx, bcmax]).astype(np.float32)

XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
           colsample_bytree=0.6, reg_lambda=3.0, min_child_weight=5,
           eval_metric="logloss", n_jobs=-1, tree_method="hist")

# per-fold PCA(train frames) -> aggregate -> xgb -> OOF
fid = subject_folds(gv, 5); oof = np.zeros(len(common))
for fold, (trm, vam) in enumerate(iter_folds(fid)):
    tr, va = np.where(trm)[0], np.where(vam)[0]
    tr_rows = np.concatenate([rows_by_clip[common[i]] for i in tr])
    pca = PCA(n_components=PCA_K, svd_solver="randomized", random_state=0).fit(HSE[tr_rows])
    def feats(ids):
        return np.array([agg_clip(pca.transform(HSE[rows_by_clip[common[i]]])) for i in ids])
    Xtr, Xva = feats(tr), feats(va)
    clf = XGBClassifier(**XGB).fit(Xtr, yv[tr])
    oof[va] = clf.predict_proba(Xva)[:, 1]
    print(f"  fold{fold} pca+agg+xgb done", flush=True)

if track == 1:
    thr_pred = np.zeros(len(common), int)
    for trm, vam in iter_folds(fid):
        tr, va = np.where(trm)[0], np.where(vam)[0]
        thr = M.tune_threshold(yv[tr], oof[tr]); thr_pred[va] = (oof[va] >= thr).astype(int)
    print(f"\nHSE stream solo clip macro-F1: {M.primary(track, yv, thr_pred, oof):.3f}", flush=True)
print(f"HSE stream solo clip AUC: {roc_auc_score(yv, oof):.4f}", flush=True)

# save OOF for fusion reuse
pd.DataFrame({"participant": [k[0] for k in common], "video": [k[1] for k in common],
              "oof_hse": oof}).to_csv(CACHE_DIR / f"hse_oof_t{track}.csv", index=False)
print(f"saved hse_oof_t{track}.csv", flush=True)
