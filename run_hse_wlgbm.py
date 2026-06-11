"""Faithful repro of philix's winning HSEmotion stream IN OUR honest 5-fold subject CV:
per-participant z-score -> per-fold PCA(emb) -> ws-windowed stats -> LGBM -> max-agg per clip.
Produces an OOF stream we own. Usage: run_hse_wlgbm.py <track>
"""
import sys, re, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from errhri_features.config import CACHE_DIR

track = int(sys.argv[1])
NPZ = f"/home/ic2/research/errhri/philix_feats/t{track}_feats.npz"
WS, SLIDE = (25, 5) if track == 1 else (10, 2)
PCA_K, LOGITS = 128, 10
LGBM = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=31,
            max_depth=6, min_child_samples=60, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=1.0, n_jobs=-1,
            random_state=42, verbosity=-1)

ref = FeatureBank(track, ["au"]).load()
keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups))
def qid(v): return re.sub(r"_\d+$", "", str(v))

z = np.load(NPZ, allow_pickle=True)
pid = z["participant_ids"].astype(str); vid = z["official_video_ids"].astype(str)
fnum = z["frame_nums"].astype(int)
mp_dim = int(z["mp_dim"][0])
X = z["features"].astype(np.float32)
mp = X[:, :mp_dim]; hse = X[:, mp_dim:]
emb = hse[:, :hse.shape[1]-LOGITS]; logit = hse[:, hse.shape[1]-LOGITS:]
print(f"T{track}: {X.shape[0]} frames | mp{mp.shape[1]} emb{emb.shape[1]} logit{logit.shape[1]}", flush=True)

# per-participant z-score (label-free) on mp+logit (PCA handles emb scale)
def pz_cols(A, p):
    out = np.empty_like(A)
    for u in np.unique(p):
        m = p == u; mu, sd = A[m].mean(0), A[m].std(0) + 1e-6
        out[m] = (A[m] - mu) / sd
    return out
mp = pz_cols(mp, pid); logit = pz_cols(logit, pid); emb = pz_cols(emb, pid)

ourkey = {(p, qid(v)): (p, v) for (p, v) in keys}
rows = defaultdict(list)
for i, (p, v) in enumerate(zip(pid, vid)):
    ok = ourkey.get((p, qid(v)))
    if ok is not None: rows[ok].append(i)
for k in rows:
    a = np.array(rows[k]); rows[k] = a[np.argsort(fnum[a])]
common = [k for k in keys if k in rows]
yv = np.array([yk[k] for k in common]); gv = np.array([gk[k] for k in common])
print(f"matched {len(common)}/{len(keys)}", flush=True)

def wstats(w, s, n):
    mean, std = w.mean(0), w.std(0); mn, mx = w.min(0), w.max(0)
    slope = w[-1] - w[0]
    vel = np.abs(np.diff(w, axis=0)).mean(0) if len(w) > 1 else np.zeros(w.shape[1], np.float32)
    pos = np.array([s/max(n,1), (s+len(w))/max(n,1)], np.float32)
    return np.concatenate([mean, std, mn, mx, slope, vel, pos]).astype(np.float32)

def win_table(clip_keys, Fmat):
    Xs, vmap = [], []
    for ci, k in enumerate(clip_keys):
        f = Fmat[rows[k]]; n = len(f)
        starts = list(range(0, n-WS+1, SLIDE)) or [0]
        for st in starts:
            Xs.append(wstats(f[st:st+WS] if n >= WS else f, st, n)); vmap.append(ci)
    return np.asarray(Xs, np.float32), np.asarray(vmap)

oof = np.zeros(len(common))
for fold, u in enumerate(np.unique(gv)):          # LOPO to match philix protocol
    tr, va = np.where(gv != u)[0], np.where(gv == u)[0]
    tr_rows = np.concatenate([rows[common[i]] for i in tr])
    pca = PCA(n_components=PCA_K, svd_solver="randomized", random_state=42).fit(emb[tr_rows])
    F = np.concatenate([mp, pca.transform(emb), logit], axis=1).astype(np.float32)
    Xtr, vtr = win_table([common[i] for i in tr], F)
    Xva, vva = win_table([common[i] for i in va], F)
    clf = LGBMClassifier(**LGBM).fit(Xtr, yv[tr][vtr])
    wp = clf.predict_proba(Xva)[:, 1]
    for j, ci in enumerate(va):
        oof[ci] = wp[vva == j].max()
    print(f"  fold{fold} done ({len(Xtr)} train win)", flush=True)

fid = subject_folds(gv, 5)
if track == 1:
    pred = np.zeros(len(common), int)
    for trm, vam in iter_folds(fid):
        tr, va = np.where(trm)[0], np.where(vam)[0]
        thr = M.tune_threshold(yv[tr], oof[tr]); pred[va] = (oof[va] >= thr).astype(int)
    print(f"\nHSE-wlgbm solo clip macro-F1: {M.primary(track, yv, pred, oof):.3f}", flush=True)
print(f"HSE-wlgbm solo clip AUC: {roc_auc_score(yv, oof):.4f}", flush=True)
pd.DataFrame({"participant": [k[0] for k in common], "video": [k[1] for k in common],
              "oof_hse_wlgbm": oof}).to_csv(CACHE_DIR / f"hse_wlgbm_oof_t{track}.csv", index=False)
print(f"saved hse_wlgbm_oof_t{track}.csv", flush=True)
