"""T1 proper single models — open-source, reproducible, NOT an ensemble of separate models.
Goal: a real model that rivals the 0.685 calibrated stack without being a stack. Candidates:
  C2) tuned mid-fusion MLP (per-modality encoder -> fuse -> head), early-stopped
  E ) FT-Transformer (rtdl_revisiting_models, MIT) on MI-top-128 feats — per-feature tokenization
  F ) TabM (MIT) — ONE model with k=32 parameter-efficient internal heads (ensemble-in-a-model)
All: subject-grouped 5-fold, subject-grouped INNER early stopping (vital at N=1319), TEMPERATURE
calibration (fit scalar T on train logits — frozen, deployable, no test labels), per-fold threshold,
official eval.py. Reference: per-stream calibrated stack = 0.685; best block-aware single so far = 0.632.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

torch.set_num_threads(8)
track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ALL_MODS = ["au", "au_graph", "gaze", "pose", "blend", "audio", "faceemb"]
mods = [m for m in ALL_MODS if (CACHE_DIR / f"{m}_t{track}.csv").exists()]

blocks, slices, off = [], [], 0
y = g = nf = keys = None
for m in mods:
    bank = FeatureBank(track, [m]).load()
    Xm, y, g = bank.matrix(select="all", leak_clean=True, normalize=True)
    if keys is None:
        keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str))); nf = bank.n_frames
    blocks.append(Xm.astype(np.float32)); slices.append(slice(off, off + Xm.shape[1])); off += Xm.shape[1]
X = np.hstack(blocks)
print(f"=== T{track}: X = {X.shape}, blocks = {[s.stop - s.start for s in slices]} ===", flush=True)


def official(pred, prob, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, pr, n in zip(keys, y, pred, prob, nf):
        n = int(max(n, ws))
        for f in range(1, n + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((n - ws) // slide + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - pr), float(pr)))
    d = tempfile.mkdtemp(prefix="t1prop_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True); print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-300:])


def temperature(logit_tr, ytr):
    """Fit a single scalar T>0 on train logits (frozen at deploy, no test labels)."""
    t = torch.zeros(1, requires_grad=True)  # log T
    lt = torch.tensor(logit_tr, dtype=torch.float32); yy = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=60)
    lossf = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = lossf(lt / t.exp(), yy); loss.backward(); return loss
    opt.step(closure)
    return float(t.exp().detach())


def train_torch(build, forward_logits, Xtr, ytr, Xva, max_ep=200, patience=25, lr=2e-3, wd=3e-4, bs=256):
    """Generic trainer: subject-grouped inner split -> early stop by inner AUC -> temperature scale.
    Returns calibrated (prob_tr_full, prob_va)."""
    # inner subject-grouped split (use group ids carried alongside via closure var gtr)
    gtr = train_torch.gtr
    ufid = subject_folds(gtr, 4)
    in_tr = np.where(ufid != 0)[0]; in_va = np.where(ufid == 0)[0]
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    dev = "cpu"
    net = build().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(spw)))
    Xt = torch.tensor(Xtr[in_tr]); yt = torch.tensor(ytr[in_tr], dtype=torch.float32)
    Xv = torch.tensor(Xtr[in_va]); yv = ytr[in_va]
    best_auc, best_state, bad = -1, None, 0
    for ep in range(max_ep):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(forward_logits(net, Xt[idx]), yt[idx]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(forward_logits(net, Xv)).numpy()
        auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        lg_tr = forward_logits(net, torch.tensor(Xtr)).numpy()
        lg_va = forward_logits(net, torch.tensor(Xva)).numpy()
    T = temperature(lg_tr, ytr)
    return 1 / (1 + np.exp(-lg_tr / T)), 1 / (1 + np.exp(-lg_va / T))


def evaluate(build, forward_logits, tag, **kw):
    fid = subject_folds(g, 5); prob = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        train_torch.gtr = g[tr]
        sel = kw.get("select_fn", lambda a, b: np.arange(a.shape[1]))(X[tr], y[tr])
        b = (lambda d=sel: build(len(d)))  # build takes n_features
        ptr, pva = train_torch(b, forward_logits, X[tr][:, sel], y[tr], X[va][:, sel],
                               max_ep=kw.get("max_ep", 200), patience=kw.get("patience", 25),
                               lr=kw.get("lr", 2e-3), wd=kw.get("wd", 3e-4))
        prob[va] = pva
        thr = M.tune_threshold(y[tr], ptr); pred[va] = (pva >= thr).astype(int)
    print(f"  [{tag}] clip-primary {M.primary(track, y, pred, prob):.3f}", flush=True)
    official(pred, prob, tag)


# ---- C2) tuned mid-fusion MLP (per-block encoder) ---------------------------
def build_midfusion(_nf, h=24, drop=0.4):
    dims = [s.stop - s.start for s in slices]
    class MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList([nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)) for d in dims])
            self.head = nn.Sequential(nn.Linear(h * len(dims), h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
        def forward(self, x):
            return self.head(torch.cat([e(x[:, s]) for e, s in zip(self.enc, slices)], 1)).squeeze(1)
    return MF()


def mi_top(k):
    def f(Xtr, ytr):
        mi = mutual_info_classif(Xtr, ytr, random_state=0)
        return np.argsort(-mi)[:k]
    return f


# ---- E) FT-Transformer ------------------------------------------------------
def build_ft(nfeat):
    from rtdl_revisiting_models import FTTransformer
    return FTTransformer(n_cont_features=nfeat, cat_cardinalities=[], d_out=1, **FTTransformer.get_default_kwargs())


def ft_logits(net, x):
    return net(x, None).squeeze(1)


# ---- F) TabM ----------------------------------------------------------------
def build_tabm(nfeat):
    from tabm import TabM
    return TabM(n_num_features=nfeat, cat_cardinalities=[], d_out=1, num_embeddings=None,
                n_blocks=3, d_block=512, dropout=0.1, arch_type="tabm", k=32, start_scaling_init="random-signs")


def tabm_logits(net, x):
    out = net(x)  # (B, k, 1)
    return out.mean(1).squeeze(1)  # average the k heads -> single logit


print("\n--- proper open-source single models (early-stopped + temperature-calibrated) ---", flush=True)
evaluate(lambda nf=None: build_midfusion(nf), lambda net, x: net(x), "C2_midfusion_MLP_tuned",
         max_ep=300, patience=30, lr=2e-3, wd=1e-3)
evaluate(build_ft, ft_logits, "E_FTTransformer_top128", select_fn=mi_top(128), max_ep=200, patience=25, lr=1e-3, wd=1e-5)
evaluate(build_tabm, tabm_logits, "F_TabM_allfeat", max_ep=200, patience=25, lr=2e-3, wd=3e-4)
print("\nReference: stack 0.685 | flat L1 0.599 | block-norm L1 0.619 | midfusion(untuned) 0.632", flush=True)
