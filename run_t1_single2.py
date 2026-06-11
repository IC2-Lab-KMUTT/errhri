"""T1 single-model round 2 — the FAIR contest. run_t1_single showed flat models lose to the stack,
but that's because they throw away block structure. Here we test SINGLE models that RESPECT modality
blocks (the stack's only real advantage as a prior), to see if one model can match 0.685 without the
per-stream ensemble. Same caches, same subject folds, same official eval.py.

Each modality is loaded + per-subject-normalized + leak_cleaned SEPARATELY (exactly like a stream),
then concatenated with KNOWN block boundaries. Models:
  A) per-block PCA -> concat -> L1-logreg     (block-aware linear, dim-reduced)
  B) per-block L2-normalize -> L1-logreg       (block-balanced so wide blocks don't hog the L1 budget)
  C) mid-fusion MLP: per-modality encoder (Linear->ReLU->Dropout) -> concat -> head   (JOINT, torch)
  D) TabPFN v2 on MI-top-300 (if installed)    (small-N tabular foundation model)
All selection/fitting is TRAIN-FOLD ONLY. Reference: calibrated stack = 0.685.
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ALL_MODS = ["au", "au_graph", "gaze", "pose", "blend", "audio", "faceemb"]
mods = [m for m in ALL_MODS if (CACHE_DIR / f"{m}_t{track}.csv").exists()]

# per-modality matrices (each normalized + leak_cleaned like a stream) -> concat with block slices
blocks, slices, off = [], [], 0
y = g = nf = keys = None
for m in mods:
    bank = FeatureBank(track, [m]).load()
    Xm, y, g = bank.matrix(select="all", leak_clean=True, normalize=True)
    if y is not None and keys is None:
        keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str)))
        nf = bank.n_frames
    blocks.append(Xm); slices.append((m, slice(off, off + Xm.shape[1]))); off += Xm.shape[1]
X = np.hstack(blocks)
print(f"=== T{track}: X = {X.shape}, blocks = {[(m, s.stop - s.start) for m, s in slices]} ===", flush=True)


def official(pred, prob, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, pr, n in zip(keys, y, pred, prob, nf):
        n = int(max(n, ws))
        for f in range(1, n + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((n - ws) // slide + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - pr), float(pr)))
    d = tempfile.mkdtemp(prefix="t1s2_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True); print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-300:])


def evaluate(predict_fold, tag):
    """predict_fold(Xtr,ytr,Xva)->prob_va. Subject-grouped 5-fold; threshold tuned on train."""
    fid = subject_folds(g, 5); prob = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        ptr, pva = predict_fold(X[tr], y[tr], X[va])
        prob[va] = pva
        thr = M.tune_threshold(y[tr], ptr)
        pred[va] = (pva >= thr).astype(int)
    print(f"  [{tag}] clip-primary {M.primary(track, y, pred, prob):.3f}", flush=True)
    official(pred, prob, tag)


# ---- A) per-block PCA -> L1-logreg ------------------------------------------
def pca_l1(Xtr, ytr, Xva, k=15):
    Ztr, Zva = [], []
    for _, sl in slices:
        kk = min(k, sl.stop - sl.start)
        p = PCA(n_components=kk, random_state=0).fit(Xtr[:, sl])
        Ztr.append(p.transform(Xtr[:, sl])); Zva.append(p.transform(Xva[:, sl]))
    Ztr, Zva = np.hstack(Ztr), np.hstack(Zva)
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000,
                             class_weight="balanced").fit(Ztr, ytr)
    return clf.predict_proba(Ztr)[:, 1], clf.predict_proba(Zva)[:, 1]


# ---- B) per-block L2-normalize -> L1-logreg ---------------------------------
def blockscaled_l1(Xtr, ytr, Xva):
    Xtr2, Xva2 = Xtr.copy(), Xva.copy()
    for _, sl in slices:
        nrm = np.linalg.norm(Xtr[:, sl], axis=0, keepdims=True) / np.sqrt(max(Xtr.shape[0], 1))
        nrm[nrm < 1e-8] = 1.0
        Xtr2[:, sl] = Xtr[:, sl] / nrm; Xva2[:, sl] = Xva[:, sl] / nrm
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000,
                             class_weight="balanced").fit(Xtr2, ytr)
    return clf.predict_proba(Xtr2)[:, 1], clf.predict_proba(Xva2)[:, 1]


# ---- C) mid-fusion MLP (per-block encoder -> fuse -> head) -------------------
def midfusion_mlp(Xtr, ytr, Xva, h=16, epochs=80, wd=1e-3, drop=0.5, lr=1e-3):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    dims = [sl.stop - sl.start for _, sl in slices]
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    class MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList([nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)) for d in dims])
            self.head = nn.Sequential(nn.Linear(h * len(dims), h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))

        def forward(self, x):
            zs = [enc(x[:, sl]) for enc, (_, sl) in zip(self.enc, slices)]
            return self.head(torch.cat(zs, 1)).squeeze(1)

    dev = "cpu"  # fusion venv torch lacks cuBLAS support for the Pascal 1080Ti; matrices are tiny
    net = MF().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(spw), device=dev))
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.tensor(Xva, dtype=torch.float32, device=dev)
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(xt), yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(net(xt)).cpu().numpy(); pva = torch.sigmoid(net(xv)).cpu().numpy()
    return ptr, pva


# ---- D) TabPFN (optional) ---------------------------------------------------
def tabpfn_top(Xtr, ytr, Xva, k=300):
    from tabpfn import TabPFNClassifier
    mi = mutual_info_classif(Xtr, ytr, random_state=0)
    sel = np.argsort(-mi)[:k]
    clf = TabPFNClassifier(device="cpu")
    clf.fit(Xtr[:, sel], ytr)
    return clf.predict_proba(Xtr[:, sel])[:, 1], clf.predict_proba(Xva[:, sel])[:, 1]


def guarded(fn, tag):
    try:
        evaluate(fn, tag)
    except Exception as e:
        import traceback
        print(f"  [{tag}] ERR {type(e).__name__}: {str(e)[:150]}", flush=True)
        traceback.print_exc()


print("\n--- single models that RESPECT block structure ---")
guarded(lambda a, b, c: pca_l1(a, b, c, k=15), "A_blockPCA15_L1logreg")
guarded(blockscaled_l1, "B_blockscaled_L1logreg")
guarded(lambda a, b, c: midfusion_mlp(a, b, c), "C_midfusion_MLP")
import importlib.util as u
if u.find_spec("tabpfn"):
    guarded(lambda a, b, c: tabpfn_top(a, b, c, k=300), "D_TabPFN_top300")
else:
    print("  [D_TabPFN] not installed — skipped")

print("\nReference: per-stream calibrated stack = 0.685 | best flat single (L1 all feats) = 0.599")
