"""T1 Direction-B (strong channel) — temporal reaction-localizer on the NEW dense sequences.

run_t1_temporal.py validated the idea on the weak 32-frame blendshape `traj` (best gru_attn+bc
AUC 0.724, fused stack 0.696). This runs the SAME models on the STRONG dense channels that B0 just
saved: au_seq (48 frames x 41 OpenGraphAU probabilities) + gaze_seq (48 frames x 10 gaze/EAR/head-
pose). Concatenated -> (N, 48, 51). Same honest harness: subject-grouped 5-fold + inner early-stop +
temperature calibration + official eval. OOF saved to temporal_dense_oof_t{track}.csv for fusion.

  python run_t1_temporal_dense.py [track]
"""
import sys, os, re, subprocess, tempfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from errhri_features import metrics as M
from errhri_features.splits import subject_folds, iter_folds
from errhri_features.datasets import load_index
from errhri_features.config import CACHE_DIR
from pipelines.sequences import SequenceBank
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE

torch.set_num_threads(8)
track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
_CELL = re.compile(r"^t(\d+)__(.+)$")


def load_seq_cache(name, track):
    """Read a t{tt}__{channel} wide cache -> (df_keys, tensor (N,L,C), channels)."""
    fp = CACHE_DIR / f"{name}_t{track}.csv"
    df = pd.read_csv(fp)
    df["participant"] = df.participant.astype(str); df["video"] = df.video.astype(str)
    ts, chs = set(), []
    for c in df.columns:
        m = _CELL.match(str(c))
        if m:
            ts.add(int(m.group(1)))
            if m.group(2) not in chs:
                chs.append(m.group(2))
    L = max(ts) + 1
    cols = [f"t{t:02d}__{ch}" for t in range(L) for ch in chs]
    flat = np.nan_to_num(df[cols].to_numpy(np.float32))
    X = flat.reshape(len(df), L, len(chs))
    keys = list(zip(df.participant, df.video))
    return keys, X, chs, L


# ---- assemble dense (N, L, C) aligned on the official index ------------------
idx = load_index(track)
idx["participant"] = idx.participant.astype(str); idx["video"] = idx.video.astype(str)
caches = {}
for nm in ("au_seq", "gaze_seq"):
    keys, X, chs, L = load_seq_cache(nm, track)
    caches[nm] = (dict(zip(keys, X)), chs, L)
    print(f"  [{nm}] {X.shape} channels={len(chs)}", flush=True)
L = caches["au_seq"][2]
assert caches["gaze_seq"][2] == L, "au_seq / gaze_seq frame length mismatch"

# keep clips present in BOTH dense caches + the index
common = [(p, v) for p, v in zip(idx.participant, idx.video)
          if (p, v) in caches["au_seq"][0] and (p, v) in caches["gaze_seq"][0]]
sub = idx.set_index(["participant", "video"]).loc[common].reset_index()
y = sub.label.to_numpy(int)
g = sub.participant.to_numpy()
nf = sub.n_frames.to_numpy(float)
keys = common
X = np.concatenate(
    [np.stack([caches["au_seq"][0][k] for k in keys]),
     np.stack([caches["gaze_seq"][0][k] for k in keys])], axis=2).astype(np.float32)  # (N, L, 51)
X = SequenceBank._subject_norm(X, g).astype(np.float32)                # label-free per-subject z
print(f"=== T{track}: dense X = {X.shape} (N, L, C={X.shape[2]}), "
      f"{int((y==0).sum())} control / {int((y==1).sum())} failure ===", flush=True)


def add_baseline_contrast(Xseq, k=8):
    base = Xseq[:, :k].mean(axis=1, keepdims=True)
    return np.concatenate([Xseq, Xseq - base], axis=2).astype(np.float32)


def temperature(logit_tr, ytr):
    t = torch.zeros(1, requires_grad=True)
    lt = torch.tensor(logit_tr, dtype=torch.float32); yy = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=60); lossf = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = lossf(lt / t.exp(), yy); loss.backward(); return loss
    opt.step(closure); return float(t.exp().detach())


class MIL_LSE(nn.Module):
    def __init__(self, c, h=48, drop=0.4, r=5.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
        self.r = r
    def forward(self, x):
        fl = self.enc(x).squeeze(-1)
        return torch.logsumexp(self.r * fl, dim=1) / self.r


class GRU_attn(nn.Module):
    def __init__(self, c, h=64, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(c, h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * h, 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * h, 1))
    def forward(self, x):
        hs, _ = self.gru(x)
        w = torch.softmax(self.att(hs).squeeze(-1), 1).unsqueeze(-1)
        return self.head((hs * w).sum(1)).squeeze(-1)


def train_seq(build, Xtr, ytr, Xva, gtr, max_ep=300, patience=30, lr=2e-3, wd=1e-3, bs=64):
    ufid = subject_folds(gtr, 4)
    in_tr = np.where(ufid != 0)[0]; in_va = np.where(ufid == 0)[0]
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    net = build(Xtr.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(spw)))
    Xt = torch.tensor(Xtr[in_tr], dtype=torch.float32); yt = torch.tensor(ytr[in_tr], dtype=torch.float32)
    Xv = torch.tensor(Xtr[in_va], dtype=torch.float32); yv = ytr[in_va]
    best, best_state, bad = -1, None, 0
    for ep in range(max_ep):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx_b = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xt[idx_b]), yt[idx_b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv)).numpy()
        auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        if auc > best + 1e-4:
            best, best_state, bad = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        lg_tr = net(torch.tensor(Xtr, dtype=torch.float32)).numpy()
        lg_va = net(torch.tensor(Xva, dtype=torch.float32)).numpy()
    T = temperature(lg_tr, ytr)
    return 1 / (1 + np.exp(-lg_tr / T)), 1 / (1 + np.exp(-lg_va / T))


def official(pred, prob, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, pr, n in zip(keys, y, pred, prob, nf):
        n = int(max(n, ws))
        for f in range(1, n + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((n - ws) // slide + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - pr), float(pr)))
    d = tempfile.mkdtemp(prefix="t1dn_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    pd.DataFrame(gt_rows, columns=["participant_id","video_id","frame_id","y_true"]).to_csv(gtp, index=False)
    pd.DataFrame(sub_rows, columns=["participant_id","video_id","window_id","y_pred","y_prob_0","y_prob_1"]).to_csv(subp, index=False)
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(f"\n########## {tag} ##########", flush=True); print(out.stdout)
    if out.returncode != 0:
        print("[stderr]", out.stderr[-300:])


def evaluate(build, Xseq, tag, **kw):
    fid = subject_folds(g, 5); prob = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        ptr, pva = train_seq(build, Xseq[tr], y[tr], Xseq[va], g[tr], **kw)
        prob[va] = pva
        thr = M.tune_threshold(y[tr], ptr); pred[va] = (pva >= thr).astype(int)
    print(f"  [{tag}] clip-primary {M.primary(track, y, pred, prob):.3f}  AUC {roc_auc_score(y, prob):.3f}", flush=True)
    official(pred, prob, tag)
    return prob


Xc = add_baseline_contrast(X)
oof = {}
print("\n--- dense temporal reaction-localizer (au_seq+gaze_seq, 48-frame x 51ch) ---", flush=True)
oof["mil_lse"]     = evaluate(lambda c: MIL_LSE(c),  X,  "MIL_LSE", lr=2e-3, wd=1e-3)
oof["mil_lse_bc"]  = evaluate(lambda c: MIL_LSE(c),  Xc, "MIL_LSE+baseline_contrast", lr=2e-3, wd=1e-3)
oof["gru_attn"]    = evaluate(lambda c: GRU_attn(c), X,  "GRU_attn", lr=1e-3, wd=1e-4)
oof["gru_attn_bc"] = evaluate(lambda c: GRU_attn(c), Xc, "GRU_attn+baseline_contrast", lr=1e-3, wd=1e-4)

out = pd.DataFrame({"participant": [k[0] for k in keys], "video": [k[1] for k in keys]})
for n, p in oof.items():
    out[f"oof_{n}"] = p
out.to_csv(CACHE_DIR / f"temporal_dense_oof_t{track}.csv", index=False)
print(f"\nsaved OOF -> temporal_dense_oof_t{track}.csv", flush=True)
print("Reference: weak-traj gru_attn_bc AUC 0.724 / mF1 0.605 ; fused stack 0.696", flush=True)
