"""T1 Direction-B step B1a — temporal REACTION-LOCALIZER on the existing 32-frame `traj` cache (no
re-extraction). Tests the core hypothesis: a model with a LOCALIZATION bias (peak/LSE pooling over
frames) beats the whole-clip GRU and the tabular aggregates, because a bystander reaction is a brief
burst (control = no burst). traj = MediaPipe blendshape trajectory (N,32,16) — weaker than AUs but
dense enough to validate. If this proves out, re-extract dense AU+gaze+pose per-frame for the strong
version. OOF probs are saved for Direction-A fusion into the calibrated stack.

Models (subject-grouped 5-fold + inner early-stop + temperature calibration + official eval):
  MIL_LSE  : per-frame encoder -> per-frame reaction score -> log-sum-exp pool (picks the peak frame)
  GRU_attn : the existing bi-GRU + attention pooling (recurrent baseline, same data) — the A/B test
Both run with and without BASELINE-CONTRAST channels (x_t - clip's own neutral baseline).
"""
import sys, os, subprocess, tempfile
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from errhri_features import metrics as M
from errhri_features.splits import subject_folds, iter_folds
from pipelines.sequences import SequenceBank
from pipelines.official import REPO_EVAL, FPS, WS, SLIDE
from errhri_features.config import CACHE_DIR

torch.set_num_threads(8)
track = int(sys.argv[1]) if len(sys.argv) > 1 else 1
bank = SequenceBank(track).load()
X, y, g = bank.matrix(normalize=True)              # (N, L, C)
X = X.astype(np.float32)
keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str)))
nf = bank.n_frames
print(f"=== T{track}: traj X = {X.shape} (N, L, C) ===", flush=True)


def add_baseline_contrast(Xseq, k=6):
    """Append per-frame deviation from the clip's own neutral baseline (mean of first k frames).
    Removes per-subject expressiveness; highlights the reaction onset. -> (N, L, 2C)."""
    base = Xseq[:, :k].mean(axis=1, keepdims=True)
    return np.concatenate([Xseq, Xseq - base], axis=2).astype(np.float32)


def official(pred, prob, tag):
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, pr, n in zip(keys, y, pred, prob, nf):
        n = int(max(n, ws))
        for f in range(1, n + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        for w in range(max((n - ws) // slide + 1, 1)):
            sub_rows.append((pid, vid, w, int(yp), float(1 - pr), float(pr)))
    d = tempfile.mkdtemp(prefix="t1tmp_")
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
    t = torch.zeros(1, requires_grad=True)
    lt = torch.tensor(logit_tr, dtype=torch.float32); yy = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=60); lossf = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = lossf(lt / t.exp(), yy); loss.backward(); return loss
    opt.step(closure); return float(t.exp().detach())


# ---- models -----------------------------------------------------------------
class MIL_LSE(nn.Module):
    """Per-frame reaction score -> log-sum-exp pool over time (soft max -> peak frame)."""
    def __init__(self, c, h=48, drop=0.4, r=5.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(c, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
        self.r = r
    def forward(self, x):                                  # x: (B, L, C)
        fl = self.enc(x).squeeze(-1)                       # (B, L) per-frame logit
        return torch.logsumexp(self.r * fl, dim=1) / self.r  # (B,) peak-pooled clip logit


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
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
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
        lg_tr = net(torch.tensor(Xtr, dtype=torch.float32)).numpy(); lg_va = net(torch.tensor(Xva, dtype=torch.float32)).numpy()
    T = temperature(lg_tr, ytr)
    return 1 / (1 + np.exp(-lg_tr / T)), 1 / (1 + np.exp(-lg_va / T))


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
print("\n--- temporal reaction-localizer vs GRU (traj 32-frame) ---", flush=True)
oof["mil_lse"]      = evaluate(lambda c: MIL_LSE(c), X,  "MIL_LSE", lr=2e-3, wd=1e-3)
oof["mil_lse_bc"]   = evaluate(lambda c: MIL_LSE(c), Xc, "MIL_LSE+baseline_contrast", lr=2e-3, wd=1e-3)
oof["gru_attn"]     = evaluate(lambda c: GRU_attn(c), X,  "GRU_attn", lr=1e-3, wd=1e-4)
oof["gru_attn_bc"]  = evaluate(lambda c: GRU_attn(c), Xc, "GRU_attn+baseline_contrast", lr=1e-3, wd=1e-4)

# save OOF probs for Direction-A fusion into the calibrated stack
out = pd.DataFrame({"participant": [k[0] for k in keys], "video": [k[1] for k in keys]})
for n, p in oof.items():
    out[f"oof_{n}"] = p
out.to_csv(CACHE_DIR / f"temporal_oof_t{track}.csv", index=False)
print(f"\nsaved OOF -> temporal_oof_t{track}.csv", flush=True)
print("Reference: stack 0.685 | FT single 0.633 | tabular-aggregate ceiling ~0.63", flush=True)
