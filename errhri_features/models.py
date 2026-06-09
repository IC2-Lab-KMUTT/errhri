"""The model zoo — the estimators we actually validated, as drop-in `CVEvaluator` models.

Anything with `fit(X, y)` + `predict_proba(X) -> (n, 2)` plugs into `CVEvaluator`/`late_fusion`.
Two kinds live here:

* **`make_xgb`** — the gradient-boosted tree on the *aggregated* `FeatureBank` matrix. This is the
  `au` / `audio` / `embed` / facial-static streams (the bulk of the fusion ensemble).
* **`ClipGRUClassifier`** — the whole-clip ordered-temporal GRU on the *raw resampled trajectory*
  (`SequenceBank`, not FeatureBank). This is the `facial_gru` stream — the strongest single T1
  stream (macro-F1 0.623) and the only model that sees temporal *order* rather than summary stats.
  Fusing it with the tree streams is what lifts T1 to 0.674 (see signal_map.FUSION).

The GRU runs on CPU by default (the validated setup: extraction GPUs are often too old for the
venv's CUDA torch). Pass device="cuda" if your torch/GPU match.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Tree stream — aggregated FeatureBank matrix
# --------------------------------------------------------------------------- #
def make_xgb(scale_pos_weight=None, **kw):
    """The validated XGBoost config for the aggregated-feature streams.

    `scale_pos_weight` handles T1's 87/13 imbalance — pass `(y==0).sum()/(y==1).sum()`.
    Returns a fresh estimator; hand the *callable* `lambda: make_xgb(spw)` to CVEvaluator.run.
    """
    from xgboost import XGBClassifier
    params = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
                  tree_method="hist", eval_metric="logloss")
    params.update(kw)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**params)


# --------------------------------------------------------------------------- #
#  Whole-clip temporal GRU — raw resampled trajectory (SequenceBank)
# --------------------------------------------------------------------------- #
class ClipGRUClassifier:
    """Ordered-temporal GRU over a clip's resampled channel trajectory.

    Input X is a (N, L, C) array (N clips, L resampled timesteps, C channels) — exactly what
    `SequenceBank.matrix()` returns. A bi-GRU encodes the sequence, attention pools it, and a
    linear head emits one logit. Order matters: the failure smile arrives *later* and decays
    differently than the control smile, which the summary-stat trees can't see.

    sklearn-style, so it drops straight into CVEvaluator.run_matrix(... X=seq ...).
    """

    def __init__(self, hidden=64, layers=1, epochs=40, lr=1e-3, dropout=0.3, bidir=True,
                 batch=64, pos_weight=None, seed=0, device="cpu"):
        self.hidden, self.layers, self.epochs, self.lr = hidden, layers, epochs, lr
        self.dropout, self.bidir, self.batch = dropout, bidir, batch
        self.pos_weight, self.seed, self.device = pos_weight, seed, device
        self._net = None

    # -- internal torch module ------------------------------------------------
    def _build(self, n_ch):
        import torch
        from torch import nn

        class _Net(nn.Module):
            def __init__(s, c, h, nl, drop, bidir):
                super().__init__()
                s.gru = nn.GRU(c, h, nl, batch_first=True, bidirectional=bidir,
                               dropout=drop if nl > 1 else 0.0)
                d = h * (2 if bidir else 1)
                s.att = nn.Linear(d, 1)               # attention pooling over time
                s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 1))

            def forward(s, x):
                h, _ = s.gru(x)                       # (B, L, d)
                w = torch.softmax(s.att(h).squeeze(-1), dim=1).unsqueeze(-1)
                return s.head((h * w).sum(1)).squeeze(-1)   # (B,)

        torch.manual_seed(self.seed)
        return _Net(n_ch, self.hidden, self.layers, self.dropout, self.bidir).to(self.device)

    def fit(self, X, y):
        import torch
        from torch import nn
        X = np.nan_to_num(np.asarray(X, np.float32)); y = np.asarray(y, np.float32)
        self._net = self._build(X.shape[2])
        pw = self.pos_weight
        if pw is None:
            pos = max(1.0, float((y == 1).sum()))
            pw = float((y == 0).sum()) / pos
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=self.device))
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        Xt = torch.tensor(X, device=self.device); yt = torch.tensor(y, device=self.device)
        n = len(Xt); rng = np.random.default_rng(self.seed)
        self._net.train()
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for i in range(0, n, self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                out = self._net(Xt[idx])
                loss_fn(out, yt[idx]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        X = np.nan_to_num(np.asarray(X, np.float32))
        self._net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self._net(torch.tensor(X, device=self.device))).cpu().numpy()
        return np.column_stack([1.0 - p, p])
