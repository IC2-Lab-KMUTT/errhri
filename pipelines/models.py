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


# --------------------------------------------------------------------------- #
#  ROCKET — random convolutional kernels + linear head (sequence model)
# --------------------------------------------------------------------------- #
class RocketClassifier:
    """Dependency-free ROCKET for short multivariate series (the MiniRocket idea without sktime).

    Random 1-D conv kernels over time → PPV + global-max pooling → standardize → logistic. A third
    inductive bias distinct from the GRU (recurrence) and the trees (summary stats): random
    convolutional features, the ERR@HRI-relevant "fast + regularizing on small data" view. PPV
    (fraction of positive activations) is the regularizing feature MiniRocket relies on.

    Input X: (N, L, C). sklearn-style fit / predict_proba.
    """

    def __init__(self, n_kernels=1000, kernel_lengths=(3, 5, 7), seed=0, C=1.0,
                 class_weight="balanced"):
        self.n_kernels, self.kernel_lengths = n_kernels, kernel_lengths
        self.seed, self.C, self.class_weight = seed, C, class_weight
        self._kernels = None

    def _make_kernels(self, L, n_ch):
        rng = np.random.default_rng(self.seed)
        ks = []
        for _ in range(self.n_kernels):
            k = int(rng.choice(self.kernel_lengths))
            max_d = max(1, (L - 1) // (k - 1)) if k > 1 else 1
            d = int(rng.integers(1, max_d + 1))
            if (k - 1) * d + 1 > L:
                d = 1
            ch = int(rng.integers(0, n_ch))
            w = rng.standard_normal(k); w -= w.mean()
            ks.append((ch, w, d, float(rng.standard_normal())))
        return ks

    def _transform(self, X):
        N, L, _ = X.shape
        feats = np.empty((N, len(self._kernels) * 2), np.float32)
        for i, (ch, w, d, b) in enumerate(self._kernels):
            k = len(w); span = (k - 1) * d + 1
            npos = L - span + 1
            if npos < 1:
                feats[:, 2 * i] = 0.0; feats[:, 2 * i + 1] = 0.0; continue
            conv = np.full((N, npos), b, np.float32)
            xc = X[:, :, ch]
            for j in range(k):
                conv += w[j] * xc[:, j * d: j * d + npos]
            feats[:, 2 * i] = (conv > 0).mean(1)        # PPV
            feats[:, 2 * i + 1] = conv.max(1)           # global max
        return feats

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        X = np.nan_to_num(np.asarray(X, np.float32))
        self._kernels = self._make_kernels(X.shape[1], X.shape[2])
        F = self._transform(X)
        self._scaler = StandardScaler().fit(F)
        self._clf = LogisticRegression(max_iter=1000, C=self.C,
                                       class_weight=self.class_weight).fit(self._scaler.transform(F), y)
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(np.asarray(X, np.float32))
        return self._clf.predict_proba(self._scaler.transform(self._transform(X)))


# --------------------------------------------------------------------------- #
#  The zoo: pick a tabular model by name + params -> a zero-arg factory
# --------------------------------------------------------------------------- #
def _logreg(**p):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, class_weight="balanced", **p)


def _rf(scale_pos_weight=None, **p):
    from sklearn.ensemble import RandomForestClassifier
    p.setdefault("n_estimators", 400)
    p.setdefault("class_weight", "balanced_subsample")
    return RandomForestClassifier(**p)  # scale_pos_weight is N/A for RF; swallowed + ignored


# name -> builder(**params) -> estimator. Add your own model here in one line.
MODEL_ZOO = {"xgb": make_xgb, "logreg": _logreg, "rf": _rf}


def make_model(name: str, **params):
    """Return a ZERO-ARG factory `() -> estimator` for `CVEvaluator.run` — the validated default
    config, overridable per call. e.g. `make_model("xgb", max_depth=4, scale_pos_weight=spw)`.

    The temporal GRU is sequence-based (not a FeatureBank model), so it is not in this tabular zoo —
    use `recipes.run_temporal(...)` for it.
    """
    if name not in MODEL_ZOO:
        raise KeyError(f"unknown model '{name}'; options: {sorted(MODEL_ZOO)} (GRU -> run_temporal)")
    builder = MODEL_ZOO[name]
    return lambda: builder(**params)
