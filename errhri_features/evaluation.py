"""Subject-grouped CV evaluation (the "group eval") + late fusion.

`CVEvaluator.run` is the honest scoreboard: subject-grouped K-fold, per-fold threshold tuning on
TRAIN only, primary metric + subject-bootstrap 95% CI + length-leak check. Your model only needs
`fit(X, y)` and `predict_proba(X) -> (n, 2)` (any sklearn-style estimator, or wrap your own).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .splits import subject_folds, iter_folds
from . import metrics as M
from .config import TRACKS


@dataclass
class Report:
    track: int
    primary: float
    ci: tuple
    auc: float
    f1_neg: float
    leak: float
    oof_prob: np.ndarray = field(repr=False)
    oof_pred: np.ndarray = field(repr=False)

    def __str__(self):
        name = TRACKS[self.track]["metric"]
        return (f"[T{self.track}] {name}={self.primary:.3f} CI[{self.ci[0]:.3f},{self.ci[1]:.3f}] "
                f"auc={self.auc:.3f} f1_neg={self.f1_neg:.3f} leak={self.leak:+.3f}")


def _proba1(model, X):
    p = model.predict_proba(X)
    return p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p)


class CVEvaluator:
    def __init__(self, track: int, n_splits: int = 5):
        self.track = track
        self.n_splits = n_splits

    def run(self, model_factory, bank, select="signal", normalize=True, leak_clean=True,
            tune=True) -> Report:
        """model_factory: zero-arg callable returning a fresh estimator each fold."""
        X, y, groups = bank.matrix(select=select, normalize=normalize, leak_clean=leak_clean)
        nfr = bank.n_frames
        return self.run_matrix(model_factory, X, y, groups, nfr, tune=tune)

    def run_matrix(self, model_factory, X, y, groups, n_frames, tune=True) -> Report:
        fid = subject_folds(groups, self.n_splits)
        oof_p = np.zeros(len(y)); oof_pred = np.zeros(len(y), int)
        for tr, va in iter_folds(fid):
            model = model_factory()
            model.fit(X[tr], y[tr])
            p_tr, p_va = _proba1(model, X[tr]), _proba1(model, X[va])
            oof_p[va] = p_va
            thr = M.tune_threshold(y[tr], p_tr) if tune else 0.5
            oof_pred[va] = (p_va >= thr).astype(int)
        vm = M.video_metrics(self.track, y, oof_pred, oof_p)
        ci = M.subject_bootstrap_ci(self.track, groups, y, oof_pred, oof_p)
        from .leak import length_corr
        return Report(self.track, vm["primary"], ci, vm["auc"], vm["f1_neg"],
                      length_corr(oof_p, n_frames), oof_p, oof_pred)


def late_fusion(track, oof_probs: dict, y, groups, n_frames, method="mean"):
    """Fuse per-stream OOF probability vectors. method='mean' (parameter-free) or 'stack'
    (logistic meta-learner, subject-grouped meta-OOF). Returns a Report."""
    from sklearn.linear_model import LogisticRegression
    names = list(oof_probs); P = np.column_stack([oof_probs[n] for n in names])
    fid = subject_folds(groups)
    if method == "mean":
        pf = P.mean(1)
    else:
        pf = np.zeros(len(y))
        for tr, va in iter_folds(fid):
            lr = LogisticRegression(max_iter=500).fit(P[tr], y[tr])
            pf[va] = lr.predict_proba(P[va])[:, 1]
    pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(y[tr], pf[tr])
        pred[va] = (pf[va] >= thr).astype(int)
    vm = M.video_metrics(track, y, pred, pf)
    ci = M.subject_bootstrap_ci(track, groups, y, pred, pf)
    from .leak import length_corr
    return Report(track, vm["primary"], ci, vm["auc"], vm["f1_neg"],
                  length_corr(pf, n_frames), pf, pred)
