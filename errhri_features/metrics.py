"""Official metrics + subject-level confidence intervals + threshold tuning.

Track 1 = macro-F1 (failure & control treated symmetrically -> the rare control class matters);
Track 2 = AUC. Clip labels are video-level, so a clip is one evaluation unit. CIs are computed by
bootstrapping over SUBJECTS (not clips) because clips within a subject are correlated.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from .config import TRACKS


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def f1_negative(y, pred) -> float:
    """F1 of the minority/negative (control / well-handled) class — the hard one on T1."""
    return float(f1_score(y, pred, pos_label=0, zero_division=0))


def auc(y, prob) -> float:
    try:
        return float(roc_auc_score(y, prob))
    except ValueError:
        return 0.5


def primary(track: int, y, pred, prob) -> float:
    return macro_f1(y, pred) if TRACKS[track]["metric"] == "macro_f1" else auc(y, prob)


def video_metrics(track: int, y, pred, prob) -> dict:
    return dict(primary=primary(track, y, pred, prob), macro_f1=macro_f1(y, pred),
                auc=auc(y, prob), f1_neg=f1_negative(y, pred))


def tune_threshold(y_train, p_train) -> float:
    """Pick the probability threshold that maximizes macro-F1 on TRAIN (never on val)."""
    grid = np.percentile(p_train, np.linspace(5, 95, 46))
    return float(max(grid, key=lambda t: f1_score(y_train, (p_train >= t).astype(int),
                                                   average="macro", zero_division=0)))


def subject_bootstrap_ci(track, groups, y, pred, prob, n_boot=1000, seed=0):
    """95% CI for the primary metric by resampling SUBJECTS with replacement."""
    groups = np.asarray(groups); y = np.asarray(y)
    pred = np.asarray(pred); prob = np.asarray(prob)
    subs = np.unique(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([np.where(groups == s)[0] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(primary(track, y[idx], pred[idx], prob[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
