"""Length-leak guard.

Track 1 is delivered as raw mp4 -> frame_count is proportional to clip duration, and
duration ALONE scores macro-F1 ~0.70 (control clips are longer than failure clips). That is the
single biggest trap in this dataset: any feature correlated with `n_frames` is a duration proxy,
not facial skill, and using it is cheating. These helpers flag/strip such features.
"""
from __future__ import annotations
import numpy as np
from .config import LEAK_THRESHOLD


def length_corr(x, n_frames) -> float:
    """Pearson corr between a feature (or OOF score) and clip length."""
    x = np.asarray(x, float); n = np.asarray(n_frames, float)
    if np.std(x) < 1e-12 or np.std(n) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, n)[0, 1])


def is_clean(x, n_frames, threshold: float = LEAK_THRESHOLD) -> bool:
    return abs(length_corr(x, n_frames)) <= threshold


def clean_columns(X, columns, n_frames, threshold: float = LEAK_THRESHOLD):
    """Return (indices, names) of columns whose |corr with n_frames| <= threshold."""
    keep_i, keep_n = [], []
    X = np.asarray(X, float)
    for j, c in enumerate(columns):
        if abs(length_corr(X[:, j], n_frames)) <= threshold:
            keep_i.append(j); keep_n.append(c)
    return keep_i, keep_n


def report(oof_score, n_frames) -> str:
    rho = length_corr(oof_score, n_frames)
    tag = "LEAK" if abs(rho) > LEAK_THRESHOLD else "clean"
    return f"length-leak corr={rho:+.3f} [{tag}]"
