"""Subject-grouped cross-validation splits.

CRITICAL: never split clips from the same participant across train/val — that leaks subject
identity and inflates scores. Always group by participant.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold


def subject_folds(groups, n_splits: int = 5) -> np.ndarray:
    """Return a per-sample fold id (0..n_splits-1) under subject-grouped K-fold.

    Deterministic for a fixed `groups` order, so every model/stream evaluated with the same
    `groups` gets identical folds — required for fair fusion / complementarity comparisons.
    """
    groups = np.asarray(groups)
    fid = np.full(len(groups), -1, int)
    dummy = np.zeros((len(groups), 1))
    for k, (_, va) in enumerate(GroupKFold(n_splits).split(dummy, groups=groups)):
        fid[va] = k
    return fid


def iter_folds(fid: np.ndarray):
    """Yield (train_mask, val_mask) for each fold."""
    for k in range(int(fid.max()) + 1):
        yield fid != k, fid == k
