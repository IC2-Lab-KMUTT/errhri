"""Per-semantic-dimension signal, split static vs dynamics — reproduces DIMENSION_VERDICT.

For each facial/behavioural dimension (smile, brow, eye, mouth/jaw, nose, gaze, head-pose,
expression, audio) it fits a subject-grouped multivariate model on JUST that dimension's features
and reports the AUC, separately for the static levels and the dynamics. The lesson it re-derives:
on Track 1 the dynamics beat the static levels almost everywhere; on Track 2 only smile clears
chance by much.

    python -m analysis.dimension_breakdown
"""
from __future__ import annotations
import numpy as np
from errhri_features import FeatureBank, CVEvaluator
from errhri_features.featurebank import per_subject_norm
from errhri_features.leak import clean_columns
from .dimensions import dimension_of, stat_family

MODALITIES = ["au", "audio"]


def _lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, C=0.5)


def _subset_auc(track, bank, cols, leak_clean=True):
    if not cols:
        return None
    idx = [bank.feature_columns().index(c) for c in cols]
    Xall = np.nan_to_num(bank.df[bank.feature_columns()].to_numpy(float))
    X = Xall[:, idx]
    names = list(cols)
    if leak_clean:
        keep, names = clean_columns(X, names, bank.n_frames)
        if not keep:
            return None
        X = X[:, keep]
    X = per_subject_norm(X, bank.groups)
    rep = CVEvaluator(track).run_matrix(_lr, X, bank.y, bank.groups, bank.n_frames)
    return rep.auc, len(names)


def run(track):
    bank = FeatureBank(track, MODALITIES).load()
    audio = set(bank.audio_cols)
    cols = bank.feature_columns()
    dims = {}
    for c in cols:
        dims.setdefault(dimension_of(c, audio), []).append(c)
    print(f"\n=== Track {track}: per-dimension AUC (subject-grouped, length-clean) ===")
    print(f"{'dimension':14} {'all':>6} {'static':>8} {'dynamics':>9}   n")
    rows = []
    for dim, dcols in sorted(dims.items()):
        if dim in ("embed", "other"):
            continue
        st = [c for c in dcols if stat_family(c) == "static"]
        dy = [c for c in dcols if stat_family(c) in ("dynamics", "timing")]
        a_all = _subset_auc(track, bank, dcols)
        a_st = _subset_auc(track, bank, st)
        a_dy = _subset_auc(track, bank, dy)
        f = lambda r: f"{r[0]:.3f}" if r else "  ·  "
        print(f"{dim:14} {f(a_all):>6} {f(a_st):>8} {f(a_dy):>9}   {len(dcols)}")
        rows.append((dim, a_all, a_st, a_dy))
    return rows


def main():
    for t in (1, 2):
        run(t)


if __name__ == "__main__":
    main()
