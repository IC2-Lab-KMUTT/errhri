"""Granular timing-only signal check — reproduces TIMING_VERDICT.

Two questions: (1) how far can the *timing* features alone go (onset/peak/magnitude/decay of the
reaction), with the duration leak stripped? (2) which timing STAT carries it? It pools every
`<channel>.<stat>` timing feature, scores the timing-only matrix subject-grouped + length-clean,
then ranks the timing stats by univariate separability. Re-derives: reaction magnitude (auc),
onset latency, amplitude and early-bias are the load-bearing timing signal on T1.

    python -m analysis.timing_features
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank, CVEvaluator
from errhri_features.featurebank import per_subject_norm
from errhri_features.leak import length_corr

MODALITIES = ["au", "blend"]   # blend carries the granular TIMING_CHANNELS; falls back to au-only


def _lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, C=0.5)


def run(track):
    try:
        bank = FeatureBank(track, MODALITIES).load()
    except FileNotFoundError:
        bank = FeatureBank(track, ["au"]).load()
    cols = bank.feature_columns()
    timing = [c for c in cols if "." in c]           # timing features use a dotted suffix
    if not timing:
        print(f"Track {track}: no timing features cached (need the `blend` extractor).")
        return
    idx = [cols.index(c) for c in timing]
    Xall = np.nan_to_num(bank.df[cols].to_numpy(float))
    X = per_subject_norm(Xall[:, idx], bank.groups)
    y, nfr = bank.y, bank.n_frames

    rep = CVEvaluator(track).run_matrix(_lr, X, y, bank.groups, nfr)
    print(f"\n=== Track {track}: timing-only ({len(timing)} feats) ===")
    print(f"  multivariate {rep}")

    # rank by the timing STAT (auc, onset_frac, peak_frac, ...) aggregated across channels
    stats = {}
    for j, c in enumerate(timing):
        stat = c.split(".")[-1]
        col = X[:, j]
        sep = 0.5 if np.std(col) < 1e-9 else max(roc_auc_score(y, col), 1 - roc_auc_score(y, col))
        stats.setdefault(stat, []).append((sep, abs(length_corr(Xall[:, idx[j]], nfr))))
    print(f"  {'stat':12} {'mean sep':>9} {'max sep':>8} {'mean leak':>10}")
    for stat, vals in sorted(stats.items(), key=lambda kv: -np.mean([v[0] for v in kv[1]])):
        seps = [v[0] for v in vals]; leaks = [v[1] for v in vals]
        print(f"  {stat:12} {np.mean(seps):>9.3f} {np.max(seps):>8.3f} {np.mean(leaks):>10.2f}")


def main():
    for t in (1, 2):
        run(t)


if __name__ == "__main__":
    main()
