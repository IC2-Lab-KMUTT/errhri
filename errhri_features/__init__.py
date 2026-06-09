"""errhri_features — shared feature-extraction + honest-evaluation toolkit for ERR@HRI 3.0.

Quickstart:
    from errhri_features import FeatureBank, CVEvaluator
    from xgboost import XGBClassifier

    bank = FeatureBank(track=1, modalities=["au", "audio", "embed"]).load()
    ev = CVEvaluator(track=1)
    report = ev.run(lambda: XGBClassifier(n_estimators=300, max_depth=3,
                                          scale_pos_weight=6.6, tree_method="hist"),
                    bank, select="signal", leak_clean=True)
    print(report)   # subject-grouped macro-F1 + 95% CI + AUC + length-leak

See README.md for the full API and the curated signal map (SIGNAL_INVENTORY.md).
"""
from .featurebank import FeatureBank, per_subject_norm
from .evaluation import CVEvaluator, Report, late_fusion
from .splits import subject_folds, iter_folds
from . import metrics, leak, signal_map, submission, config

__all__ = ["FeatureBank", "per_subject_norm", "CVEvaluator", "Report", "late_fusion",
           "subject_folds", "iter_folds", "metrics", "leak", "signal_map", "submission", "config"]
