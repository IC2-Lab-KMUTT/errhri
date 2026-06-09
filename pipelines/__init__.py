"""pipelines — the modifiable model + recipe layer that sits ON TOP of the core.

The core (`errhri_features`) turns video into signal features and evaluates honestly, and it stays
stable. This package is where you experiment: swap a model, change params, recompose streams — all
without touching the core signal pipeline.

    models    the estimator zoo:  make_model("xgb"/"logreg"/"rf", **params) + ClipGRUClassifier
    sequences SequenceBank — raw resampled trajectory tensor for the temporal GRU
    recipes   param-driven runners: run_stream, run_fusion, run_temporal  (Stream config)

    from pipelines.recipes import Stream, run_stream, run_fusion
    run_stream(1, Stream(modalities=("au", "audio"), model="xgb", params={"max_depth": 4}))
"""
from .models import make_model, make_xgb, ClipGRUClassifier, MODEL_ZOO
from .sequences import SequenceBank
from .recipes import Stream, run_stream, run_fusion, run_temporal

__all__ = ["make_model", "make_xgb", "ClipGRUClassifier", "MODEL_ZOO", "SequenceBank",
           "Stream", "run_stream", "run_fusion", "run_temporal"]
