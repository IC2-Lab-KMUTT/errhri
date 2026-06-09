"""FeatureBank: load + merge cached modality features, select, and per-subject normalize.

This is the object teammates start from. It merges the per-modality feature caches
(`<modality>_t<track>.csv`, each keyed by participant,video) with the clip index (label,
n_frames), and serves a clean (X, y, groups) matrix with the signal-tier and length-leak filters
applied. The heavy extraction is done once (see extractors / scripts.extract_all); this is cheap.

    bank = FeatureBank(track=1, modalities=["au", "audio", "embed"]).load()
    X, y, groups = bank.matrix(select="signal", leak_clean=True)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CACHE_DIR
from .datasets import load_index
from . import signal_map, leak

_BOOK = {"participant", "video", "label", "n_frames", "n_det", "start", "end", "Unnamed: 0"}


def per_subject_norm(X: np.ndarray, groups) -> np.ndarray:
    """Center/scale each feature within each subject (robust, median/IQR). LABEL-FREE: uses only
    a subject's own clips, so it mirrors deployment (you always have the test subject's clips)."""
    X = np.asarray(X, float); groups = np.asarray(groups)
    Xn = X.copy()
    for s in np.unique(groups):
        m = groups == s
        med = np.median(X[m], 0)
        iqr = np.percentile(X[m], 75, 0) - np.percentile(X[m], 25, 0)
        iqr[iqr < 1e-9] = 1.0
        Xn[m] = (X[m] - med) / iqr
    return np.nan_to_num(Xn)


class FeatureBank:
    def __init__(self, track: int, modalities=("au", "audio", "embed"), cache_dir: Path = None):
        self.track = track
        self.modalities = list(modalities)
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.df = None
        self.audio_cols: list[str] = []

    def load(self) -> "FeatureBank":
        df = load_index(self.track)[["participant", "video", "label", "n_frames"]]
        for mod in self.modalities:
            fp = self.cache_dir / f"{mod}_t{self.track}.csv"
            if not fp.exists():
                raise FileNotFoundError(f"missing feature cache {fp} — run extractors first")
            m = pd.read_csv(fp)
            m["participant"] = m.participant.astype(str); m["video"] = m.video.astype(str)
            cols = [c for c in m.columns if c not in _BOOK]
            if mod == "audio":
                self.audio_cols = [c for c in cols if pd.api.types.is_numeric_dtype(m[c])]
            df = df.merge(m[["participant", "video"] + cols], on=["participant", "video"],
                          how="left", suffixes=("", f"_{mod}"))
        self.df = df.reset_index(drop=True)
        signal_map._AUDIO_HINT.update(self.audio_cols)
        return self

    # --- accessors -----------------------------------------------------------
    @property
    def y(self):
        return self.df.label.to_numpy(int)

    @property
    def groups(self):
        return self.df.participant.to_numpy()

    @property
    def n_frames(self):
        return self.df.n_frames.to_numpy(float)

    def feature_columns(self):
        return [c for c in self.df.columns if c not in _BOOK
                and pd.api.types.is_numeric_dtype(self.df[c])]

    def matrix(self, select="all", normalize=True, leak_clean=False):
        """Return (X, y, groups). `select`: 'all' | 'signal' | explicit list of columns.
        normalize=per-subject; leak_clean=drop duration-proxy columns (|corr n_frames|>0.30)."""
        if self.df is None:
            self.load()
        cols = self.feature_columns()
        if isinstance(select, (list, tuple)):
            cols = [c for c in select if c in cols]
        elif select == "signal":
            cols = signal_map.select_features(cols, level="signal", track=self.track)
        X = np.nan_to_num(self.df[cols].to_numpy(float))
        if leak_clean:
            keep_i, cols = leak.clean_columns(X, cols, self.n_frames)
            X = X[:, keep_i]
        if normalize:
            X = per_subject_norm(X, self.groups)
        self._last_cols = cols
        return X, self.y, self.groups

    @property
    def columns(self):
        return getattr(self, "_last_cols", self.feature_columns())
