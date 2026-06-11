"""SequenceBank: the FeatureBank analogue for the temporal GRU.

FeatureBank serves the flat aggregated matrix (for trees). SequenceBank serves the raw resampled
trajectory tensor (N, L, C) that `models.ClipGRUClassifier` consumes — loaded from the `traj`
cache (see extractors/trajectory.py). Same conventions: merges the clip index for labels/groups,
optional per-subject channel normalization (label-free), leak-safe (trajectories are length-
*normalized*, so shape carries the signal, not duration).

    seq = SequenceBank(track=1).load()
    X, y, groups = seq.matrix()                 # X: (N, L, C)
    ev = CVEvaluator(track=1)
    rep = ev.run_matrix(lambda: ClipGRUClassifier(), X, y, groups, seq.n_frames)
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from errhri_features.config import CACHE_DIR
from errhri_features.datasets import load_index

_CELL = re.compile(r"^t(\d+)__(.+)$")


class SequenceBank:
    def __init__(self, track: int, cache_dir: Path = None):
        self.track = track
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.df = None
        self.L = 0
        self.channels: list[str] = []

    def load(self) -> "SequenceBank":
        fp = self.cache_dir / f"traj_t{self.track}.csv"
        if not fp.exists():
            raise FileNotFoundError(f"missing trajectory cache {fp} — run the `traj` extractor "
                                    "(scripts.extract_all --modalities traj)")
        traj = pd.read_csv(fp)
        traj["participant"] = traj.participant.astype(str); traj["video"] = traj.video.astype(str)
        idx = load_index(self.track)[["participant", "video", "label", "n_frames"]]
        idx["participant"] = idx.participant.astype(str); idx["video"] = idx.video.astype(str)
        self.df = idx.merge(traj, on=["participant", "video"], how="inner").reset_index(drop=True)
        ts, chs = set(), []
        for c in traj.columns:
            m = _CELL.match(str(c))
            if m:
                ts.add(int(m.group(1)))
                if m.group(2) not in chs:
                    chs.append(m.group(2))
        self.L, self.channels = (max(ts) + 1 if ts else 0), chs
        return self

    @property
    def y(self):
        return self.df.label.to_numpy(int)

    @property
    def groups(self):
        return self.df.participant.to_numpy()

    @property
    def n_frames(self):
        return self.df.n_frames.to_numpy(float)

    def tensor(self) -> np.ndarray:
        """(N, L, C) trajectory tensor."""
        cols = [f"t{t:02d}__{ch}" for t in range(self.L) for ch in self.channels]
        flat = np.nan_to_num(self.df[cols].to_numpy(float))
        return flat.reshape(len(self.df), self.L, len(self.channels))

    def matrix(self, normalize=True):
        """Return (X, y, groups) with X = (N, L, C). normalize=per-subject per-channel (z over
        that subject's clips+time) — label-free, mirrors deployment, removes per-person baseline."""
        X = self.tensor()
        if normalize:
            X = self._subject_norm(X, self.groups)
        return X, self.y, self.groups

    @staticmethod
    def _subject_norm(X, groups):
        groups = np.asarray(groups); Xn = X.astype(float).copy()
        for s in np.unique(groups):
            m = groups == s
            block = X[m]                                   # (ns, L, C)
            mu = block.mean(axis=(0, 1), keepdims=True)
            sd = block.std(axis=(0, 1), keepdims=True); sd[sd < 1e-9] = 1.0
            Xn[m] = (block - mu) / sd
        return np.nan_to_num(Xn)


# continuous per-frame channels available in the `au_frames` cache (skip presence binaries + expr)
_AU_FRAME_CHANNELS = (
    ["au1_int", "au2_int", "au4_int", "au5_int", "au6_int", "au9_int", "au12_int", "au15_int",
     "au17_int", "au20_int", "au25_int", "au26_int"]
    + ["gaze_yaw", "gaze_pitch", "pitch", "yaw", "roll"]
    + ["geo_mouth_open", "geo_mouth_width", "geo_eye_open_l", "geo_eye_open_r", "geo_brow_eye_l",
       "geo_brow_eye_r", "geo_inner_brow", "geo_nose_lip", "geo_jaw_open", "geo_lipcorner_asym",
       "geo_eye_asym"])


def _resample(values, L):
    """Linear-resample a 1-D channel to L points (interp NaNs, pad constants)."""
    v = pd.Series(values).interpolate(limit_direction="both").to_numpy(float)
    if not np.isfinite(v).any():
        return np.zeros(L)
    v = np.nan_to_num(v, nan=float(np.nanmean(v)))
    if len(v) == 1:
        return np.repeat(v, L)
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)), v)


class FrameSequenceBank:
    """Per-frame trajectory tensor from the `au_frames` cache (raw libreface per-frame rows) —
    no `traj`/FaceLandmarker needed. Gives the temporal models a real multivariate AU/pose/geometry
    time series (≈10 frames × 28 channels) to chew on. Same (X, y, groups) interface as SequenceBank.

        fs = FrameSequenceBank(track=1).load()
        X, y, groups = fs.matrix()          # X: (N, L, C)
    """

    def __init__(self, track, cache_dir=None, L=10, channels=None, cache_name="au_frames"):
        self.track = track
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_name = cache_name
        self.L = L
        self.channels = list(channels) if channels else list(_AU_FRAME_CHANNELS)
        self.df = None
        self._tensor = None

    def load(self):
        fp = self.cache_dir / f"{self.cache_name}_t{self.track}.csv"
        if not fp.exists():
            raise FileNotFoundError(f"missing per-frame cache {fp}")
        fr = pd.read_csv(fp)
        fr["participant"] = fr.participant.astype(str); fr["video"] = fr.video.astype(str)
        chans = [c for c in self.channels if c in fr.columns]
        self.channels = chans
        order = "frame" if "frame" in fr.columns else None
        traj = {}
        for key, g in fr.groupby(["participant", "video"]):
            if order:
                g = g.sort_values(order)
            arr = np.stack([_resample(g[c].to_numpy(float), self.L) for c in chans], axis=1)
            traj[key] = arr                                  # (L, C)
        idx = load_index(self.track)[["participant", "video", "label", "n_frames"]]
        idx["participant"] = idx.participant.astype(str); idx["video"] = idx.video.astype(str)
        keep = [(p, v) in traj for p, v in zip(idx.participant, idx.video)]
        self.df = idx[keep].reset_index(drop=True)
        self._tensor = np.stack([traj[(p, v)] for p, v in zip(self.df.participant, self.df.video)])
        return self

    @property
    def y(self):
        return self.df.label.to_numpy(int)

    @property
    def groups(self):
        return self.df.participant.to_numpy()

    @property
    def n_frames(self):
        return self.df.n_frames.to_numpy(float)

    def matrix(self, normalize=True):
        X = self._tensor
        if normalize:
            X = SequenceBank._subject_norm(X, self.groups)
        return X, self.y, self.groups
