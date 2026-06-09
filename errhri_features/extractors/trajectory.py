"""TrajectoryExtractor: per-clip resampled channel trajectory for the temporal GRU.

The aggregated extractors (`blend`, `au`) collapse a clip to summary stats — fine for trees, but
the whole-clip GRU needs the *ordered* trajectory. This caches each clip's discriminative channels
(config.TIMING_CHANNELS — smile/jaw/blink/gaze/pose/brow) resampled to a fixed length L, flattened
to columns `t<k>__<channel>` so it rides the same parallel/resume/cache machinery as every other
extractor. `SequenceBank` reshapes it back to (N, L, C).

Reuses BlendshapeExtractor's per-frame MediaPipe pass, so it needs the same FaceLandmarker model
(env ERRHRI_FACE_MODEL). Output cache: `traj_t<track>.csv`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .blendshape import BlendshapeExtractor, _euler, _geometry
from ..datasets import video_path, sample_frames
from ..config import BLENDSHAPES, POSE, TIMING_CHANNELS


def _resample(values: np.ndarray, L: int) -> np.ndarray:
    """Linear-resample a (T,) channel to L points (ffill/bfill NaNs first). T may be < or > L."""
    v = pd.Series(values).interpolate(limit_direction="both").to_numpy(float)
    if not np.isfinite(v).any():
        return np.zeros(L)
    v = np.nan_to_num(v, nan=float(np.nanmean(v)))
    if len(v) == 1:
        return np.repeat(v, L)
    xp = np.linspace(0.0, 1.0, len(v))
    return np.interp(np.linspace(0.0, 1.0, L), xp, v)


class TrajectoryExtractor(BlendshapeExtractor):
    name = "traj"

    def __init__(self, s: int = 32, workers: int = 8, cache_dir=None, model_path: str = None,
                 L: int = 32, channels=None):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir, model_path=model_path)
        self.L = L
        # ordered (name, blendshape/pose key) pairs — the discriminative trajectory channels
        self.channels = list(channels) if channels else list(TIMING_CHANNELS.items())

    def extract_clip(self, track, participant, video) -> dict:
        import cv2
        from .blendshape import _mp_name
        rows = []
        for k, fr in sample_frames(video_path(track, participant, video), self.s):
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            res = self.lm.detect(img)
            if not res.face_blendshapes:
                continue
            row = {"frame": k}
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            for name in BLENDSHAPES:
                row[name] = float(bs.get(_mp_name(name), np.nan))
            if res.facial_transformation_matrixes:
                row["yaw"], row["pitch"], row["roll"] = _euler(res.facial_transformation_matrixes[0])
            rows.append(row)
        if not rows:
            return {"n_det": 0}
        df = pd.DataFrame(rows).sort_values("frame")
        out = {}
        for nm, key in self.channels:
            series = df[key].to_numpy(float) if key in df else np.array([np.nan])
            traj = _resample(series, self.L)
            for t in range(self.L):
                out[f"t{t:02d}__{nm}"] = traj[t]
        return out
