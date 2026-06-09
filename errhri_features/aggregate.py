"""Turn a per-frame trajectory into per-clip features: STATIC levels + DYNAMICS + TIMING.

Lesson from the analysis: on Track 1 the DYNAMICS (std/range/velocity/slope) beat the static
levels in almost every facial dimension, and the granular TIMING (onset/peak/magnitude) is the
strongest length-clean signal. So every continuous channel gets all three families.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .timing import timing_features

STAT_KEYS = ["mean", "std", "median", "min", "max", "p10", "p90", "range", "iqr",
             "slope", "vel", "delta"]


def summary_stats(v: np.ndarray, prefix: str) -> dict:
    """12 level+dynamics stats for one channel's values over the sampled frames."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in STAT_KEYS}
    o = {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std()),
         f"{prefix}_median": float(np.median(v)), f"{prefix}_min": float(v.min()),
         f"{prefix}_max": float(v.max()), f"{prefix}_p10": float(np.percentile(v, 10)),
         f"{prefix}_p90": float(np.percentile(v, 90)), f"{prefix}_range": float(v.max() - v.min()),
         f"{prefix}_iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
         f"{prefix}_slope": float(np.polyfit(np.arange(v.size), v, 1)[0]) if v.size > 1 else 0.0,
         f"{prefix}_vel": float(np.mean(np.abs(np.diff(v)))) if v.size > 1 else 0.0,
         f"{prefix}_delta": float(v[-1] - v[0])}
    return o


def aggregate(frames: pd.DataFrame, continuous, presence=None, timing_map=None,
              categorical=None, categories=None, order_col="frame") -> dict:
    """Aggregate a per-frame DataFrame for one clip into a flat feature dict.

    continuous : list of column names -> summary_stats (level + dynamics)
    presence   : list of column names (0/1) -> rate / ever / #transitions
    timing_map : {feat_prefix: column} -> granular relative timing features
    categorical: a column of string labels -> per-category fractions + entropy + #switches
    """
    if order_col in frames:
        frames = frames.sort_values(order_col)
    out = {"n_det": int(len(frames))}
    for c in (continuous or []):
        out.update(summary_stats(pd.to_numeric(frames.get(c), errors="coerce").to_numpy(float), c))
    for c in (presence or []):
        v = pd.to_numeric(frames.get(c), errors="coerce").to_numpy(float); v = v[np.isfinite(v)]
        out[f"{c}_rate"] = float(v.mean()) if v.size else 0.0
        out[f"{c}_ever"] = float(v.max()) if v.size else 0.0
        out[f"{c}_ntrans"] = float(np.sum(np.diff((v > 0.5).astype(int)) == 1)) if v.size > 1 else 0.0
    for prefix, col in (timing_map or {}).items():
        t = timing_features(pd.to_numeric(frames.get(col), errors="coerce").to_numpy(float))
        for k, val in t.items():
            out[f"{prefix}.{k}"] = val
    if categorical and categorical in frames:
        ex = [str(x) for x in frames[categorical].tolist() if str(x) and str(x) != "nan"]
        tot = max(len(ex), 1); fr = []
        for e in (categories or []):
            f = sum(1 for x in ex if x == e) / tot
            out[f"expr_{e.lower()}"] = float(f); fr.append(f)
        p = np.array([f for f in fr if f > 0])
        out["expr_entropy"] = float(-(p * np.log(p)).sum()) if p.size else 0.0
        out["expr_nswitch"] = float(sum(1 for i in range(1, len(ex)) if ex[i] != ex[i - 1]))
    return out
