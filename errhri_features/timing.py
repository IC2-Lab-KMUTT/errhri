"""Granular, leak-clean TIME-BASED reaction features from a per-frame channel trajectory.

Answers: for a reaction channel (smile, blink, gaze, head-pose...), WHEN does it happen and for
HOW LONG? All features are RELATIVE (fraction of clip / rates) so they do not re-introduce the
duration leak. On Track 1 these are the strongest length-clean signal (timing-only AUC ~0.79);
the carriers are reaction MAGNITUDE (auc/amp) and ONSET/PEAK timing — duration & burst-rate are
weak (and slightly leaky), so prefer the former.
"""
from __future__ import annotations
import numpy as np

TIMING_STATS = ["peak_frac", "onset_frac", "offset_frac", "dur_frac", "rise", "decay",
                "ncross_rate", "early_bias", "auc", "amp"]


def timing_features(x: np.ndarray) -> dict:
    """Compute the 10 relative timing stats for one channel trajectory x[0..T-1]."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    out = {k: 0.0 for k in TIMING_STATS}
    if T < 4:
        return out
    b = float(np.percentile(x, 25)); mx = float(x.max()); amp = mx - b
    out["amp"] = amp
    out["auc"] = float(np.mean(np.maximum(x - b, 0.0)))      # mean elevation magnitude
    if amp < 1e-9:
        return out
    thr = b + 0.5 * amp
    above = x > thr
    tt = np.arange(T) / (T - 1)
    out["peak_frac"] = float(tt[int(np.argmax(x))])          # time-to-peak (relative)
    if above.any():
        idx = np.where(above)[0]
        out["onset_frac"] = float(tt[idx[0]])                # onset latency (relative)
        out["offset_frac"] = float(tt[idx[-1]])
        out["dur_frac"] = float(above.mean())                # fraction elevated (weak/leaky)
        out["rise"] = amp / max(out["peak_frac"] - out["onset_frac"], 1.0 / T)
        out["decay"] = amp / max(out["offset_frac"] - out["peak_frac"], 1.0 / T)
        out["ncross_rate"] = float(np.sum(np.diff(above.astype(int)) == 1)) / T  # burst rate
        half = T // 2
        out["early_bias"] = float(above[:half].mean() - above[half:].mean())     # early vs late
    return out
