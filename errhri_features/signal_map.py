"""Curated signal/noise verdicts — knowing what NOT to feed is half the work at N=23-36 subjects.

Measured with subject-grouped CV, length-leak checked, on both tracks (see SIGNAL_INVENTORY.md).
Use `select_features(columns, level=...)` to filter a feature matrix to a signal tier, and
`DIMENSION_VERDICT` / `TIMING_VERDICT` to reason about what to keep. Verdicts are AUC-based.

VERDICT levels:  SIGNAL > WEAK > NOISE  (and LEAK = forbidden duration proxy, handled by leak.py)
"""
from __future__ import annotations

# --- per-dimension verdicts (Track 1 / Track 2 multivariate AUC) -------------
DIMENSION_VERDICT = {
    # dimension:        (t1_auc, t2_auc, verdict, note)
    "expression":       (0.682, 0.545, "SIGNAL", "expr_entropy/nswitch = instability; top compact T1 signal"),
    "mouth_jaw":        (0.652, 0.512, "SIGNAL", "open/jaw-drop dynamics"),
    "brow":             (0.652, 0.521, "SIGNAL", "raise/frown; static~dyn"),
    "smile":            (0.648, 0.588, "SIGNAL", "cheek/lip-corner; best T2 dimension"),
    "eye":              (0.647, 0.541, "SIGNAL", "blink/aperture dynamics"),
    "nose":             (0.624, 0.486, "SIGNAL_MOD", "wrinkle/sneer"),
    "head_pose":        (0.611, 0.525, "SIGNAL_MOD", "yaw/pitch/roll movement"),
    "gaze":             (0.609, 0.468, "SIGNAL_MOD", "look-direction; slightly more relevant on T2"),
    "audio":            (0.606, 0.491, "WEAK_T1_NOISE_T2", "eGeMAPS; orthogonal -> keep for T1 fusion only"),
    "embed":            (0.674, 0.508, "WEAK_ORTHO", "DINOv2; lifts T1 fusion; noise on T2"),
    "other_lipmouth":   (0.536, 0.448, "NOISE", "negative-affect lips (frown/stretch/press) — drop"),
}

# --- timing-stat verdicts (which time-based aspects carry signal) ------------
TIMING_VERDICT = {
    "auc":         ("SIGNAL", "reaction magnitude — strongest"),
    "onset_frac":  ("SIGNAL", "onset latency — strong, length-clean"),
    "amp":         ("SIGNAL", "amplitude"),
    "early_bias":  ("SIGNAL", "early-vs-late — failure smile is later"),
    "peak_frac":   ("SIGNAL", "time-to-peak"),
    "offset_frac": ("WEAK", ""),
    "decay":       ("WEAK_LEAKY", "borderline length-correlated"),
    "dur_frac":    ("WEAK_LEAKY", "reaction length — borderline length-correlated"),
    "rise":        ("NOISE", ""),
    "ncross_rate": ("NOISE_LEAKY", "burstiness — near-noise + leaky"),
}

# stream-level fusion result (subject-grouped, length-clean) ------------------
FUSION = {
    "track1": {"facial_gru": 0.623, "facial_static": 0.622, "au": 0.602, "audio": 0.579,
               "embed": 0.580, "fusion_mean": 0.666, "fusion_stack": 0.674,
               "duration_leak_bar": 0.702, "official_baseline": 0.502, "metric": "macro_f1"},
    "track2": {"facial_gru": 0.556, "au": 0.545, "fusion_mean": 0.556,
               "official_baseline": 0.564, "metric": "auc", "note": "all streams weak; fusion flat"},
}

# substrings marking the NOISE feature groups to drop by default
_NOISE_SUBSTRINGS = ["au14", "au15", "au17", "au20", "au23", "au24",  # negative-affect lip AUs
                     "noseSneer", "mouthFrown", "mouthPress", "mouthStretch"]
_NOISE_TIMING = [".rise", ".ncross_rate"]


def select_features(columns, level: str = "signal", track: int = 1):
    """Filter feature column names by signal tier.

    level='all'    -> everything
    level='signal' -> drop the measured NOISE groups (negative-affect lips, rise/burst timing);
                      on T2 also drop audio + embed (≈chance there).
    Returns the kept column names (order preserved). Pair with leak.clean_columns for the
    duration-leak guard.
    """
    if level == "all":
        return list(columns)
    keep = []
    for c in columns:
        cl = str(c)
        if any(s in cl for s in _NOISE_SUBSTRINGS):
            continue
        if any(cl.endswith(s) or (s[1:] in cl and cl.split("_")[-1] in ("rise", "ncrossrate"))
               for s in _NOISE_TIMING):
            continue
        if track == 2 and (cl.startswith("emb_") or cl in _AUDIO_HINT or "_aud" in cl):
            continue
        keep.append(c)
    return keep


_AUDIO_HINT = set()  # populated by FeatureBank when audio columns are known
