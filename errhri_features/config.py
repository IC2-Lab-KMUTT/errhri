"""Central configuration: tracks, channels, AU lists, model names.

Edit `RAW_ROOTS` / `CACHE_DIR` to point at your data + feature cache. Everything else is the
validated schema used across the toolkit.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- paths (override via env or by editing) ----------------------------------
CACHE_DIR = Path(os.environ.get("ERRHRI_CACHE", Path(__file__).resolve().parent.parent / "cache"))
# raw mp4 roots, per track:  <root>/<participant>/<video>.mp4
RAW_ROOTS = {
    1: Path(os.environ.get("ERRHRI_T1_ROOT", "data/track1/trainval")),
    2: Path(os.environ.get("ERRHRI_T2_ROOT", "data/track2/trainval")),
}

# --- track definitions -------------------------------------------------------
# T1 "BAD": failure vs control, macro-F1, majority-vote agg, 87/13 imbalance.
# T2 "Bad Idea": well vs poorly, AUC, max-prob agg, balanced.
TRACKS = {
    1: dict(name="BAD", metric="macro_f1", agg="majority", fps=5,
            window_size=10, slide=5),          # window_size <= 2*fps (T1 rule)
    2: dict(name="BadIdea", metric="auc", agg="maxprob", fps=3,
            window_size=30, slide=30),         # T2 has no window cap -> whole clip
}

# --- libreface FACS Action Units ---------------------------------------------
AU_INTENSITY = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
AU_PRESENCE = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
AU_NAME = {1: "inner-brow-raise", 2: "outer-brow-raise", 4: "brow-lower(frown)",
           5: "upper-lid-raise", 6: "cheek-raise(smile)", 7: "lid-tighten", 9: "nose-wrinkle",
           10: "upper-lip-raise", 12: "lip-corner-pull(smile)", 14: "dimple",
           15: "lip-corner-depress", 17: "chin-raise", 20: "lip-stretch", 23: "lip-tighten",
           24: "lip-press", 25: "lips-part", 26: "jaw-drop"}
EXPRESSIONS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

# --- MediaPipe geometry: framing-invariant distance ratios (a,b) / inter-ocular
GEOMETRY = [("mouth_open", 13, 14), ("mouth_width", 61, 291), ("eye_open_l", 159, 145),
            ("eye_open_r", 386, 374), ("brow_eye_l", 105, 159), ("brow_eye_r", 334, 386),
            ("inner_brow", 55, 285), ("nose_lip", 1, 13), ("jaw_open", 152, 13)]
INTER_OCULAR = (33, 263)

# --- MediaPipe ARKit blendshapes (52) + pose (3) -----------------------------
BLENDSHAPES = [
    "browDownL", "browDownR", "browInnerUp", "browOuterUpL", "browOuterUpR", "cheekPuff",
    "cheekSquintL", "cheekSquintR", "eyeBlinkL", "eyeBlinkR", "eyeLookDownL", "eyeLookDownR",
    "eyeLookInL", "eyeLookInR", "eyeLookOutL", "eyeLookOutR", "eyeLookUpL", "eyeLookUpR",
    "eyeSquintL", "eyeSquintR", "eyeWideL", "eyeWideR", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleL", "mouthDimpleR", "mouthFrownL", "mouthFrownR",
    "mouthFunnel", "mouthLeft", "mouthLowerDownL", "mouthLowerDownR", "mouthPressL", "mouthPressR",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileL", "mouthSmileR", "mouthStretchL", "mouthStretchR",
    "mouthUpperUpL", "mouthUpperUpR", "noseSneerL", "noseSneerR"]
POSE = ["yaw", "pitch", "roll"]

# key blendshape channels for granular timing features (the discriminative ones)
TIMING_CHANNELS = {
    "smile_cheekPuff": "cheekPuff", "smile_mouthSmileL": "mouthSmileL",
    "smile_dimpleL": "mouthDimpleL", "jawOpen": "jawOpen",
    "blink_L": "eyeBlinkL", "gaze_lookOutL": "eyeLookOutL", "gaze_lookDownL": "eyeLookDownL",
    "gaze_lookInL": "eyeLookInL", "head_yaw": "yaw", "head_pitch": "pitch", "head_roll": "roll",
    "brow_innerUp": "browInnerUp", "brow_downL": "browDownL", "nose_sneerL": "noseSneerL",
    "mouthStretchL": "mouthStretchL", "mouthFrownL": "mouthFrownL",
}

# --- deep embedding ----------------------------------------------------------
EMBED_MODEL = "vit_small_patch14_dinov2.lvd142m"
EMBED_IMG = 224

# leak guard: any feature whose |corr with n_frames| exceeds this is a duration proxy
LEAK_THRESHOLD = 0.30
