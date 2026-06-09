"""Shared semantic-dimension map: feature name -> facial/behavioural dimension.

Used by feature_report and dimension_breakdown so both group features the same way. Order matters
(first match wins) because some substrings overlap (gaze_yaw vs head yaw).

AU tokens ("au6", "au12", ...) are matched with a non-digit lookahead so "au1" does NOT also match
au10/au12/au14/au15/au17, and so it works for BOTH naming styles a cache might use
(`au6int_mean` and `au6_int_mean`).
"""
from __future__ import annotations
import re

# (dimension, [tokens]). First hit wins. Names are lowercased before matching.
# A token of the form "au<digits>" is matched as a whole AU id (digit-boundary aware); any other
# token is a plain substring.
_RULES = [
    ("gaze",          ["gaze_", "eyelook", "lookout", "lookin", "lookdown", "lookup"]),
    ("head_pose",     ["head_", "yaw_", "pitch_", "roll_"]),
    ("smile",         ["au6", "au12", "au14", "cheek", "smile", "dimple", "lipcorner_asym",
                       "mouthsmile"]),
    ("brow",          ["au1", "au2", "au4", "brow", "inner_brow"]),
    ("eye",           ["au5", "au7", "blink", "eyeblink", "eye_open", "eyewide", "squint",
                       "eye_asym", "lid"]),
    ("mouth_jaw",     ["au25", "au26", "jaw", "mouth_open", "mouth_width", "lips", "mouthclose",
                       "mouthfunnel", "nose_lip"]),
    ("nose",          ["au9", "au10", "nose", "sneer", "upper-lip", "mouthupper"]),
    ("expression",    ["expr_"]),
    ("other_lipmouth", ["au15", "au17", "au20", "au23", "au24", "mouthfrown", "mouthpress",
                        "mouthstretch", "mouthroll", "mouthshrug", "mouthpucker", "pucker",
                        "funnel", "shrug"]),
]
_AU = re.compile(r"^au\d+$")


def _hit(token: str, f: str) -> bool:
    if _AU.match(token):                       # AU id: match the number with a digit boundary
        return re.search(token + r"(?!\d)", f) is not None
    return token in f


def dimension_of(feature: str, audio_cols=()) -> str:
    f = str(feature).lower()
    if feature in audio_cols or f.startswith("audio") or "_aud" in f:
        return "audio"
    if f.startswith("emb_"):
        return "embed"
    for dim, subs in _RULES:
        if any(_hit(s, f) for s in subs):
            return dim
    return "other"


def stat_family(feature: str) -> str:
    """static level vs dynamics vs timing — the within-channel split."""
    f = str(feature).lower()
    if "." in f:                       # timing features use a dotted suffix
        return "timing"
    if any(f.endswith("_" + k) for k in ("std", "range", "iqr", "slope", "vel", "delta",
                                         "ntrans", "nswitch", "entropy")):
        return "dynamics"
    return "static"
