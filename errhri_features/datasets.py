"""Dataset index + frame sampling + face cropping (shared by all extractors).

Index CSV schema (one row per clip):  participant,video,label,n_frames
  - participant : subject id (used for grouped CV)
  - video       : clip id (filename stem, no extension)
  - label       : 0/1 ground-truth
  - n_frames    : raw frame count (the duration-leak signal; kept ONLY for the leak guard)
Raw video path resolves to  RAW_ROOTS[track]/<participant>/<video>.mp4
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CACHE_DIR, RAW_ROOTS


def load_index(track: int) -> pd.DataFrame:
    df = pd.read_csv(CACHE_DIR / f"index_t{track}.csv")
    df["participant"] = df.participant.astype(str)
    df["video"] = df.video.astype(str)
    return df


def video_path(track: int, participant: str, video: str) -> Path:
    return RAW_ROOTS[track] / str(participant) / f"{video}.mp4"


def sample_frames(video_path, s: int):
    """Evenly sample up to `s` frames; returns list of (frame_index, BGR image)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release(); return []
    out = []
    for k, fi in enumerate(np.linspace(0, n - 1, min(s, n)).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            out.append((k, fr))
    cap.release()
    return out


class FaceCropper:
    """MediaPipe face detection -> square crop with margin (falls back to center crop)."""

    def __init__(self, margin: float = 0.3, min_conf: float = 0.3):
        import mediapipe as mp
        self.fd = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                            min_detection_confidence=min_conf)
        self.margin = margin

    def crop(self, frame):
        import cv2
        h, w = frame.shape[:2]
        res = self.fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            bb = res.detections[0].location_data.relative_bounding_box
            x, y = int(bb.xmin * w), int(bb.ymin * h)
            bw, bh = int(bb.width * w), int(bb.height * h)
            mx, my = int(bw * self.margin), int(bh * self.margin)
            c = frame[max(0, y - my):y + bh + my, max(0, x - mx):x + bw + mx]
        else:
            sde = min(h, w)
            c = frame[(h - sde) // 2:(h + sde) // 2, (w - sde) // 2:(w + sde) // 2]
        return c if c.size else frame
