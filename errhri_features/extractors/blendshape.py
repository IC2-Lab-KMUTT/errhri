"""MediaPipe FaceLandmarker extractor: 52 ARKit blendshapes + head pose (yaw/pitch/roll) +
framing-invariant geometry, with the dense trajectory aggregated to static + dynamics + granular
TIMING features. This is the facial backbone (smile/pose/gaze/blink trajectories).

Needs the FaceLandmarker model file (`face_landmarker_v2_with_blendshapes.task`); set its path via
env ERRHRI_FACE_MODEL or pass model_path=. Download:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .base import BaseExtractor
from ..datasets import video_path, sample_frames
from ..aggregate import aggregate
from ..config import BLENDSHAPES, POSE, GEOMETRY, INTER_OCULAR, TIMING_CHANNELS, EXPRESSIONS


def _euler(matrix) -> tuple:
    """yaw,pitch,roll (deg) from a 4x4 facial transformation matrix."""
    R = np.array(matrix).reshape(4, 4)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy)); yaw = 0.0
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
    return float(yaw), float(pitch), float(roll)


def _geometry(lms) -> dict:
    def dist(i, j):
        return float(np.hypot(lms[i].x - lms[j].x, lms[i].y - lms[j].y))
    io = dist(*INTER_OCULAR)
    g = {}
    if io < 1e-6:
        for nm, _, _ in GEOMETRY:
            g[f"geo_{nm}"] = np.nan
        g["geo_lipcorner_asym"] = np.nan; g["geo_eye_asym"] = np.nan
        return g
    for nm, a, b in GEOMETRY:
        g[f"geo_{nm}"] = dist(a, b) / io
    g["geo_lipcorner_asym"] = abs(dist(61, 1) - dist(291, 1)) / io
    g["geo_eye_asym"] = abs(dist(159, 145) - dist(386, 374)) / io
    return g


class BlendshapeExtractor(BaseExtractor):
    name = "blend"

    def __init__(self, s: int = 16, workers: int = 8, cache_dir=None, model_path: str = None):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir)
        self.model_path = model_path or os.environ.get("ERRHRI_FACE_MODEL",
                                                        "face_landmarker.task")

    def init_worker(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            num_faces=1)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.mp = mp

    def extract_clip(self, track, participant, video) -> dict:
        import cv2
        cont = list(BLENDSHAPES) + list(POSE) + [f"geo_{nm}" for nm, _, _ in GEOMETRY] \
            + ["geo_lipcorner_asym", "geo_eye_asym"]
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
                # MediaPipe uses lowerCamel like "mouthSmileLeft"; our short names map by prefix
                row[name] = float(bs.get(_mp_name(name), np.nan))
            if res.facial_transformation_matrixes:
                yaw, pitch, roll = _euler(res.facial_transformation_matrixes[0])
                row["yaw"], row["pitch"], row["roll"] = yaw, pitch, roll
            row.update(_geometry(res.face_landmarks[0]))
            rows.append(row)
        if not rows:
            return {"n_det": 0}
        return aggregate(pd.DataFrame(rows), continuous=cont, timing_map=TIMING_CHANNELS)


def _mp_name(short: str) -> str:
    """Map our short blendshape names (mouthSmileL) to MediaPipe names (mouthSmileLeft)."""
    return short.replace("L", "Left").replace("R", "Right") if short[-1] in "LR" else short
