"""FACS Action Unit extractor via libreface (12 AU intensities + 12 presence + expression +
gaze + head pose + framing-invariant geometry), aggregated to per-clip level.

Requires libreface (CPU torch is fine — see README for the install caveat).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseExtractor
from ..datasets import video_path, sample_frames
from ..aggregate import aggregate
from ..config import AU_INTENSITY, AU_PRESENCE, EXPRESSIONS, GEOMETRY, INTER_OCULAR


def _geometry(r: dict) -> dict:
    def pt(i):
        return np.array([r.get(f"lm_mp_{i}_x", np.nan), r.get(f"lm_mp_{i}_y", np.nan)], float)
    def dist(i, j):
        return float(np.linalg.norm(pt(i) - pt(j)))
    io = dist(*INTER_OCULAR)
    g = {}
    if not np.isfinite(io) or io < 1e-6:
        for nm, _, _ in GEOMETRY:
            g[f"geo_{nm}"] = np.nan
        g["geo_lipcorner_asym"] = np.nan; g["geo_eye_asym"] = np.nan
        return g
    for nm, a, b in GEOMETRY:
        g[f"geo_{nm}"] = dist(a, b) / io
    g["geo_lipcorner_asym"] = abs(dist(61, 1) - dist(291, 1)) / io
    g["geo_eye_asym"] = abs(dist(159, 145) - dist(386, 374)) / io
    return g


class ActionUnitExtractor(BaseExtractor):
    name = "au"

    def init_worker(self):
        import torch
        torch.set_num_threads(1)
        import libreface
        self.lf = libreface

    def extract_clip(self, track, participant, video) -> dict:
        import tempfile, os, cv2
        cont = [f"au{n}_int" for n in AU_INTENSITY] + ["gaze_yaw", "gaze_pitch", "pitch", "yaw", "roll"]
        cont += [f"geo_{nm}" for nm, _, _ in GEOMETRY] + ["geo_lipcorner_asym", "geo_eye_asym"]
        pres = [f"au{n}_pr" for n in AU_PRESENCE]
        rows = []
        with tempfile.TemporaryDirectory() as td:
            for k, fr in sample_frames(video_path(track, participant, video), self.s):
                p = os.path.join(td, f"{k}.jpg"); cv2.imwrite(p, fr)
                try:
                    r = self.lf.get_facial_attributes_image(p, device="cpu")
                except Exception:
                    continue
                ai = r.get("au_intensities", {}); ap = r.get("detected_aus", {})
                row = {"frame": k}
                for n in AU_INTENSITY:
                    row[f"au{n}_int"] = float(ai.get(f"au_{n}_intensity", np.nan))
                for n in AU_PRESENCE:
                    row[f"au{n}_pr"] = float(ap.get(f"au_{n}", np.nan))
                row["expr"] = r.get("facial_expression", "")
                for c in ("gaze_yaw", "gaze_pitch", "pitch", "yaw", "roll"):
                    row[c] = float(r.get(c, np.nan))
                row.update(_geometry(r))
                rows.append(row)
        if not rows:
            return {"n_det": 0}
        return aggregate(pd.DataFrame(rows), continuous=cont, presence=pres,
                         categorical="expr", categories=EXPRESSIONS)
