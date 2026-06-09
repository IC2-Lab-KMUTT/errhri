"""Deep face-appearance embedding extractor: DINOv2 ViT-S/14 on the cropped face, pooled over
frames (mean + std). The only LEARNED (non-engineered) representation; weak solo but orthogonal
to the geometric streams -> lifts Track-1 fusion. Noise on Track 2.

Requires timm + torch (CPU is fine, ~0.05s/frame) + mediapipe (face crop).
"""
from __future__ import annotations
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path, sample_frames, FaceCropper
from ..config import EMBED_MODEL, EMBED_IMG

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class EmbeddingExtractor(BaseExtractor):
    name = "embed"

    def init_worker(self):
        import torch, timm
        torch.set_num_threads(1)
        self.torch = torch
        self.model = timm.create_model(EMBED_MODEL, pretrained=True, num_classes=0,
                                       img_size=EMBED_IMG).eval()
        self.cropper = FaceCropper()

    def _prep(self, frame):
        import cv2
        c = self.cropper.crop(frame)
        c = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (EMBED_IMG, EMBED_IMG))
        c = c.astype(np.float32) / 255.0
        return ((c - _MEAN) / _STD).transpose(2, 0, 1)

    def extract_clip(self, track, participant, video) -> dict:
        crops = [self._prep(fr) for _, fr in sample_frames(video_path(track, participant, video), self.s)]
        if not crops:
            return {"n_det": 0}
        with self.torch.no_grad():
            emb = self.model(self.torch.tensor(np.stack(crops), dtype=self.torch.float32)).numpy()
        mean, std = emb.mean(0), emb.std(0)
        out = {"n_det": len(crops)}
        for i in range(emb.shape[1]):
            out[f"emb_mean_{i:03d}"] = float(mean[i]); out[f"emb_std_{i:03d}"] = float(std[i])
        return out
