"""DINOv2 dense face-embedding dynamics — the frozen-backbone facial channel done correctly.

The first FER attempt failed for two reasons: (1) a tiny FER2013-trained 7-emotion ViT — a weak,
low-dimensional bottleneck — and (2) only 16 sampled frames, which averages the brief reaction spike
away. This extractor fixes both:

  * BETTER MODEL: a frozen **DINOv2** backbone (self-supervised SOTA visual features, 768-d) instead
    of a 7-class emotion head. Its embedding captures expression, gaze, head pose and mouth state in
    a far richer space than emotion logits.
  * DENSE + DYNAMIC: sample many frames (default s=48) and summarise the embedding TRAJECTORY with
    reaction-aware stats. The key signals are identity-INVARIANT temporal dynamics — per-dim std and
    peak frame-to-frame velocity (how much the face *changes*), plus a global embedding-motion energy
    — exactly the "did the bystander react" cue that static emotion means miss. Per-dim mean is kept
    too (identity carried there is stripped by downstream per-subject normalisation).

Frozen, GPU, one model per worker. Proper MediaPipe face crops. Cache: `faceemb_t<track>.csv`.
"""
from __future__ import annotations
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path, sample_frames

DINO_MODEL = "facebook/dinov2-base"   # 768-d; strong frozen self-supervised features


def _agg_embed(E):
    """E: (T, D) embedding trajectory -> identity-robust reaction-aware summary."""
    T, D = E.shape
    k = max(1, T // 4)
    dif = np.diff(E, axis=0) if T > 1 else np.zeros((1, D))
    vel = np.abs(dif)
    out = {}
    m, sd = E.mean(0), E.std(0)
    mxv = vel.max(0)
    apex = np.sort(E, axis=0)[-k:].mean(0)                     # top-k mean per dim (peak hold)
    for i in range(D):
        out[f"fe_mean_{i:03d}"] = float(m[i])                  # identity (stripped by subj-norm)
        out[f"fe_std_{i:03d}"] = float(sd[i])                  # temporal spread (reaction)
        out[f"fe_maxvel_{i:03d}"] = float(mxv[i])              # peak change (reaction spike)
        out[f"fe_apex_{i:03d}"] = float(apex[i])
    # global face-motion energy in embedding space (identity-invariant)
    step = np.linalg.norm(dif, axis=1)
    out["fe_motion_energy"] = float(step.sum())
    out["fe_motion_peak"] = float(step.max())
    out["fe_motion_std"] = float(step.std())
    return out


class FaceEmbExtractor(BaseExtractor):
    name = "faceemb"

    def __init__(self, s: int = 48, workers: int = 1, cache_dir=None, device: str = "cuda",
                 model: str = DINO_MODEL):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir)
        self.device = device
        self.model_name = model

    def init_worker(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        try:
            from ..datasets import FaceCropper
            self.cropper = FaceCropper()
        except Exception as e:
            from .fer import _HaarCropper
            print(f"  [faceemb] MediaPipe cropper unavailable ({type(e).__name__}); Haar fallback", flush=True)
            self.cropper = _HaarCropper()
        self._mean = np.array(self.proc.image_mean, np.float32)
        self._std = np.array(self.proc.image_std, np.float32)
        sz = self.proc.crop_size if hasattr(self.proc, "crop_size") else self.proc.size
        self._size = sz.get("height", sz.get("shortest_edge", 224)) if isinstance(sz, dict) else 224

    def _prep(self, frame):
        import cv2
        c = self.cropper.crop(frame)
        c = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (self._size, self._size))
        c = c.astype(np.float32) / 255.0
        return ((c - self._mean) / self._std).transpose(2, 0, 1)

    def extract_clip(self, track, participant, video) -> dict:
        frames = sample_frames(video_path(track, participant, video), self.s)
        crops = [self._prep(fr) for _, fr in frames]
        if not crops:
            return {"n_det": 0}
        embs = []
        with self.torch.no_grad():
            for i in range(0, len(crops), 32):                 # batch to fit GPU
                x = self.torch.tensor(np.stack(crops[i:i + 32]), dtype=self.torch.float32,
                                      device=self.device)
                out = self.model(pixel_values=x)
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()   # CLS token (T_i, 768)
                embs.append(cls)
        E = np.concatenate(embs, 0)
        row = {"n_det": len(crops)}
        row.update(_agg_embed(E))
        return row
