"""Frozen facial-expression-recognition (FER) CNN/ViT features — the learned expression channel.

The official metric is bottlenecked by the CONTROL class: telling a calm bystander from a reacting
one. Hand-crafted AUs cap out at F1(control)≈0.40. This extractor adds a *frozen pretrained* FER
transformer (trained on millions of expression images) as a complementary representation — the
"CNN done right" (frozen, so no 36-subject overfit, unlike the from-scratch BadNet baseline).

Per sampled frame we run the FER ViT on the cropped face and keep BOTH:
  * the 7 emotion probabilities (angry/disgust/fear/happy/neutral/sad/surprise), and
  * the penultimate CLS expression embedding.
Then we aggregate over the clip with REACTION-aware statistics — mean/std/max/apex(top-k)/peak-
velocity/time-of-peak for each emotion (the surprise/amusement spike a failure provokes), plus
mean/std of the embedding. Per-subject normalisation downstream strips identity from the embedding.

Runs on GPU (one model per worker). Cache: `fer_t<track>.csv`.
"""
from __future__ import annotations
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path, sample_frames

FER_MODEL = "trpakov/vit-face-expression"
_EMOS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class _HaarCropper:
    """Dependency-light face crop (OpenCV Haar) — no mediapipe. Center-crop fallback."""

    def __init__(self, margin: float = 0.3):
        import cv2
        self.cv2 = cv2
        self.cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.margin = margin

    def crop(self, frame):
        h, w = frame.shape[:2]
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self.cc.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces):
            x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
            mx, my = int(bw * self.margin), int(bh * self.margin)
            c = frame[max(0, y - my):y + bh + my, max(0, x - mx):x + bw + mx]
        else:
            sde = min(h, w)
            c = frame[(h - sde) // 2:(h + sde) // 2, (w - sde) // 2:(w + sde) // 2]
        return c if c.size else frame


def _agg_emotion(P):
    """P: (T, 7) emotion-prob trajectory -> reaction-aware stats per emotion."""
    T = P.shape[0]
    k = max(1, T // 4)
    vel = np.abs(np.diff(P, axis=0)).max(0) if T > 1 else np.zeros(P.shape[1])
    apex = np.sort(P, axis=0)[-k:].mean(0)
    out = {}
    for j, e in enumerate(_EMOS):
        col = P[:, j]
        out[f"fer_{e}_mean"] = float(col.mean())
        out[f"fer_{e}_std"] = float(col.std())
        out[f"fer_{e}_max"] = float(col.max())
        out[f"fer_{e}_apex"] = float(apex[j])
        out[f"fer_{e}_maxvel"] = float(vel[j])
        out[f"fer_{e}_tmax"] = float(col.argmax() / max(1, T))
    return out


class FERExtractor(BaseExtractor):
    name = "fer"

    def __init__(self, s: int = 16, workers: int = 1, cache_dir=None, device: str = "cuda"):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir)
        self.device = device

    def init_worker(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(FER_MODEL)
        self.model = AutoModelForImageClassification.from_pretrained(
            FER_MODEL, output_hidden_states=True).to(self.device).eval()
        id2label = self.model.config.id2label
        # map model's label order -> our canonical _EMOS order
        self.order = [next(i for i, l in id2label.items() if l.lower().startswith(e[:4])) for e in _EMOS]
        # prefer MediaPipe face detection (better crops); fall back to Haar if mp.solutions is
        # unavailable (broken on mediapipe>=0.10.31 under numpy 2.x — pin mediapipe==0.10.21 numpy<2)
        try:
            from ..datasets import FaceCropper
            self.cropper = FaceCropper()
        except Exception as e:
            print(f"  [fer] MediaPipe cropper unavailable ({type(e).__name__}); using Haar fallback", flush=True)
            self.cropper = _HaarCropper()
        self._mean = np.array(self.proc.image_mean, np.float32)
        self._std = np.array(self.proc.image_std, np.float32)
        self._size = self.proc.size.get("height", 224)

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
        with self.torch.no_grad():
            x = self.torch.tensor(np.stack(crops), dtype=self.torch.float32, device=self.device)
            out = self.model(pixel_values=x)
            probs = self.torch.softmax(out.logits, dim=1).cpu().numpy()[:, self.order]  # (T,7) reordered
            cls = out.hidden_states[-1][:, 0, :].cpu().numpy()                          # (T, D)
        row = {"n_det": len(crops)}
        row.update(_agg_emotion(probs))
        m, sd = cls.mean(0), cls.std(0)
        for i in range(cls.shape[1]):
            row[f"fer_emb_mean_{i:03d}"] = float(m[i]); row[f"fer_emb_std_{i:03d}"] = float(sd[i])
        return row
