"""OpenGraphAU dense AU dynamics — fast single-forward FACS Action Units (41 AUs).

py-feat's modular detector caps at ~1.5 fps because its face detector (img2pose) is hardcoded and
slow, which makes DENSE AU sampling infeasible (the shipped AU cache used only 10 frames/clip, which
averages the brief facial reaction spike away). This extractor swaps in OpenGraphAU's MEFARG network:
ONE backbone forward pass -> 41 AU probabilities, measured at >500 fps on a 1080Ti. Paired with the
same fast MediaPipe face crop used by `faceemb`, dense AU sampling (s>=48) becomes cheap.

Each clip's (T, 41) AU-probability trajectory is summarised with reaction-aware temporal stats —
per-AU mean / std / peak / apex / onset-velocity plus a global AU-motion energy. These are
identity-light and reaction-sensitive (the "did the bystander react" cue), exactly what a denser
sampling of a brief facial spike is meant to amplify.

Requires the OpenGraphAU repo (set OPENGRAPHAU_DIR; default ~/research/errhri/OpenGraphAU) plus the
ResNet-50 stage-1 checkpoint. Frozen, GPU, one model per worker. Cache: `au_graph_t<track>.csv`.

Model: Luo et al., "Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action
Unit Recognition", IJCAI-22 (OpenGraphAU toolbox, 41-AU hybrid-dataset weights).
"""
from __future__ import annotations
import os
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path, sample_frames

OGAU_DIR = os.environ.get("OPENGRAPHAU_DIR", os.path.expanduser("~/research/errhri/OpenGraphAU"))
CKPT = os.environ.get(
    "OPENGRAPHAU_CKPT",
    os.path.join(OGAU_DIR, "checkpoints", "OpenGraphAU-ResNet50_first_stage.pth"),
)

# 41 AU names in model output order (27 main + 14 left/right sub-AUs).
AU_NAMES = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU11", "AU12", "AU13",
            "AU14", "AU15", "AU16", "AU17", "AU18", "AU19", "AU20", "AU22", "AU23", "AU24",
            "AU25", "AU26", "AU27", "AU32", "AU38", "AU39", "AUL1", "AUR1", "AUL2", "AUR2",
            "AUL4", "AUR4", "AUL6", "AUR6", "AUL10", "AUR10", "AUL12", "AUR12", "AUL14", "AUR14"]


def _agg_au(A):
    """A: (T, 41) AU-probability trajectory -> reaction-aware per-AU summary."""
    T, D = A.shape
    k = max(1, T // 4)
    dif = np.diff(A, axis=0) if T > 1 else np.zeros((1, D))
    vel = np.abs(dif)
    m, sd = A.mean(0), A.std(0)
    mx = A.max(0)                                   # apex activation (brief reactions)
    mxv = vel.max(0)                                # peak onset velocity
    apex = np.sort(A, axis=0)[-k:].mean(0)          # robust peak hold (top-k mean)
    out = {}
    for i, nm in enumerate(AU_NAMES):
        out[f"{nm}_mean"] = float(m[i])
        out[f"{nm}_std"] = float(sd[i])
        out[f"{nm}_max"] = float(mx[i])
        out[f"{nm}_maxvel"] = float(mxv[i])
        out[f"{nm}_apex"] = float(apex[i])
    step = np.linalg.norm(dif, axis=1)              # global AU-state motion energy
    out["au_motion_energy"] = float(step.sum())
    out["au_motion_peak"] = float(step.max())
    out["au_motion_std"] = float(step.std())
    return out


class AUGraphExtractor(BaseExtractor):
    name = "au_graph"

    def __init__(self, s: int = 48, workers: int = 4, cache_dir=None, device: str = "cuda",
                 arc: str = "resnet50", ckpt: str = CKPT, ogau_dir: str = OGAU_DIR):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir)
        self.device = device
        self.arc = arc
        self.ckpt = ckpt
        self.ogau_dir = ogau_dir

    def init_worker(self):
        import sys
        import torch
        from torchvision import transforms
        if self.ogau_dir not in sys.path:
            sys.path.insert(0, self.ogau_dir)
        import model.resnet as _rn
        _rn.models_dir = os.path.join(self.ogau_dir, "pretrain_models")  # absolute backbone init path
        from model.ANFL import MEFARG
        from utils import load_state_dict
        self.torch = torch
        net = MEFARG(num_main_classes=27, num_sub_classes=14, backbone=self.arc,
                     neighbor_num=4, metric="dots")
        net = load_state_dict(net, self.ckpt)
        self.model = net.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToPILImage(),                                   # RGB uint8 HWC ndarray
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            from ..datasets import FaceCropper
            self.cropper = FaceCropper()
        except Exception as e:
            from .fer import _HaarCropper
            print(f"  [au_graph] MediaPipe cropper unavailable ({type(e).__name__}); Haar fallback",
                  flush=True)
            self.cropper = _HaarCropper()

    def _prep(self, frame):
        import cv2
        c = self.cropper.crop(frame)
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        return self.tf(rgb).numpy()

    def extract_clip(self, track, participant, video) -> dict:
        frames = sample_frames(video_path(track, participant, video), self.s)
        crops = [self._prep(fr) for _, fr in frames]
        if not crops:
            return {"n_det": 0}
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(crops), 64):
                x = self.torch.tensor(np.stack(crops[i:i + 64]), dtype=self.torch.float32,
                                      device=self.device)
                pred = self.model(x)
                if not self.torch.is_tensor(pred):
                    pred = pred[0]
                outs.append(pred.cpu().numpy())
        A = np.concatenate(outs, 0)                  # (T, 41)
        row = {"n_det": len(crops)}
        row.update(_agg_au(A))
        return row
