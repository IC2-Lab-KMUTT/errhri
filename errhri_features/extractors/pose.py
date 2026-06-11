"""Head + upper-body pose dynamics — a genuinely orthogonal channel to facial AUs.

Facial AUs (py-feat) cap out at F1(control)~0.40: they describe muscle activation but not *gross*
head/body reaction — the nod, the recoil, the head-turn-away, the freeze, the lean-back that a
bystander does when the robot fails. None of that is in the AU / blendshape / facial-geometry
streams. This extractor adds it.

Per sampled frame we run MediaPipe Pose (33 upper-body landmarks; on a webcam clip we reliably get
nose/eyes/ears/mouth/shoulders) and derive FRAMING-INVARIANT scalar signals — head roll/yaw/pitch
proxies, head offset from the shoulder line (nod / lean), shoulder tilt, inter-ocular & shoulder
width (proximity), ear-visibility asymmetry (turning away). Each signal's trajectory is summarised
with REACTION-aware stats: mean/std/range/apex(top-k)/peak-velocity/peak-acceleration/time-of-peak,
plus a global head-motion-energy. Per-subject normalisation downstream strips posture identity.

Density matters: a reaction is a brief event, so we sample densely (default s=64) — unlike a sparse
16-frame grab that averages the spike away. Pose runs on CPU; use several workers. Cache:
`pose_t<track>.csv`.
"""
from __future__ import annotations
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path, sample_frames

# MediaPipe Pose landmark indices (upper body, reliably visible on a face-webcam clip)
NOSE, L_EYE, R_EYE = 0, 2, 5
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SH, R_SH = 11, 12
_EPS = 1e-6


def _ang(dy, dx):
    return float(np.arctan2(dy, dx))


def _frame_signals(lm, vis):
    """lm: (33,2) normalised xy; vis: (33,) visibility. -> dict of framing-invariant scalars."""
    nose = lm[NOSE]; le, re = lm[L_EYE], lm[R_EYE]
    lear, rear = lm[L_EAR], lm[R_EAR]
    ml, mr = lm[MOUTH_L], lm[MOUTH_R]
    lsh, rsh = lm[L_SH], lm[R_SH]
    eye_mid = (le + re) / 2.0
    mouth_mid = (ml + mr) / 2.0
    sh_mid = (lsh + rsh) / 2.0
    iod = np.linalg.norm(le - re) + _EPS
    sh_w = np.linalg.norm(lsh - rsh)
    scale = sh_w if sh_w > 0.02 else iod * 3.0          # body scale; fall back to face scale
    scale += _EPS
    ear_d = np.linalg.norm(lear - rear) + _EPS
    return {
        "head_roll":   _ang(re[1] - le[1], re[0] - le[0]),          # eye-line tilt
        "sh_tilt":     _ang(rsh[1] - lsh[1], rsh[0] - lsh[0]),      # shoulder-line tilt
        "yaw_asym":    float((np.linalg.norm(nose - lear) - np.linalg.norm(nose - rear)) / ear_d),
        "head_x":      float((nose[0] - sh_mid[0]) / scale),       # horizontal head offset
        "head_y":      float((nose[1] - sh_mid[1]) / scale),       # vertical (nod / slump)
        "pitch_proxy": float((mouth_mid[1] - eye_mid[1]) / iod),   # face foreshortening (pitch)
        "iod":         float(iod),                                  # camera proximity (face)
        "sh_width":    float(sh_w),                                 # camera proximity (body) / lean
        "ear_vis_asym": float(abs(vis[L_EAR] - vis[R_EAR])),        # head turned away
        "head_size":   float(np.linalg.norm(nose - eye_mid) / scale),
        "eye_sh_d":    float((eye_mid[1] - sh_mid[1]) / scale),     # head-above-shoulder (lean back/fwd)
    }


_SIGNALS = ["head_roll", "sh_tilt", "yaw_asym", "head_x", "head_y", "pitch_proxy",
            "iod", "sh_width", "ear_vis_asym", "head_size", "eye_sh_d"]


def _agg(traj):
    """traj: (T, K) per-signal trajectory -> reaction-aware stats per signal + global motion."""
    T = traj.shape[0]
    k = max(1, T // 4)
    vel = np.abs(np.diff(traj, axis=0)) if T > 1 else np.zeros((1, traj.shape[1]))
    acc = np.abs(np.diff(traj, axis=0, n=2)) if T > 2 else np.zeros((1, traj.shape[1]))
    apex = np.sort(traj, axis=0)[-k:].mean(0)
    out = {}
    for j, s in enumerate(_SIGNALS):
        c = traj[:, j]
        out[f"pose_{s}_mean"] = float(c.mean())
        out[f"pose_{s}_std"] = float(c.std())
        out[f"pose_{s}_range"] = float(c.max() - c.min())
        out[f"pose_{s}_apex"] = float(apex[j])
        out[f"pose_{s}_maxvel"] = float(vel[:, j].max())
        out[f"pose_{s}_maxacc"] = float(acc[:, j].max())
        out[f"pose_{s}_tmax"] = float(c.argmax() / max(1, T))
    # global head-motion energy (translational): how much the head moved overall
    hx, hy = _SIGNALS.index("head_x"), _SIGNALS.index("head_y")
    out["pose_motion_energy"] = float(np.sqrt(vel[:, hx] ** 2 + vel[:, hy] ** 2).sum())
    out["pose_motion_peak"] = float(np.sqrt(vel[:, hx] ** 2 + vel[:, hy] ** 2).max())
    return out


class PoseExtractor(BaseExtractor):
    name = "pose"

    def __init__(self, s: int = 64, workers: int = 6, cache_dir=None, complexity: int = 1):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir)
        self.complexity = complexity

    def init_worker(self):
        import mediapipe as mp
        self.pose = mp.solutions.pose.Pose(static_image_mode=True,
                                           model_complexity=self.complexity,
                                           min_detection_confidence=0.3)

    def extract_clip(self, track, participant, video) -> dict:
        import cv2
        frames = sample_frames(video_path(track, participant, video), self.s)
        rows, n_det = [], 0
        for _, fr in frames:
            res = self.pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue
            pts = res.pose_landmarks.landmark
            lm = np.array([[p.x, p.y] for p in pts], np.float32)
            vis = np.array([p.visibility for p in pts], np.float32)
            rows.append(_frame_signals(lm, vis))
            n_det += 1
        if not rows:
            return {"n_det": 0}
        traj = np.array([[r[s] for s in _SIGNALS] for r in rows], np.float32)
        out = {"n_det": n_det}
        out.update(_agg(traj))
        return out
