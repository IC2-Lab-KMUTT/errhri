"""Dense eye-gaze + 6DoF head-pose dynamics — the breadth channel dense AUs lack.

The old py-feat `au` cache carried gaze + head pose but only at 10 sparse frames; `au_graph` is dense
but AU-occurrence-only (no gaze, no head pose). This extractor fills the gap: a DEDICATED dense gaze
+ head-pose stream from MediaPipe FaceMesh (refine_landmarks=True gives the 10 iris landmarks).

Per sampled frame we derive framing-invariant scalars:
  * eye gaze: iris offset within each eye socket -> horizontal/vertical gaze, binocular average, L/R
    asymmetry, and aversion magnitude (how far the eyes look from centre — the "glance away" cue);
  * eyelid: eye-aspect-ratio per eye + mean (blink / squint / widen — startle and surprise);
  * head pose: yaw/pitch/roll via solvePnP on a canonical 3-D face (the nod / recoil / turn-away).

Each trajectory is summarised with REACTION-aware stats (mean/std/range/apex/peak-velocity/
peak-acceleration/time-of-peak) + global gaze- and head-motion energy. Dense (s=48) so the brief
reaction glance/recoil is not averaged away. CPU MediaPipe; several workers. Cache: `gaze_t<track>.csv`.
"""
from __future__ import annotations
import numpy as np
from .base import BaseExtractor
from ..datasets import video_path, sample_frames

_EPS = 1e-6
# FaceMesh (478-pt, refine_landmarks=True) indices
L_OUT, L_IN = 33, 133          # left eye outer/inner corner
R_IN, R_OUT = 362, 263         # right eye inner/outer corner
L_TOP, L_BOT = 159, 145        # left eye lid top/bottom
R_TOP, R_BOT = 386, 374        # right eye lid top/bottom
L_IRIS, R_IRIS = 468, 473      # iris centres
NOSE_TIP, CHIN = 1, 152
MOUTH_L, MOUTH_R = 61, 291

# canonical 3-D model points (mm) for solvePnP head pose
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),# left eye outer
    (225.0, 170.0, -135.0), # right eye outer
    (-150.0, -150.0, -125.0),# left mouth
    (150.0, -150.0, -125.0), # right mouth
], np.float64)


def _gaze_ratio(iris, inner, outer, top, bot):
    """iris offset within the eye box, normalised: ~0 centred, +/- look toward a corner/lid."""
    ex = outer - inner
    ew = np.linalg.norm(ex) + _EPS
    eye_c = (inner + outer) / 2.0
    h = float(np.dot(iris - eye_c, ex / ew) / (ew / 2.0))   # horizontal gaze, [-1,1]-ish
    ev = bot - top
    eh = np.linalg.norm(ev) + _EPS
    lid_c = (top + bot) / 2.0
    v = float(np.dot(iris - lid_c, ev / eh) / (eh / 2.0))   # vertical gaze
    ear = float(eh / ew)                                     # eye-aspect-ratio (openness)
    return h, v, ear


def _head_pose(lm_px, w, h):
    """solvePnP yaw/pitch/roll (degrees) from 6 facial points; (0,0,0) on failure."""
    import cv2
    img_pts = np.array([lm_px[NOSE_TIP], lm_px[CHIN], lm_px[L_OUT], lm_px[R_OUT],
                        lm_px[MOUTH_L], lm_px[MOUTH_R]], np.float64)
    f = float(w)
    cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img_pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    return yaw, pitch, roll


def _frame_signals(lm, w, h):
    """lm: (478,2) normalised xy -> framing-invariant gaze + head-pose scalars."""
    p = lm  # normalised
    lh, lv, lear = _gaze_ratio(p[L_IRIS], p[L_IN], p[L_OUT], p[L_TOP], p[L_BOT])
    rh, rv, rear = _gaze_ratio(p[R_IRIS], p[R_IN], p[R_OUT], p[R_TOP], p[R_BOT])
    gh, gv = (lh + rh) / 2.0, (lv + rv) / 2.0
    lm_px = lm * np.array([w, h], np.float32)
    yaw, pitch, roll = _head_pose(lm_px, w, h)
    return {
        "gaze_h": gh, "gaze_v": gv,
        "gaze_mag": float(np.hypot(gh, gv)),            # aversion from centre
        "gaze_asym": float(lh - rh),                    # vergence / L-R disagreement
        "ear_l": lear, "ear_r": rear, "ear_mean": float((lear + rear) / 2.0),
        "yaw": yaw, "pitch": pitch, "roll": roll,
    }


_SIGNALS = ["gaze_h", "gaze_v", "gaze_mag", "gaze_asym", "ear_l", "ear_r", "ear_mean",
            "yaw", "pitch", "roll"]


def _agg(traj):
    """traj: (T, K) -> reaction-aware stats per signal + global gaze/head motion energy."""
    T = traj.shape[0]
    k = max(1, T // 4)
    vel = np.abs(np.diff(traj, axis=0)) if T > 1 else np.zeros((1, traj.shape[1]))
    acc = np.abs(np.diff(traj, axis=0, n=2)) if T > 2 else np.zeros((1, traj.shape[1]))
    apex = np.sort(traj, axis=0)[-k:].mean(0)
    out = {}
    for j, s in enumerate(_SIGNALS):
        c = traj[:, j]
        out[f"gaze_{s}_mean"] = float(c.mean())
        out[f"gaze_{s}_std"] = float(c.std())
        out[f"gaze_{s}_range"] = float(c.max() - c.min())
        out[f"gaze_{s}_apex"] = float(apex[j])
        out[f"gaze_{s}_maxvel"] = float(vel[:, j].max())
        out[f"gaze_{s}_maxacc"] = float(acc[:, j].max())
        out[f"gaze_{s}_tmax"] = float(c.argmax() / max(1, T))
    gx, gy = _SIGNALS.index("gaze_h"), _SIGNALS.index("gaze_v")
    out["gaze_motion_energy"] = float(np.sqrt(vel[:, gx] ** 2 + vel[:, gy] ** 2).sum())
    out["gaze_motion_peak"] = float(np.sqrt(vel[:, gx] ** 2 + vel[:, gy] ** 2).max())
    yi, pi = _SIGNALS.index("yaw"), _SIGNALS.index("pitch")
    out["head_motion_energy"] = float(np.sqrt(vel[:, yi] ** 2 + vel[:, pi] ** 2).sum())
    return out


class GazeExtractor(BaseExtractor):
    name = "gaze"

    def __init__(self, s: int = 48, workers: int = 6, cache_dir=None):
        super().__init__(s=s, workers=workers, cache_dir=cache_dir)

    def init_worker(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.3)

    def extract_clip(self, track, participant, video) -> dict:
        import cv2
        frames = sample_frames(video_path(track, participant, video), self.s)
        rows, n_det = [], 0
        for _, fr in frames:
            h, w = fr.shape[:2]
            res = self.mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            pts = res.multi_face_landmarks[0].landmark
            lm = np.array([[p.x, p.y] for p in pts], np.float32)
            if lm.shape[0] < 478:
                continue
            rows.append(_frame_signals(lm, w, h))
            n_det += 1
        if not rows:
            return {"n_det": 0}
        traj = np.array([[r[s] for s in _SIGNALS] for r in rows], np.float32)
        out = {"n_det": n_det}
        out.update(_agg(traj))
        return out
