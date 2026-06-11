"""Dense per-frame SEQUENCE caches — same pipelines as `au_graph` / `gaze`, but SAVE the trajectory.

The aggregated extractors sampled 48 frames, computed per-frame AU/gaze/head-pose values, then kept
only summary stats — the sequence itself was discarded, which blocked every temporal model on the
strong channels (au_frames is only 10 frames; traj is blendshapes only). These subclasses re-run the
identical per-frame pipelines and write the full trajectory, linearly resampled to a fixed L so the
cache is a flat wide CSV in the same `t{tt:02d}__{channel}` format as `traj` (SequenceBank-style).

  au_seq   : OpenGraphAU 41 AU probabilities per frame  -> 48 x 41 cols  (GPU, dream-venv)
  gaze_seq : 10 gaze/EAR/head-pose signals per frame    -> 48 x 10 cols  (CPU MediaPipe)

One-time job; afterwards every temporal experiment reads these caches.
"""
from __future__ import annotations
import numpy as np
from .au_graph import AUGraphExtractor, AU_NAMES
from .gaze import GazeExtractor, _frame_signals, _SIGNALS
from ..datasets import video_path, sample_frames


def _resample_traj(A, L):
    """A: (T, C) -> (L, C) by per-channel linear interpolation (T may vary with detection)."""
    T, C = A.shape
    if T == L:
        return A
    if T == 1:
        return np.repeat(A, L, axis=0)
    xs, xt = np.linspace(0, 1, L), np.linspace(0, 1, T)
    return np.stack([np.interp(xs, xt, A[:, c]) for c in range(C)], axis=1)


def _flat(A, names):
    """(L, C) trajectory -> {t{tt:02d}__{name}: value} flat dict (traj-cache convention)."""
    out = {}
    for t in range(A.shape[0]):
        for c, nm in enumerate(names):
            out[f"t{t:02d}__{nm}"] = float(A[t, c])
    return out


class AUSeqExtractor(AUGraphExtractor):
    name = "au_seq"

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
        A = np.concatenate(outs, 0)                                   # (T, 41)
        row = {"n_det": len(crops)}
        row.update(_flat(_resample_traj(A, self.s), AU_NAMES))
        return row


class GazeSeqExtractor(GazeExtractor):
    name = "gaze_seq"

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
        traj = np.array([[r[s] for s in _SIGNALS] for r in rows], np.float32)   # (T, 10)
        out = {"n_det": n_det}
        out.update(_flat(_resample_traj(traj, self.s), _SIGNALS))
        return out
