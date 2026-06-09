"""Official windowed submission format + video-level scoring (the "submission eval").

The challenge wants one CSV row per SLIDING WINDOW per clip:
    participant_id,video_id,window_id,y_pred,y_prob_0,y_prob_1
plus declared fps,window_size,slide. Video score: Track 1 = MAJORITY VOTE of y_pred -> macro-F1;
Track 2 = MAX y_prob_1 -> AUC. Constraints: slide<=window_size (both); window_size<=2*fps (T1).

For a clip-level model, replicate the clip's prediction across its windows — the video-level
aggregation then reproduces the clip-level score. For a windowed model, pass per-window probs.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from .config import TRACKS
from . import metrics as M


def windows_for_clip(n_frames: int, window_size: int, slide: int):
    bounds, s = [], 0
    while s < n_frames:
        bounds.append((s, min(s + window_size, n_frames)))
        if s + window_size >= n_frames:
            break
        s += slide
    return bounds or [(0, max(n_frames, 1))]


def _check_constraints(track, fps, window_size, slide):
    assert slide <= window_size, "slide must be <= window_size"
    if track == 1:
        assert window_size <= 2 * fps, "Track 1: window_size must be <= 2*fps"


def write_submission(track, keys, clip_prob, clip_pred, n_frames, path,
                     fps=None, window_size=None, slide=None) -> pd.DataFrame:
    """keys: list of (participant, video); clip_prob/clip_pred/n_frames: aligned arrays.
    Emits the official per-window CSV (clip prediction replicated across its windows)."""
    cfg = TRACKS[track]
    fps = fps or cfg["fps"]; window_size = window_size or cfg["window_size"]; slide = slide or cfg["slide"]
    _check_constraints(track, fps, window_size, slide)
    rows = []
    for (pid, vid), p1, yp, nf in zip(keys, clip_prob, clip_pred, n_frames):
        for wi, _ in enumerate(windows_for_clip(int(nf), window_size, slide)):
            rows.append(dict(participant_id=pid, video_id=vid, window_id=wi,
                             y_pred=int(yp), y_prob_0=float(1 - p1), y_prob_1=float(p1)))
    df = pd.DataFrame(rows)
    df.attrs.update(fps=fps, window_size=window_size, slide=slide)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def aggregate_video(track, submission_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-window rows to one row per video using the official aggregation."""
    out = []
    for (pid, vid), g in submission_df.groupby(["participant_id", "video_id"]):
        if TRACKS[track]["agg"] == "majority":
            yp = int(round(g.y_pred.mean()))           # majority vote
            prob = float(g.y_prob_1.mean())
        else:
            prob = float(g.y_prob_1.max())             # max prob
            yp = int(prob >= 0.5)
        out.append(dict(participant=str(pid), video=str(vid), y_pred=yp, y_prob_1=prob))
    return pd.DataFrame(out)


def official_score(track, submission_df, gt_df) -> dict:
    """Video-level official metric from a submission + ground truth (participant,video,label)."""
    vid = aggregate_video(track, submission_df)
    gt = gt_df.copy()
    gt["participant"] = gt.participant.astype(str); gt["video"] = gt.video.astype(str)
    m = vid.merge(gt[["participant", "video", "label"]], on=["participant", "video"])
    y = m.label.to_numpy(int)
    return M.video_metrics(track, y, m.y_pred.to_numpy(int), m.y_prob_1.to_numpy(float))


def run_official_evaluator(script_path, submission_csv, gt_csv) -> str:
    """Shell out to the vendored official eval script if you have it (drop it in official/)."""
    r = subprocess.run(["python", str(script_path), str(submission_csv), str(gt_csv)],
                       capture_output=True, text=True)
    return r.stdout + r.stderr
