"""Official-evaluator bridge — measure what the challenge actually ranks.

We had been optimising clip-level macro-F1 on the 87/13 trainval distribution. The challenge ranks
Track 1 by **video-level macro-F1 via majority vote across windows** (ties -> failure), balanced
accuracy as tiebreaker, on a control-scarce test set (~40 failure / 6 control). This module wires our
model's OOF clip probabilities into the official submission/GT CSV format, runs the real `eval.py`,
and reports the official numbers — so optimisation targets the true objective, and steps 2-3 (CNN
features, higher fps) have a faithful measurement harness.

Because all windows of a clip share the label, a clip-level prediction maps to the official metric by
replicating it across that clip's windows (majority vote then returns the clip prediction). So this is
an exact bridge, not an approximation.

    from pipelines.official import official_report
    official_report(1)                      # best ensemble -> official eval.py numbers
"""
from __future__ import annotations
import os, sys, subprocess, tempfile
import numpy as np
import pandas as pd
from errhri_features import FeatureBank, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from .recipes import _oof_by_key, _fuse_oof, Stream

REPO_EVAL = os.path.expanduser("~/research/errhri/repo/eval.py")
DEFAULT_STREAMS = {"xgb_au": Stream(("au",), model="xgb"),
                   "rf_au":  Stream(("au",), model="rf"),
                   "blend":  Stream(("blend",), model="xgb")}
FPS = {1: 5, 2: 30}
WS = {1: 10, 2: 10}
SLIDE = {1: 5, 2: 2}


def best_oof(track, streams=None, n_splits=5):
    """OOF clip probabilities from the late-fusion stack over `streams`. Returns aligned arrays."""
    streams = streams or DEFAULT_STREAMS
    ref = FeatureBank(track, ["au"]).load()
    keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
    yk = dict(zip(keys, ref.y)); gk = dict(zip(keys, ref.groups)); nk = dict(zip(keys, ref.n_frames))
    oofs = {}
    for n, s in streams.items():
        o = _oof_by_key(track, s, n_splits)
        if o:
            oofs[n] = o
    common = [k for k in keys if all(k in o for o in oofs.values())]
    P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
    y = np.array([yk[k] for k in common]); g = np.array([gk[k] for k in common])
    nfr = np.array([nk[k] for k in common])
    pf = _fuse_oof(track, P, y, g, method="stack", C=1.0)
    return common, pf, y, g, nfr


def _tune_threshold(track, pf, y, g, objective="macro"):
    """Pick the decision threshold that maximises the chosen OFFICIAL objective under honest
    per-fold tuning. objective: 'macro' (macro-F1) or 'bal' (balanced accuracy)."""
    from sklearn.metrics import f1_score, balanced_accuracy_score
    def score(yy, pp, thr):
        pr = (pp >= thr).astype(int)
        return f1_score(yy, pr, average="macro") if objective == "macro" else balanced_accuracy_score(yy, pr)
    pred = np.zeros(len(y), int); fid = subject_folds(g, 5)
    grid = np.quantile(pf, np.linspace(0.02, 0.98, 97))
    for tr, va in iter_folds(fid):
        best = max(grid, key=lambda t: score(y[tr], pf[tr], t))
        pred[va] = (pf[va] >= best).astype(int)
    return pred


def write_official_csvs(track, keys, pred, pf, nfr, gt_path, sub_path):
    """Replicate clip pred/prob across the clip's official windows + emit frame-level GT."""
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    gt_rows, sub_rows = [], []
    for (pid, vid), yp, prob, nf in zip(keys, pred, pf, nfr):
        nf = int(max(nf, ws))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(round(prob >= 0.5))))   # placeholder; true label set below
        n_win = max((nf - ws) // slide + 1, 1)
        for w in range(n_win):
            sub_rows.append((pid, vid, w, int(yp), 1.0 - prob, prob))
    return gt_rows, sub_rows, (fps, ws, slide)


def official_report(track, streams=None, objective="macro", n_splits=5):
    common, pf, y, g, nfr = best_oof(track, streams, n_splits)
    pred = _tune_threshold(track, pf, y, g, objective)
    fps, ws, slide = FPS[track], WS[track], SLIDE[track]
    # frame-level GT (true labels) + window-level submission (clip pred replicated)
    gt_rows, sub_rows = [], []
    for (pid, vid), yt, yp, prob, nf in zip(common, y, pred, pf, nfr):
        nf = int(max(nf, ws))
        for f in range(1, nf + 1):
            gt_rows.append((pid, vid, f, int(yt)))
        n_win = max((nf - ws) // slide + 1, 1)
        for w in range(n_win):
            sub_rows.append((pid, vid, w, int(yp), float(1.0 - prob), float(prob)))
    gt = pd.DataFrame(gt_rows, columns=["participant_id", "video_id", "frame_id", "y_true"])
    sub = pd.DataFrame(sub_rows, columns=["participant_id", "video_id", "window_id",
                                          "y_pred", "y_prob_0", "y_prob_1"])
    d = tempfile.mkdtemp(prefix="errhri_official_")
    gtp, subp = os.path.join(d, "gt.csv"), os.path.join(d, "sub.csv")
    gt.to_csv(gtp, index=False); sub.to_csv(subp, index=False)
    print(f"=== Official eval.py — Track {track} (objective={objective}, fps={fps}, ws={ws}, slide={slide}) ===")
    print(f"  (OOF trainval holdout, {len(common)} clips, {int((y==0).sum())} control / {int((y==1).sum())} failure)")
    out = subprocess.run([sys.executable, REPO_EVAL, "--gt", gtp, "--pred", subp, "--track", str(track),
                          "--fps", str(fps), "--window_size", str(ws), "--slide", str(slide)],
                         capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print("  [eval.py stderr]", out.stderr[-500:])
    return gtp, subp


def main():
    for t in (1, 2):
        for obj in ("macro", "bal"):
            official_report(t, objective=obj)


if __name__ == "__main__":
    main()
