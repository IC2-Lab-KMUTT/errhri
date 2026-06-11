"""Apex / reaction-spike modelling — the formulation we previously skipped.

The BAD task is a brief *reaction* to a failure: a surprise/amusement apex that lasts ~1-2 s inside
a 14-22 s clip. Whole-clip MEAN aggregation (our FeatureBank streams) washes that spike out — which
is exactly why the aggregated-AU ceiling sits at ~0.67 macro-F1 and the greedy optimiser rejected
every temporal stream. This module tests the alternative the literature (and the leaderboard's
"macro-F1 + max-vote") points to:

  * `run_apex`            — clip vector built from APEX / velocity / acceleration / top-k pooling of
                            the per-frame AU trajectory (not mean/std). The reaction *magnitude* and
                            *sharpness*, kept instead of averaged away.
  * `run_window_maxvote`  — a window-level classifier: slice each clip into short windows, label each
                            with the clip label, predict per-window reaction probability, aggregate to
                            the clip by MAX (does ANY window look like a reaction?). The direct
                            max-vote formulation. Honest subject-grouped CV throughout.

All on the py-feat AU frame cache (`au_frames`). Run:  python -m pipelines.apex
"""
from __future__ import annotations
import numpy as np
from errhri_features import CVEvaluator, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from .models import make_model
from .sequences import FrameSequenceBank
from .recipes import _spw


def _all_metrics(track, y, pred, p):
    from sklearn.metrics import f1_score
    return (f"macroF1={f1_score(y, pred, average='macro'):.3f} "
            f"wF1={f1_score(y, pred, average='weighted'):.3f} "
            f"F1pos={f1_score(y, pred, pos_label=1):.3f} "
            f"F1neg={f1_score(y, pred, pos_label=0):.3f}")


# --------------------------------------------------------------------------- #
#  Apex / dynamics clip vector
# --------------------------------------------------------------------------- #
def apex_feats(X):
    """X: (N, L, C) per-frame AU trajectory -> clip features keeping the REACTION, not the mean.

    Per channel: top-k (apex) mean, global max/range, |velocity| and |acceleration| peaks, the best
    sliding-window mean (the reaction window), and the normalised time-of-peak (when the reaction
    happens). These are length-invariant and target the spike a mean would erase."""
    N, L, C = X.shape
    k = max(1, L // 10)
    w = max(2, L // 5)
    topk = np.sort(X, axis=1)[:, -k:, :].mean(1)             # apex magnitude
    mx = X.max(1); mn = X.min(1); rng = mx - mn
    std = X.std(1)
    vel = np.diff(X, axis=1) if L > 1 else np.zeros((N, 1, C))
    vmax = np.abs(vel).max(1); vmean = np.abs(vel).mean(1); vstd = vel.std(1)
    acc = np.diff(vel, axis=1) if vel.shape[1] > 1 else np.zeros((N, 1, C))
    amax = np.abs(acc).max(1)
    csum = np.cumsum(X, axis=1)
    wins = (csum[:, w:, :] - csum[:, :-w, :]) / w            # sliding-window means
    winmax = wins.max(1) if wins.shape[1] > 0 else mx        # best reaction window
    tmax = X.argmax(1) / max(1, L)                           # WHEN the apex occurs
    return np.concatenate([topk, mx, rng, std, vmax, vmean, vstd, amax, winmax, tmax], axis=1)


def run_apex(track, n_splits=5, L=64, return_oof=False):
    fs = FrameSequenceBank(track, L=L).load()
    X, y, g = fs.matrix(); X = np.nan_to_num(np.asarray(X, np.float32))
    print(f"  [apex] T{track} seq shape={X.shape}")
    F = apex_feats(X)
    rep = CVEvaluator(track, n_splits).run_matrix(
        make_model("xgb", scale_pos_weight=_spw(y)), F, y, g, fs.n_frames)
    pred = np.zeros(len(y), int); fid = subject_folds(g, n_splits)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(y[tr], rep.oof_prob[tr]); pred[va] = (rep.oof_prob[va] >= thr).astype(int)
    print(f"  [apex solo] T{track}: {rep}  | {_all_metrics(track, y, pred, rep.oof_prob)}")
    if return_oof:
        keys = list(zip(fs.df.participant.astype(str), fs.df.video.astype(str)))
        return dict(zip(keys, rep.oof_prob)), y, g
    return rep


# --------------------------------------------------------------------------- #
#  Window-level model + MAX-vote aggregation
# --------------------------------------------------------------------------- #
def _window_features(seg):
    """seg: (w, C) -> a window descriptor (level + spread + peak + motion)."""
    vel = np.diff(seg, axis=0) if seg.shape[0] > 1 else np.zeros((1, seg.shape[1]))
    return np.concatenate([seg.mean(0), seg.std(0), seg.max(0), seg.min(0), np.abs(vel).max(0)])


def run_window_maxvote(track, win=None, stride=None, n_splits=5, L=64, return_oof=False):
    fs = FrameSequenceBank(track, L=L).load()
    X, y, g = fs.matrix(); X = np.nan_to_num(np.asarray(X, np.float32))
    N, L, C = X.shape
    win = win or max(3, L // 4); stride = stride or max(1, win // 2)
    starts = list(range(0, max(1, L - win + 1), stride)) or [0]
    WF, wy, wg, wclip = [], [], [], []
    for i in range(N):
        for s in starts:
            WF.append(_window_features(X[i, s:s + win, :]))
            wy.append(y[i]); wg.append(g[i]); wclip.append(i)
    WF = np.asarray(WF, np.float32); wy = np.array(wy); wg = np.array(wg); wclip = np.array(wclip)
    print(f"  [maxvote] T{track} {len(starts)} win/clip (win={win},stride={stride}) -> {WF.shape} window samples")

    clip_p = np.full(N, np.nan)
    fid = subject_folds(wg, n_splits)                       # group by SUBJECT across windows
    for tr, va in iter_folds(fid):
        m = make_model("xgb", scale_pos_weight=_spw(wy[tr]))()
        m.fit(WF[tr], wy[tr])
        p = m.predict_proba(WF[va])[:, 1]
        for ci in np.unique(wclip[va]):                    # MAX-vote aggregate to clip
            clip_p[ci] = p[wclip[va] == ci].max()
    # honest clip-level threshold tuning (clip_p is already subject-held-out OOF)
    pred = np.zeros(N, int); fidc = subject_folds(g, n_splits)
    for tr, va in iter_folds(fidc):
        thr = M.tune_threshold(y[tr], clip_p[tr]); pred[va] = (clip_p[va] >= thr).astype(int)
    from errhri_features.leak import length_corr
    from errhri_features.evaluation import Report
    vm = M.video_metrics(track, y, pred, clip_p); ci = M.subject_bootstrap_ci(track, g, y, pred, clip_p)
    rep = Report(track, vm["primary"], ci, vm["auc"], vm["f1_neg"], length_corr(clip_p, fs.n_frames),
                 clip_p, pred)
    print(f"  [window+maxvote] T{track}: {rep}  | {_all_metrics(track, y, pred, clip_p)}")
    if return_oof:
        keys = list(zip(fs.df.participant.astype(str), fs.df.video.astype(str)))
        return dict(zip(keys, clip_p)), y, g
    return rep


def main():
    for t in (1, 2):
        for L in (32, 64, 128):
            print(f"\n=== Track {t}: apex / max-vote reaction modelling (L={L}) ===")
            run_apex(t, L=L)
            run_window_maxvote(t, L=L)


if __name__ == "__main__":
    main()
