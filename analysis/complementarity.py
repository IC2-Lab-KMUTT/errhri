"""Cross-stream fusion potential — the methodology behind the FUSION table.

The point we kept re-learning: judge a stream by how its ERRORS decorrelate from the others, not by
its solo score. A weak stream (audio sep ~0.58, embed ~0.51) earns its place in the ensemble if it
is *wrong on different clips*. This runs every stream through the SAME subject-grouped folds,
collects out-of-fold probabilities, and reports:

  solo        each stream's own primary metric
  prob corr   pairwise correlation of OOF probabilities (low = diverse)
  err corr    pairwise correlation of |y - p| (low = errors decorrelate = fusion headroom)
  oracle      fraction of clips at least one stream gets right (upper bound of a perfect selector)
  fusion      mean + logistic-stack late fusion (the validated T1 win: 0.623 single -> 0.674)

Streams: tree streams (au / audio / embed) via XGB on FeatureBank; the facial_gru stream (whole-
clip temporal GRU on SequenceBank) is included automatically if the `traj` cache exists.

    python -m analysis.complementarity
"""
from __future__ import annotations
import numpy as np
from errhri_features import FeatureBank, CVEvaluator, late_fusion, metrics as M
from errhri_features.config import CACHE_DIR
from pipelines.models import make_xgb, ClipGRUClassifier
from pipelines.sequences import SequenceBank

TREE_STREAMS = ["au", "audio", "embed"]


def _key(df):
    return list(zip(df.participant.astype(str), df.video.astype(str)))


def _tree_oof(track, mod, spw):
    bank = FeatureBank(track, [mod]).load()
    X, y, groups = bank.matrix(select="signal", leak_clean=True)
    if X.shape[1] == 0:                # e.g. audio/embed on T2 are dropped as noise -> no columns
        return None
    rep = CVEvaluator(track).run_matrix(lambda: make_xgb(spw), X, y, groups, bank.n_frames)
    return dict(zip(_key(bank.df), rep.oof_prob))


def _gru_oof(track, spw):
    seq = SequenceBank(track).load()
    X, y, groups = seq.matrix()
    rep = CVEvaluator(track).run_matrix(
        lambda: ClipGRUClassifier(pos_weight=spw, epochs=40), X, y, groups, seq.n_frames)
    return dict(zip(_key(seq.df), rep.oof_prob))


def run(track):
    ref = FeatureBank(track, ["au"]).load()
    keys = _key(ref.df)
    y = dict(zip(keys, ref.y)); grp = dict(zip(keys, ref.groups))
    nfr = dict(zip(keys, ref.n_frames))
    spw = (ref.y == 0).sum() / max(1, (ref.y == 1).sum())

    streams = {}
    for mod in TREE_STREAMS:
        try:
            oof = _tree_oof(track, mod, spw)
        except FileNotFoundError:
            print(f"  [{mod}] cache missing — skipped", flush=True); continue
        if oof is None:
            print(f"  [{mod}] no signal columns on T{track} — skipped", flush=True); continue
        streams[mod] = oof
        print(f"  [{mod}] OOF done", flush=True)
    if (CACHE_DIR / f"traj_t{track}.csv").exists():
        streams["facial_gru"] = _gru_oof(track, spw)
        print("  [facial_gru] OOF done", flush=True)

    common = set(keys)
    for s in streams.values():
        common &= set(s)
    common = [k for k in keys if k in common]
    if len(streams) < 2:
        print(f"Track {track}: need >=2 streams to compare."); return
    yv = np.array([y[k] for k in common])
    gv = np.array([grp[k] for k in common])
    nv = np.array([nfr[k] for k in common])
    P = {s: np.array([streams[s][k] for k in common]) for s in streams}
    names = list(P)

    print(f"\n=== Track {track}: complementarity ({len(common)} clips, {len(names)} streams) ===")
    print("  solo:")
    err = {}
    for s in names:
        pred = (P[s] >= M.tune_threshold(yv, P[s])).astype(int)
        print(f"    {s:12} {M.primary(track, yv, pred, P[s]):.3f}")
        err[s] = np.abs(yv - P[s])

    print("  prob corr / err corr (off-diagonal; low = complementary):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pc = np.corrcoef(P[a], P[b])[0, 1]
            ec = np.corrcoef(err[a], err[b])[0, 1]
            print(f"    {a:12}~{b:12} prob {pc:+.2f}  err {ec:+.2f}")

    # oracle: at least one stream correct (per-stream tuned threshold)
    correct = np.zeros(len(common), bool)
    for s in names:
        pred = (P[s] >= M.tune_threshold(yv, P[s])).astype(int)
        correct |= (pred == yv)
    print(f"  oracle coverage (>=1 stream correct): {correct.mean():.3f}")

    for method in ("mean", "stack"):
        rep = late_fusion(track, P, yv, gv, nv, method=method)
        print(f"  fusion[{method}]  {rep}")


def main():
    for t in (1, 2):
        run(t)


if __name__ == "__main__":
    main()
