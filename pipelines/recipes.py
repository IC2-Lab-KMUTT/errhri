"""Recipes — the modifiable layer. Run a stream / fusion / temporal-GRU / submission by editing a
config, not the core. Every recipe is a thin wrapper over the core (`FeatureBank`, `CVEvaluator`,
`late_fusion`) so the core signal pipeline stays untouched while you experiment here.

Change behaviour by changing PARAMS, e.g.::

    from pipelines.recipes import Stream, run_stream, run_fusion, run_temporal

    run_stream(1, Stream(modalities=("au", "audio"), model="xgb", params={"max_depth": 4}))
    run_stream(1, Stream(modalities=("au",), model="logreg"))           # swap the model
    run_fusion(1, [Stream(("au",)), Stream(("audio",)), Stream(("embed",))], method="stack")
    run_fusion(1, [...], include_temporal=True, gru={"hidden": 96, "epochs": 60})
    run_temporal(1, hidden=64, epochs=40)

`python -m pipelines.recipes` runs a small demo on whatever caches you have.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from errhri_features import FeatureBank, CVEvaluator, late_fusion
from .models import make_model, ClipGRUClassifier
from .sequences import SequenceBank


@dataclass
class Stream:
    """One feature stream + model. The unit you compose and tweak.

    modalities : which caches to merge, e.g. ("au",) or ("au", "audio", "embed")
    model      : a key in models.MODEL_ZOO ("xgb" | "logreg" | "rf")
    select     : "signal" (drop measured noise) | "all" | explicit column list
    leak_clean : strip duration-proxy features (|corr n_frames| > 0.30) — keep True on T1
    params     : overrides passed straight to the model builder (max_depth, C, ...)
    """
    modalities: tuple = ("au",)
    model: str = "xgb"
    select: str = "signal"
    leak_clean: bool = True
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "+".join(self.modalities) + f":{self.model}"


def _spw(y):
    return (y == 0).sum() / max(1, (y == 1).sum())


def run_stream(track, stream: Stream = None, n_splits=5):
    """Train+evaluate one stream. Returns a core `Report` (primary metric + CI + AUC + leak)."""
    stream = stream or Stream()
    bank = FeatureBank(track, stream.modalities).load()
    params = dict(stream.params)
    if stream.model in ("xgb", "rf"):
        params.setdefault("scale_pos_weight", _spw(bank.y))
    factory = make_model(stream.model, **params)
    return CVEvaluator(track, n_splits).run(factory, bank, select=stream.select,
                                            leak_clean=stream.leak_clean)


def run_temporal(track, n_splits=5, **gru_params):
    """Whole-clip temporal GRU on the raw trajectory (`traj` cache). gru_params -> ClipGRUClassifier
    (hidden, epochs, lr, dropout, bidir, ...)."""
    seq = SequenceBank(track).load()
    X, y, groups = seq.matrix()
    spw = _spw(y)
    return CVEvaluator(track, n_splits).run_matrix(
        lambda: ClipGRUClassifier(pos_weight=spw, **gru_params), X, y, groups, seq.n_frames)


def _oof_by_key(track, stream, n_splits):
    bank = FeatureBank(track, stream.modalities).load()
    X, y, g = bank.matrix(select=stream.select, leak_clean=stream.leak_clean)
    if X.shape[1] == 0:
        return None
    params = dict(stream.params)
    if stream.model in ("xgb", "rf"):
        params.setdefault("scale_pos_weight", _spw(y))
    rep = CVEvaluator(track, n_splits).run_matrix(make_model(stream.model, **params),
                                                  X, y, g, bank.n_frames)
    keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str)))
    return dict(zip(keys, rep.oof_prob))


def run_fusion(track, streams, method="stack", include_temporal=False, gru=None, n_splits=5):
    """Late-fuse several streams (+ optionally the temporal GRU). method='mean' | 'stack'.
    Streams are aligned by (participant, video), so a partial `traj` cache can't misalign them."""
    ref = FeatureBank(track, ["au"]).load()
    keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
    y = dict(zip(keys, ref.y)); grp = dict(zip(keys, ref.groups)); nfr = dict(zip(keys, ref.n_frames))

    oofs = {}
    for s in streams:
        o = _oof_by_key(track, s, n_splits)
        if o is None:
            print(f"  [{s.name}] no signal columns on T{track} — skipped"); continue
        oofs[s.name] = o
    if include_temporal:
        seq = SequenceBank(track).load()
        X, sy, sg = seq.matrix()
        rep = CVEvaluator(track, n_splits).run_matrix(
            lambda: ClipGRUClassifier(pos_weight=_spw(sy), **(gru or {})), X, sy, sg, seq.n_frames)
        skeys = list(zip(seq.df.participant.astype(str), seq.df.video.astype(str)))
        oofs["facial_gru"] = dict(zip(skeys, rep.oof_prob))

    if len(oofs) < 2:
        raise ValueError(f"need >=2 usable streams to fuse, got {list(oofs)}")
    common = [k for k in keys if all(k in o for o in oofs.values())]
    P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
    yv = np.array([y[k] for k in common]); gv = np.array([grp[k] for k in common])
    nv = np.array([nfr[k] for k in common])
    return late_fusion(track, P, yv, gv, nv, method=method)


def main():
    for t in (1, 2):
        print(f"\n=== Track {t} ===")
        print("  au stream     ", run_stream(t, Stream(("au",))))
        try:
            print("  au+audio fuse ", run_fusion(t, [Stream(("au",)), Stream(("audio",))]))
        except (FileNotFoundError, ValueError) as e:
            print("  fusion skipped:", e)


if __name__ == "__main__":
    main()
