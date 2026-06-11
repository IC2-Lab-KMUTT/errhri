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
from errhri_features import FeatureBank, CVEvaluator, late_fusion, metrics as M
from errhri_features.splits import subject_folds, iter_folds
from .models import make_model, ClipGRUClassifier, RocketClassifier
from .sequences import SequenceBank, FrameSequenceBank


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


# --------------------------------------------------------------------------- #
#  Research recipe: ROCKET temporal stream + RF + calibrated fusion
# --------------------------------------------------------------------------- #
def run_temporal_rocket(track, n_splits=5, **kw):
    """ROCKET (random conv kernels) on the per-frame AU/pose/geometry trajectory (`au_frames`
    cache). A temporal view with a different inductive bias from the GRU and the trees."""
    fs = FrameSequenceBank(track).load()
    X, y, g = fs.matrix()
    return CVEvaluator(track, n_splits).run_matrix(lambda: RocketClassifier(**kw), X, y, g,
                                                   fs.n_frames)


def _oof_rocket(track, n_splits, **kw):
    fs = FrameSequenceBank(track).load()
    X, y, g = fs.matrix()
    rep = CVEvaluator(track, n_splits).run_matrix(lambda: RocketClassifier(**kw), X, y, g, fs.n_frames)
    keys = list(zip(fs.df.participant.astype(str), fs.df.video.astype(str)))
    return dict(zip(keys, rep.oof_prob))


def _oof_gru(track, n_splits, gru_kw=None):
    """facial-GRU stream: whole-clip temporal GRU on the per-frame **py-feat AU** trajectory
    (`au_frames` via FrameSequenceBank). Proper FACS AUs, not MediaPipe blendshapes — AU is the
    validated facial signal (task #93), so the temporal facial stream rides the AU channels too."""
    fs = FrameSequenceBank(track).load()
    X, y, g = fs.matrix()
    rep = CVEvaluator(track, n_splits).run_matrix(
        lambda: ClipGRUClassifier(pos_weight=_spw(y), **(gru_kw or {})), X, y, g, fs.n_frames)
    keys = list(zip(fs.df.participant.astype(str), fs.df.video.astype(str)))
    return dict(zip(keys, rep.oof_prob))


def _fuse_probs(track, P, y, groups, n_frames, method="stack", calibrate=False):
    """Late-fuse stream OOF prob dict (already aligned arrays). If calibrate, isotonic-recalibrate
    each stream INSIDE the subject-grouped meta CV (train-only fit) before stacking — honest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    names = list(P); fid = subject_folds(groups, n_splits=5)
    pf = np.zeros(len(y))
    for tr, va in iter_folds(fid):
        cols_tr, cols_va = [], []
        for n in names:
            p = P[n]
            if calibrate:
                iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
                cols_tr.append(iso.transform(p[tr])); cols_va.append(iso.transform(p[va]))
            else:
                cols_tr.append(p[tr]); cols_va.append(p[va])
        Atr, Ava = np.column_stack(cols_tr), np.column_stack(cols_va)
        if method == "mean":
            pf[va] = Ava.mean(1)
        else:
            pf[va] = LogisticRegression(max_iter=500).fit(Atr, y[tr]).predict_proba(Ava)[:, 1]
    pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(y[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    from errhri_features.evaluation import Report
    from errhri_features.leak import length_corr
    vm = M.video_metrics(track, y, pred, pf)
    ci = M.subject_bootstrap_ci(track, groups, y, pred, pf)
    return Report(track, vm["primary"], ci, vm["auc"], vm["f1_neg"],
                  length_corr(pf, n_frames), pf, pred)


def run_research_fusion(track, n_splits=5, rocket_kw=None, include_facial=False, gru_kw=None):
    """The research recipe: assemble diverse streams (RF, XGB, audio, embed, ROCKET-temporal, and
    — with include_facial — the temporal facial_gru stream on the py-feat AU trajectory), report
    per-stream solo + error-decorrelation, then fuse (uncalibrated vs isotonic-calibrated) with
    per-stream drop ablations. All facial signal is proper FACS AU (au / au_frames), never MediaPipe
    blendshapes (weaker + duplicative). include_facial=True adds facial_gru (needs au_frames cache)."""
    ref = FeatureBank(track, ["au"]).load()
    keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
    y = dict(zip(keys, ref.y)); grp = dict(zip(keys, ref.groups)); nfr = dict(zip(keys, ref.n_frames))

    builders = {
        "rf_au":   lambda: _oof_by_key(track, Stream(("au",), model="rf"), n_splits),
        "xgb_au":  lambda: _oof_by_key(track, Stream(("au",), model="xgb"), n_splits),
        "audio":   lambda: _oof_by_key(track, Stream(("audio",), model="xgb"), n_splits),
        "embed":   lambda: _oof_by_key(track, Stream(("embed",), model="xgb"), n_splits),
        "rocket":  lambda: _oof_rocket(track, n_splits, **(rocket_kw or {})),
    }
    if include_facial:
        builders["facial_gru"] = lambda: _oof_gru(track, n_splits, gru_kw)
    oofs = {}
    for name, build in builders.items():
        try:
            o = build()
        except FileNotFoundError:
            print(f"  [{name}] cache missing — skipped"); continue
        if o is None:
            print(f"  [{name}] no signal columns on T{track} — skipped"); continue
        oofs[name] = o; print(f"  [{name}] OOF done", flush=True)

    common = [k for k in keys if all(k in o for o in oofs.values())]
    P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
    yv = np.array([y[k] for k in common]); gv = np.array([grp[k] for k in common])
    nv = np.array([nfr[k] for k in common]); names = list(P)

    print(f"\n=== Track {track}: research fusion ({len(common)} clips, {len(names)} streams) ===")
    print("  solo:")
    err = {}
    for s in names:
        pred = (P[s] >= M.tune_threshold(yv, P[s])).astype(int)
        print(f"    {s:8} {M.primary(track, yv, pred, P[s]):.3f}")
        err[s] = np.abs(yv - P[s])
    print("  error corr vs rocket / rf (low = complementary):")
    for s in names:
        line = "    " + f"{s:8}"
        for ref_s in ("rocket", "rf_au"):
            if ref_s in err and ref_s != s:
                line += f"  {ref_s}:{np.corrcoef(err[s], err[ref_s])[0,1]:+.2f}"
        print(line)
    correct = np.zeros(len(common), bool)
    for s in names:
        correct |= ((P[s] >= M.tune_threshold(yv, P[s])).astype(int) == yv)
    print(f"  oracle (>=1 stream correct): {correct.mean():.3f}")

    out = {}
    for method in ("mean", "stack"):
        for cal in (False, True):
            tag = f"{method}{'+cal' if cal else ''}"
            rep = _fuse_probs(track, P, yv, gv, nv, method=method, calibrate=cal)
            out[tag] = rep; print(f"  fusion[{tag:9}] {rep}")
    # all-but-rocket / all-but-rf ablations (does each new stream add?)
    for drop in ("rocket", "rf_au"):
        if drop in P and len(P) > 2:
            sub = {k: v for k, v in P.items() if k != drop}
            rep = _fuse_probs(track, sub, yv, gv, nv, method="stack", calibrate=True)
            print(f"  ablation[-{drop:7} stack+cal] {rep}")
    return out


# --------------------------------------------------------------------------- #
#  MAX fusion: full stream zoo (every modality x model) + optimized fusion
# --------------------------------------------------------------------------- #
def _seq_bank(track, source):
    """source 'auf' -> per-frame py-feat AU trajectory; 'traj' -> landmark trajectory."""
    return FrameSequenceBank(track).load() if source == "auf" else SequenceBank(track).load()


def _oof_seq(track, n_splits, kind, source, **kw):
    bank = _seq_bank(track, source)
    X, y, g = bank.matrix()
    if kind == "rocket":
        fac = lambda: RocketClassifier(**kw)
    else:
        fac = lambda: ClipGRUClassifier(pos_weight=_spw(y), **kw)
    rep = CVEvaluator(track, n_splits).run_matrix(fac, X, y, g, bank.n_frames)
    keys = list(zip(bank.df.participant.astype(str), bank.df.video.astype(str)))
    return dict(zip(keys, rep.oof_prob))


def _score_oof(track, oof, yk):
    keys = list(oof); p = np.array([oof[k] for k in keys]); y = np.array([yk[k] for k in keys])
    pred = (p >= M.tune_threshold(y, p)).astype(int)
    return M.primary(track, y, pred, p)


def _best_tab(track, modalities, model, grid, n_splits, yk):
    """Pick the grid config with the best OOF primary metric. Returns (oof, params, score)."""
    best = None
    for params in grid:
        o = _oof_by_key(track, Stream(modalities, model=model, params=dict(params)), n_splits)
        if o is None:
            return None
        s = _score_oof(track, o, yk)
        if best is None or s > best[2]:
            best = (o, params, s)
    return best


def _best_seq(track, kind, source, grid, n_splits, yk):
    best = None
    for params in grid:
        try:
            o = _oof_seq(track, n_splits, kind, source, **params)
        except FileNotFoundError:
            return None
        s = _score_oof(track, o, yk)
        if best is None or s > best[2]:
            best = (o, params, s)
    return best


def _fuse_oof(track, P, y, groups, method="stack", C=1.0, penalty="l2"):
    """Honest subject-grouped meta-CV: fit the fuser on TRAIN folds only, emit OOF fused probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    names = list(P); fid = subject_folds(groups, n_splits=5); pf = np.zeros(len(y))
    for tr, va in iter_folds(fid):
        Atr = np.column_stack([P[n][tr] for n in names])
        Ava = np.column_stack([P[n][va] for n in names])
        if method == "mean":
            pf[va] = Ava.mean(1)
        elif method == "wmean":
            w = np.array([roc_auc_score(y[tr], P[n][tr]) if len(set(y[tr])) > 1 else 0.5
                          for n in names])
            w = np.clip(w - 0.5, 0, None)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
            pf[va] = Ava @ w
        else:
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            lr = LogisticRegression(max_iter=2000, C=C, penalty=penalty,
                                    solver=solver).fit(Atr, y[tr])
            pf[va] = lr.predict_proba(Ava)[:, 1]
    return pf


def _report_pf(track, y, groups, nfr, pf):
    from errhri_features.evaluation import Report
    from errhri_features.leak import length_corr
    fid = subject_folds(groups, n_splits=5); pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(y[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    vm = M.video_metrics(track, y, pred, pf)
    ci = M.subject_bootstrap_ci(track, groups, y, pred, pf)
    return Report(track, vm["primary"], ci, vm["auc"], vm["f1_neg"],
                  length_corr(pf, nfr), pf, pred)


def _primary_pf(track, y, groups, pf):
    fid = subject_folds(groups, n_splits=5); pred = np.zeros(len(y), int)
    for tr, va in iter_folds(fid):
        thr = M.tune_threshold(y[tr], pf[tr]); pred[va] = (pf[va] >= thr).astype(int)
    return M.primary(track, y, pred, pf)


def _greedy(track, P, y, groups, method="stack", C=1.0):
    """Forward-selection: keep adding the stream that most improves honest meta-CV primary."""
    remaining = list(P); chosen = []; best = -1.0
    while remaining:
        scored = []
        for s in remaining:
            sub = {k: P[k] for k in chosen + [s]}
            pf = P[s] if len(sub) == 1 else _fuse_oof(track, sub, y, groups, method=method, C=C)
            scored.append((_primary_pf(track, y, groups, pf), s))
        scored.sort(reverse=True, key=lambda t: t[0])
        if scored[0][0] <= best + 1e-4:
            break
        best = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
    return chosen, best


def run_max_fusion(track, n_splits=5, gru_kw=None):
    """The kitchen-sink optimizer: build EVERY modality x model stream (overlapping ones included),
    light per-stream grid tuning (chosen on OOF primary — mild optimism, noted), then sweep fusion
    strategies (mean / AUC-weighted / L2-stack over a C grid / sparse L1-stack / greedy forward
    selection) inside the honest subject-grouped meta-CV. Reports every fusion + the best.

    Facial signal is py-feat AU (au / au_frames) for the primary streams; the MediaPipe landmark
    trajectory + blendshape streams are kept ONLY as extra orthogonal views (they were additive),
    never as the main facial representation. GRU is not gridded (CPU cost) — one solid config."""
    gru_cfg = gru_kw or {"hidden": 64, "epochs": 40}
    ref = FeatureBank(track, ["au"]).load()
    keys = list(zip(ref.df.participant.astype(str), ref.df.video.astype(str)))
    yk = dict(zip(keys, ref.y)); grp = dict(zip(keys, ref.groups)); nfr = dict(zip(keys, ref.n_frames))

    # (name, kind, spec, grid) — every modality through every model that suits it
    XGB_G = [{"max_depth": 3}, {"max_depth": 4, "n_estimators": 500},
             {"max_depth": 2, "n_estimators": 500, "learning_rate": 0.03}]
    plan = [
        ("xgb_au",   "tab", (("au",), "xgb"),    XGB_G),
        ("rf_au",    "tab", (("au",), "rf"),     [{}, {"max_depth": 8}, {"n_estimators": 600}]),
        ("logreg_au","tab", (("au",), "logreg"), [{"C": 0.3}, {"C": 1.0}]),
        ("audio",    "tab", (("audio",), "xgb"), [{"max_depth": 3}, {"max_depth": 4}]),
        ("embed",    "tab", (("embed",), "xgb"), [{"max_depth": 3}, {"max_depth": 4}]),
        ("blend_mp", "tab", (("blend",), "xgb"), [{"max_depth": 3}, {"max_depth": 4}]),
        ("rocket_au","seq", ("rocket", "auf"),   [{"n_kernels": 1000}, {"n_kernels": 3000}]),
        ("rocket_tj","seq", ("rocket", "traj"),  [{"n_kernels": 1000}, {"n_kernels": 3000}]),
        ("gru_au",   "seq", ("gru", "auf"),      [gru_cfg]),
        ("gru_tj",   "seq", ("gru", "traj"),     [gru_cfg]),
    ]
    oofs = {}; chosen_params = {}; solo = {}
    for name, kind, spec, grid in plan:
        try:
            if kind == "tab":
                res = _best_tab(track, spec[0], spec[1], grid, n_splits, yk)
            else:
                res = _best_seq(track, spec[0], spec[1], grid, n_splits, yk)
        except FileNotFoundError:
            res = None
        if res is None:
            print(f"  [{name}] unavailable on T{track} — skipped"); continue
        o, params, score = res
        oofs[name] = o; chosen_params[name] = params; solo[name] = score
        print(f"  [{name:9}] solo={score:.3f}  params={params}", flush=True)

    common = [k for k in keys if all(k in o for o in oofs.values())]
    P = {n: np.array([o[k] for k in common]) for n, o in oofs.items()}
    yv = np.array([yk[k] for k in common]); gv = np.array([grp[k] for k in common])
    nv = np.array([nfr[k] for k in common])

    print(f"\n=== Track {track}: MAX fusion ({len(common)} clips, {len(P)} streams) ===")
    results = {}
    def _emit(tag, pf):
        rep = _report_pf(track, yv, gv, nv, pf); results[tag] = rep
        print(f"  fusion[{tag:16}] {rep}", flush=True)

    _emit("mean", _fuse_oof(track, P, yv, gv, method="mean"))
    _emit("wmean", _fuse_oof(track, P, yv, gv, method="wmean"))
    for C in (0.1, 0.3, 1.0, 3.0):
        _emit(f"stack_l2_C{C}", _fuse_oof(track, P, yv, gv, method="stack", C=C, penalty="l2"))
    _emit("stack_l1_C1.0", _fuse_oof(track, P, yv, gv, method="stack", C=1.0, penalty="l1"))
    chosen, gbest = _greedy(track, P, yv, gv, method="stack", C=1.0)
    print(f"  greedy selected ({gbest:.3f} oof-primary): {chosen}", flush=True)
    _emit("greedy_stack", _fuse_oof(track, {k: P[k] for k in chosen}, yv, gv, method="stack", C=1.0)
          if len(chosen) > 1 else P[chosen[0]])

    best_tag = max(results, key=lambda t: results[t].primary if hasattr(results[t], "primary")
                   else 0)
    print(f"\n  >>> BEST: {best_tag}  {results[best_tag]}", flush=True)
    return {"results": results, "solo": solo, "params": chosen_params,
            "greedy": chosen, "best": best_tag}


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
