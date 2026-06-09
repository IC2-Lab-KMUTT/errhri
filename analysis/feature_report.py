"""Granular per-feature strength table -> FEATURE_STRENGTH.md (Track 1 + Track 2 side by side).

For EVERY engineered feature (AU + audio; the anonymous DINOv2 dims are excluded as un-namable),
report, per track:
  sep   univariate separability  = max(AUC, 1-AUC) of the per-subject-normalized feature vs label
  dir   sign of mean(failure/poorly) - mean(control/well) on the raw feature  (+ = higher in error)
  leak  |corr with n_frames|     — the duration-proxy flag (>0.30 = tainted, T1 only really)

Grouped by semantic dimension, sorted by T1 separability. This is the raw material behind the
curated verdicts in signal_map.py — teammates can re-rank, disagree, and pick their own features.

    python -m analysis.feature_report            # writes FEATURE_STRENGTH.md
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from errhri_features import FeatureBank
from errhri_features.featurebank import per_subject_norm
from errhri_features.leak import length_corr
from .dimensions import dimension_of, stat_family

MODALITIES = ["au", "audio"]
OUT = Path(__file__).resolve().parent.parent / "FEATURE_STRENGTH.md"
DIM_ORDER = ["expression", "smile", "mouth_jaw", "brow", "eye", "nose", "head_pose", "gaze",
             "audio", "other_lipmouth", "other"]


def _per_feature(track):
    bank = FeatureBank(track, MODALITIES).load()
    cols = bank.feature_columns()
    Xraw = np.nan_to_num(bank.df[cols].to_numpy(float))
    Xn = per_subject_norm(Xraw, bank.groups)
    y, nfr = bank.y, bank.n_frames
    audio = set(bank.audio_cols)
    rows = {}
    for j, c in enumerate(cols):
        col = Xn[:, j]
        if np.std(col) < 1e-9:
            sep = 0.5
        else:
            a = roc_auc_score(y, col)
            sep = max(a, 1 - a)
        pos, neg = Xraw[y == 1, j], Xraw[y == 0, j]
        direction = "+" if (np.nanmean(pos) - np.nanmean(neg)) >= 0 else "-"
        rows[c] = dict(sep=sep, dir=direction, leak=abs(length_corr(Xraw[:, j], nfr)),
                       dim=dimension_of(c, audio), fam=stat_family(c))
    return rows


def main():
    print("computing per-feature strength (T1)...", flush=True)
    t1 = _per_feature(1)
    print("computing per-feature strength (T2)...", flush=True)
    t2 = _per_feature(2)
    feats = sorted(set(t1) | set(t2),
                   key=lambda f: t1.get(f, {}).get("sep", 0.5), reverse=True)
    by_dim = {}
    for f in feats:
        dim = (t1.get(f) or t2.get(f))["dim"]
        by_dim.setdefault(dim, []).append(f)

    L = ["# Feature strength — granular per-feature correlation breakdown",
         "",
         "Univariate signal strength of **every engineered feature** on both tracks, so you can make",
         "your own keep/drop calls instead of trusting the curated `signal_map.py` verdicts. Generated",
         "by `analysis/feature_report.py` from the feature caches — re-run it on your own data.",
         "",
         "- **sep** = univariate separability, `max(AUC, 1-AUC)` of the per-subject-normalized feature",
         "  vs the label (0.50 = noise, higher = more discriminative *alone*; fusion can still rescue a",
         "  low-sep feature if it is *orthogonal* — see `analysis/complementarity.py`).",
         "- **dir** = `+` feature is higher in the error class (failure / poorly-handled), `-` lower.",
         "- **leak** = `|corr(feature, n_frames)|`; **> 0.30 is a duration proxy — do not use on T1**",
         "  (the `leak_clean=True` guard strips these automatically).",
         "- **fam** = static level / dynamics (std·range·slope·velocity) / timing (onset·peak·magnitude).",
         "",
         "> Track-1 reference bars: official baseline macro-F1 **0.502**, our honest fusion **0.674**,",
         "> the forbidden duration-only leak **0.702**. Track-2: baseline AUC **0.564** (signal is weak;",
         "> most per-feature `sep` sit near 0.50 — that is the finding, not a bug).",
         ""]
    for dim in DIM_ORDER + [d for d in by_dim if d not in DIM_ORDER]:
        if dim not in by_dim:
            continue
        fs = sorted(by_dim[dim], key=lambda f: t1.get(f, {}).get("sep", 0.5), reverse=True)
        t1seps = [t1[f]["sep"] for f in fs if f in t1]
        head = f"## {dim}  ({len(fs)} feats"
        if t1seps:
            head += f", T1 best sep {max(t1seps):.3f}"
        L += ["", head + ")", "",
              "| feature | fam | T1 sep | T1 dir | T1 leak | T2 sep | T2 dir |",
              "|---|---|---|---|---|---|---|"]
        for f in fs:
            a, b = t1.get(f), t2.get(f)
            fam = (a or b)["fam"]
            def cell(d, k, fmt="{:.3f}"):
                return fmt.format(d[k]) if d else "·"
            leak = (f"{a['leak']:.2f}" + ("⚠" if a and a["leak"] > 0.30 else "")) if a else "·"
            L.append(f"| `{f}` | {fam} | {cell(a,'sep')} | {a['dir'] if a else '·'} | {leak} "
                     f"| {cell(b,'sep')} | {b['dir'] if b else '·'} |")
    OUT.write_text("\n".join(L) + "\n")
    n = len(feats)
    print(f"wrote {OUT}  ({n} features, {len(by_dim)} dimensions)")


if __name__ == "__main__":
    main()
