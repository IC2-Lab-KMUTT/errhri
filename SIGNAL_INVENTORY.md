# ERR@HRI 3.0 — Signal vs Noise Inventory

> Living document. Catalogs every input/feature family we have tested, its measured
> discriminative strength, and a verdict: **SIGNAL** (keep), **WEAK** (small but real — keep
> for the union), **NOISE** (drop), or **LEAK** (forbidden — absolute-duration proxy, must
> never be a feature). The end goal is to take **all** signal (however small) into one
> multi-modal architecture, then tune. Last updated mid-AU-extraction.

## Tasks & metrics

| | Track 1 "BAD" | Track 2 "Bad Idea" |
|---|---|---|
| label | failure vs control | well vs poorly handled |
| metric | **macro-F1** | **AUC** |
| video aggregation | majority vote | max prob |
| balance | 87 / 13 (imbalanced) | 53 / 47 (balanced) |
| size | 1319 clips / 36 subj | 685 clips / 23 subj |

## Reference bars (honest, subject-grouped CV)

| bar | T1 macro-F1 | T2 AUC | note |
|---|---|---|---|
| official challenge baseline | 0.502 | 0.564 | provided |
| **duration LEAK** | **0.702** | 0.591\* | forbidden; clip length only |
| facial whole-clip GRU (best honest) | **0.638** | — | 21 discriminative ARKit channels |
| facial attention-MIL | 0.630 | 0.546 | |
| audio (eGeMAPS) | 0.528 | 0.491 | weak/none |

\* T2 duration 0.591 is a raw-video artifact; the official T2 clips are trimmed, so the
duration leak is effectively **T1-only**.

---

## ⚠️ Methodology note — single-model metric deltas UNDERSTATE signal

**Do not read "model A 0.638 vs model B 0.630" as "the thing A adds is worth only 0.008."**
A single scalar metric is the *net* of gains and losses against the other model; it hides
whether the two models are right about the **same** clips or **different** ones.

- Two models both at ~0.63 can have **weakly-correlated errors** → each is correct on clips the
  other misses. Their *union of competence* is much larger than either alone.
- **Fusion gain scales with error decorrelation, not with the metric gap.** Decorrelated 0.63 +
  0.63 models can fuse well above 0.63; perfectly-correlated ones cannot.
- Therefore signal contribution must be judged by **prediction/error correlation** and an
  **oracle / late-fusion estimate** across models (GRU vs MIL vs XGBoost vs audio vs AU), not by
  ranking single-model scores.

**Concrete measurement (planned/ongoing — `complementarity.py`):** dump per-clip OOF probability
vectors for every model, then report (a) Pearson/Spearman between probability vectors, (b)
correlation of the **error indicators** `1[wrong]`, (c) **oracle upper bound** (clip counted
correct if *any* model is right), and (d) realized **late-fusion** (mean / logistic-stack of OOF
probs) vs `max(single models)`. The same lens applies to every modality in this doc: audio's
0.606 univariate AUC may be **orthogonal** to facial and so contribute to fusion far beyond its
solo score — that is exactly why WEAK signals are kept.

> Rule of thumb adopted here: a feature/model earns a place in the final ensemble if it is either
> (i) strong solo, **or** (ii) weak solo but **decorrelated** from the backbone. Solo score alone
> never decides inclusion.

---

## SIGNAL — confirmed discriminative, length-clean (KEEP)

| feature family | source | track | strength / evidence | notes |
|---|---|---|---|---|
| **Amused / positive smile** | MediaPipe blendshapes: `cheekPuff`, `mouthSmile_L/R`, `dimple_L/R` | T1 (carries to T2) | top discriminative channels in separator + SHAP; drive the 0.638 GRU | reaction-to-failure often a nervous/amused smile |
| **Head pose** | `pitch`, `yaw` (framing-invariant) | T1, T2 | repeatedly in top features; survives leak removal | looking down/away on error |
| **Gaze aversion** | gaze direction | T1, T2 | discriminative in separator | aversion on failure |
| **Blink** | eye-blink blendshapes | T1 | present in top set | |
| **Velocity / temporal change** | per-frame Δ of blendshapes | T1 | adds over static means | dynamics, not absolute level |
| **Ordered temporal trajectory** | whole-clip GRU over resampled L frames | T1 | GRU 0.638, static MIL/XGB ~0.630 — but see ⚠️ methodology note: the small *net* gap understates the signal; GRU and static models likely capture **partially non-overlapping** clips | the honest single-model facial ceiling |
| **Threshold tuning** | per-fold macro-F1 threshold | T1 | meaningful lift under 87/13 imbalance | post-hoc, applies to any model |
| **AU dynamics & expression instability** | libreface AU6/mouth-width/expr **std·range·velocity·entropy·#switches** | **T1** | multivariate **AUC 0.744** (length-clean); top univariate 0.69 | richest length-clean ranking signal; see RESULT section. Feed dynamics, not static levels |

## WEAK — small but real signal (KEEP for the union, don't rely on alone)

| feature family | source | track | strength | notes |
|---|---|---|---|---|
| **Audio prosody** | openSMILE eGeMAPSv02 (88 functionals) | **T1 only** | univariate AUC **0.606** (multivariate macro-F1 0.528) | 85% clips near-silent (headphones); weak but >chance on T1 |
| **Seed ensembling** | mean of GRU seeds | T1 | +0.010 @0.5 | washes out after threshold tuning |
| **Deep face embedding** | DINOv2 ViT-S/14 on face crop, PCA-in-fold | **T1** | solo macro-F1 0.580 / AUC 0.674; **lifts 5-stream fusion 0.655→0.674** | learned pixel representation, orthogonal (err-corr 0.20 vs facial); the one non-engineered input. Noise on T2 (0.508) |

## NOISE — no signal beyond chance (DROP)

| feature family | source | track | evidence | why |
|---|---|---|---|---|
| Audio | eGeMAPS / energy-silence | **T2** | AUC 0.491 | near-silent; no signal |
| Negative affect / frown | `browDown`, frown blendshapes | T1 | not discriminative | counter-intuitive but measured |
| Spike / change-point presence | naive SpikeModel | T1 | ≈ chance | a "spike" exists in control too |
| Global motion | frame-diff motion energy | T1 | not discriminative | |
| Raw all-feature dump | all 60 blendshapes, untuned | T1 | overfits at N=36 subj | needs feature selection |
| Frame size / position | `size`, `cx`, `cy` | both | framing-dependent | recording-setup artifact, excluded |

## LEAK — forbidden (NEVER a feature) ⚠️

| feature | corr with n_frames | inflation caused | status |
|---|---|---|---|
| **clip duration** (`n_frames`, `len_sec`) | — | T1 0.702 alone | cheating; excluded |
| `dyn_len_sec` | +0.70 | hybrid 0.678 → 0.636 when removed | excluded (CLEAN_DYN) |
| `dyn_nbursts` | +0.52 | dynamics 0.676 → 0.607 when removed | excluded (CLEAN_DYN) |

> The 0.67–0.70 numbers seen early were the **duration leak**, not facial skill. Any new
> feature is checked for `|corr(feature, n_frames)| > 0.30` and discounted if it is a
> duration proxy.

---

## RESULT — proper FACS AUs (libreface), 382 features — DONE

libreface per-frame (S=10): 12 AU intensities, 12 AU presence, 8 expressions, gaze, head pose,
11 framing-invariant landmark-geometry ratios — each ×12 stats (mean/std/median/min/max/p10/p90/
range/IQR/**slope**/**velocity**/**delta**) = **382 features**. Cached raw per-frame in
`au_frames_t{track}.csv`; re-aggregate via `au_aggregate.py`.

### Track 1 — **STRONG, length-clean** → SIGNAL (best ranking signal found)
| metric | value | note |
|---|---|---|
| multivariate XGBoost **AUC** | **0.744** | leak −0.044 (NOT duration); vs duration-AUC 0.639, facial ~0.64 |
| macro-F1 @0.5 / @tuned | 0.608 / 0.542 [0.504,0.582] | threshold-unstable under 87/13 imbalance |

Top length-clean univariate features (corr_len ≈ 0):

| feature | AUC | meaning |
|---|---|---|
| `expr_entropy` | 0.691 | expression **instability** over clip |
| `geomouth_width_std/range/vel` | 0.69 | mouth-width **dynamics** |
| `expr_neutral` (−) / `expr_happiness` (+) | 0.68 / 0.67 | less neutral / more amused on failure |
| `expr_nswitch` | 0.657 | # expression switches |
| `au6int_iqr` | 0.660 | AU6 (cheek-raiser/smile) intensity variability |

**Carriers are DYNAMICS & INSTABILITY (std/range/velocity/entropy/#switches), not static AU
levels** — the reaction is a temporal change-event. Confirms why temporal order matters.
⚠️ The 0.744 is **AUC** (strong ranker); the T1 metric is **macro-F1** where AU = 0.54–0.61, so
this is "richest length-clean ranking signal," not yet "beats 0.638 macro-F1." Calibration +
fusion is the path to convert ranking → macro-F1.

### Track 2 — **WEAK** → keep for fusion only
multivariate AUC **0.545** (≈ facial gmil 0.546, < official 0.564), leak +0.066 clean. Brow/mouth
geometry weakly predictive (~0.58 univariate). No breakthrough; T2 stays hard.

> Implication: on T1, **AU dynamics + expression instability** is now a primary stream — likely
> decorrelated from the static blendshape backbone (different aggregation, validated AU model).
> `complementarity.py` (task #94) will confirm fusion lift. Static AU *levels* underperform their
> *dynamics* — feed slope/velocity/range/entropy, not just means, to the final model.

---

## PER-DIMENSION + GRANULAR TIMING → see `DIMENSIONS_AND_TIMING.md`

Detailed companion: every facial/audio channel grouped into semantic dimensions (smile, mouth/jaw,
brow, eye, nose, gaze, head-pose, expression, audio) with a static-vs-dynamics split, plus granular
per-channel timing (onset/peak/duration/rise/decay/burst/magnitude). Headlines:
- **T1: dynamics > static almost everywhere**; top dims expression-instability 0.682, mouth/brow/
  smile/eye ~0.65; negative-affect lips = NOISE (0.536). **Timing-only AUC 0.788, length-clean** —
  carriers are smile **magnitude (auc/amp)** + **onset/peak timing**; duration/burst-rate weak.
- **T2: all dims weak (≤0.59)**; only smile (0.588) clears chance; timing-only 0.580.

## COMPLEMENTARITY / FUSION RESULTS (`complementarity.py`, 2026-06-09) — methodology proven

Shared subject-grouped 5-fold OOF probs per stream → prob/error correlation + fusion. (CPU: the
1080 Ti is sm_61, incompatible with the venv's cuDNN-9.2 torch, so GRU ran on CPU.)

### Track 1 — FUSION BEATS THE CEILING ✅ (5 streams incl. deep embedding)
Solo macro-F1: facial-gru 0.623 · facial-static 0.622 · au 0.602 · audio 0.579 · embed 0.580.
Prob-correlation: **audio ⟂ everything (0.05–0.10)**, embed vs facial 0.27–0.36, au vs facial
0.36–0.46, gru vs static 0.57. Oracle: ≥1 stream correct on **97.5%** of clips (single-best acc
80.4%) — large hidden headroom.

| fusion | macro-F1 | AUC | leak | note |
|---|---|---|---|---|
| 4-stream mean (no embed) | 0.657 | 0.797 | −0.095 | facial-gru+static+au+audio |
| **5-stream mean-prob** | **0.666** [0.629,0.705] | 0.804 | −0.092 | + DINOv2 embed |
| **5-stream logistic stack** | **0.674** [0.637,0.713] | 0.804 | −0.099 | best honest result |

→ **+0.051 over best single (0.623), clears the 0.638 facial ceiling, length-clean, closing on the
0.702 duration-leak bar with ZERO leak.** Both "weak" streams earn their place by orthogonality:
audio is the most decorrelated (error-corr 0.18), the DINOv2 embedding is a different
*representation* (error-corr 0.20) and adds +0.019 to the stack. Methodology proven twice: flat
solo deltas hid real fusion gains; decorrelation drove them.

### Track 2 — fusion ~flat (streams decorrelated but too weak)
Solo AUC: facial-gru 0.556 · facial-static 0.546 · au 0.545 · **audio 0.491 (<chance)**.
Prob-correlation 0.04–0.31 (decorrelated) but each stream too weak: mean-prob **0.558** ≈ best
single 0.556; logistic stack 0.544 (audio drags). → **DROP audio on T2**; fusion needs a stronger
base stream first.

---

## Implications for the final architecture

0. **MEASURED FINAL DIRECTION (T1):** late-fuse **facial-gru + facial-static + AU + audio + DINOv2
   embed** (logistic stack) → **macro-F1 0.674**, length-clean — beats the 0.638 facial ceiling and
   every single stream, closing on the 0.702 duration-leak bar with honest signal. This is the
   validated T1 architecture skeleton; next is calibration + tuning. **T2:** facial only
   (gru+static+AU), **drop audio AND embed** (both ≈chance there); fusion flat (0.556) — needs a
   stronger base stream.
   **COVERAGE: all input TYPES now tested** — engineered (AU/blendshape/geometry/pose/gaze/expr,
   static+dynamics+timing+sequence), audio prosody, AND learned pixel representation (DINOv2). The
   only untested items are low-value/speculative: precise 3D iris-gaze, rPPG, higher-fps micro-expr,
   ASR speech content.
1. **Facial is the backbone** (0.638 honest). Smile + head-pose + gaze + temporal order are the
   load-bearing signals.
2. **Fuse on complementarity, not solo score** — include a stream if it is strong **or**
   decorrelated from the backbone. T1 audio (0.606), AU-derived signal, and the static MIL/XGB
   heads likely each cover clips the GRU misses; late-fusion/stacking should beat
   `max(single models)`. Measure with `complementarity.py` before committing the final head.
3. **Keep BOTH the sequence model and the static heads** — they are not redundant; the GRU
   (temporal order) and MIL/XGBoost (static aggregates) capture partially non-overlapping signal,
   so fuse them rather than picking the single best.
4. **Imbalance handling** (threshold tuning, class weights) is mandatory on T1.
5. **Hard exclusions:** duration and all `size/cx/cy` framing features stay out — they are
   leaks/artifacts, not skill.
6. **T2 is harder & balanced** — audio is dead there; rely on facial + AU only.
