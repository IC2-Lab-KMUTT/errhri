# ERR@HRI 3.0 — Signal Inventory

Every input/feature family we tested, its **measured** strength, and a verdict. The goal: take
**all** real signal (however small) into one multi-modal model, then tune.

**How to read the numbers.** Two metrics appear throughout — don't mix them up:
- **macro-F1 / AUC (official)** — the *challenge* metrics, video-level, subject-grouped CV. T1 is
  **macro-F1** (majority-vote aggregation), T2 is **AUC** (max-prob). These are what actually count.
- **sep-AUC (diagnostic)** — a feature/dimension's univariate or multivariate *separability* (an
  AUC ranker). Useful to compare signals, but a high sep-AUC ≠ a high macro-F1 (a strong ranker
  still needs calibration + a tuned threshold to score macro-F1). Where we quote a sep-AUC for a
  T1 stream, we also give its **macro-F1** so the comparison is honest.

**Verdicts.** **PRIMARY** (load-bearing) · **SUPPORTING** (real, keep) · **WEAK-ORTHOGONAL** (small
solo but decorrelated → earns its place in fusion) · **NOISE** (drop) · **LEAK** (forbidden
duration proxy — never a feature).

Companions: per-feature table → **`FEATURE_STRENGTH.md`**; granular per-channel timing →
**`DIMENSIONS_AND_TIMING.md`**.

---

## 1 · Tasks & metrics

| | Track 1 "BAD" | Track 2 "Bad Idea" |
|---|---|---|
| label | failure vs control | well vs poorly handled |
| **official metric** | **macro-F1** | **AUC** |
| video aggregation | majority vote | max prob |
| balance | 87 / 13 (imbalanced) | 53 / 47 (balanced) |
| size | 1319 clips / 36 subj | 685 clips / 23 subj |

---

## 2 · Headline — what we honestly achieved

Subject-grouped CV, length-leak clean. **This is the scoreboard; the duration row is NOT a target —
it is the leak we refuse to use.**

| | T1 macro-F1 | T2 AUC | |
|---|---|---|---|
| Official challenge baseline | 0.502 | 0.564 | provided |
| **Our best honest result** | **0.674** | 0.558 | T1: 5-stream fusion · T2: facial only |
| ~~duration-only "leak"~~ | ~~0.702~~ | — | ❌ forbidden (clip length, not skill) — excluded everywhere |

**T1: macro-F1 0.674** [CI 0.637–0.713], AUC 0.804, length-leak −0.099 — **+0.17 over the 0.502
baseline and above the 0.638 single-model facial ceiling, with zero leak.** Achieved by late-fusing
five complementary streams (§7). The 0.702 "duration" number is higher only because it cheats; we
beat the *honest* problem, which is what matters.

**T2 stays hard:** balanced and largely facial; best honest AUC ≈ 0.558 (≈ official baseline). Audio
and the deep embedding are at chance here and are dropped; fusion is flat because every stream is
weak (§7).

---

## 3 · Signal catalog (Track 1 streams)

Solo = that stream alone, subject-grouped CV, signal-tier features, length-clean. macro-F1 is the
official metric; sep-AUC given where it adds context.

| stream | source | T1 macro-F1 (solo) | T1 sep-AUC | verdict | why |
|---|---|---|---|---|---|
| **Facial temporal (GRU)** | whole-clip ordered trajectory of 21 ARKit channels | **0.623** | — | **PRIMARY** | sees temporal *order*; the honest single-model ceiling |
| **Facial static (MIL/XGB)** | per-clip blendshape aggregates | 0.622 | — | **PRIMARY** | static levels+dynamics; non-redundant with the GRU |
| **AU dynamics (libreface)** | 12 AU + 12 presence + 8 expr + gaze/pose/geometry, ×12 stats (382) | 0.602 | **0.744** | **PRIMARY** | richest length-clean *ranker*; carriers are dynamics & instability, not static levels |
| **DINOv2 embedding** | ViT-S/14 on face crop, PCA-in-fold | 0.580 | 0.674 | **WEAK-ORTHOGONAL** | learned pixel representation; error-corr ≈0.20 vs facial → adds +0.019 to the stack |
| **Audio prosody** | openSMILE eGeMAPSv02 (88 functionals) | 0.579 | 0.606 | **WEAK-ORTHOGONAL** | 85% clips near-silent (headphones); most decorrelated stream (error-corr 0.18) |

**What inside the face carries it** (the load-bearing behaviours, from separator + SHAP + AU):
- **Amused / nervous smile** — `cheekPuff`, `mouthSmile_L/R`, `dimple`, AU6/AU12 — reaction to
  one's own failure is often a smile. Top discriminative channel; drives the GRU.
- **Expression instability** — `expr_entropy`, `expr_nswitch` — the face *changes more* on failure.
- **Head pose** (pitch/yaw, framing-invariant) and **gaze aversion** — looking down/away on error.
- **Blink** and **mouth-width dynamics** — present in the top set.
- **Carriers are DYNAMICS, not static levels:** feed std / range / velocity / slope / entropy /
  #switches, not just means. This is why temporal order helps.

---

## 4 · Timing signals (Track 1) — *when* the reaction happens

Granular per-channel timing (relative to clip length, so leak-clean by construction). **Timing
features alone score sep-AUC 0.788, length-clean** — the reaction's *shape in time* is real signal.

| timing feature | verdict | what it measures |
|---|---|---|
| `auc` (reaction magnitude) | **SIGNAL — strongest** | area under the smile/jaw trajectory |
| `amp` | **SIGNAL** | peak amplitude of the reaction |
| `onset_frac` | **SIGNAL** | onset latency — when the reaction starts (length-clean) |
| `early_bias` | **SIGNAL** | early-vs-late mass — the failure smile arrives **later** |
| `peak_frac` | **SIGNAL** | time-to-peak |
| `offset_frac` | WEAK | when it ends |
| `dur_frac`, `decay` | WEAK-LEAKY | reaction length — borderline length-correlated, use with care |
| `rise`, `ncross_rate` | NOISE | slope-in / burstiness — at/near chance |

→ The signal is **magnitude + onset/peak timing of the smile**, not how long or how bursty it is.
Per-channel detail in `DIMENSIONS_AND_TIMING.md`.

---

## 5 · Per-dimension strength (diagnostic sep-AUC)

Multivariate AUC per semantic dimension, subject-grouped, length-clean. Shows *where* in the face
the signal lives, and the **T1 ≫ T2** gap. (These are rankers; the official macro-F1 is reported at
the stream/fusion level above.)

| dimension | T1 sep-AUC | T2 sep-AUC | verdict |
|---|---|---|---|
| expression (instability) | **0.682** | 0.545 | PRIMARY |
| smile (cheek / lip-corner) | 0.648 | **0.588** | PRIMARY (T1) · best T2 dim |
| mouth / jaw | 0.652 | 0.512 | SUPPORTING |
| brow (raise / frown) | 0.652 | 0.521 | SUPPORTING |
| eye (blink / aperture) | 0.647 | 0.541 | SUPPORTING |
| nose (wrinkle / sneer) | 0.624 | 0.486 | SUPPORTING |
| head pose | 0.611 | 0.525 | SUPPORTING |
| gaze | 0.609 | 0.468 | SUPPORTING |
| audio prosody | 0.606 | 0.491 | WEAK-ORTHOGONAL (T1) · NOISE (T2) |
| deep embedding (DINOv2) | 0.674 | 0.508 | WEAK-ORTHOGONAL (T1) · NOISE (T2) |
| negative-affect lips (frown/press/stretch) | 0.536 | 0.448 | **NOISE** |

On T1, **dynamics beat static levels in almost every dimension.** On T2 only smile clears chance —
hence T2 is facial-only and hard.

---

## 6 · Noise & forbidden

**NOISE — at/near chance, drop:**

| family | track | evidence |
|---|---|---|
| Audio (eGeMAPS / energy-silence) | **T2** | AUC 0.491 — near-silent, no signal |
| Negative-affect lips (browDown, frown, press, stretch) | T1 | sep-AUC 0.536 — measured non-discriminative |
| Spike / change-point presence (naive SpikeModel) | T1 | ≈ chance — a "spike" exists in control too |
| Global motion (frame-diff energy) | T1 | not discriminative |
| Raw all-feature dump (60 blendshapes, untuned) | T1 | overfits at N=36 subj — needs selection |
| Frame size / position (`size`, `cx`, `cy`) | both | recording-setup artifact |

**LEAK — forbidden, NEVER a feature ⚠️:**

| feature | corr with `n_frames` | effect | status |
|---|---|---|---|
| **clip duration** (`n_frames`, `len_sec`) | — | T1 macro-F1 0.702 *alone* | excluded — cheating |
| `dyn_len_sec` | +0.70 | inflates 0.678 → 0.636 when removed | excluded |
| `dyn_nbursts` | +0.52 | inflates 0.676 → 0.607 when removed | excluded |

T1 clips are raw mp4 → controls run longer than failures, so anything correlated with length is a
duration proxy, not skill. Guard: drop any feature with `|corr(feature, n_frames)| > 0.30`
(`leak_clean=True`); keep every reported `leak` near 0.

---

## 7 · Why fuse on complementarity (not solo score)

**A single metric hides whether two models are right about the same clips or different ones.** Two
streams both at ~0.62 with *decorrelated errors* each cover clips the other misses — their union is
far larger than either alone. **Fusion gain scales with error decorrelation, not the metric gap.**
So a stream earns a place if it is *strong* **or** *weak-but-decorrelated* from the backbone — solo
score alone never decides. (Measured with `analysis/complementarity.py`: per-stream OOF probs →
prob/error correlation, oracle coverage, late fusion.)

**Track 1 — fusion clears the ceiling ✅**

Prob-correlation: audio ⟂ everything (0.05–0.10), embed vs facial 0.27–0.36, AU vs facial 0.36–0.46,
GRU vs static 0.57. **Oracle: ≥1 stream correct on 97.5% of clips** (single-best 80.4%) — large
hidden headroom.

| fusion (length-clean) | T1 macro-F1 | AUC | leak |
|---|---|---|---|
| 4-stream mean (gru+static+au+audio) | 0.657 | 0.797 | −0.095 |
| 5-stream mean-prob (+ embed) | 0.666 [0.629, 0.705] | 0.804 | −0.092 |
| **5-stream logistic stack** | **0.674** [0.637, 0.713] | 0.804 | −0.099 |

Both "weak" streams earn their place by orthogonality: audio is the most decorrelated (error-corr
0.18); the DINOv2 embedding is a different *representation* (error-corr 0.20) and adds +0.019.

**Track 2 — fusion flat.** Streams are decorrelated (prob-corr 0.04–0.31) but each too weak:
mean-prob AUC 0.558 ≈ best single (facial-gru 0.556); audio (0.491, <chance) drags the stack. →
**drop audio on T2**; it needs a stronger base stream before fusion helps.

---

## 8 · Final architecture direction

1. **T1 — late-fuse five complementary streams** (logistic stack): facial-GRU + facial-static + AU
   dynamics + audio + DINOv2 embedding → **macro-F1 0.674**, length-clean. This is the validated
   skeleton; next is calibration + tuning (convert the strong AUC rankers into macro-F1).
2. **T2 — facial only** (GRU + static + AU); **drop audio and embedding** (both ≈ chance). Fusion is
   flat (≈0.558) — the priority is a stronger base stream, not more streams.
3. **Keep both the sequence model and the static heads** — GRU (temporal order) and MIL/XGB (static
   aggregates) capture partially non-overlapping clips; fuse, don't pick one.
4. **Feed dynamics + timing, not static levels** — std/range/velocity/entropy/#switches and smile
   magnitude + onset/peak timing are the carriers.
5. **Imbalance handling is mandatory on T1** (class weights + per-fold threshold tuning).
6. **Hard exclusions:** duration and all `size/cx/cy` framing features — leaks/artifacts, not skill.

**Coverage: all input *types* tested** — engineered (AU / blendshape / geometry / pose / gaze /
expression, in static + dynamics + timing + sequence forms), audio prosody, and a learned pixel
representation (DINOv2). Remaining gaps are low-value/speculative: precise 3D iris-gaze, rPPG,
higher-fps micro-expression, ASR speech content.
