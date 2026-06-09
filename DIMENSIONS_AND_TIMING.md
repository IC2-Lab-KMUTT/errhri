# ERR@HRI — Per-Dimension Signal Map + Granular Timing Breakdown

> Companion to `SIGNAL_INVENTORY.md`. Two analyses, both tracks, all subject-grouped 5-fold,
> length-leak checked:
> 1. **Dimension breakdown** (`dimension_breakdown.py`) — every facial/audio channel grouped into
>    semantic dimensions, with a STATIC-level vs TEMPORAL-dynamics split.
> 2. **Granular timing** (`timing_features.py`) — per reaction channel: onset, time-to-peak,
>    duration, rise/decay, burst-rate, magnitude — all relative (leak-clean), from dense per-frame
>    trajectories.
> All numbers are AUC (comparable across tracks). "static" = level stats (mean/median/min/max/
> p10/p90/rate/ever); "dyn" = std/range/IQR/slope/velocity/delta/#transitions/#switches/entropy.

---

## 1. DIMENSION BREAKDOWN

### Track 1 (failure vs control) — DYNAMICS beat STATIC almost everywhere
| dimension | n | AUC | static | **dyn** | leak | best feature | verdict |
|---|--:|--:|--:|--:|--:|---|---|
| **expression** (global instability) | 10 | **0.682** | 0.668 | 0.675 | −0.05 | `expr_entropy` 0.691 (+fail) | **SIGNAL** |
| **mouth_jaw** (open/drop) | 48 | 0.652 | 0.606 | **0.668** | +0.02 | `geomouth_open_std` 0.655 | **SIGNAL** |
| **brow** (raise/frown) | 81 | 0.652 | 0.608 | 0.606 | +0.01 | `au2int_iqr` 0.627 | **SIGNAL** |
| **smile** (cheek/lip-corner) | 54 | 0.648 | 0.632 | **0.672** | −0.01 | `geomouth_width_std` 0.689 | **SIGNAL** |
| **eye** (blink/aperture) | 51 | 0.647 | 0.570 | **0.648** | −0.04 | `au5int_iqr` 0.613 | **SIGNAL** |
| **nose** (wrinkle/sneer) | 27 | 0.624 | 0.593 | **0.654** | +0.02 | `geonose_lip_iqr` 0.664 | signal (mod) |
| **head_pose** (yaw/pitch/roll) | 36 | 0.611 | 0.585 | 0.599 | +0.05 | `roll_std` 0.648 | signal (mod) |
| **gaze** (look-direction) | 24 | 0.609 | 0.570 | 0.592 | −0.06 | `gaze_pitch_slope` 0.609 | signal (mod) |
| **audio** (eGeMAPS) | 93 | 0.606 | 0.606 | — | −0.15 | `loudness_stddevNorm` 0.597 | WEAK (orthogonal — keep) |
| **other_lipmouth** (frown/stretch/press = neg-affect) | 51 | 0.536 | 0.534 | 0.545 | +0.01 | `au15int_iqr` 0.582 | **NOISE** |

**Reading:** the discriminative facial dimensions all sit ~0.61–0.68, and in 6 of 8 the **dyn
column ≥ static** — the reaction lives in the *change/variability*, not the resting level. The
single strongest compact signal is **expression instability** (`expr_entropy`, 0.691). The
negative-affect lip group (depressor/stretch/press/dimple = frowning) is **noise (0.536)** — it is
*positive* smile/amusement, not negativity, that separates the classes. Everything is length-clean
(|leak| ≤ 0.06) except audio's silence∝length artifact (−0.15, and audio is weak anyway).

### Track 2 (well vs poorly handled) — all dimensions WEAK
| dimension | n | AUC | static | dyn | leak | best feature | verdict |
|---|--:|--:|--:|--:|--:|---|---|
| **smile** | 54 | **0.588** | 0.547 | 0.566 | −0.02 | `geomouth_width_iqr` 0.584 | weak-best |
| expression | 10 | 0.545 | 0.548 | 0.469 | −0.02 | `expr_fear` 0.455 | weak |
| eye | 51 | 0.541 | 0.550 | 0.520 | +0.04 | `geoeye_open_l_median` | weak |
| head_pose | 36 | 0.525 | 0.542 | 0.486 | +0.03 | `roll_iqr` 0.576 | weak |
| brow | 81 | 0.521 | 0.524 | 0.490 | +0.00 | `geobrow_eye_l_max` | ~noise |
| mouth_jaw | 48 | 0.512 | 0.461 | 0.534 | +0.03 | `au25int_p10` | ~noise |
| nose | 27 | 0.486 | 0.465 | 0.536 | +0.06 | `geonose_lip_std` 0.581 | ~noise |
| audio | 93 | 0.491 | 0.491 | — | −0.13 | — | NOISE |
| gaze | 24 | 0.468 | 0.445 | 0.510 | +0.06 | `gaze_yaw_delta` | ~noise |
| other_lipmouth | 51 | 0.448 | 0.427 | 0.469 | +0.05 | `au17int_iqr` | NOISE |

**Reading:** only **smile (0.588)** clears 0.56; everything else is ≈ chance. T2 carries far less
facial signal than T1 — expected (balanced, harder, no length leak to lean on).

---

## 2. GRANULAR TIMING BREAKDOWN (the missing time-based features)

Per channel, leak-clean relative timing: `onset_frac` (latency), `peak_frac` (time-to-peak),
`offset_frac`, `dur_frac` (fraction elevated), `rise`/`decay` steepness, `ncross_rate` (burst
rate), `early_bias` (early vs late half), `auc` (mean elevation), `amp` (amplitude).

### Track 1 — timing-only multivariate AUC = **0.788, fully length-clean** (beats GRU ~0.75, AU 0.744)
Which timing *aspect* carries signal (mean |AUC−0.5| across channels, higher = stronger):
| timing stat | strength | max leak | verdict |
|---|--:|--:|---|
| **`auc`** (mean elevation magnitude) | 0.095 | 0.09 | **strongest — magnitude** |
| **`onset_frac`** (onset latency) | 0.089 | 0.13 | **strong, length-clean — WHEN it starts matters** |
| `amp` (amplitude) | 0.071 | 0.15 | strong |
| `early_bias` (early vs late) | 0.065 | 0.16 | real — failure smile is LATER |
| `peak_frac` (time-to-peak) | 0.064 | 0.08 | real — WHEN it peaks |
| `offset_frac` | 0.044 | 0.08 | weak |
| `decay` | 0.048 | 0.27 | weak + borderline leaky |
| `dur_frac` (reaction length) | 0.038 | 0.27 | weak + borderline leaky |
| `rise` | 0.029 | 0.21 | near-noise |
| `ncross_rate` (burstiness) | 0.023 | 0.29 | near-noise + leaky |

Top single timing features: `smile_mouthSmileL.auc` 0.699 · `cheekPuff.auc` 0.690 ·
`mouthSmileL.amp` 0.678 · **`cheekPuff.onset_frac` 0.675** · `mouthSmileL.onset_frac` 0.660 ·
`cheekPuff.peak_frac` 0.651. `early_bias` is **−fail** (smile concentrated in the LATER half of
failure clips — the reaction follows the error).

**Verdict T1:** the time-based signal = **reaction MAGNITUDE (auc/amp) + ONSET/PEAK timing of the
smile channel**, length-clean. Reaction *duration* and *burst-rate* are weak and slightly
length-contaminated → low priority (and the duration ones flirt with the leak — handle with care).

### Track 2 — timing-only multivariate AUC = 0.580 (leak-clean 0.570), ≈ its best dimension
Weak overall; strongest stat types are `rise` 0.043 and `peak_frac` 0.040. Notable that
**gaze/head-yaw timing** surfaces here (`gaze_lookOutL.auc` 0.571, `head_yaw.peak_frac` inv,
`smile.early_bias` 0.581 −fail) where they were minor on T1 — but all individually weak (~0.58).

---

## 3. CURATED SIGNAL → PIPELINE IMPLICATIONS

**Feed the model (T1), in priority order:**
1. **Smile channel as a temporal trajectory** — its magnitude (`auc`/`amp`) and **onset/peak
   timing** are the single richest length-clean signal (timing-only AUC 0.788). cheekPuff +
   mouthSmile + dimple.
2. **Expression instability** — `expr_entropy`, `expr_nswitch` (compact, 0.69).
3. **Dynamics over levels for every facial channel** — std/range/velocity/slope of smile, mouth,
   eye, brow, nose (dyn ≥ static everywhere). Do NOT rely on static means.
4. **Mouth/jaw-open + brow + eye dynamics** (~0.65 each).
5. **Head-pose + gaze dynamics** (moderate ~0.61) — and gaze matters slightly more on T2.
6. **Audio** — keep as an orthogonal late-fusion stream on T1 only (weak solo 0.606 but
   decorrelated; proven to lift fusion). Drop on T2.

**Do NOT feed (noise / overfitting risk at N=23–36 subj):**
- Negative-affect lip AUs (`other_lipmouth`: frown/stretch/press/depress) — noise both tracks.
- `ncross_rate`, `rise` timing — near-noise.
- `dur_frac`/`decay` timing — weak AND borderline length-leaky; include only with a leak guard.
- Audio on T2; absolute durations / `size`/`cx`/`cy` (always — leak/artifact).

**Architecture consequence:** a model that ingests the **ordered smile/mouth/brow/eye trajectory**
(so it can read onset/peak/magnitude) + a few explicit timing scalars (onset_frac, peak_frac, auc,
early_bias per smile channel) + expression-instability + AU dynamics, late-fused with audio (T1),
is the configuration the evidence supports. Timing-only already hits AUC 0.788 length-clean — the
open work is converting that ranking into macro-F1 (calibration + the fusion at 0.657) toward the
0.702 leak bar with honest signal. **T2** needs a stronger base stream first; smile-dynamics +
gaze/head timing are the only footholds.
