# ERR@HRI 3.0 — method: extraction, temporal modelling, ensembling

How every feature stream is extracted, how the temporal models work, and how they're
calibrated and fused into the best Track-1 pipeline (video macro-F1 **0.696**) and the
Track-2 pipeline (AUC ~0.601). File references point at the `IC2-Lab-KMUTT/errhri` repo,
branch `feat/dense-modalities-and-recipes`.

## 1. Shared extraction backbone (every modality)
- **Frame sampling** (`errhri_features/datasets.py::sample_frames`): a **fixed number `s` of frames
  evenly spaced across the clip** (OpenCV `CAP_PROP_POS_FRAMES` seeks), *not* a fixed fps. This is
  central to the leak defense — a fixed frame budget makes every feature **length-invariant**, so
  nothing secretly encodes clip duration. A 14 s and a 22 s clip both yield `s` frames.
- **Face crop** (`datasets.py::FaceCropper`): MediaPipe `FaceDetection(model_selection=1, conf=0.3)`,
  square crop + 30% margin, center-crop fallback.
- **Aggregation philosophy** (`errhri_features/aggregate.py`): a "BAD" reaction is a ~1–2 s spike
  inside a 14–22 s clip, so a plain mean washes it out. Every continuous channel is collapsed to
  **level + dynamics** (mean/std/median/min/max/p10/p90/range/iqr/slope/`vel`/`delta`) **+ apex**
  (top-k mean), **peak velocity/acceleration**, **time-of-peak**, and a global **motion-energy**.
- **Two cross-cutting rules:** facial signal = FACS **AUs**, never raw MediaPipe blendshapes/landmarks
  as the primary facial representation (blend/traj kept only as extra orthogonal views); and
  **per-subject normalization** (label-free z-score over each subject's clips) strips identity /
  baseline expressiveness and mirrors deployment.

## 2. Per-modality extraction
| Stream | Library / model | Frames (`s`) | Raw or processed | What's stored |
|---|---|---|---|---|
| `au` | **libreface** FACS | 8–10 (sparse) | per-frame AU intensities + presence + expression + gaze + head-pose + framing-invariant geometry → `aggregate()` | 12 summary stats/channel + presence rate/ever/ntrans + expr fractions/entropy/switches |
| `au_graph` | **OpenGraphAU** MEFARG (ResNet-50), IJCAI-22 | **48 (dense)**, GPU | 41 AU probabilities, **one forward pass** (>500 fps) → `_agg_au` | per-AU mean/std/max/maxvel/apex + AU-motion energy/peak/std |
| `gaze` | **MediaPipe FaceMesh** (refine_landmarks, 478 pts + iris) | **48 (dense)**, CPU | iris-offset gaze h/v/mag/asym, eye-aspect-ratio, head pose via `solvePnP` | per-signal mean/std/range/apex/maxvel/maxacc/tmax + gaze & head motion energy |
| `pose` | **MediaPipe Pose** (33 landmarks) | **64 (dense)**, CPU | framing-invariant head/shoulder geometry (nod, recoil, lean, turn-away) | reaction-aware stats + pose motion energy |
| `blend` | **MediaPipe FaceLandmarker** (52 ARKit blendshapes + 4×4 transform matrix) | 16 | blendshapes + Euler pose + geometry → `aggregate()` with **timing** features | static + dynamics + granular onset/peak/magnitude timing |
| `traj` | same FaceLandmarker, **trajectory kept** | 32 → resampled **L=32** | ordered discriminative channels (smile/jaw/blink/gaze/pose/brow), **not aggregated** | flat `t{k}__{channel}` wide CSV (the temporal cache) |
| `faceemb` | **DINOv2-base** (frozen, 768-d CLS token) | **48 (dense)**, GPU | embedding trajectory → `_agg_embed` | per-dim mean/std/maxvel/apex + embedding motion energy |
| `fer` | FER emotion net (+ Haar-crop variant) | 16 | 7-emotion probabilities | weak; kept as orthogonal view |
| `audio` | **openSMILE eGeMAPSv02** | whole clip | ffmpeg→16 kHz mono wav → 88 affective functionals + RMS/silence probe | ~85% clips near-silent (headphones) → weak-but-orthogonal on T1, noise on T2 |
| `au_seq` / `gaze_seq` **(NEW)** | OpenGraphAU / FaceMesh | **48 (dense)** | **full 48-frame trajectory saved** (`t{tt}__{channel}`) | the strong dense sequences the aggregated caches threw away |

## 3. Temporal modelling
`pipelines/sequences.py::SequenceBank` reshapes the flat `traj` cache back to a tensor **(N, L=32,
C=16)**, per-subject normalized. Then `run_t1_temporal.py`:
1. **Baseline-contrast** (`add_baseline_contrast`, k=6): append each frame minus the clip's *own*
   neutral baseline (mean of the first 6 frames) → **(N, L, 2C)**. The single biggest temporal lever:
   removes per-person expressiveness, exposes the reaction onset. Lifted GRU AUC 0.693 → **0.724**.
2. **Two localizer models** (both biased toward "a brief peak somewhere"):
   - **`MIL_LSE`** — per-frame MLP encoder → per-frame logit → **log-sum-exp pooling**
     (`logsumexp(r·f)/r`, r=5), a soft max that picks the peak frame.
   - **`GRU_attn`** — bidirectional GRU + attention pooling + head.
3. **Honest training** (`train_seq`): subject-grouped **5-fold outer** CV; an **inner 4-fold** holdout
   for early-stopping on AUC; `pos_weight` for class imbalance; AdamW; then **temperature scaling**
   (LBFGS on train logits) calibrates each model.
4. OOF clip probabilities for all 4 variants (`mil_lse`, `mil_lse_bc`, `gru_attn`, `gru_attn_bc`) are
   saved to `temporal_oof_t1.csv` for fusion.

## 4. Ensembling — the 0.696 recipe
Late-fusion **calibrated stack**, not one big model (`run_t1_fuse_temporal.py::fuse`):
1. **Per-stream OOF probabilities.** Each tabular stream (`au, au_graph, gaze, pose, blend, audio`) →
   `FeatureBank` merges its caches, `select="signal"` drops measured-noise columns, `leak_clean` drops
   any feature with |corr(n_frames)| > 0.30 → **XGBoost** (`scale_pos_weight`) → subject-grouped 5-fold
   → OOF prob per clip, keyed by `(participant, video)`. The 4 temporal streams enter as precomputed OOF.
2. **Per-fold isotonic calibration.** Inside each meta-CV fold, every stream's probabilities are
   recalibrated with `IsotonicRegression` **fit on the training fold only**, then applied to train+val.
   Fit-on-train + frozen → **real-time deployable, NOT a test leak** (same logic as temperature scaling).
3. **Logistic stack.** Calibrated per-stream columns are stacked; a `LogisticRegression` meta-learner is
   fit on the train fold → fused OOF probability.
4. **Greedy forward selection.** Repeatedly add the stream that most improves honest meta-CV macro-F1,
   stop when none helps. Selected `[au, blend, gaze, au_graph, pose, gru_attn_bc, mil_lse_bc, mil_lse]`
   — kept **all three** baseline-contrast/temporal streams, dropped audio.
5. **Per-fold threshold** (`metrics.tune_threshold` on train OOF) → applied to val.
6. **Official metric** (`pipelines/official.py`): clip pred/prob is replicated across the clip's windows
   and scored by the challenge `eval.py` as **video-level macro-F1 via majority vote** (exact bridge —
   all windows of a clip share the label).

**Result:** greedy fused = **0.6961** vs 6-stream stack without temporal = **0.6848**.
Track 2 uses the same machinery with an AUC objective (`run_t2_v3.py`, ~0.601).

**Why a stack, not one model:** streams have wildly different scales/densities and the data is tiny +
control-scarce (173 control / 1146 failure). Per-stream calibration + a meta-learner over already-good
per-block probabilities rescues weak-but-orthogonal streams. A single model over the raw concat plateaus
at 0.635 (FT-Transformer) / 0.558 (flat XGBoost) — see `run_t1_single*.py` / `run_t1_proper*.py`.

## 5. Leak guard (`errhri_features/leak.py`)
Track 1 is raw mp4 → frame count ∝ duration, and **duration alone scores macro-F1 ~0.70** (control
clips run longer). Any feature correlated with `n_frames` is a duration proxy, not facial skill, and
using it is cheating + barred by the organisers as non-real-time. `length_corr` / `clean_columns` flag
and strip any feature with |corr(n_frames)| > 0.30. Fixed frame sampling (§1) + length-normalized
trajectories (§3) keep shape, not duration, as the signal.

---

# Session 2026-06-11 — HSEmotion adoption, LOPO migration, owned bests

## New best results (honest, official eval, fully owned — no teammate-OOF dependence)
| Track | metric | old | **new owned best** | recipe |
|---|---|---|---|---|
| T1 | video macro-F1 | 0.696 | **0.7365** | LOPO stack: `HSE_owned + gaze + au_xgb + pose + audio + blend` (`run_t1_lopo_stack.py`) |
| T2 | video AUC | 0.576 | **0.6262** | `HSE_tail60_pz + gaze + HSE_tail90_pz + TD_gru_attn_pz` (`run_t2_owned.py`) |

## The HSEmotion stream (the new signal class)
Per-frame **MediaPipe(68: blendshapes+headpose+geometry) + HSEmotion `enet_b0_8_va_mtl`** (1280-d AffectNet
embedding + 8 emotion logits + valence/arousal = 1290-d ×2 pooling = 2560-d), sampled at 5 fps over the whole
clip. Recipe (philix's, reproduced + audited): per-participant z-score of ALL columns (incl. embedding) →
PCA-128 on the embedding block → sliding-window stats (mean/std/min/max/q25/q75/slope/|vel|/pos; T1 ws25/slide10,
T2 ws10/slide2 on tail-60/90 segments) → leak_guard (drop |corr n_frames|>0.30) → class-weighted LGBM
(`scale_pos_weight`) → **mean** aggregation over windows → LOPO. Solo: T1 0.820 AUC / 0.704 F1; T2 tail60 0.605.
Why it wins: it is a *learned affect representation* — a different representation space from all our geometric
streams, hence genuinely orthogonal in fusion. Build scripts: `his_lgbm_slim.py` (T1), `run_t2_owned.py` (T2).

## Protocol upgrades adopted
- **LOPO replaces 5-fold** everywhere (tabular streams + meta-CV + thresholds): +0.018 AUC / +0.033 F1 on
  identical features. Not a leak — more train data per fold, matches train-on-all-trainval deployment.
- **pz-ranking** (per-participant z-score of *video scores*): label-free, legitimate for batch-AUC submissions;
  worth +0.02 on T2 streams.

## Teammate-pipeline audit (philix) — leaks found and quantified
His LOPO model CV and features are honest. Three selection-on-labels leaks inflate his *reported* numbers:
1. **T1 decision rule** — `tune_rule` picks global/ppct/pz threshold on full OOF labels: 0.728 honest → 0.7433 (+0.015).
2. **T1 window+param grid** — window-size sweep and LGBM grid scored on full OOF labels: ~+0.03 AUC (his 0.851
   vs 0.820 for the fixed recipe).
3. **T2 fusion weights** — >100-candidate pz-fusion grid, `candidates[0]` on full labels: 0.6315 reported vs
   0.6119 nested; his best single honest stream (0.6238) actually beats his own nested fusion.

## Negative results (representation > pipeline, confirmed 3 ways)
- His window-table template on OUR geometry seqs (au_seq+gaze_seq): 0.689 AUC — *worse* than our old temporal
  streams (~0.72). The template doesn't rescue weak representations.
- T2 gaze-scanpath reading dynamics (full-fps iris, saccades/regressions/fixations/blinks, 5s tails): 0.513 ≈
  chance. (`run_t2_scanpath.py`)
- T2 clips are **silent** (whisper: no speech) → the stimulus text that defines the T2 label is not observable.
  T2's ~0.63 cap is an observability limit, structural — not a feature plateau.

## Ops gotchas
- LGBM `n_jobs=-1` on the shared 12-core box → 35-thread thrash (>1h for a 5-min job). Use `n_jobs=4` +
  `OMP_NUM_THREADS=4`.
- mediapipe ≥0.10.3x drops legacy `mp.solutions` — use dream-venv (0.10.21) for FaceMesh extractors.
- `df.to_numpy()` may return read-only views; `.copy()` before in-place normalization.
