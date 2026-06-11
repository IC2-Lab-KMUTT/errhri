# RUN.md — ERR@HRI 3.0 ops cheat-sheet (server, venvs, caches, runs, learnings)

The single page to re-orient on this project. How to reach the GPU box, which venv runs what,
where the caches live, how to extract a feature stream, how to score it through the official
evaluator, and the hard-won gotchas. Keep it updated; this is the memory.

> **DUA / honesty constraints (non-negotiable).** The ERR@HRI dataset is DUA-governed: never commit
> videos, per-clip frame counts, or label CSVs. `.gitignore` excludes all data + `cache_backup/`.
> **Never use clip length / duration as a feature** — the organizers' wording: *"use of clip length
> as a feature directly is not feasible for real time"*. `leak_clean=True` (default) strips any column
> with |corr(n_frames)| > 0.30. "Honest" = would work in real time given enough compute (no model-size
> ceiling), not that it must be small/fast.

---

## 1. Server access

```bash
ssh ic2@ic2-system-product-name        # Tailscale SSH; may prompt re-auth periodically
```
- Tailscale re-auth mid-session is normal. **Detached jobs survive it** — always launch long jobs with
  `setsid ... > /tmp/x.log 2>&1 < /dev/null & disown` (not a bare `&`, which dies on disconnect).
- Foreground SSH commands that run long will auto-background in this harness; that's fine.

## 2. Key paths (server)

| What | Path |
|---|---|
| Feature package (server) | `/home/ic2/research/errhri-features/` (`errhri_features/` inside) |
| Feature package (local)  | `~/Khush/Projects/Research/errhri-features/` |
| **Cache dir** (all `*_t{1,2}.csv`) | `/home/ic2/research/errhri/train2/cache/` |
| Cache backup (gitignored) | `/home/ic2/research/errhri/cache_backup/` |
| Local cache mirror | `errhri-features/cache_backup/` |
| Official evaluator | `/home/ic2/research/errhri/repo/eval.py` |
| OpenGraphAU repo + weights | `/home/ic2/research/errhri/OpenGraphAU/` (`checkpoints/`, `pretrain_models/`) |
| **Raw video — Track 1** | `/home/ic2/research/errhri/raw/d2/trainval/` |
| **Raw video — Track 2** | `/home/ic2/research/errhri/raw/d1/trainval/` |

> ⚠️ **`d1`/`d2` are SWAPPED vs the track number.** T1 → `d2`, T2 → `d1`. Getting this wrong gives an
> empty extraction.

## 3. Virtualenvs (server) — which one runs what

| venv | Stack | Use for | Notes |
|---|---|---|---|
| `~/research/errhri/venv` | CPU, sklearn + xgboost | **fusion / official eval** (`pipelines.*`, `run_*_fusion.py`) | the modelling/scoring venv |
| `~/dream-venv` | GPU torch **2.5.1+cu121** + DINOv2 + mediapipe **0.10.21** + timm | **GPU extractors**: `faceemb`, `pose`, `au_graph` | works on the 1080Ti (Pascal) |
| `~/research/errhri/auenv` | libreface (CPU) | legacy py-feat-free AU | ~1.75 s/frame — too slow, avoid |
| `~/featv2` | py3.11, torch 2.5.1+cu121, py-feat **0.6.2** (v1) | reference only | v1 Detector ~1 fps |
| `~/featv3` | py3.11, torch **2.6.0+cu124**, py-feat **0.7.0** (from git) | reference only | 0.7.0 Detector still ~1.5–1.8 fps (see §7) |
| `uv` | `~/.local/bin/uv` | package installs | use `-p <venv>/bin/python` |

**GPU:** 2× GTX 1080 Ti (Pascal **sm_61**), 11 GB each.
- torch **cu121 / cu124 work**. torch **cu128 / cu130 / cu13 DROP Pascal** — do not install them.
- Pin `CUDA_VISIBLE_DEVICES=0` for single-GPU jobs (py-feat's DataParallel splits a batch across both
  GPUs and crashes; one process per GPU is the workaround if you want both).

## 4. Standard env block (paste before any run)

```bash
export ERRHRI_CACHE=/home/ic2/research/errhri/train2/cache
export ERRHRI_T1_ROOT=/home/ic2/research/errhri/raw/d2/trainval
export ERRHRI_T2_ROOT=/home/ic2/research/errhri/raw/d1/trainval
export OPENGRAPHAU_DIR=/home/ic2/research/errhri/OpenGraphAU
export CUDA_VISIBLE_DEVICES=0
```

## 5. Caches (in `ERRHRI_CACHE`, keyed by `participant,video`; all gitignored)

| cache | stream | what | feats |
|---|---|---|---|
| `au_t{1,2}.csv` | `au` | py-feat v1 AU, 10 frames/clip (original) | — |
| `au_frames_t{1,2}.csv` | (seq) | per-frame AU trajectory (GRU/ROCKET) | — |
| `blend_t{1,2}.csv` | `blend` | MediaPipe blendshapes | — |
| `traj_t{1,2}.csv` | `traj` | MediaPipe landmark trajectory (seq) | — |
| `audio_t`, `embed_t` | `audio`,`embed` | prosody / generic embed | — |
| `pose_t{1,2}.csv` | `pose` | MediaPipe head/body pose dynamics (dense s=64) | 80 |
| `faceemb_t{1,2}.csv` | `faceemb` | **DINOv2-base** dense face-embedding dynamics (s=48) | 3076 |
| `au_graph_t{1,2}.csv` | `au_graph` | **OpenGraphAU** dense 41-AU dynamics (s=48) ← newest | 209 |
| `fer_*`, `faceemb_hi_*` | — | ruled-out FER / s=128 density probe | — |

`BaseExtractor.run(track)` **resumes from cache** (skips already-done clips) and flushes every 200.

## 6. How to run things

**Extract a GPU stream (dense, both tracks):**
```bash
cd /home/ic2/research/errhri-features        # (env block from §4 first)
~/dream-venv/bin/python -c "
from errhri_features.extractors import AUGraphExtractor   # or FaceEmbExtractor / PoseExtractor
for t in (1,2): AUGraphExtractor(s=48, workers=6).run(t)"
```
- `au_graph` full run: **~18 min Track 1, ~8 min Track 2** (6 workers, ~39 frames/s end-to-end).
- Convenience script: `bash /home/ic2/research/errhri/run_au_graph.sh` (extracts both tracks + backs up).

**Score a stream set through the official evaluator (video macro-F1, the ranked metric):**
```bash
~/research/errhri/venv/bin/python run_au_graph_fusion.py 1    # Track 1; arg2 = substring filter
```
`run_au_graph_fusion.py` compares baseline / +au_graph / +faceemb+pose / all via `pipelines.official`.
Add a stream in code as `Stream(("<cache_name>",), model="xgb")`.

**Detached long job pattern:**
```bash
setsid bash my_job.sh > /tmp/my_job.log 2>&1 < /dev/null & disown
# poll: grep -q "DONE" /tmp/my_job.log
```

**Always back up new caches:**
```bash
cp /home/ic2/research/errhri/train2/cache/<new>_t*.csv /home/ic2/research/errhri/cache_backup/
```

## 7. Learnings / gotchas (don't relearn these)

- **py-feat has no fast "Detectorv2".** PyPI caps at 0.6.2; GitHub main is unreleased 0.7.0 whose
  default `Detector` is still the modular chain with a **hardcoded `img2pose` face detector** (~1.5–1.8
  fps). The docs' "single multi-task net, substantially faster" is **not in the shipped code**. →
  Use **OpenGraphAU** (MEFARG ResNet-50): one forward → 41 AUs, **558 fps** model / **~39 fps**
  end-to-end on the 1080Ti (~26× faster e2e than py-feat). That is the `au_graph` stream.
- **OpenGraphAU setup:** `git clone https://github.com/lingjivoo/OpenGraphAU`; ResNet-50 stage-1
  weights via `gdown 11xh9r2e4qCpWEtQ-ptJGWut_TQ0_AmSp` → `checkpoints/`; ImageNet init
  `resnet50-19c8e357.pth` → `pretrain_models/` (overwritten by the full ckpt but needed to instantiate);
  needs `timm`. Preprocess = Resize(256)→CenterCrop(224)→ImageNet-norm. Output = 41 sigmoid AU probs.
- **`d1`/`d2` video roots are swapped** vs track number (T1→d2, T2→d1).
- **mediapipe ≥0.10.31 + numpy 2.x breaks `mp.solutions`.** Pin `mediapipe==0.10.21 numpy==1.26.4`
  (already in dream-venv). torch 2.5.1+cu121 GPU still works alongside it.
- **Pascal torch:** cu121/cu124 OK; cu128/cu130/cu13 silently drop sm_61.
- **Detached jobs survive Tailscale re-auth** (setsid/nohup). A bare `&` does not.

## 8. Results so far — Track 1 (official video macro-F1, OOF trainval holdout, 173 control / 1146 failure)

| stream set | macro-F1 | control-F1 | bal-acc | AUC |
|---|---|---|---|---|
| `au_graph` solo | 0.607 | 0.356 | 0.610 | 0.680 |
| baseline (au+rf+blend) | ~0.650 | 0.40 | 0.644 | 0.78 |
| **baseline + au_graph** | **0.665** | **0.436** | 0.654 | 0.790 |
| baseline + au_graph (xgb+rf) | 0.661 | 0.426 | 0.646 | 0.790 |
| base + faceemb + pose (prev best) | 0.665 | 0.436 | 0.654 | 0.801 |
| base + faceemb + pose + au_graph | 0.663 | 0.433 | 0.651 | 0.801 |

(eval.py prints two metric blocks per config — window-level then video-level majority-vote; numbers
above are the higher/window block, video-level tracks ~0.01 lower. Comparison conclusions identical.)

**Verdict on `au_graph` quality (the real question, not speed) — CORRECTED by the swap test:**
- `au_graph` is **not strictly higher quality** — it is a different trade: more temporal **density**
  on the AU channel (48 vs 10 frames) but **narrower**. The old `au` cache is **multi-channel** (AU
  *intensities* + **gaze** (24 cols) + **head pose** pitch/yaw/roll (60 cols)); `au_graph` is **only
  41 AU occurrence probabilities** — no gaze, no head pose, no intensity calibration.
- **Swap test (replace old au with au_graph, both as xgb+rf core + blend):**
  - baseline (old multi-channel au): **0.662 / 0.654** (window / video macro-F1), control 0.428
  - SWAP au→au_graph: **0.640 / 0.615** ⬇ — loses gaze+head-pose, dense AU-probs don't compensate
  - SWAP + faceemb + pose: 0.670 / 0.652 (window up, video down vs prior → noise)
  - SWAP + keep old au too: 0.664 / 0.651
- So: **adding au_graph on top = tie** (its AU info already present as intensities → redundant, +variance);
  **swapping it in = worse** (drops gaze + head pose). No config reliably beats the prior best ~0.665.
- Standing finding holds: **the ceiling is in the signal, not the extractor.** T1 is effectively a
  45-stimulus / 5-control problem; calm-failure clips are confusable with control. The old multi-channel
  py-feat AU stream is already a strong facial summary; a denser-but-narrower AU-only stream is a
  sideways move. Structural control-F1 wall ~0.44 unmoved. Run: `run_au_graph_swap.py`.

## 8c. BREAKTHROUGH — T1 with dense `gaze` + calibrated greedy fusion (best so far)

Added a dedicated **dense gaze + 6DoF head-pose** stream (`gaze`, MediaPipe FaceMesh iris + solvePnP,
s=48, 74 feats) — the breadth channel `au_graph` lacked. Then ran calibrated (isotonic) greedy
forward-selection over the full zoo through the official evaluator (`run_t1_maxfuse.py`).

Greedy trace (clip-primary): au_xgb → +blend 0.659 → **+gaze 0.672** → +au_graph 0.681 → +pose 0.682
→ +audio 0.685. Winner = `[au_xgb, blend, gaze, au_graph, pose, audio]`, isotonic-calibrated stack:

| level | macro-F1 | control-F1 | bal-acc | AUC |
|---|---|---|---|---|
| **WINDOW** | **0.7025** | 0.504 | 0.695 | 0.799 |
| **VIDEO (official ranked)** | **0.6848** | 0.456 | 0.691 | 0.787 |

vs prior best ~0.655 video / ~0.665 window. **Two levers, both real:** (1) the new dense `gaze`
stream is genuinely additive (+0.013, greedy picked it 3rd) — confirms gaze/head-pose breadth, not just
AU density, was the missing piece; (2) **isotonic calibration + greedy selection** (baseline core alone
0.665 → 0.676 just from calibration). Control-F1 finally moved off the structural ~0.44 wall (video
0.456, window 0.504). Caveat: greedy selects on OOF (mild optimism); video-level 0.685 is the honest
ranked number, clip/window 0.70. Run: `run_t1_maxfuse.py`.

## 8b. Results — Track 2 (official AUC, OOF trainval, 360 control / 325 failure, balanced)

| stream set | AUC |
|---|---|
| `au_graph` solo | **0.47–0.49 (below random)** |
| baseline (au+rf+blend) | 0.53–0.55 |
| baseline + au_graph | 0.52–0.54 (slightly down) |
| base + faceemb + pose | 0.52–0.54 (slightly down) |

**T2 verdict:** facial AUs carry **no usable signal** on Track 2 — `au_graph` solo is below chance, and
adding it (or faceemb+pose) slightly *hurts* the baseline. This matches the prior finding that T2's
real signal is the **landmark-trajectory GRU** (`gru_tj` ≈ 0.583 AUC, the actual T2 best), not facial
AU. Note the AU-centric `baseline` here (~0.55) is itself weak on T2; the proper T2 model isn't in this
comparison. → For T2, do **not** add facial-AU streams; pursue `gru_tj` + calibration instead.

## 8d. T2 FINAL — 0.6007 official video AUC (target ≥0.60 hit)

Three iterations (`run_t2_boost.py` → `run_t2_v2.py` → `run_t2_v3.py`):
- v1: greedy_cal `[gru_h64, gru_h128b, blend, gaze]` → **0.594**. Found: config-diverse GRU pair works;
  the new dense `gaze` stream is the 2nd-best T2 solo (0.568) — real signal, unlike facial AUs.
- v2: seed-ensembled GRU + prosody → **0.564 WORSE**. Lessons: seed-averaging one config blurs the
  GRU's useful variance (ens 0.570 < single 0.583); `audio` adds nothing on T2 even with select=all;
  more streams ≠ better on 685 noisy clips.
- v3: kept v1's four streams, swept the FUSION: **`rank_wmean` = rank-normalize each stream per fold,
  AUC-weighted mean → 0.6007 official video AUC** (beats every logistic stack; stack_C3.0_cal 0.596).
  Drop-one: every stream contributes (drop gru_h64 → 0.568, drop blend → 0.587, drop gaze → 0.593).

**T2 recipe:** `gru_tj(h64) + gru_tj(h128b,bidir) + gaze + blend`, rank-normalized AUC-weighted mean.
Caveat: fusion method chosen on OOF AUC (mild selection optimism); rank_wmean has no fitted params
beyond AUC weights, so the risk is small.

## 8e. Single-model-over-everything vs the stack (settled with receipts) — `run_t1_single.py`
Challenge: "why not ONE tuned model on a denoised concat of all features instead of a per-stream
ensemble?" Tested honestly — concat ALL 7 tabular modalities into one matrix (FeatureBank merge +
per-subject norm + leak_clean → X = 1319×4849), noise removal fit on TRAIN fold only (drop
near-constant → prune |corr|>0.95 redundant pairs → univariate MI top-k), ONE model, same subject
folds, same official `eval.py` as the 0.685 stack.

| approach | model | official video macro-F1 |
|---|---|---|
| naive concat (all 4849) | XGBoost | 0.558 |
| MI top-300 | XGBoost | 0.578 |
| MI top-150 | XGBoost | 0.578 |
| MI top-80 | XGBoost | 0.574 |
| MI top-40 | XGBoost | 0.562 |
| MI top-150 | HistGBM | 0.543 |
| **all 4849** | **L1-logreg** | **0.599** ← best single model |
| **6 blocks** | **isotonic-cal LR stack** | **0.685** ← per-stream ensemble (best) |

Findings: (1) **trees can't exploit the wide concat** — dense blocks (faceemb/au_graph) starve the
split budget; naive XGB is *worst* (0.558), selection only claws back to ~0.578. (2) **A sparse linear
model over ALL features is the best single model (0.599)** — beats every tree variant and every
selection cut; L1 does its own soft per-feature weighting, robust to noise. (3) **The calibrated stack
still wins by ~0.086** — its edge isn't capacity, it's per-stream isotonic calibration + a meta-learner
over already-good per-block probabilities, which rescues the small orthogonal streams (gaze, audio)
that a monolithic model drowns out. In this control-scarce, small-N (1319 clips / 36 subjects),
heterogeneous-block regime the ensemble is a *regularization / inductive-bias* device, not a weaker
model class. NOTE: not a universal law — with enough data a jointly-trained mid-fusion model would
overtake it; the ensemble wins *because* N is small and blocks are scale-heterogeneous.

### 8e-round2 — single models that RESPECT block structure (`run_t1_single2.py`)
The fair contest: give ONE model the modality-block prior the stack hard-codes. Each modality loaded +
per-subject-normalized + leak_cleaned separately (like a stream), concatenated with KNOWN block slices
(au 382 / au_graph 208 / gaze 73 / pose 79 / blend 940 / audio 92 / faceemb 3075).

| single model | official video macro-F1 |
|---|---|
| flat L1-logreg (no block info, from 8e) | 0.599 |
| per-block PCA(15) → L1-logreg | 0.570 |
| per-block-normalized → L1-logreg | 0.619 |
| **mid-fusion MLP** (per-modality encoder → fuse → head, torch, CPU) | **0.632** |
| TabPFN top-300 | gated — needs PriorLabs `TABPFN_TOKEN` (not run) |
| **reference: per-stream calibrated stack** | **0.685** |

Verdict: block structure IS most of the gap. Flat 0.599 → block-normalized 0.619 → joint mid-fusion
0.632 (closes ~38% of the 0.086 gap). But the stack still wins by ~0.053 — the residual is per-stream
isotonic calibration + per-block models matched to each block's structure, which a single small-N model
can't fully replicate without overfitting. GOTCHA: fusion-venv torch (2.12+cu130) throws
CUBLAS_STATUS_ARCH_MISMATCH on the Pascal 1080Ti (cu13 dropped sm_61) → force CPU
(`CUDA_VISIBLE_DEVICES=""`) for sklearn-venv torch models.

### 8e-round3 — proper open-source deep-tabular single models (`run_t1_proper.py`)
Reproducible (MIT) models, NOT ensembles of separate models. Subject-grouped 5-fold + subject-grouped
INNER early-stopping + temperature calibration (frozen on train → deployable). torch 2.12 on CPU.
Installed: `rtdl_revisiting_models` (FT-Transformer), `tabm`. TabPFN rejected (gated: needs PriorLabs
`TABPFN_TOKEN` account → bad for a reproducibility section).

| model | official video macro-F1 |
|---|---|
| mid-fusion MLP (early-stopped + temp-cal) | 0.618 |
| **FT-Transformer (MI-top-128 features)** | **0.633** |
| TabM (all 4849 feats, raw) | 0.616 |
| untuned mid-fusion MLP (round2) | 0.632 |
| **per-stream calibrated stack** | **0.685** |

Read: best reproducible SINGLE model = FT-Transformer 0.633, ties the untuned mid-fusion MLP, still
~0.05 under the stack. TabM UNDERPERFORMED — fed 4849 raw feats; its paper recipe needs MI-feature-
selection + PiecewiseLinearEmbeddings (numerical embeddings), not run yet → likely leaving points on
the table. CALIBRATION IS DEPLOYABLE (fit-on-train, frozen, no test labels) — confirmed, not a leak.
NOTE the stack itself is ALSO fully open-source/reproducible (plain sklearn isotonic+logistic) and
real-time deployable — "reproducible" was only ever a TabPFN problem, not a stack problem.
NEXT: TabM-done-right (MI-top-~200 + PiecewiseLinearEmbeddings); FT-Transformer top-256 + seed-ensemble.

### 8f. Direction A — temporal + baseline-contrast fused into the stack (NEW BEST) `run_t1_temporal.py` + `run_t1_fuse_temporal.py`
Finding: the per-frame temporal models had failed on T1 not from GRU-vs-MIL but from MISSING
BASELINE-CONTRAST (deviation from the clip's own neutral). On the 32-frame `traj` (blendshapes only),
baseline-contrast lifts AUC: GRU_attn 0.693->0.724, MIL_LSE 0.631->0.681 (best single-stream AUC on T1).
Solo macro-F1 stays ~0.60 (weak channel, threshold-limited) but the high AUC + orthogonality make it a
strong FUSION stream. Isotonic-calibrated greedy over the 6-stack + 4 temporal streams:

| config | official video macro-F1 |
|---|---|
| 6-stream stack (control) | 0.6848 |
| **+ temporal/baseline-contrast (T_gru_attn_bc, T_mil_lse_bc, T_mil_lse)** | **0.6961** (window 0.7045) |

Greedy picked all 3 contrast/temporal streams (+0.011). Lever #2 (baseline-contrast) validated; the
temporal stream is additive. NEXT: B0 = re-extract DENSE per-frame AU+gaze+pose (the strong channel;
current `traj` is blendshapes only, au_frames is just 10 frames) -> temporal+contrast there should beat
the blendshape version and lift further. GOTCHA: SequenceBank.matrix() returns float64 -> cast float32
for torch. Local background monitor tasks die between turns; use server-side setsid nohup + poll.

## FINAL SCOREBOARD (official evaluator, OOF trainval)
| track | metric | before this session | after |
|---|---|---|---|
| T1 | video macro-F1 (ranked) | ~0.655 | **0.696** (stack + temporal/baseline-contrast; was 0.685 stack-only) |
| T1 | single-model best (reproducible) | — | FT-Transformer 0.635 (MI-top-256); flat L1 0.599; tabular single-model ceiling ~0.635 |
| T2 | video AUC (ranked) | 0.583 | **0.6007** |

## 9. Open / next

- Run `run_au_graph_fusion.py 2` for the Track-2 (AUC) verdict — T2 has more real headroom.
- Higher-value honest levers than more facial features: threshold calibration to the 13% test prior,
  positive-unlabeled framing (control = reliable negative), stimulus-fold CIs.
- Caches durably backed up (server `cache_backup/` + local mirror) — never re-extract for new fusion
  experiments.

## 2026-06-11 session — owned bests + bundle v2
- **T1 owned best 0.7365** (`run_t1_lopo_stack.py`, OOFs in `lopo_oof_t1.csv`); **T2 owned best 0.6262**
  (`run_t2_owned.py`, OOFs in `hse_owned_oof_t2.csv`). Recipes + audit in METHOD.md §2026-06-11.
- HSEmotion per-frame matrices (philix's, with consent): server `/home/ic2/research/errhri/philix_feats/t{1,2}_feats.npz`.
- **Bundle v2** (all 43 cache CSVs + HSE matrices + scripts, AES-256 enc, same password as v1):
  server `/home/ic2/research/errhri/bundle/errhri_bundle_v2.zip.enc` (+ .sha256). Server-only — do NOT host publicly (DUA).
- Run LGBM with `n_jobs=4 OMP_NUM_THREADS=4` on ic2 (n_jobs=-1 thrashes). FaceMesh extractors: dream-venv
  (mediapipe 0.10.21; 0.10.35 dropped mp.solutions).
