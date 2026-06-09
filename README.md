# errhri-features

Shared **feature-extraction + honest-evaluation** toolkit for the **ERR@HRI 3.0** (ICMI'26)
facial error-detection challenge. The hard half — *what signal exists, what's noise, and how to
evaluate without fooling yourself* — is already done and baked in. You bring a model and pick
features; the toolkit handles extraction, subject-grouped CV, the length-leak guard, bootstrap
CIs, and the official windowed submission.

> Curated findings live in **`SIGNAL_INVENTORY.md`** (signal/noise/fusion) and
> **`DIMENSIONS_AND_TIMING.md`** (per-dimension + granular timing). Read those first — knowing
> which features are mediocre/overlapping/noise is what keeps you from overfitting at N=23–36
> subjects.

## What's inside

| modality (cache `*_t<track>.csv`) | extractor | signal |
|---|---|---|
| `au` — 12 AU intensities + 12 presence + 8 expressions + gaze + pose + geometry (×12 stats) | libreface | **strong T1** (dynamics) |
| `blend` — 52 ARKit blendshapes + pose + geometry + **granular timing** | MediaPipe FaceLandmarker | facial backbone |
| `audio` — eGeMAPSv02 (88) + energy/silence | openSMILE | weak-orthogonal (T1 fusion only) |
| `embed` — DINOv2 ViT-S/14 face embedding (mean+std) | timm | weak-orthogonal (T1 fusion only) |

Every continuous channel gets **level + dynamics + timing** stats. On T1 the dynamics beat the
static levels and timing-only already scores AUC ≈ 0.79 (length-clean); late-fusing all streams
hits macro-F1 **0.674** vs the 0.638 facial ceiling, honest (the 0.702 duration baseline is a
LEAK — see below).

## Install

torch must be the **CPU build, installed first**, or libreface/timm will drag in a multi-GB CUDA
stack (and on older GPUs the CUDA torch won't even run). Two-venv setup recommended:

```bash
python -m venv venv && . venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .                      # makes `errhri_features` importable
# blendshape model (one-time):
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
export ERRHRI_FACE_MODEL=$PWD/face_landmarker.task
```

## Data layout

```
ERRHRI_CACHE/index_t1.csv         # participant,video,label,n_frames  (one row per clip)
ERRHRI_T1_ROOT/<participant>/<video>.mp4
```
Point the toolkit at your data:
```bash
export ERRHRI_CACHE=/path/to/cache
export ERRHRI_T1_ROOT=/path/to/track1/trainval
export ERRHRI_T2_ROOT=/path/to/track2/trainval
```

## Extract features (once)

```bash
python -m scripts.extract_all --track all --modalities au audio embed blend --workers 8
```
Resumes from cache. Writes `au_t1.csv`, `audio_t1.csv`, … to `ERRHRI_CACHE`.

## Use it (the part you write)

```python
from errhri_features import FeatureBank, CVEvaluator, late_fusion
from xgboost import XGBClassifier

bank = FeatureBank(track=1, modalities=["au", "audio", "embed"]).load()
spw  = (bank.y == 0).sum() / (bank.y == 1).sum()          # class imbalance (T1 is 87/13)

ev = CVEvaluator(track=1)                                  # subject-grouped 5-fold
rep = ev.run(lambda: XGBClassifier(scale_pos_weight=spw, tree_method="hist"),
             bank, select="signal", leak_clean=True)       # drop noise + duration-proxy feats
print(rep)        # macro-F1 + 95% CI + AUC + f1_neg + length-leak
```

`select=`: `"all"` · `"signal"` (drops measured-noise groups; drops audio+embed on T2) · or an
explicit column list. `leak_clean=True` drops any feature with |corr(·, n_frames)| > 0.30.

**Late fusion** (the validated T1 win):
```python
oof = {m: ev.run(lambda: XGBClassifier(scale_pos_weight=spw, tree_method="hist"),
                 FeatureBank(1, [m]).load(), select="signal", leak_clean=True).oof_prob
       for m in ["au", "audio", "embed"]}
print(late_fusion(1, oof, bank.y, bank.groups, bank.n_frames, method="stack"))
```

`scripts/example_pipeline.py` is a full copy-paste reference (streams → fusion → submission).

## Evaluation

- **Group eval** (`CVEvaluator`): subject-grouped K-fold, per-fold threshold tuning on TRAIN
  only, primary metric (T1 macro-F1 / T2 AUC), subject-bootstrap 95% CI, length-leak readout.
- **Submission eval** (`errhri_features.submission`): writes the official per-window CSV
  (`participant_id,video_id,window_id,y_pred,y_prob_0,y_prob_1`), enforces the window constraints
  (slide ≤ window_size; T1 window ≤ 2·fps), and scores it at video level (T1 majority-vote
  macro-F1 / T2 max-prob AUC). Drop the organizers' official script into `official/` and call
  `submission.run_official_evaluator(...)` to cross-check.

## ⚠️ The one rule: duration is a LEAK, never a feature

Track 1 clips are raw mp4 → `n_frames` ∝ duration, and **duration alone scores macro-F1 ~0.70**
(controls run longer than failures). It is not facial skill. The toolkit excludes it everywhere
and `leak_clean=True` strips any feature that proxies it. Keep `leak` near 0 in every report.

## Extend it

- **New features** → subclass `extractors.BaseExtractor` (implement `init_worker` + `extract_clip`),
  add to `extractors.REGISTRY`. You get parallelism/resume/caching for free.
- **Stronger AU model / more prosody / different encoder** → swap the extractor; the FeatureBank,
  CV, leak guard, and fusion don't change.
- **New model / pipeline** → any `fit`/`predict_proba` estimator into `CVEvaluator`; or build a
  sequence model on the raw trajectories (smile/pose channels) — the timing analysis says onset +
  peak + magnitude of the smile are the load-bearing signal, so make them first-class.
