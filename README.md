# errhri-features

Shared **feature-extraction + honest-evaluation** toolkit for the **ERR@HRI 3.0** (ICMI'26)
facial error-detection challenge. The hard half — *what signal exists, what's noise, and how to
evaluate without fooling yourself* — is already done and baked in. You bring a model and pick
features; the toolkit handles extraction, subject-grouped CV, the length-leak guard, bootstrap
CIs, and the official windowed submission.

> Curated findings live in **`SIGNAL_INVENTORY.md`** (signal/noise/fusion) and
> **`DIMENSIONS_AND_TIMING.md`** (per-dimension + granular timing). For the **granular,
> per-feature correlation-strength breakdown** of *every* engineered feature on both tracks —
> univariate separability, error-class direction, and the duration-leak flag, grouped by dimension
> — see **`FEATURE_STRENGTH.md`** (regenerate it yourself with `python -m analysis.feature_report`).
> Read those first — knowing which features are mediocre/overlapping/noise is what keeps you from
> overfitting at N=23–36 subjects, and the per-feature table lets you make your own keep/drop calls
> instead of trusting our verdicts.

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

## Repository layout — three layers, separated on purpose

```
errhri_features/   CORE — video -> signal features -> honest evaluation.   Stable; don't edit to experiment.
                   (extractors, FeatureBank, CVEvaluator, leak guard, metrics, signal_map, submission)
pipelines/         YOUR LAYER — models + param-driven recipes. Swap a model / change params / recompose
                   streams here without touching the core.  (models zoo, SequenceBank, recipes)
analysis/          SIGNAL STUDIES — reproduce/challenge the signal map on your own cache.
                   (feature_report -> FEATURE_STRENGTH.md, dimension_breakdown, timing_features, complementarity)
scripts/           OPERATIONAL — build_index, extract_all, example_pipeline.
```

The split is the point: the core never changes when you try a new model — you only edit a `Stream`
config or a recipe param in `pipelines/`. That keeps the signal pipeline readable while you iterate.

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

## Build the cache (you start with nothing but the videos)

The cache is **not** shipped (it's DUA-governed). Generate it from your copy of the data:

```bash
# 1. the clip index — labels parsed from filenames + frame counts. EVERYTHING reads this.
python -m scripts.build_index --track all          # -> index_t1.csv, index_t2.csv

# 2. the feature caches — resumes if interrupted
python -m scripts.extract_all --track all --modalities au audio embed blend traj --workers 8
```
`build_index` must run first (extractors and `FeatureBank` read `index_t<track>.csv` for the clip
list + labels). Then `extract_all` writes `au_t1.csv`, `audio_t1.csv`, `traj_t1.csv`, … to
`ERRHRI_CACHE`. `traj` is the per-frame trajectory cache for the temporal GRU; `au`/`embed`/`blend`/
`traj` need CPU torch + the FaceLandmarker model (see Install).

## Use it (the part you write) — the recipe layer

You compose `Stream`s and fuse them; defaults are the validated ones, and you change behaviour by
changing **params**, never the core:

```python
from pipelines.recipes import Stream, run_stream, run_fusion, run_temporal

# one stream — swap the model or override its params right here
run_stream(1, Stream(modalities=("au", "audio"), model="xgb", params={"max_depth": 4}))
run_stream(1, Stream(("au",), model="logreg"))          # MODEL_ZOO: xgb | logreg | rf

# late fusion (the validated T1 win); add the temporal GRU when the `traj` cache exists
run_fusion(1, [Stream(("au",)), Stream(("audio",)), Stream(("embed",))], method="stack")
run_fusion(1, [Stream(("au",)), Stream(("audio",))], include_temporal=True, gru={"hidden": 96})

# the whole-clip temporal GRU alone — strongest single T1 stream (sees ORDER, not summary stats)
run_temporal(1, hidden=64, epochs=40)
```

Each returns a core `Report` (primary metric + 95 % CI + AUC + f1_neg + length-leak). A `Stream`'s
knobs: `modalities`, `model` (a key in `pipelines.models.MODEL_ZOO`), `select`
(`"all"` · `"signal"` — drops measured-noise groups, and audio+embed on T2 — · or an explicit column
list), `leak_clean` (strip |corr(·, n_frames)| > 0.30), and `params` (forwarded to the model).

**Add a model** in one line: drop a `name -> builder` into `pipelines/models.py:MODEL_ZOO`, then
`Stream(model="yourname")`. The core, the leak guard, the CV and the fusion don't change.

Under the hood it's still just the core — `pipelines/recipes.py` wraps `FeatureBank` + `CVEvaluator`
+ `late_fusion`, so you can always drop down to them directly. `scripts/example_pipeline.py` is a
full copy-paste reference (streams → fusion → submission).

## The analysis pipelines (what we actually ran)

`analysis/` reproduces every experiment behind the curated signal map — run them on your own cache
to re-derive (and challenge) the verdicts:

| `python -m analysis.<x>` | what it does |
|---|---|
| `feature_report` | per-feature univariate strength table → **`FEATURE_STRENGTH.md`** (both tracks) |
| `dimension_breakdown` | per-semantic-dimension AUC, static vs dynamics split |
| `timing_features` | granular timing-only signal (onset / peak / magnitude), length-clean |
| `complementarity` | cross-stream fusion potential: prob/error decorrelation, oracle, late fusion |

`complementarity` is the methodology that matters: it judges a stream by how its **errors
decorrelate** from the others (fusion headroom), not by its solo score — which is why weak-but-
orthogonal `audio`/`embed` earn a place in the T1 ensemble that lifts 0.623 → 0.674.

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
- **New model** → add a `name -> builder` line to `pipelines/models.py:MODEL_ZOO`, then
  `Stream(model="yourname")`. Any `fit`/`predict_proba` estimator works. The temporal GRU on the
  raw trajectory already lives in `pipelines/models.py` (`ClipGRUClassifier`) + `pipelines/
  sequences.py` (`SequenceBank`) — use `recipes.run_temporal`.
- **New recipe** → add a function to `pipelines/recipes.py`; it wraps the core, so the signal
  pipeline stays untouched. **New finding** → add an `analysis/` module beside the existing four.
