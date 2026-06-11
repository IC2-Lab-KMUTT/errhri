# Max-fusion optimizer — results & the honest ceiling

Run: `pipelines.recipes.run_max_fusion(track)` — full stream zoo × per-stream grid tuning ×
5 fusion strategies, all inside honest subject-grouped GroupKFold meta-CV. Caches: py-feat AU
(`au`, `au_frames`), audio, embed, MediaPipe blend/traj. Leak guard active throughout.

## Track 1 (macro-F1, 1319 clips / 36 subjects, 87/13 imbalance)

Tuned solos: xgb_au 0.644 · blend_mp 0.644 · rf_au 0.630 · gru_tj 0.625 · logreg_au 0.614 ·
rocket_tj 0.600 · rocket_au 0.584 · gru_au 0.578 · embed 0.565 · audio 0.531

| fusion | macro-F1 | CI | leak |
|---|---|---|---|
| mean (10 streams) | 0.650 | [.614,.683] | −0.061 |
| AUC-weighted mean | 0.659 | [.628,.687] | −0.061 |
| L2-stack (C=0.3) | 0.656 | [.618,.690] | −0.073 |
| **L1-stack (sparse)** | **0.667** | [.631,.700] | −0.080 |
| greedy forward-select | 0.673 | [.630,.715] | −0.074 |

greedy kept only **xgb_au + blend_mp + rf_au** (3 aggregated-facial streams; rejected every
temporal stream).

## Track 2 (AUC, 685 clips / 23 subjects, balanced)

Tuned solos: gru_tj **0.583** (best) · rf_au 0.565 · xgb_au 0.563 · logreg 0.554 · rocket_au
0.552 · blend 0.548 · gru_au 0.524 · rocket_tj 0.515. Best fusion = greedy selecting **gru_tj
alone = 0.583**. Mean/wmean ≈ 0.573–0.576.

## Verdict — the ceiling is in the SIGNAL, not the models

1. **T1 honest ceiling ≈ 0.67 macro-F1**, essentially all **aggregated facial AU**. The greedy
   optimizer *rejected every temporal stream* — once xgb_au + rf_au + blend are present, GRU and
   ROCKET add nothing. Temporal order does not beat robust AU summary stats on T1.
2. Earlier "ROCKET is additive (+0.017)" was relative to a *weaker* mix; against the tuned
   aggregated-AU core it is dominated. Corrected finding.
3. Tuning + regularized fusion (L1-stack) > adding model types. Architecture has plateaued.
4. T2 is small/noisy: landmark-traj GRU alone is the whole story; fusion can't reliably help.
5. Leak clean everywhere (−0.06 to −0.08 on T1). Official baseline T1 0.502, T2 0.564.

## UPDATE — official evaluator + the real bottleneck (the actual thing to fix)

Read the official `repo/eval.py` + `EVALUATION.md`. **Track 1 is ranked by video-level macro-F1 via
MAJORITY vote across windows** (ties→failure), balanced-accuracy tiebreaker. The official **test set
is ~40 failure / 6 control** — so macro-F1 is *entirely* controlled by the ~6 control clips.
Official baseline = **BadNet, a from-scratch CNN on raw frames, macro-F1 0.502** (worse than our AU
0.66). Windows capped at `≤2×fps`; up to 30 fps allowed (we extract at 5).

Built `pipelines/official.py` — runs our ensemble through the real `eval.py`. Result (OOF trainval
holdout, 1146 failure / 173 control), macro objective:

  video-level macro-F1 = **0.661**, balanced-acc 0.645, F1(failure)=0.922, **F1(control)=0.400**,
  control recall 61/173, AUC 0.782.

**The entire gap to 0.8 is the control class** (F1 0.40 → need ~0.68). Tuning the threshold for
balanced-accuracy raises bal-acc to 0.673 but DROPS macro-F1 to 0.577 — so operating-point tricks
cannot fix it. Only features that separate control from failure better can. → pivot to (1) official
eval as the optimisation target [done], (2) frozen pretrained face-CNN per-frame features fused with
AU [next, GPU via `dream-venv` torch 2.5.1+cu121 on the 1080Ti], (3) higher-fps re-extraction.

## UPDATE 2 — frozen face-CNN (FER ViT) tested: does NOT break the control wall

Step 2 of the plan: added a **frozen pretrained FER transformer** (`trpakov/vit-face-expression`,
7 emotions + 768-d CLS embedding, per-frame on GPU, reaction-aware pooling) as the "CNN done right"
(frozen → no 36-subject overfit, unlike from-scratch BadNet). Cache `fer_t<track>.csv` (1578 feats).
Measured through the official eval bridge. Also fixed the **mediapipe `solutions` bug** (mediapipe
≥0.10.31 + numpy 2.x drops the submodule → bad Haar-only crops); pinned **mediapipe==0.10.21 +
numpy==1.26.4** in dream-venv (torch 2.5.1+cu121 GPU survives) and re-extracted with proper
mediapipe face crops.

| Config | crop | **video macro-F1** | F1(control) | bal-acc | window macro-F1 |
|---|---|---|---|---|---|
| FER only | Haar | 0.589 | 0.306 | 0.605 | 0.599 |
| FER only | mediapipe | 0.605 | 0.333 | 0.623 | 0.626 |
| **Baseline (au+rf+blend)** | — | **0.656** | 0.394 | 0.643 | 0.658 |
| Baseline + FER | Haar | 0.653 | 0.391 | 0.645 | 0.656 |
| Baseline + FER | mediapipe | 0.647 | 0.396 | **0.660** | **0.660** |

**Verdict:** a frozen generic expression CNN is a *weaker, lower-dimensional* signal than tuned AUs
and does **not** lift the officially-ranked **video-level macro-F1** (0.647 ≤ 0.656). Proper
mediapipe crops were real (FER-only 0.589→0.605; in-fusion window control-F1 0.413→0.435, bal-acc
0.643→0.660) but the headline metric and the ~0.40 control-F1 wall are unchanged. **Frozen
pretrained face features are ruled out as the lever to 0.8.** The remaining hypotheses: (a) different
*modality* (body/head pose, speech prosody/content, robot/dialogue state), (b) end-to-end fine-tuned
video model (risky on 36 subjects; from-scratch BadNet only got 0.502), (c) higher fps for finer
micro-expression dynamics + more windows per clip (step 3).

## UPDATE 3 — pose + DINOv2 (FER done right) FINALLY move the control wall

Two new orthogonal streams, both densely sampled (s=48–64 frames, vs the failed FER's 16):
- **pose**: MediaPipe Pose head roll/yaw/pitch + shoulder/lean dynamics, framing-invariant,
  reaction-aware (`pose_t<track>.csv`, 80 feats).
- **faceemb**: frozen **DINOv2-base** embedding (768-d) on proper MediaPipe face crops, summarised by
  identity-invariant temporal dynamics — per-dim std / peak-velocity / apex + global motion energy
  (`faceemb_t<track>.csv`, 3076 feats). This is the FER redo "with a better model, done correctly".

Official video-level macro-F1 (single late-fusion stack run, T1):

| Config | macro-F1 | F1(control) | bal-acc | AUC |
|---|---|---|---|---|
| pose only | 0.579 | 0.271 | 0.581 | 0.627 |
| faceemb only (DINOv2) | 0.591 | 0.295 | 0.595 | 0.691 |
| **baseline (au+rf+blend)** | 0.650 | 0.388 | 0.644 | 0.782 |
| baseline + pose | 0.661 | 0.410 | 0.661 | 0.790 |
| **baseline + faceemb** | **0.669** | 0.422 | 0.664 | 0.783 |
| baseline + pose + faceemb | 0.666 | **0.426** | **0.676** | 0.791 |
| all (+old FER) | 0.662 | 0.422 | 0.677 | 0.790 |

**This is the first genuine movement on the control class.** F1(control) lifts 0.388 → 0.426 and
bal-acc 0.644 → 0.676 — the ~0.40 control wall that pure facial-AU fusion could not crack. Both new
streams are additive; the old sparse FER2013 model is now redundant (subsumed by DINOv2) and slightly
hurts.

**Correction to UPDATE 2:** "frozen face-CNN ruled out" was wrong *as a general claim*. The earlier
failure was a WEAK model (tiny FER2013 7-emotion ViT) + SPARSE sampling (16 frames averaging the
reaction spike away). A STRONG frozen backbone (DINOv2) + DENSE sampling + temporal-dynamics
aggregation **does** help (+0.019 macro-F1, +0.04 control-F1). Frozen pretrained features work — the
backbone and sampling density were the bug, exactly as suspected.

Caches are durably backed up (server `cache_backup/` + local `cache_backup/`, both gitignored), so
new fusion experiments never re-run extraction. Best operating point so far:
**baseline+pose+faceemb** (control-F1 0.426, bal-acc 0.676) or **baseline+faceemb** (macro-F1 0.669).

**Open question (under investigation):** a third party reportedly reached ~0.8 honest on T1.
That is far outside this feature set's ceiling → strongly implies either (a) additional modalities
we are not using (body/head pose, speech prosody/content, robot-interaction/dialogue state), (b)
end-to-end video learning capturing micro-expressions AU misses, or (c) a less strict eval split.
See research session notes.
