#!/bin/bash
# B0 — dense per-frame sequence extraction (SAVES the trajectories this time).
# Serialized: au_seq (GPU) then gaze_seq (CPU) to avoid CPU oversubscription. Both tracks.
set -x
export ERRHRI_CACHE=/home/ic2/research/errhri/train2/cache
export ERRHRI_T1_ROOT=/home/ic2/research/errhri/raw/d2/trainval
export ERRHRI_T2_ROOT=/home/ic2/research/errhri/raw/d1/trainval
export OPENGRAPHAU_DIR=/home/ic2/research/errhri/OpenGraphAU
export CUDA_VISIBLE_DEVICES=0
PY=/home/ic2/dream-venv/bin/python
cd /home/ic2/research/errhri-features

$PY - <<'EOF'
from errhri_features.extractors import AUSeqExtractor
for track in (1, 2):
    AUSeqExtractor(s=48, workers=4).run(track)
EOF

$PY - <<'EOF'
from errhri_features.extractors import GazeSeqExtractor
for track in (1, 2):
    GazeSeqExtractor(s=48, workers=6).run(track)
EOF

cp /home/ic2/research/errhri/train2/cache/au_seq_t*.csv \
   /home/ic2/research/errhri/train2/cache/gaze_seq_t*.csv \
   /home/ic2/research/errhri/train2/cache_backup/ 2>/dev/null
echo "DENSE_SEQ ALL DONE"
