#!/bin/bash
# MCM-DM Full Pipeline: tile embedding → Morse features → train → test
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Usage:
#   bash run.sh [DATASET] [CUDA_ID] [SEED] [EPOCHS]
#
# Examples:
#   bash run.sh Crime2026 0 42 100
#   bash run.sh Twitter   0 42 300

set -e

DATASET=${1:-Crime2026}
CUDA_ID=${2:-0}
SEED=${3:-42}
EPOCHS=${4:-100}
KNN_K=5

# Zoom level per dataset (OSM tile resolution)
case $DATASET in
    Earthquake)  ZOOM=12 ;;
    COVID19)     ZOOM=12 ;;
    Tornado)     ZOOM=7  ;;
    eMAS)        ZOOM=7  ;;
    Crime2026)   ZOOM=12 ;;
    Traffic)     ZOOM=12 ;;
    Citibike)    ZOOM=15 ;;
    Twitter)     ZOOM=14 ;;
    Wildfire_CA) ZOOM=8  ;;
    *)
        echo "Unknown dataset: $DATASET. Please set ZOOM manually: ZOOM=12 bash run.sh ..."
        exit 1 ;;
esac

PYTHON=${PYTHON:-python3}
EMB_FILE="tile_embeddings_${DATASET}_z${ZOOM}.pt"
MORSE_FILE="morse_features_${DATASET}_z${ZOOM}.pt"
MODEL_DIR="models_mm/${DATASET}_ts500_morse_temporal_seed${SEED}"

echo "============================================"
echo " MCM-DM Pipeline"
echo " Dataset : $DATASET  Zoom : $ZOOM"
echo " CUDA    : $CUDA_ID  Seed : $SEED  Epochs : $EPOCHS"
echo "============================================"

# Step 1: Extract tile embeddings (downloads OSM tiles + runs Qwen2-VL)
if [ -f "$EMB_FILE" ]; then
    echo "[Step 1] Tile embeddings already exist: $EMB_FILE — skipping."
else
    echo "[Step 1] Extracting tile embeddings..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID $PYTHON extract_tile_embeddings.py \
        --dataset $DATASET --zoom $ZOOM --data_root dataset
    echo "[Step 1] Done → $EMB_FILE"
fi

# Step 2: Build Morse complex features
if [ -f "$MORSE_FILE" ]; then
    echo "[Step 2] Morse features already exist: $MORSE_FILE — skipping."
else
    echo "[Step 2] Building Morse features (k=$KNN_K)..."
    $PYTHON build_morse_features.py \
        --emb_file $EMB_FILE \
        --out_file $MORSE_FILE \
        --k $KNN_K
    echo "[Step 2] Done → $MORSE_FILE"
fi

# Step 3: Train
echo "[Step 3] Training for $EPOCHS epochs..."
CUDA_VISIBLE_DEVICES=$CUDA_ID $PYTHON train_mm_morse_temporal.py \
    --dataset $DATASET --zoom $ZOOM \
    --emb_file $EMB_FILE --morse_file $MORSE_FILE \
    --timesteps 500 --samplingsteps 500 \
    --total_epochs $EPOCHS \
    --alpha 0.05 --beta_topo 0.001 --beta_topo_t 0.0005 \
    --seed $SEED --cuda_id 0
echo "[Step 3] Training done. Checkpoints saved to $MODEL_DIR/"

# Step 4: Test
echo "[Step 4] Running test..."
CUDA_VISIBLE_DEVICES=$CUDA_ID $PYTHON test_morse_pure.py \
    --dataset $DATASET --zoom $ZOOM \
    --emb_file $EMB_FILE --morse_file $MORSE_FILE \
    --timesteps 500 --samplingsteps 500 \
    --model_ckpt          ${MODEL_DIR}/model_best.pkl \
    --morse_ckpt          ${MODEL_DIR}/morse_best.pkl \
    --gate_ckpt           ${MODEL_DIR}/gate_best.pkl \
    --temporal_morse_ckpt ${MODEL_DIR}/temporal_morse_best.pkl \
    --cuda_id 0 --seed $SEED
echo "[Step 4] Done."
echo "============================================"
