#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-blackbox-finetune-recall45}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -r blackbox_finetune_recall45/requirements.txt

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -m blackbox_finetune_recall45.gpu --cuda-device "$CUDA_DEVICE"
if [[ "$MODE" == "diagnose" ]]; then
  exit 0
fi
DATA_DIR="${DATA_DIR:-blackbox_finetune_recall45/data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-blackbox_finetune_recall45/data_validation}"
OUTPUT_DIR="${OUTPUT_DIR:-blackbox_finetune_recall45/runs/qwen2.5-0.5b-blackbox-recall45-lora}"
MIN_POSITIVE_RECALL="${MIN_POSITIVE_RECALL:-0.45}"
TRAIN_SEED="${TRAIN_SEED:-20260545}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_START="20110101"
  TRAIN_END="20151231"
  VALIDATION_START="20260101"
  VALIDATION_END="20260131"
  POS_LIMIT="${SMOKE_POSITIVE_LIMIT:-12}"
  EPOCHS="${EPOCHS:-3}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
  GRAD_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
else
  TRAIN_START="20110101"
  TRAIN_END="20251231"
  VALIDATION_START="20260101"
  VALIDATION_END="20260430"
  POS_LIMIT="${POSITIVE_LIMIT:-}"
  EPOCHS="${EPOCHS:-1}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
  GRAD_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
fi

BUILD_ARGS=(--output-dir "$DATA_DIR" --start-date "$TRAIN_START" --end-date "$TRAIN_END" --negative-ratio 1.0)
VAL_ARGS=(--output-dir "$VALIDATION_DATA_DIR" --start-date "$VALIDATION_START" --end-date "$VALIDATION_END" --negative-ratio 1.0)
if [[ -n "$POS_LIMIT" ]]; then
  BUILD_ARGS+=(--positive-limit "$POS_LIMIT")
fi
if [[ "$MODE" == "smoke" ]]; then
  VAL_ARGS+=(--positive-limit "$POS_LIMIT")
fi

python -m blackbox_finetune_recall45.build_dataset "${BUILD_ARGS[@]}"
python -m blackbox_finetune_recall45.build_validation_dataset "${VAL_ARGS[@]}"
python -m blackbox_finetune_recall45.train --base-model "$BASE_MODEL" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --max-seq-length "$MAX_SEQ_LENGTH" --epochs "$EPOCHS" --batch-size 1 --gradient-accumulation-steps "$GRAD_STEPS" --learning-rate "$LEARNING_RATE" --max-grad-norm "$MAX_GRAD_NORM" --checkpoint-every "$CHECKPOINT_EVERY" --train-seed "$TRAIN_SEED" --cuda-device "$CUDA_DEVICE"
python -m blackbox_finetune_recall45.evaluate --base-model "$BASE_MODEL" --adapter-dir "$OUTPUT_DIR/adapter" --data-dir "$VALIDATION_DATA_DIR" --threshold 0.50 --min-positive-recall "$MIN_POSITIVE_RECALL" --cuda-device "$CUDA_DEVICE" --max-seq-length "$MAX_SEQ_LENGTH"
python -m unittest tests.test_blackbox_finetune_recall45 -v
