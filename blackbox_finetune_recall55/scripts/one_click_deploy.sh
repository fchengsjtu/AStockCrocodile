#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

env_flag() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

dataset_ready() {
  [[ -f "$1/train.jsonl" && -f "$1/test.jsonl" ]]
}

run_dataset_build_if_needed() {
  local dir="$1"
  local label="$2"
  local force_value="${3:-}"
  shift 3
  if dataset_ready "$dir" && ! env_flag "$force_value"; then
    echo "Using cached $label dataset in $dir; set REBUILD_DATASET=1 to rebuild all datasets."
    return 0
  fi
  "$@"
}

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-blackbox-finetune-recall55}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -r blackbox_finetune_recall55/requirements.txt

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -m blackbox_finetune_recall55.gpu --cuda-device "$CUDA_DEVICE"
if [[ "$MODE" == "diagnose" ]]; then
  exit 0
fi
MIN_POSITIVE_RECALL="${MIN_POSITIVE_RECALL:-0.60}"
TRAIN_SEED="${TRAIN_SEED:-20260560}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
OOM_PATIENCE="${OOM_PATIENCE:-20}"
NONFINITE_SKIP_LIMIT="${NONFINITE_SKIP_LIMIT:-100}"
NONFINITE_BACKOFF_EVERY="${NONFINITE_BACKOFF_EVERY:-10}"
LR_BACKOFF_FACTOR="${LR_BACKOFF_FACTOR:-0.5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-6}"
RESUME_ADAPTER_DIR="${RESUME_ADAPTER_DIR:-}"
SAMPLE_MODE="${SAMPLE_MODE:-long}"
DEFAULT_DATA_DIR="blackbox_finetune_recall55/data_no_partial_week_$SAMPLE_MODE"
DEFAULT_VALIDATION_DATA_DIR="blackbox_finetune_recall55/data_evaluation_no_partial_week_$SAMPLE_MODE"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$DEFAULT_VALIDATION_DATA_DIR}"
if [[ "$SAMPLE_MODE" == "short" ]]; then
  DEFAULT_OUTPUT_DIR="blackbox_finetune_recall55/runs/qwen2.5-0.5b-blackbox-recall55-short-lora"
elif [[ "$SAMPLE_MODE" == "xlong" ]]; then
  DEFAULT_OUTPUT_DIR="blackbox_finetune_recall55/runs/qwen2.5-0.5b-blackbox-recall55-xlong-lora"
elif [[ "$SAMPLE_MODE" == "xxlong" ]]; then
  DEFAULT_OUTPUT_DIR="blackbox_finetune_recall55/runs/qwen2.5-0.5b-blackbox-recall55-xxlong-lora"
else
  DEFAULT_OUTPUT_DIR="blackbox_finetune_recall55/runs/qwen2.5-0.5b-blackbox-recall55-long-lora"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-3.0}"
if [[ "$SAMPLE_MODE" == "short" ]]; then
  DEFAULT_MAX_SEQ_LENGTH=1024
  DEFAULT_CHECKPOINT_EVERY=500
elif [[ "$SAMPLE_MODE" == "xlong" ]]; then
  DEFAULT_MAX_SEQ_LENGTH=3072
  DEFAULT_CHECKPOINT_EVERY=500
elif [[ "$SAMPLE_MODE" == "xxlong" ]]; then
  DEFAULT_MAX_SEQ_LENGTH=4096
  DEFAULT_CHECKPOINT_EVERY=500
else
  DEFAULT_MAX_SEQ_LENGTH=2048
  DEFAULT_CHECKPOINT_EVERY=500
fi
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$DEFAULT_CHECKPOINT_EVERY}"

if [[ "$MODE" == "smoke" ]]; then
  DEFAULT_TRAIN_START="20200101"
  DEFAULT_TRAIN_END="20211231"
  DEFAULT_VALIDATION_START="20260101"
  DEFAULT_VALIDATION_END="20260131"
  TRAIN_START="${TRAIN_START_DATE:-$DEFAULT_TRAIN_START}"
  TRAIN_END="${TRAIN_END_DATE:-$DEFAULT_TRAIN_END}"
  VALIDATION_START="${VALIDATION_START_DATE:-${TEST_START_DATE:-$DEFAULT_VALIDATION_START}}"
  VALIDATION_END="${VALIDATION_END_DATE:-${TEST_END_DATE:-$DEFAULT_VALIDATION_END}}"
  POS_LIMIT="${SMOKE_POSITIVE_LIMIT:-12}"
  EPOCHS="${EPOCHS:-3}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}"
  GRAD_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
else
  DEFAULT_TRAIN_START="20200101"
  DEFAULT_TRAIN_END="20251231"
  DEFAULT_VALIDATION_START="20260101"
  DEFAULT_VALIDATION_END="20260430"
  TRAIN_START="${TRAIN_START_DATE:-$DEFAULT_TRAIN_START}"
  TRAIN_END="${TRAIN_END_DATE:-$DEFAULT_TRAIN_END}"
  VALIDATION_START="${VALIDATION_START_DATE:-${TEST_START_DATE:-$DEFAULT_VALIDATION_START}}"
  VALIDATION_END="${VALIDATION_END_DATE:-${TEST_END_DATE:-$DEFAULT_VALIDATION_END}}"
  POS_LIMIT="${POSITIVE_LIMIT:-}"
  EPOCHS="${EPOCHS:-1}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}"
  GRAD_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
fi

BUILD_ARGS=(--output-dir "$DATA_DIR" --start-date "$TRAIN_START" --end-date "$TRAIN_END" --negative-ratio "$NEGATIVE_RATIO" --sample-mode "$SAMPLE_MODE")
VAL_ARGS=(--output-dir "$VALIDATION_DATA_DIR" --start-date "$VALIDATION_START" --end-date "$VALIDATION_END" --negative-ratio "$NEGATIVE_RATIO" --sample-mode "$SAMPLE_MODE")
if [[ -n "$POS_LIMIT" ]]; then
  BUILD_ARGS+=(--positive-limit "$POS_LIMIT")
fi
if [[ "$MODE" == "smoke" ]]; then
  VAL_ARGS+=(--positive-limit "$POS_LIMIT")
fi

run_dataset_build_if_needed "$DATA_DIR" training "${REBUILD_DATASET:-}" python -m blackbox_finetune_recall55.build_dataset "${BUILD_ARGS[@]}"
run_dataset_build_if_needed "$VALIDATION_DATA_DIR" validation "${REBUILD_VALIDATION_DATASET:-${REBUILD_DATASET:-}}" python -m blackbox_finetune_recall55.build_validation_dataset "${VAL_ARGS[@]}"
TRAIN_ARGS=(--base-model "$BASE_MODEL" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --max-seq-length "$MAX_SEQ_LENGTH" --epochs "$EPOCHS" --batch-size 1 --gradient-accumulation-steps "$GRAD_STEPS" --learning-rate "$LEARNING_RATE" --max-grad-norm "$MAX_GRAD_NORM" --checkpoint-every "$CHECKPOINT_EVERY" --oom-patience "$OOM_PATIENCE" --nonfinite-skip-limit "$NONFINITE_SKIP_LIMIT" --nonfinite-backoff-every "$NONFINITE_BACKOFF_EVERY" --lr-backoff-factor "$LR_BACKOFF_FACTOR" --min-learning-rate "$MIN_LEARNING_RATE" --train-seed "$TRAIN_SEED" --cuda-device "$CUDA_DEVICE")
if [[ -n "$RESUME_ADAPTER_DIR" ]]; then
  TRAIN_ARGS+=(--resume-adapter-dir "$RESUME_ADAPTER_DIR")
fi
if env_flag "${REBUILD_TOKEN_CACHE:-}"; then
  TRAIN_ARGS+=(--rebuild-token-cache)
fi
if env_flag "${NO_AUTO_RESUME:-}"; then
  TRAIN_ARGS+=(--no-auto-resume)
fi
python -m blackbox_finetune_recall55.train "${TRAIN_ARGS[@]}"
python -m blackbox_finetune_recall55.evaluate --base-model "$BASE_MODEL" --adapter-dir "$OUTPUT_DIR/adapter" --data-dir "$VALIDATION_DATA_DIR" --threshold 0.50 --min-positive-recall "$MIN_POSITIVE_RECALL" --cuda-device "$CUDA_DEVICE" --max-seq-length "$MAX_SEQ_LENGTH"
python -m unittest tests.test_blackbox_finetune_recall55 -v
