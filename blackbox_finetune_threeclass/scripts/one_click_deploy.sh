#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source blackbox_finetune_threeclass/scripts/set_wsl_env.sh
MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

env_flag() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -r blackbox_finetune_threeclass/requirements.txt
python -m blackbox_finetune_threeclass.gpu --cuda-device "$CUDA_DEVICE"

if [[ "$MODE" == "diagnose" ]]; then
  exit 0
fi

POSITIVE_LIMIT=""
TRAIN_EPOCHS="$EPOCHS"
if [[ "$MODE" == "smoke" ]]; then
  POSITIVE_LIMIT="--positive-limit ${SMOKE_POSITIVE_LIMIT:-20}"
  TRAIN_EPOCHS="${SMOKE_EPOCHS:-0.02}"
fi

echo "==== Three-class project configuration ===="
for name in \
  MODE BASE_MODEL VENV_DIR SAMPLE_MODE MAX_SEQ_LENGTH \
  TRAIN_START_DATE TRAIN_END_DATE VALIDATION_START_DATE VALIDATION_END_DATE \
  DATA_DIR VALIDATION_DATA_DIR OUTPUT_DIR INITIAL_BINARY_ADAPTER_DIR EPOCHS BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS LEARNING_RATE WEIGHT_DECAY MAX_GRAD_NORM \
  LORA_RANK LORA_DROPOUT CHECKPOINT_EVERY TRAIN_SEED EVAL_RANDOM_SEED \
  EVAL_SAMPLE_METHOD EVAL_MAX_SAMPLES EVAL_PRECISION_TOP_K \
  EVAL_PRECISION_THRESHOLD NEGATIVE_WEIGHT NEUTRAL_WEIGHT EVAL_THRESHOLD_TOP_RATIO \
  SAMPLE_BOTTOM_BAND_RATIO ON_THE_FLY_TOKENIZE CUDA_DEVICE \
  HF_LOCAL_FILES_ONLY HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TRUST_REMOTE_CODE \
  REBUILD_DATASET CANDIDATE_BATCH_SIZE MYSQL_QUERY_RETRIES; do
  printf '  %s=%s\n' "$name" "${!name}"
done
echo "  CLASS_RATIO=positive:negative:neutral=1:2:10"
echo "  LABEL_RULE=first trigger in next 3 trading days: +20% / -6% / neither"
echo "==========================================="

if env_flag "$REBUILD_DATASET" || [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/test.jsonl" ]]; then
  # shellcheck disable=SC2086
  python -m blackbox_finetune_threeclass.build_dataset \
    --output-dir "$DATA_DIR" \
    --start-date "$TRAIN_START_DATE" \
    --end-date "$TRAIN_END_DATE" \
    --sample-mode "$SAMPLE_MODE" \
    --seed "$TRAIN_SEED" \
    --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
    --mysql-query-retries "$MYSQL_QUERY_RETRIES" \
    $POSITIVE_LIMIT
else
  echo "Using cached training dataset: $DATA_DIR"
fi

if env_flag "$REBUILD_DATASET" || [[ ! -f "$VALIDATION_DATA_DIR/test.jsonl" ]]; then
  # shellcheck disable=SC2086
  python -m blackbox_finetune_threeclass.build_validation_dataset \
    --output-dir "$VALIDATION_DATA_DIR" \
    --start-date "$VALIDATION_START_DATE" \
    --end-date "$VALIDATION_END_DATE" \
    --sample-mode "$SAMPLE_MODE" \
    --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
    --mysql-query-retries "$MYSQL_QUERY_RETRIES" \
    $POSITIVE_LIMIT
else
  echo "Using cached validation dataset: $VALIDATION_DATA_DIR"
fi

if [[ "$MODE" == "dataset-only" ]]; then
  exit 0
fi

TRAIN_ARGS=(
  --base-model "$BASE_MODEL"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --epochs "$TRAIN_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM"
  --lora-rank "$LORA_RANK"
  --lora-dropout "$LORA_DROPOUT"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --train-seed "$TRAIN_SEED"
  --eval-max-samples "$EVAL_MAX_SAMPLES"
  --eval-precision-top-k "$EVAL_PRECISION_TOP_K"
  --eval-precision-threshold "$EVAL_PRECISION_THRESHOLD"
  --cuda-device "$CUDA_DEVICE"
)
if env_flag "$ON_THE_FLY_TOKENIZE"; then
  TRAIN_ARGS+=(--on-the-fly-tokenize)
fi
if [[ -n "$INITIAL_BINARY_ADAPTER_DIR" ]]; then
  if [[ "$INITIAL_BINARY_ADAPTER_DIR" == [A-Za-z]:\\* ]] && command -v wslpath >/dev/null 2>&1; then
    INITIAL_BINARY_ADAPTER_DIR="$(wslpath -u "$INITIAL_BINARY_ADAPTER_DIR")"
  fi
  TRAIN_ARGS+=(--initial-binary-adapter-dir "$INITIAL_BINARY_ADAPTER_DIR")
fi
python -m blackbox_finetune_threeclass.train "${TRAIN_ARGS[@]}"

if [[ "${RUN_FINAL_EVAL:-1}" == "1" ]]; then
  python -m blackbox_finetune_threeclass.evaluate \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUTPUT_DIR/adapter" \
    --data-dir "$VALIDATION_DATA_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --cuda-device "$CUDA_DEVICE"
fi

python -m unittest tests.test_blackbox_finetune_threeclass -v
