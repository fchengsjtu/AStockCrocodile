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

normalize_dataset_path() {
  local path="$1"
  if [[ "$path" == [A-Za-z]:\\* ]] && command -v wslpath >/dev/null 2>&1; then
    path="$(wslpath -u "$path")"
  elif [[ "$path" != /* ]]; then
    path="$ROOT_DIR/$path"
  fi
  printf '%s\n' "$path"
}

prepare_explicit_datasets() {
  local configured=0
  local value
  for value in "$TRAIN_DATASET_PATH" "$TEST_DATASET_PATH" "$VALIDATION_DATASET_PATH"; do
    [[ -n "$value" ]] && configured=$((configured + 1))
  done
  if [[ "$configured" == "0" ]]; then
    return 0
  fi
  if [[ "$configured" != "3" ]]; then
    echo "TRAIN_DATASET_PATH, TEST_DATASET_PATH, and VALIDATION_DATASET_PATH must be set together." >&2
    exit 2
  fi

  TRAIN_DATASET_PATH="$(normalize_dataset_path "$TRAIN_DATASET_PATH")"
  TEST_DATASET_PATH="$(normalize_dataset_path "$TEST_DATASET_PATH")"
  VALIDATION_DATASET_PATH="$(normalize_dataset_path "$VALIDATION_DATASET_PATH")"
  EXPLICIT_DATASET_WORK_DIR="$(normalize_dataset_path "$EXPLICIT_DATASET_WORK_DIR")"
  for value in "$TRAIN_DATASET_PATH" "$TEST_DATASET_PATH" "$VALIDATION_DATASET_PATH"; do
    if [[ ! -f "$value" ]]; then
      echo "Explicit dataset file does not exist: $value" >&2
      exit 2
    fi
  done

  DATA_DIR="$EXPLICIT_DATASET_WORK_DIR/training"
  VALIDATION_DATA_DIR="$EXPLICIT_DATASET_WORK_DIR/validation"
  mkdir -p "$DATA_DIR" "$VALIDATION_DATA_DIR"
  ln -sfn "$TRAIN_DATASET_PATH" "$DATA_DIR/train.jsonl"
  ln -sfn "$TEST_DATASET_PATH" "$DATA_DIR/test.jsonl"
  ln -sfn "$VALIDATION_DATASET_PATH" "$VALIDATION_DATA_DIR/test.jsonl"
  EXPLICIT_DATASETS_ENABLED=1
  return 0
}

recall_target_value() {
  python - "$1" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except Exception:
    value = 0.60
if value > 1:
    value /= 100.0
value = min(max(value, 0.0), 1.0)
print(f"{value:.2f}")
PY
}

recall_target_tag() {
  python - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"recall{round(value * 100):02d}")
PY
}

model_label_value() {
  recall_target_value "$1"
}

model_tag() {
  recall_target_tag "$1"
}

run_dataset_build_if_needed() {
  local dir="$1"
  local label="$2"
  local force_value="${3:-}"
  local expected_signature="$4"
  local signature_file="$dir/.dataset_signature"
  shift 4
  if dataset_ready "$dir" && ! env_flag "$force_value"; then
    if [[ -f "$signature_file" ]] && [[ "$(cat "$signature_file")" == "$expected_signature" ]]; then
      echo "Using cached $label dataset in $dir; configuration signature matches."
      return 0
    fi
    echo "Cached $label dataset configuration does not match; rebuilding $dir."
  fi
  "$@"
  printf '%s\n' "$expected_signature" > "$signature_file"
  DATASET_REBUILT=1
}

train_supports_arg() {
  local arg="$1"
  [[ "$TRAIN_HELP" == *"$arg"* ]]
}

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-blackbox-finetune-recall60}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -r blackbox_finetune_recall60/requirements.txt

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
python -m blackbox_finetune_recall60.gpu --cuda-device "$CUDA_DEVICE"
if [[ "$MODE" == "diagnose" ]]; then
  exit 0
fi
MODEL_LABEL_VALUE="$(model_label_value "${MODEL_LABEL_VALUE:-${RECALL_TARGET:-${MIN_POSITIVE_RECALL:-0.60}}}")"
MODEL_TAG="$(model_tag "$MODEL_LABEL_VALUE")"
# Deprecated compatibility only: these keep older default paths and scripts
# working. They are not validation or training objectives.
RECALL_TARGET="${RECALL_TARGET:-$MODEL_LABEL_VALUE}"
RECALL_TAG="${RECALL_TAG:-$MODEL_TAG}"
MIN_POSITIVE_RECALL="${MIN_POSITIVE_RECALL:-$MODEL_LABEL_VALUE}"
PRECISION_TOP_K="${PRECISION_TOP_K:-10}"
PRECISION_THRESHOLD="${PRECISION_THRESHOLD:-${MIN_PRECISION_AT_20:-${PRECISION_AT_20_TARGET:-0.40}}}"
RANDOM_SEED="${RANDOM_SEED:-937498347}"
TRAIN_SEED="${TRAIN_SEED:-$RANDOM_SEED}"
EVAL_RANDOM_SEED="${EVAL_RANDOM_SEED:-20260530}"
export EVAL_RANDOM_SEED
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
USE_4BIT="${USE_4BIT:-1}"
ON_THE_FLY_TOKENIZE="${ON_THE_FLY_TOKENIZE:-1}"
OOM_PATIENCE="${OOM_PATIENCE:-20}"
NONFINITE_SKIP_LIMIT="${NONFINITE_SKIP_LIMIT:-100}"
NONFINITE_BACKOFF_EVERY="${NONFINITE_BACKOFF_EVERY:-10}"
LR_BACKOFF_FACTOR="${LR_BACKOFF_FACTOR:-0.5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-6}"
RESUME_ADAPTER_DIR="${RESUME_ADAPTER_DIR:-}"
INITIAL_ADAPTER_DIR="${INITIAL_ADAPTER_DIR:-}"
if [[ -n "$INITIAL_ADAPTER_DIR" && "$INITIAL_ADAPTER_DIR" == *\\* ]] && command -v wslpath >/dev/null 2>&1; then
  INITIAL_ADAPTER_DIR="$(wslpath -u "$INITIAL_ADAPTER_DIR")"
fi
SAMPLE_MODE="${SAMPLE_MODE:-xlong}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-5.0}"
NEGATIVE_TAG="$(python - "$NEGATIVE_RATIO" <<'PY'
import sys
value = float(sys.argv[1])
text = ("%g" % value).replace(".", "_")
print(f"neg{text}")
PY
)"
DEFAULT_DATA_DIR="blackbox_finetune_recall60/data_no_partial_week_${MODEL_TAG}_${SAMPLE_MODE}_${NEGATIVE_TAG}"
DEFAULT_VALIDATION_DATA_DIR="blackbox_finetune_recall60/data_evaluation_no_partial_week_${MODEL_TAG}_${SAMPLE_MODE}_${NEGATIVE_TAG}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$DEFAULT_VALIDATION_DATA_DIR}"
DEFAULT_OUTPUT_DIR="blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-${MODEL_TAG}-${SAMPLE_MODE}-lora"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-}"
TEST_DATASET_PATH="${TEST_DATASET_PATH:-}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-}"
EXPLICIT_DATASET_WORK_DIR="${EXPLICIT_DATASET_WORK_DIR:-$OUTPUT_DIR/input_datasets}"
EXPLICIT_DATASETS_ENABLED=0
prepare_explicit_datasets
if [[ "$SAMPLE_MODE" == "short" ]]; then
  DEFAULT_MAX_SEQ_LENGTH=1024
  DEFAULT_CHECKPOINT_EVERY=100
elif [[ "$SAMPLE_MODE" == "xlong" ]]; then
  DEFAULT_MAX_SEQ_LENGTH=3072
  DEFAULT_CHECKPOINT_EVERY=100
elif [[ "$SAMPLE_MODE" == "xxlong" ]]; then
  DEFAULT_MAX_SEQ_LENGTH=4096
  DEFAULT_CHECKPOINT_EVERY=100
else
  DEFAULT_MAX_SEQ_LENGTH=2048
  DEFAULT_CHECKPOINT_EVERY=100
fi
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-${CHECKOUT_EVERY:-$DEFAULT_CHECKPOINT_EVERY}}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.48}"
EVAL_THRESHOLD_POSITION="${EVAL_THRESHOLD_POSITION:-0.2}"
EVAL_PRECISION_TOP_K="${EVAL_PRECISION_TOP_K:-$PRECISION_TOP_K}"
EVAL_PRECISION_THRESHOLD="${EVAL_PRECISION_THRESHOLD:-${EVAL_MIN_PRECISION_AT_20:-$PRECISION_THRESHOLD}}"
EVAL_SAMPLE_METHOD="${EVAL_SAMPLE_METHOD:-random}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
export EVAL_SAMPLE_METHOD
POSITIVE_LOSS_WEIGHT="${POSITIVE_LOSS_WEIGHT:-1.0}"
NEGATIVE_LOSS_WEIGHT="${NEGATIVE_LOSS_WEIGHT:-1.0}"
FP_DYNAMIC_PENALTY="${FP_DYNAMIC_PENALTY:-0}"
FP_PENALTY_WEIGHT="${FP_PENALTY_WEIGHT:-1.0}"
FP_THRESHOLD_EMA_ALPHA="${FP_THRESHOLD_EMA_ALPHA:-0.2}"
FP_THRESHOLD_MIN="${FP_THRESHOLD_MIN:-0.40}"
FP_THRESHOLD_MAX="${FP_THRESHOLD_MAX:-0.65}"
PRECISION_TAG="$(python - "$EVAL_PRECISION_TOP_K" "$EVAL_PRECISION_THRESHOLD" <<'PY'
import sys
k = max(1, int(float(sys.argv[1])))
t = float(sys.argv[2])
if t > 1:
    t /= 100.0
t = min(max(t, 0.0), 1.0)
print(f"top{k}_precision{round(t * 100):03d}")
PY
)"
FINAL_PRECISION_TAG="$(python - "$PRECISION_TOP_K" "$PRECISION_THRESHOLD" <<'PY'
import sys
k = max(1, int(float(sys.argv[1])))
t = float(sys.argv[2])
if t > 1:
    t /= 100.0
t = min(max(t, 0.0), 1.0)
print(f"top{k}_precision{round(t * 100):03d}")
PY
)"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR/evaluations/$PRECISION_TAG}"

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
  EPOCHS="${EPOCHS:-0.3}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}"
  GRAD_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
else
  DEFAULT_TRAIN_START="20230101"
  DEFAULT_TRAIN_END="20241231"
  DEFAULT_VALIDATION_START="20260101"
  DEFAULT_VALIDATION_END="20260530"
  TRAIN_START="${TRAIN_START_DATE:-$DEFAULT_TRAIN_START}"
  TRAIN_END="${TRAIN_END_DATE:-$DEFAULT_TRAIN_END}"
  VALIDATION_START="${VALIDATION_START_DATE:-${TEST_START_DATE:-$DEFAULT_VALIDATION_START}}"
  VALIDATION_END="${VALIDATION_END_DATE:-${TEST_END_DATE:-$DEFAULT_VALIDATION_END}}"
  POS_LIMIT="${POSITIVE_LIMIT:-}"
  EPOCHS="${EPOCHS:-0.3}"
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}"
  GRAD_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
fi

cat <<EOF
Project blackbox environment:
  MODE=$MODE
  MODEL_TAG=$MODEL_TAG
  TRAINING_LOSS=causal_lm_cross_entropy
  EVALUATION_GATE=precision@$EVAL_PRECISION_TOP_K >= $EVAL_PRECISION_THRESHOLD
  SAMPLE_MODE=$SAMPLE_MODE
  NEGATIVE_RATIO=$NEGATIVE_RATIO
  NEGATIVE_TAG=$NEGATIVE_TAG
  TRAIN_START=$TRAIN_START
  TRAIN_END=$TRAIN_END
  VALIDATION_START=$VALIDATION_START
  VALIDATION_END=$VALIDATION_END
  DATA_DIR=$DATA_DIR
  VALIDATION_DATA_DIR=$VALIDATION_DATA_DIR
  TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-<unset>}
  TEST_DATASET_PATH=${TEST_DATASET_PATH:-<unset>}
  VALIDATION_DATASET_PATH=${VALIDATION_DATASET_PATH:-<unset>}
  EXPLICIT_DATASET_WORK_DIR=$EXPLICIT_DATASET_WORK_DIR
  EXPLICIT_DATASETS_ENABLED=$EXPLICIT_DATASETS_ENABLED
  OUTPUT_DIR=$OUTPUT_DIR
  MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH
  EPOCHS=$EPOCHS
  GRADIENT_ACCUMULATION_STEPS=$GRAD_STEPS
  RANDOM_SEED=$RANDOM_SEED
  TRAIN_SEED=$TRAIN_SEED
  CHECKPOINT_EVERY=$CHECKPOINT_EVERY
  LEARNING_RATE=$LEARNING_RATE
  EVAL_THRESHOLD=$EVAL_THRESHOLD
  EVAL_THRESHOLD_POSITION=$EVAL_THRESHOLD_POSITION
  EVAL_RANDOM_SEED=$EVAL_RANDOM_SEED
  EVAL_SAMPLE_METHOD=$EVAL_SAMPLE_METHOD
  EVAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES
  EVAL_PRECISION_TOP_K=$EVAL_PRECISION_TOP_K
  EVAL_PRECISION_THRESHOLD=$EVAL_PRECISION_THRESHOLD
  POSITIVE_LOSS_WEIGHT=$POSITIVE_LOSS_WEIGHT
  NEGATIVE_LOSS_WEIGHT=$NEGATIVE_LOSS_WEIGHT
  FP_DYNAMIC_PENALTY=$FP_DYNAMIC_PENALTY
  FP_PENALTY_WEIGHT=$FP_PENALTY_WEIGHT
  FP_THRESHOLD_EMA_ALPHA=$FP_THRESHOLD_EMA_ALPHA
  FP_THRESHOLD_MIN=$FP_THRESHOLD_MIN
  FP_THRESHOLD_MAX=$FP_THRESHOLD_MAX
  RESUME_ADAPTER_DIR=${RESUME_ADAPTER_DIR:-<unset>}
  INITIAL_ADAPTER_DIR=${INITIAL_ADAPTER_DIR:-<unset>}
  USE_4BIT=$USE_4BIT
  HF_LOCAL_FILES_ONLY=$HF_LOCAL_FILES_ONLY
  HF_HUB_OFFLINE=$HF_HUB_OFFLINE
  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE
  TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE
EOF

BUILD_ARGS=(--output-dir "$DATA_DIR" --start-date "$TRAIN_START" --end-date "$TRAIN_END" --negative-ratio "$NEGATIVE_RATIO" --sample-mode "$SAMPLE_MODE")
VAL_ARGS=(--output-dir "$VALIDATION_DATA_DIR" --start-date "$VALIDATION_START" --end-date "$VALIDATION_END" --negative-ratio "$NEGATIVE_RATIO" --sample-mode "$SAMPLE_MODE")
if [[ -n "$POS_LIMIT" ]]; then
  BUILD_ARGS+=(--positive-limit "$POS_LIMIT")
fi
if [[ "$MODE" == "smoke" ]]; then
  VAL_ARGS+=(--positive-limit "$POS_LIMIT")
fi

TRAIN_DATASET_SIGNATURE="kind=training;start=$TRAIN_START;end=$TRAIN_END;negative_ratio=$NEGATIVE_RATIO;sample_mode=$SAMPLE_MODE;positive_limit=${POS_LIMIT:-all};seed=$TRAIN_SEED"
VALIDATION_DATASET_SIGNATURE="kind=validation;start=$VALIDATION_START;end=$VALIDATION_END;negative_ratio=$NEGATIVE_RATIO;sample_mode=$SAMPLE_MODE;positive_limit=${POS_LIMIT:-all};seed=$TRAIN_SEED"
DATASET_REBUILT=0
if [[ "$EXPLICIT_DATASETS_ENABLED" == "1" ]]; then
  echo "Using explicit train, test, and validation dataset files; dataset generation is skipped."
else
  run_dataset_build_if_needed "$DATA_DIR" training "${REBUILD_DATASET:-}" "$TRAIN_DATASET_SIGNATURE" python -m blackbox_finetune_recall60.build_dataset "${BUILD_ARGS[@]}"
  run_dataset_build_if_needed "$VALIDATION_DATA_DIR" validation "${REBUILD_VALIDATION_DATASET:-${REBUILD_DATASET:-}}" "$VALIDATION_DATASET_SIGNATURE" python -m blackbox_finetune_recall60.build_validation_dataset "${VAL_ARGS[@]}"
fi
if [[ "$DATASET_REBUILT" == "1" && -z "$RESUME_ADAPTER_DIR" ]]; then
  NO_AUTO_RESUME=1
  echo "Dataset was rebuilt; automatic checkpoint resume is disabled for this run."
fi
TRAIN_HELP="$(python -m blackbox_finetune_recall60.train --help 2>&1 || true)"
TRAIN_ARGS=(--base-model "$BASE_MODEL" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --checkpoint-eval-data-dir "$VALIDATION_DATA_DIR" --max-seq-length "$MAX_SEQ_LENGTH" --epochs "$EPOCHS" --batch-size 1 --gradient-accumulation-steps "$GRAD_STEPS" --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY" --max-grad-norm "$MAX_GRAD_NORM" --lora-rank "$LORA_RANK" --lora-dropout "$LORA_DROPOUT" --checkpoint-every "$CHECKPOINT_EVERY" --oom-patience "$OOM_PATIENCE" --nonfinite-skip-limit "$NONFINITE_SKIP_LIMIT" --nonfinite-backoff-every "$NONFINITE_BACKOFF_EVERY" --lr-backoff-factor "$LR_BACKOFF_FACTOR" --min-learning-rate "$MIN_LEARNING_RATE" --eval-threshold "$EVAL_THRESHOLD" --eval-max-samples "$EVAL_MAX_SAMPLES" --train-seed "$TRAIN_SEED" --cuda-device "$CUDA_DEVICE")
if train_supports_arg "--eval-threshold-position"; then
  TRAIN_ARGS+=(--eval-threshold-position "$EVAL_THRESHOLD_POSITION")
else
  echo "Current train.py does not support --eval-threshold-position; skipping it."
fi
if train_supports_arg "--eval-precision-top-k"; then
  TRAIN_ARGS+=(--eval-precision-top-k "$EVAL_PRECISION_TOP_K")
else
  echo "Current train.py does not support --eval-precision-top-k; skipping it."
fi
if train_supports_arg "--eval-precision-threshold"; then
  TRAIN_ARGS+=(--eval-precision-threshold "$EVAL_PRECISION_THRESHOLD")
else
  echo "Current train.py does not support --eval-precision-threshold; skipping it."
fi
if train_supports_arg "--positive-loss-weight"; then
  TRAIN_ARGS+=(--positive-loss-weight "$POSITIVE_LOSS_WEIGHT")
fi
if train_supports_arg "--negative-loss-weight"; then
  TRAIN_ARGS+=(--negative-loss-weight "$NEGATIVE_LOSS_WEIGHT")
fi
if env_flag "$FP_DYNAMIC_PENALTY"; then
  if train_supports_arg "--fp-dynamic-penalty"; then
    TRAIN_ARGS+=(--fp-dynamic-penalty)
  else
    echo "Current train.py does not support --fp-dynamic-penalty; skipping it."
  fi
fi
if train_supports_arg "--fp-penalty-weight"; then
  TRAIN_ARGS+=(--fp-penalty-weight "$FP_PENALTY_WEIGHT")
fi
if train_supports_arg "--fp-threshold-ema-alpha"; then
  TRAIN_ARGS+=(--fp-threshold-ema-alpha "$FP_THRESHOLD_EMA_ALPHA")
fi
if train_supports_arg "--fp-threshold-min"; then
  TRAIN_ARGS+=(--fp-threshold-min "$FP_THRESHOLD_MIN")
fi
if train_supports_arg "--fp-threshold-max"; then
  TRAIN_ARGS+=(--fp-threshold-max "$FP_THRESHOLD_MAX")
fi
if [[ -n "$EVAL_OUTPUT_DIR" ]]; then
  TRAIN_ARGS+=(--eval-output-dir "$EVAL_OUTPUT_DIR")
fi
if [[ -n "$RESUME_ADAPTER_DIR" ]]; then
  TRAIN_ARGS+=(--resume-adapter-dir "$RESUME_ADAPTER_DIR")
fi
if [[ -n "$INITIAL_ADAPTER_DIR" ]]; then
  if train_supports_arg "--initial-adapter-dir"; then
    TRAIN_ARGS+=(--initial-adapter-dir "$INITIAL_ADAPTER_DIR")
  else
    echo "Current train.py does not support --initial-adapter-dir; skipping INITIAL_ADAPTER_DIR=$INITIAL_ADAPTER_DIR."
  fi
fi
if ! env_flag "$USE_4BIT"; then
  TRAIN_ARGS+=(--no-4bit)
fi
if env_flag "${REBUILD_TOKEN_CACHE:-}"; then
  TRAIN_ARGS+=(--rebuild-token-cache)
fi
if env_flag "$ON_THE_FLY_TOKENIZE"; then
  if train_supports_arg "--on-the-fly-tokenize"; then
    TRAIN_ARGS+=(--on-the-fly-tokenize)
  else
    echo "Current train.py does not support --on-the-fly-tokenize; skipping it."
  fi
fi
if env_flag "${NO_AUTO_RESUME:-}"; then
  TRAIN_ARGS+=(--no-auto-resume)
fi
python -m blackbox_finetune_recall60.train "${TRAIN_ARGS[@]}"
if env_flag "${RUN_FINAL_EVAL:-}"; then
  python -m blackbox_finetune_recall60.evaluate --base-model "$BASE_MODEL" --adapter-dir "$OUTPUT_DIR/adapter" --data-dir "$VALIDATION_DATA_DIR" --threshold 0.50 --precision-top-k "$PRECISION_TOP_K" --precision-threshold "$PRECISION_THRESHOLD" --output-dir "$OUTPUT_DIR/evaluations/$FINAL_PRECISION_TAG" --cuda-device "$CUDA_DEVICE" --max-seq-length "$MAX_SEQ_LENGTH"
else
  echo "Skipping final evaluation by default. Set RUN_FINAL_EVAL=1 to evaluate after training."
fi
python -m unittest tests.test_blackbox_finetune_recall60 -v
