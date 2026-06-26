#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

to_wsl_path() {
  local value="$1"
  if command -v wslpath >/dev/null 2>&1 && [[ "$value" =~ ^[A-Za-z]:\\ ]]; then
    wslpath -u "$value"
  else
    printf '%s\n' "$value"
  fi
}

MODEL_DIR="$(to_wsl_path "${1:-/mnt/d/Models/precision10@0.4-c2-900}")"
TRAIN_JSONL="$(to_wsl_path "${2:-D:\\Models\\precision10@0.4-3200\\negative_reshuffle\\cycle-01\\datasets\\training\\train.jsonl}")"
OUTPUT_NAME="${3:-reshuffle_from_precision10_0_4_3200_cycle01_keep30}"
EVAL_DATA_DIR="${4:-}"

KEEP_RATIO="${KEEP_RATIO:-0.30}"
SAMPLE_MODE="${SAMPLE_MODE:-xlong}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATABASE_BATCH_SIZE="${DATABASE_BATCH_SIZE:-80}"
DATABASE_MAX_ATTEMPTS="${DATABASE_MAX_ATTEMPTS:-20}"
RESHUFFLE_SEED="${RESHUFFLE_SEED:-937498347}"
STAT_TYPE="${STAT_TYPE:-short_term_surge_3d_20pct}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  elif [[ -x "/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python" ]]; then
    PYTHON_BIN="/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory does not exist: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Training JSONL does not exist: $TRAIN_JSONL" >&2
  exit 2
fi

TRAIN_DATA_DIR="$(cd "$(dirname "$TRAIN_JSONL")" && pwd)"
if [[ ! -f "$TRAIN_DATA_DIR/test.jsonl" || ! -f "$TRAIN_DATA_DIR/all.jsonl" ]]; then
  echo "Training dataset directory must contain train.jsonl, test.jsonl and all.jsonl: $TRAIN_DATA_DIR" >&2
  exit 2
fi

if [[ -z "$EVAL_DATA_DIR" ]]; then
  sibling_eval_dir="$(cd "$TRAIN_DATA_DIR/.." && pwd)/evaluation"
  if [[ -d "$sibling_eval_dir" ]]; then
    EVAL_DATA_DIR="$sibling_eval_dir"
  else
    EVAL_DATA_DIR="$TRAIN_DATA_DIR"
  fi
else
  EVAL_DATA_DIR="$(to_wsl_path "$EVAL_DATA_DIR")"
fi

if [[ ! -d "$EVAL_DATA_DIR" ]]; then
  echo "Evaluation dataset directory does not exist: $EVAL_DATA_DIR" >&2
  exit 2
fi

RUN_DIR="$MODEL_DIR/negative_reshuffle/$OUTPUT_NAME"
METADATA_DIR="$RUN_DIR/source_metadata"
METADATA_JSON="$METADATA_DIR/eval-source-datasets.json"
mkdir -p "$METADATA_DIR"

"$PYTHON_BIN" - "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR" "$METADATA_JSON" <<'PY'
import json
import sys
from pathlib import Path

training_dir, evaluation_dir, metadata_json = sys.argv[1:4]
record = {
    "original_train_dataset_path": training_dir,
    "original_eval_dataset_path": evaluation_dir,
}
path = Path(metadata_json)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"metadata written: {path}")
PY

echo "Negative reshuffle from explicit train.jsonl"
echo "  model_dir=$MODEL_DIR"
echo "  train_jsonl=$TRAIN_JSONL"
echo "  train_data_dir=$TRAIN_DATA_DIR"
echo "  eval_data_dir=$EVAL_DATA_DIR"
echo "  output_name=$OUTPUT_NAME"
echo "  keep_ratio=$KEEP_RATIO"
echo "  sample_mode=$SAMPLE_MODE"
echo "  max_seq_length=$MAX_SEQ_LENGTH"
echo "  cuda_device=$CUDA_DEVICE"
echo "  seed=$RESHUFFLE_SEED"
echo "  metadata_json=$METADATA_JSON"
echo
echo "Current train/test negative rows are excluded from database replacements."

"$PYTHON_BIN" -m blackbox_negative_reshuffle.run \
  --model-dir "$MODEL_DIR" \
  --evaluation-json "$METADATA_JSON" \
  --output-name "$OUTPUT_NAME" \
  --keep-ratio "$KEEP_RATIO" \
  --seed "$RESHUFFLE_SEED" \
  --stat-type "$STAT_TYPE" \
  --sample-mode "$SAMPLE_MODE" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --database-batch-size "$DATABASE_BATCH_SIZE" \
  --database-max-attempts "$DATABASE_MAX_ATTEMPTS" \
  --progress-every "$PROGRESS_EVERY" \
  --cuda-device "$CUDA_DEVICE"

echo
echo "Generated dataset:"
echo "  $RUN_DIR/datasets/training/train.jsonl"
echo "  $RUN_DIR/datasets/training/test.jsonl"
echo "  $RUN_DIR/datasets/training/all.jsonl"
echo "Manifest:"
echo "  $RUN_DIR/reshuffle_manifest.json"
