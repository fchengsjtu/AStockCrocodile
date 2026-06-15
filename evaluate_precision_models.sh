#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_PATHS=(
  "D:\\Models\\precision10@0.4-c1-4800"
  "D:\\Models\\precision10@0.4-c1-1000"
  "D:\\Models\\precision10@0.4-c1-1600"
)

DATASET_PATH="${1:-D:\\Models\\precision10@0.4-3200\\negative_reshuffle\\cycle-01\\checkpoint_full_evaluations_20260101_20260531\\dataset\\test.jsonl}"
RESULT_DIR="${2:-D:\\Models\\precision10@0.4-3200\\negative_reshuffle\\cycle-01\\checkpoint_full_evaluations_20260101_20260531\\batch_precision_evaluations}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-blackbox-finetune-recall60}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
THRESHOLD="${THRESHOLD:-0.50}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PRECISION_TOP_K="${PRECISION_TOP_K:-100}"
PRECISION_THRESHOLD="${PRECISION_THRESHOLD:-0}"

to_unix_path() {
  local value="$1"
  if [[ "$value" == [A-Za-z]:\\* ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -u "$value"
  else
    printf '%s\n' "$value"
  fi
}

model_slug() {
  "$PYTHON_BIN" - "$1" <<'PY'
from pathlib import Path
import re
import sys

name = Path(sys.argv[1]).name
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model")
PY
}

DATASET_UNIX="$(to_unix_path "$DATASET_PATH")"
RESULT_DIR_UNIX="$(to_unix_path "$RESULT_DIR")"

if [[ -f "$DATASET_UNIX" ]]; then
  DATA_DIR="$(dirname "$DATASET_UNIX")"
elif [[ -d "$DATASET_UNIX" && -f "$DATASET_UNIX/test.jsonl" ]]; then
  DATA_DIR="$DATASET_UNIX"
else
  echo "Evaluation dataset not found. Expected a test.jsonl file or a directory containing test.jsonl: $DATASET_PATH" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Python venv not found: $VENV_DIR" >&2
  echo "Create it with blackbox_finetune_recall60/scripts/one_click_deploy.sh diagnose/full first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
mkdir -p "$RESULT_DIR_UNIX"

SUMMARY_JSONL="$RESULT_DIR_UNIX/precision_summary.jsonl"
SUMMARY_CSV="$RESULT_DIR_UNIX/precision_summary.csv"
: > "$SUMMARY_JSONL"
printf 'model,adapter_dir,samples,positive_samples,positive_recall,precision,precision_at_5,precision_at_10,precision_at_20,precision_at_50,precision_at_100,threshold,max_seq_length,output_path\n' > "$SUMMARY_CSV"

echo "Batch precision evaluation"
echo "  data_dir=$DATA_DIR"
echo "  result_dir=$RESULT_DIR_UNIX"
echo "  base_model=$BASE_MODEL"
echo "  max_seq_length=$MAX_SEQ_LENGTH"
echo "  threshold=$THRESHOLD"
echo "  precision_top_k=$PRECISION_TOP_K"
echo "  cuda_device=$CUDA_DEVICE"

for model_path in "${MODEL_PATHS[@]}"; do
  ADAPTER_DIR="$(to_unix_path "$model_path")"
  SLUG="$(model_slug "$ADAPTER_DIR")"
  MODEL_OUTPUT_DIR="$RESULT_DIR_UNIX/$SLUG"
  mkdir -p "$MODEL_OUTPUT_DIR"
  echo "==== Evaluating $ADAPTER_DIR ===="
  "$PYTHON_BIN" -m blackbox_finetune_recall60.evaluate \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$ADAPTER_DIR" \
    --data-dir "$DATA_DIR" \
    --threshold "$THRESHOLD" \
    --precision-top-k "$PRECISION_TOP_K" \
    --precision-threshold "$PRECISION_THRESHOLD" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --output-dir "$MODEL_OUTPUT_DIR" \
    --cuda-device "$CUDA_DEVICE"

  "$PYTHON_BIN" - "$MODEL_OUTPUT_DIR/evaluation.json" "$ADAPTER_DIR" "$SUMMARY_JSONL" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

evaluation_path = Path(sys.argv[1])
adapter_dir = sys.argv[2]
summary_jsonl = Path(sys.argv[3])
summary_csv = Path(sys.argv[4])
data = json.loads(evaluation_path.read_text(encoding="utf-8"))
row = {
    "model": Path(adapter_dir).name,
    "adapter_dir": adapter_dir,
    "samples": data.get("samples", 0),
    "positive_samples": data.get("positive_samples", 0),
    "positive_recall": data.get("positive_recall", 0.0),
    "precision": data.get("precision", 0.0),
    "precision@5": data.get("precision@5", 0.0),
    "precision@10": data.get("precision@10", 0.0),
    "precision@20": data.get("precision@20", 0.0),
    "precision@50": data.get("precision@50", 0.0),
    "precision@100": data.get("precision@100", 0.0),
    "threshold": data.get("threshold", 0.0),
    "max_seq_length": data.get("max_seq_length", 0),
    "output_path": str(evaluation_path),
}
with summary_jsonl.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
with summary_csv.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "model",
            "adapter_dir",
            "samples",
            "positive_samples",
            "positive_recall",
            "precision",
            "precision@5",
            "precision@10",
            "precision@20",
            "precision@50",
            "precision@100",
            "threshold",
            "max_seq_length",
            "output_path",
        ],
    )
    writer.writerow(row)
print(
    f"{row['model']}: "
    f"precision@5={row['precision@5']:.4f} "
    f"precision@10={row['precision@10']:.4f} "
    f"precision@20={row['precision@20']:.4f} "
    f"precision@50={row['precision@50']:.4f} "
    f"precision@100={row['precision@100']:.4f}"
)
PY
done

echo "Summary JSONL: $SUMMARY_JSONL"
echo "Summary CSV: $SUMMARY_CSV"
