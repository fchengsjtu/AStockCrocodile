#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python}"
UP_ADAPTER_DIR="${1:-${UP_ADAPTER_DIR:-/mnt/d/Models/precision10@0.4-c2-900}}"
DATA_PATH="${2:-${DATA_PATH:-/mnt/d/Documents/StockInfoCrawler/blackbox_finetune_recall60/data_evaluation_no_partial_week_recall60_xlong_neg9/test.jsonl}}"
OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-/mnt/d/Documents/StockInfoCrawler/blackbox_finetune_drop6/positive_top_reshuffle_c2/joint_weight_sweep_results}}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FINAL_TOP_N="${FINAL_TOP_N:-500}"
WEIGHT_START="${WEIGHT_START:-0.20}"
WEIGHT_END="${WEIGHT_END:-0.50}"
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"

if [[ "$DATA_PATH" == *.jsonl ]]; then
  DATA_DIR="$(dirname "$DATA_PATH")"
else
  DATA_DIR="$DATA_PATH"
fi

DROP_CHECKPOINT_ROOT="/mnt/d/Documents/StockInfoCrawler/blackbox_finetune_drop6/positive_top_reshuffle_c2/checkpoints"
DROP_CHECKPOINTS=(
  "$DROP_CHECKPOINT_ROOT/update-004300"
  "$DROP_CHECKPOINT_ROOT/update-004600"
  "$DROP_CHECKPOINT_ROOT/update-004900"
)

echo "Joint up/drop weight-sweep evaluation"
echo "  project=$PROJECT_DIR"
echo "  python=$PYTHON_BIN"
echo "  up_adapter=$UP_ADAPTER_DIR"
echo "  data_dir=$DATA_DIR"
echo "  output_root=$OUTPUT_ROOT"
echo "  final_top_n=$FINAL_TOP_N"
echo "  drop_weight=${WEIGHT_START}..${WEIGHT_END} step=${WEIGHT_STEP}"
echo

mkdir -p "$OUTPUT_ROOT"
SUMMARY_FILE="$OUTPUT_ROOT/all_checkpoint_weight_sweep_summary.jsonl"
: > "$SUMMARY_FILE"

for drop_adapter in "${DROP_CHECKPOINTS[@]}"; do
  checkpoint_name="$(basename "$drop_adapter")"
  checkpoint_output="$OUTPUT_ROOT/$checkpoint_name"
  echo "==== evaluating $checkpoint_name ===="
  "$PYTHON_BIN" blackbox_finetune_recall60/scripts/joint_evaluate_up_drop_weight_sweep.py \
    --up-adapter-dir "$UP_ADAPTER_DIR" \
    --drop-adapter-dir "$drop_adapter" \
    --data-dir "$DATA_DIR" \
    --output-dir "$checkpoint_output" \
    --weight-start "$WEIGHT_START" \
    --weight-end "$WEIGHT_END" \
    --weight-step "$WEIGHT_STEP" \
    --final-top-n "$FINAL_TOP_N" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --cuda-device "$CUDA_DEVICE"

  "$PYTHON_BIN" - "$checkpoint_name" "$checkpoint_output/weight_sweep_summary.jsonl" >> "$SUMMARY_FILE" <<'PY'
import json
import sys

checkpoint_name = sys.argv[1]
summary_path = sys.argv[2]
with open(summary_path, "r", encoding="utf-8") as file:
    for line in file:
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        row["drop_checkpoint"] = checkpoint_name
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
PY
done

echo
echo "All checkpoint summaries appended to: $SUMMARY_FILE"
