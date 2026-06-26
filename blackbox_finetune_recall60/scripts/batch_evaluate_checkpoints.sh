#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Edit this tuple when you want to evaluate a different checkpoint set.
CHECKPOINT_SUBDIRS=(
  "update-000500"
  "update-000900"
  "update-001000"
)

to_wsl_path() {
  local value="$1"
  if command -v wslpath >/dev/null 2>&1 && [[ "$value" =~ ^[A-Za-z]:\\ ]]; then
    wslpath -u "$value"
  else
    printf '%s\n' "$value"
  fi
}

CHECKPOINT_ROOT="${1:-/mnt/d/Models/precision10@0.4-3200/negative_reshuffle/cycle-01/continue_from_precision10_0_4_3200/checkpoints}"
EVAL_DATA_DIR="${2:-blackbox_finetune_recall60/data_evaluation_no_partial_week_recall60_xlong_neg9}"
OUTPUT_ROOT="${3:-}"

CHECKPOINT_ROOT="$(to_wsl_path "$CHECKPOINT_ROOT")"
EVAL_DATA_DIR="$(to_wsl_path "$EVAL_DATA_DIR")"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
THRESHOLD="${EVAL_THRESHOLD:-0.48}"
PRECISION_TOP_K="${PRECISION_TOP_K:-500}"
PRECISION_THRESHOLD="${PRECISION_THRESHOLD:-0.0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  elif [[ -x "/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python" ]]; then
    PYTHON_BIN="/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$CHECKPOINT_ROOT/batch_evaluations_$(date +%Y%m%d_%H%M%S)"
else
  OUTPUT_ROOT="$(to_wsl_path "$OUTPUT_ROOT")"
fi
mkdir -p "$OUTPUT_ROOT"
SUMMARY_JSONL="$OUTPUT_ROOT/summary.jsonl"
SUMMARY_JSON="$OUTPUT_ROOT/summary.json"
: > "$SUMMARY_JSONL"

echo "Batch evaluating recall60 checkpoints"
echo "  checkpoint_root=$CHECKPOINT_ROOT"
echo "  eval_data_dir=$EVAL_DATA_DIR"
echo "  output_root=$OUTPUT_ROOT"
echo "  python=$PYTHON_BIN"
echo "  threshold=$THRESHOLD"
echo "  reported_precision=precision@5,@10,@20,@50,@100,@200,@500"

for checkpoint_subdir in "${CHECKPOINT_SUBDIRS[@]}"; do
  adapter_dir="$CHECKPOINT_ROOT/$checkpoint_subdir"
  if [[ ! -d "$adapter_dir" ]]; then
    echo "WARNING missing checkpoint directory; skipping: $adapter_dir" >&2
    "$PYTHON_BIN" - "$checkpoint_subdir" "$adapter_dir" "$SUMMARY_JSONL" <<'PY'
import json
import sys

checkpoint, adapter_dir, summary_path = sys.argv[1:4]
record = {
    "checkpoint": checkpoint,
    "adapter_dir": adapter_dir,
    "status": "missing_checkpoint",
}
with open(summary_path, "a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
PY
    continue
  fi

  output_dir="$OUTPUT_ROOT/$checkpoint_subdir"
  mkdir -p "$output_dir"
  echo
  echo "==== evaluate $checkpoint_subdir ===="

  set +e
  "$PYTHON_BIN" -m blackbox_finetune_recall60.evaluate \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$adapter_dir" \
    --data-dir "$EVAL_DATA_DIR" \
    --threshold "$THRESHOLD" \
    --precision-top-k "$PRECISION_TOP_K" \
    --precision-threshold "$PRECISION_THRESHOLD" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --cuda-device "$CUDA_DEVICE" \
    --output-dir "$output_dir"
  eval_exit_code=$?
  set -e

  result_path="$output_dir/evaluation.json"
  "$PYTHON_BIN" - "$checkpoint_subdir" "$adapter_dir" "$result_path" "$SUMMARY_JSONL" "$eval_exit_code" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, adapter_dir, result_path, summary_path, eval_exit_code = sys.argv[1:6]
result_file = Path(result_path)
if result_file.exists():
    with result_file.open("r", encoding="utf-8") as file:
        result = json.load(file)
    status = "ok" if int(eval_exit_code) == 0 else "evaluate_failed_after_result"
else:
    result = {}
    status = "evaluate_failed_no_result"
record = {
    "checkpoint": checkpoint,
    "adapter_dir": adapter_dir,
    "result_path": result_path,
    "status": status,
    "eval_exit_code": int(eval_exit_code),
    "samples": result.get("samples"),
    "positive_samples": result.get("positive_samples"),
    "tp": result.get("tp"),
    "fp": result.get("fp"),
    "tn": result.get("tn"),
    "fn": result.get("fn"),
    "precision": result.get("precision"),
    "positive_recall": result.get("positive_recall"),
}
for k in (5, 10, 20, 50, 100, 200, 500):
    record[f"precision@{k}"] = result.get(f"precision@{k}")
with open(summary_path, "a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
PY
done

"$PYTHON_BIN" - "$SUMMARY_JSONL" "$SUMMARY_JSON" <<'PY'
import json
import sys

summary_jsonl, summary_json = sys.argv[1:3]
records = []
with open(summary_jsonl, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            records.append(json.loads(line))
with open(summary_json, "w", encoding="utf-8") as file:
    json.dump(records, file, ensure_ascii=False, indent=2)
PY

echo
echo "Batch evaluation summary: $SUMMARY_JSONL"
echo "Batch evaluation summary JSON: $SUMMARY_JSON"
