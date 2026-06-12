#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/Documents/StockInfoCrawler}"
DEFAULT_CYCLE_DIR="/mnt/d/Models/precision10@0.4-3200/negative_reshuffle/cycle-01"
CYCLE_DIR="${1:-$DEFAULT_CYCLE_DIR}"
START_DATE="${2:-20260101}"
END_DATE="${3:-20260531}"
RESULT_FILE="${4:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
THRESHOLD="${THRESHOLD:-0.48}"
PRECISION_TOP_K="${PRECISION_TOP_K:-10}"
PRECISION_THRESHOLD="${PRECISION_THRESHOLD:-0.40}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-0200 1000 1500 1600 2100 2900 3000 4200 4700 4800}"
DRY_RUN="${DRY_RUN:-0}"

to_wsl_path() {
  local value="$1"
  if [[ "$value" =~ ^[A-Za-z]:\\ ]]; then
    wslpath -u "$value"
  else
    printf '%s\n' "$value"
  fi
}

CYCLE_DIR="$(to_wsl_path "$CYCLE_DIR")"
if [[ -n "$RESULT_FILE" ]]; then
  RESULT_FILE="$(to_wsl_path "$RESULT_FILE")"
fi

SOURCE_DATA_DIR="$CYCLE_DIR/datasets/evaluation"
SOURCE_TEST_PATH="$SOURCE_DATA_DIR/test.jsonl"
CHECKPOINT_DIR="$CYCLE_DIR/training/checkpoints"
RESULT_ROOT="$CYCLE_DIR/checkpoint_full_evaluations_${START_DATE}_${END_DATE}"
FILTERED_DATA_DIR="$RESULT_ROOT/dataset"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULT_ROOT/runs/$RUN_ID"
RESULT_FILE="${RESULT_FILE:-$RESULT_ROOT/results.jsonl}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_TEST_PATH" ]]; then
  echo "Evaluation dataset not found: $SOURCE_TEST_PATH" >&2
  exit 2
fi

mkdir -p "$FILTERED_DATA_DIR" "$RUN_DIR" "$(dirname "$RESULT_FILE")"

FILTER_SUMMARY="$("$PYTHON_BIN" - "$SOURCE_TEST_PATH" "$FILTERED_DATA_DIR/test.jsonl" "$START_DATE" "$END_DATE" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
start_date = sys.argv[3].replace("-", "")
end_date = sys.argv[4].replace("-", "")

count = 0
minimum = None
maximum = None
target_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
with source_path.open("r", encoding="utf-8") as source, temporary_path.open("w", encoding="utf-8", newline="\n") as target:
    for line in source:
        row = json.loads(line)
        anchor_date = str(row.get("metadata", {}).get("anchor_date", "")).replace("-", "")
        if not anchor_date or anchor_date < start_date or anchor_date > end_date:
            continue
        target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        count += 1
        minimum = anchor_date if minimum is None or anchor_date < minimum else minimum
        maximum = anchor_date if maximum is None or anchor_date > maximum else maximum
temporary_path.replace(target_path)

summary = {
    "samples": count,
    "minimum_anchor_date": minimum,
    "maximum_anchor_date": maximum,
}
(target_path.parent / "filter_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False))
PY
)"

SAMPLE_COUNT="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["samples"])' "$FILTER_SUMMARY")"
if [[ "$SAMPLE_COUNT" -le 0 ]]; then
  echo "No evaluation samples found for ${START_DATE}-${END_DATE}." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

cat <<EOF
Negative reshuffle checkpoint full evaluation
  cycle_dir=$CYCLE_DIR
  checkpoint_dir=$CHECKPOINT_DIR
  source_data_dir=$SOURCE_DATA_DIR
  filtered_data_dir=$FILTERED_DATA_DIR
  requested_dates=$START_DATE-$END_DATE
  filtered_summary=$FILTER_SUMMARY
  checkpoints=$CHECKPOINT_UPDATES
  threshold=$THRESHOLD
  precision_top_k=$PRECISION_TOP_K
  precision_threshold=$PRECISION_THRESHOLD
  max_seq_length=$MAX_SEQ_LENGTH
  result_file=$RESULT_FILE
  run_dir=$RUN_DIR
  dry_run=$DRY_RUN
EOF

append_result() {
  local checkpoint_path="$1"
  local update="$2"
  local evaluation_path="$3"
  local exit_code="$4"
  local error_message="${5:-}"
  "$PYTHON_BIN" - "$RESULT_FILE" "$checkpoint_path" "$update" "$evaluation_path" "$exit_code" "$START_DATE" "$END_DATE" "$RUN_ID" "$error_message" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

result_path = Path(sys.argv[1])
checkpoint_path = sys.argv[2]
update = int(sys.argv[3])
evaluation_path = Path(sys.argv[4]) if sys.argv[4] else None
exit_code = int(sys.argv[5])
start_date = sys.argv[6]
end_date = sys.argv[7]
run_id = sys.argv[8]
error_message = sys.argv[9]

record = {}
if evaluation_path is not None and evaluation_path.is_file():
    record = json.loads(evaluation_path.read_text(encoding="utf-8"))
record.update(
    {
        "checkpoint_update": update,
        "checkpoint_path": checkpoint_path,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "evaluation_exit_code": exit_code,
        "evaluation_run_id": run_id,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
)
if error_message:
    record["error"] = error_message
result_path.parent.mkdir(parents=True, exist_ok=True)
with result_path.open("a", encoding="utf-8", newline="\n") as output:
    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
}

completed=0
failed=0
for short_update in $CHECKPOINT_UPDATES; do
  update_number=$((10#$short_update))
  checkpoint_name="$(printf 'update-%06d' "$update_number")"
  adapter_dir="$CHECKPOINT_DIR/$checkpoint_name"
  detail_dir="$RUN_DIR/$checkpoint_name"
  evaluation_path="$detail_dir/evaluation.json"
  log_path="$detail_dir/evaluate.log"
  mkdir -p "$detail_dir"

  echo
  echo "==== Evaluating $checkpoint_name ===="
  if [[ ! -f "$adapter_dir/adapter_config.json" ]]; then
    message="Checkpoint adapter not found: $adapter_dir"
    echo "$message" >&2
    append_result "$adapter_dir" "$update_number" "" 2 "$message"
    failed=$((failed + 1))
    continue
  fi

  command=(
    "$PYTHON_BIN" -m blackbox_finetune_recall60.evaluate
    --base-model "$BASE_MODEL"
    --adapter-dir "$adapter_dir"
    --data-dir "$FILTERED_DATA_DIR"
    --threshold "$THRESHOLD"
    --precision-top-k "$PRECISION_TOP_K"
    --precision-threshold "$PRECISION_THRESHOLD"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --cuda-device "$CUDA_DEVICE"
    --output-dir "$detail_dir"
  )
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  set +e
  (
    cd "$PROJECT_DIR"
    "${command[@]}"
  ) 2>&1 | tee "$log_path"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ -f "$evaluation_path" ]]; then
    append_result "$adapter_dir" "$update_number" "$evaluation_path" "$exit_code"
    completed=$((completed + 1))
    echo "Appended $checkpoint_name result to $RESULT_FILE"
  else
    message="Evaluation produced no result JSON; see $log_path"
    append_result "$adapter_dir" "$update_number" "" "$exit_code" "$message"
    failed=$((failed + 1))
    echo "$message" >&2
  fi
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run completed. No models were evaluated and no result rows were appended."
else
  echo "Evaluation series completed: completed=$completed failed=$failed result_file=$RESULT_FILE"
fi
