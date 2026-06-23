#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

source "$SCRIPT_DIR/set_wsl_env.sh"

TARGET_UPDATES="${TARGET_UPDATES:-1000}"
export CHECKPOINT_EVERY="${SELECTED_GROUPS_CHECKPOINT_EVERY:-100}"
export SELECTED_GROUPS_ENABLED=1
export SELECTED_GROUPS_OUTPUT_DIR="${SELECTED_GROUPS_OUTPUT_DIR:-$OUTPUT_DIR/selected_groups}"
export NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"
export RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-0}"
export RESET_THREECLASS_ENV=0

bash "$SCRIPT_DIR/one_click_deploy.sh" dataset-only

if [[ ! -s "$DATA_DIR/train.jsonl" ]]; then
  echo "Missing non-empty training dataset: $DATA_DIR/train.jsonl" >&2
  exit 1
fi

TRAIN_ROWS="$(python3 - "$DATA_DIR/train.jsonl" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as file:
    print(sum(1 for line in file if line.strip()))
PY
)"

export EPOCHS="$(python3 - "$TRAIN_ROWS" "$TARGET_UPDATES" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" <<'PY'
import sys

rows = max(1, int(sys.argv[1]))
target_updates = max(1, int(sys.argv[2]))
batch_size = max(1, int(sys.argv[3]))
grad_accum = max(1, int(sys.argv[4]))
print(f"{(target_updates * batch_size * grad_accum) / rows:.12f}")
PY
)"

echo "==== Selected group training ===="
echo "  TARGET_UPDATES=$TARGET_UPDATES"
echo "  TRAIN_ROWS=$TRAIN_ROWS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "  EPOCHS=$EPOCHS"
echo "  CHECKPOINT_EVERY=$CHECKPOINT_EVERY"
echo "  SELECTED_GROUPS_OUTPUT_DIR=$SELECTED_GROUPS_OUTPUT_DIR"
echo "================================="

bash "$SCRIPT_DIR/one_click_deploy.sh" full
