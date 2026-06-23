#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

source "$SCRIPT_DIR/set_wsl_env.sh"

to_wsl_path() {
  local value="$1"
  if [[ "$value" == [A-Za-z]:\\* ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -u "$value"
  else
    printf '%s\n' "$value"
  fi
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
SELECTED_GROUPS_ROOT="$(to_wsl_path "${SELECTED_GROUPS_ROOT:-$OUTPUT_DIR/selected_groups}")"
SOURCE_CHECKPOINT_ROOT="$(to_wsl_path "${SOURCE_CHECKPOINT_ROOT:-$OUTPUT_DIR/checkpoints}")"
CONTINUED_OUTPUT_ROOT="$(to_wsl_path "${CONTINUED_OUTPUT_ROOT:-$OUTPUT_DIR/selected_group_epoch_runs}")"
COMMON_EVAL_DATA_DIR="$(to_wsl_path "${COMMON_EVAL_DATA_DIR:-D:\\Documents\\StockInfoCrawler\\blackbox_finetune_threeclass\\data_evaluation_xlong_p1_n4_u11}")"
RESULTS_JSONL="$(to_wsl_path "${RESULTS_JSONL:-${RESULTS_JSON:-$CONTINUED_OUTPUT_ROOT/evaluation_results.jsonl}}")"
CONTINUE_EPOCHS="${CONTINUE_EPOCHS:-1.0}"
CONTINUE_CHECKPOINT_EVERY="${CONTINUE_CHECKPOINT_EVERY:-100000000}"
export RUN_FINAL_EVAL=0
export SELECTED_GROUPS_ENABLED=0
export POSITIVE_PURIFICATION_ENABLED="${CONTINUE_POSITIVE_PURIFICATION_ENABLED:-0}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -r blackbox_finetune_threeclass/requirements.txt
python -m blackbox_finetune_threeclass.gpu --cuda-device "$CUDA_DEVICE"

if [[ ! -d "$SELECTED_GROUPS_ROOT" ]]; then
  echo "Missing selected groups root: $SELECTED_GROUPS_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_CHECKPOINT_ROOT" ]]; then
  echo "Missing source checkpoint root: $SOURCE_CHECKPOINT_ROOT" >&2
  exit 1
fi
if [[ ! -s "$COMMON_EVAL_DATA_DIR/test.jsonl" ]]; then
  echo "Missing common evaluation dataset: $COMMON_EVAL_DATA_DIR/test.jsonl" >&2
  exit 1
fi

mkdir -p "$CONTINUED_OUTPUT_ROOT"
: > "$RESULTS_JSONL"

echo "==== Selected group one-epoch continuation ===="
echo "  SELECTED_GROUPS_ROOT=$SELECTED_GROUPS_ROOT"
echo "  SOURCE_CHECKPOINT_ROOT=$SOURCE_CHECKPOINT_ROOT"
echo "  CONTINUED_OUTPUT_ROOT=$CONTINUED_OUTPUT_ROOT"
echo "  COMMON_EVAL_DATA_DIR=$COMMON_EVAL_DATA_DIR"
echo "  RESULTS_JSONL=$RESULTS_JSONL"
echo "  CONTINUE_EPOCHS=$CONTINUE_EPOCHS"
echo "================================================"

shopt -s nullglob
processed=0
for update_dir in "$SELECTED_GROUPS_ROOT"/update-*; do
  [[ -d "$update_dir" ]] || continue
  update_name="$(basename "$update_dir")"
  source_checkpoint="$SOURCE_CHECKPOINT_ROOT/$update_name"
  if [[ ! -f "$source_checkpoint/adapter_config.json" ]]; then
    echo "Skipping $update_name: missing checkpoint adapter at $source_checkpoint" >&2
    continue
  fi
  for group_name in top1_positive bottom1_positive; do
    group_data_dir="$update_dir/$group_name"
    train_path="$group_data_dir/train.jsonl"
    if [[ ! -s "$train_path" ]]; then
      echo "Skipping $update_name/$group_name: missing or empty $train_path" >&2
      continue
    fi
    run_output_dir="$CONTINUED_OUTPUT_ROOT/$update_name/$group_name"
    eval_output="$run_output_dir/evaluation.json"
    mkdir -p "$run_output_dir"

    echo "==== Continue $update_name/$group_name for $CONTINUE_EPOCHS epoch(s) ===="
    python -m blackbox_finetune_threeclass.train \
      --base-model "$BASE_MODEL" \
      --data-dir "$group_data_dir" \
      --checkpoint-eval-data-dir "$COMMON_EVAL_DATA_DIR" \
      --output-dir "$run_output_dir" \
      --initial-binary-adapter-dir "$source_checkpoint" \
      --max-seq-length "$MAX_SEQ_LENGTH" \
      --epochs "$CONTINUE_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning-rate "$LEARNING_RATE" \
      --weight-decay "$WEIGHT_DECAY" \
      --max-grad-norm "$MAX_GRAD_NORM" \
      --lora-rank "$LORA_RANK" \
      --lora-dropout "$LORA_DROPOUT" \
      --positive-ce-weight "$POSITIVE_CE_WEIGHT" \
      --negative-ce-weight "$NEGATIVE_CE_WEIGHT" \
      --neutral-ce-weight "$NEUTRAL_CE_WEIGHT" \
      --fp-loss-weight "$FP_LOSS_WEIGHT" \
      --neutral-fp-loss-weight "$NEUTRAL_FP_LOSS_WEIGHT" \
      --high-score-positive-bonus "$HIGH_SCORE_POSITIVE_BONUS" \
      --high-score-positive-bonus-max-multiplier "$HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER" \
      --high-score-negative-penalty-weight "$HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT" \
      --high-score-neutral-penalty-weight "$HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT" \
      --high-score-negative-margin "$HIGH_SCORE_NEGATIVE_MARGIN" \
      --high-score-neutral-margin "$HIGH_SCORE_NEUTRAL_MARGIN" \
      --positive-purification-bottom-k "$POSITIVE_PURIFICATION_BOTTOM_K" \
      --positive-purification-group-size "$POSITIVE_PURIFICATION_GROUP_SIZE" \
      --positive-purification-decay "$POSITIVE_PURIFICATION_DECAY" \
      --checkpoint-every "$CONTINUE_CHECKPOINT_EVERY" \
      --train-seed "$TRAIN_SEED" \
      --eval-max-samples "$EVAL_MAX_SAMPLES" \
      --eval-precision-top-k "$EVAL_PRECISION_TOP_K" \
      --eval-precision-threshold "$EVAL_PRECISION_THRESHOLD" \
      --cuda-device "$CUDA_DEVICE" \
      --no-auto-resume \
      --no-positive-purification-enabled \
      --on-the-fly-tokenize

    echo "==== Evaluate $update_name/$group_name ===="
    python -m blackbox_finetune_threeclass.evaluate \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$run_output_dir/adapter" \
      --data-dir "$COMMON_EVAL_DATA_DIR" \
      --max-seq-length "$MAX_SEQ_LENGTH" \
      --output "$eval_output" \
      --cuda-device "$CUDA_DEVICE"

    python - "$RESULTS_JSONL" "$eval_output" "$update_name" "$group_name" "$source_checkpoint" "$group_data_dir" "$run_output_dir" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
eval_path = Path(sys.argv[2])
record = {
    "update": sys.argv[3],
    "group": sys.argv[4],
    "source_checkpoint": sys.argv[5],
    "selected_group_data_dir": sys.argv[6],
    "continued_output_dir": sys.argv[7],
    "evaluation_path": str(eval_path),
    "evaluation": json.loads(eval_path.read_text(encoding="utf-8")),
}
with results_path.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
    processed=$((processed + 1))
  done
done

if [[ "$processed" -eq 0 ]]; then
  echo "No selected group datasets were processed." >&2
  exit 1
fi

echo "Processed selected group runs: $processed"
echo "Per-dataset evaluation results: $RESULTS_JSONL"
