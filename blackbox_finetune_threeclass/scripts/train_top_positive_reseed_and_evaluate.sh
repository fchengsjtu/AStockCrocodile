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

latest_checkpoint() {
  local checkpoint_root="$1"
  python - "$checkpoint_root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
if root.is_dir():
    for path in root.iterdir():
        match = re.fullmatch(r"update-(\d+)", path.name)
        if match and (path / "adapter_config.json").is_file():
            candidates.append((int(match.group(1)), path))
if not candidates:
    raise SystemExit(f"no checkpoint adapters found under {root}")
print(max(candidates)[1])
PY
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DATA_DIR="$(to_wsl_path "${SOURCE_DATA_DIR:-$DATA_DIR}")"
CHECKPOINT_ROOT="$(to_wsl_path "${CHECKPOINT_ROOT:-$OUTPUT_DIR/checkpoints}")"
SOURCE_CHECKPOINT_DIR="$(to_wsl_path "${SOURCE_CHECKPOINT_DIR:-}")"
if [[ -z "$SOURCE_CHECKPOINT_DIR" ]]; then
  SOURCE_CHECKPOINT_DIR="$(latest_checkpoint "$CHECKPOINT_ROOT")"
fi
COMMON_EVAL_DATA_DIR="$(to_wsl_path "${COMMON_EVAL_DATA_DIR:-D:\\Documents\\StockInfoCrawler\\blackbox_finetune_threeclass\\data_evaluation_xlong_p1_n4_u11}")"
TOP_POSITIVE_LIMIT="${TOP_POSITIVE_LIMIT:-2000}"
TOP_POSITIVE_SEED="${TOP_POSITIVE_SEED:-937498347}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
CONTINUE_EPOCHS="${CONTINUE_EPOCHS:-0.5}"
CONTINUE_CHECKPOINT_EVERY="${CONTINUE_CHECKPOINT_EVERY:-100000000}"
TOP_POSITIVE_RUN_NAME="${TOP_POSITIVE_RUN_NAME:-top${TOP_POSITIVE_LIMIT}_positive_reseeded}"
TOP_POSITIVE_ROOT="$(to_wsl_path "${TOP_POSITIVE_ROOT:-$OUTPUT_DIR/top_positive_reseed}")"
RESEEDED_DATA_DIR="$TOP_POSITIVE_ROOT/$TOP_POSITIVE_RUN_NAME/dataset"
CONTINUED_OUTPUT_DIR="$TOP_POSITIVE_ROOT/$TOP_POSITIVE_RUN_NAME/training"
EVAL_OUTPUT="$CONTINUED_OUTPUT_DIR/evaluation.json"
RESULTS_JSONL="$(to_wsl_path "${RESULTS_JSONL:-$TOP_POSITIVE_ROOT/evaluation_results.jsonl}")"

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

if [[ ! -s "$SOURCE_DATA_DIR/train.jsonl" ]]; then
  echo "Missing source training dataset: $SOURCE_DATA_DIR/train.jsonl" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_CHECKPOINT_DIR/adapter_config.json" ]]; then
  echo "Missing source checkpoint adapter: $SOURCE_CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -s "$COMMON_EVAL_DATA_DIR/test.jsonl" ]]; then
  echo "Missing common evaluation dataset: $COMMON_EVAL_DATA_DIR/test.jsonl" >&2
  exit 1
fi

mkdir -p "$RESEEDED_DATA_DIR" "$CONTINUED_OUTPUT_DIR" "$(dirname "$RESULTS_JSONL")"

echo "==== Top-positive reseed continuation ===="
echo "  SOURCE_DATA_DIR=$SOURCE_DATA_DIR"
echo "  SOURCE_CHECKPOINT_DIR=$SOURCE_CHECKPOINT_DIR"
echo "  TOP_POSITIVE_LIMIT=$TOP_POSITIVE_LIMIT"
echo "  SCORE_BATCH_SIZE=$SCORE_BATCH_SIZE"
echo "  RESEEDED_DATA_DIR=$RESEEDED_DATA_DIR"
echo "  CONTINUE_EPOCHS=$CONTINUE_EPOCHS"
echo "  COMMON_EVAL_DATA_DIR=$COMMON_EVAL_DATA_DIR"
echo "  RESULTS_JSONL=$RESULTS_JSONL"
echo "==========================================="

python - "$SOURCE_DATA_DIR" "$SOURCE_CHECKPOINT_DIR" "$RESEEDED_DATA_DIR" "$BASE_MODEL" "$MAX_SEQ_LENGTH" "$TOP_POSITIVE_LIMIT" "$TOP_POSITIVE_SEED" "$SCORE_BATCH_SIZE" <<'PY'
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from blackbox_finetune_threeclass.common import CLASS_NEGATIVE, CLASS_NEUTRAL, CLASS_POSITIVE, read_jsonl, write_jsonl
from blackbox_finetune_threeclass.inference import load_model
from blackbox_finetune_threeclass.train import _score_positive_answer_rows

source_data_dir = Path(sys.argv[1])
source_checkpoint_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
base_model = sys.argv[4]
max_seq_length = int(sys.argv[5])
top_limit = max(1, int(sys.argv[6]))
seed = int(sys.argv[7])
score_batch_size = max(1, int(sys.argv[8]))
rng = random.Random(seed)

rows = read_jsonl(source_data_dir / "train.jsonl")
by_date = defaultdict(lambda: {CLASS_POSITIVE: [], CLASS_NEGATIVE: [], CLASS_NEUTRAL: []})
positive_indices = []
for index, row in enumerate(rows):
    metadata = row.get("metadata") or {}
    label = int(metadata["label"])
    anchor_date = metadata.get("anchor_date")
    if anchor_date is None:
        continue
    by_date[anchor_date][label].append(index)
    if label == CLASS_POSITIVE:
        positive_indices.append(index)

print(f"loaded rows={len(rows)} positives={len(positive_indices)}", flush=True)
model, tokenizer = load_model(base_model, source_checkpoint_dir)
model.eval()
started = time.monotonic()
positive_scores = {}
for offset in range(0, len(positive_indices), score_batch_size):
    batch_indices = positive_indices[offset : offset + score_batch_size]
    positive_scores.update(_score_positive_answer_rows(model, tokenizer, rows, batch_indices, max_seq_length))
    done = min(offset + len(batch_indices), len(positive_indices))
    if done % max(score_batch_size * 100, score_batch_size) == 0 or done == len(positive_indices):
        elapsed = time.monotonic() - started
        print(f"positive scoring progress {done}/{len(positive_indices)} elapsed={elapsed:.1f}s", flush=True)

ranked_positive_indices = sorted(positive_indices, key=lambda index: positive_scores.get(index, float("-inf")), reverse=True)
selected_rows = []
selected_scores = []
skipped_no_negative = 0
skipped_no_neutral = 0
used_negative = set()
used_neutral = set()
date_counts = Counter()

for positive_index in ranked_positive_indices:
    if len(selected_scores) >= top_limit:
        break
    anchor_date = rows[positive_index]["metadata"]["anchor_date"]
    negative_pool = [index for index in by_date[anchor_date][CLASS_NEGATIVE] if index not in used_negative]
    neutral_pool = [index for index in by_date[anchor_date][CLASS_NEUTRAL] if index not in used_neutral]
    if len(negative_pool) < 4:
        skipped_no_negative += 1
        continue
    if len(neutral_pool) < 11:
        skipped_no_neutral += 1
        continue
    picked_negative = rng.sample(negative_pool, 4)
    picked_neutral = rng.sample(neutral_pool, 11)
    used_negative.update(picked_negative)
    used_neutral.update(picked_neutral)
    group_indices = [positive_index] + picked_negative + picked_neutral
    group_rows = [dict(rows[index]) for index in group_indices]
    for row in group_rows:
        row.pop("update_positive_weight", None)
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            row["metadata"] = dict(metadata)
            row["metadata"].pop("update_positive_weight", None)
    rng.shuffle(group_rows)
    selected_rows.extend(group_rows)
    selected_scores.append(
        {
            "index": positive_index,
            "score": positive_scores.get(positive_index),
            "scode": rows[positive_index].get("metadata", {}).get("scode"),
            "anchor_date": anchor_date,
            "rank": len(selected_scores) + 1,
        }
    )
    date_counts[anchor_date] += 1

output_dir.mkdir(parents=True, exist_ok=True)
write_jsonl(output_dir / "train.jsonl", selected_rows)
write_jsonl(output_dir / "test.jsonl", [])
write_jsonl(output_dir / "selected_positive_scores.jsonl", selected_scores)
stats = {
    "source_data_dir": str(source_data_dir),
    "source_checkpoint_dir": str(source_checkpoint_dir),
    "top_positive_limit": top_limit,
    "selected_positive_count": len(selected_scores),
    "train_rows": len(selected_rows),
    "class_counts": dict(Counter(int((row.get("metadata") or {})["label"]) for row in selected_rows)),
    "skipped_no_negative": skipped_no_negative,
    "skipped_no_neutral": skipped_no_neutral,
    "unique_dates": len(date_counts),
    "top_dates": date_counts.most_common(20),
    "score_batch_size": score_batch_size,
    "seed": seed,
    "elapsed_seconds": round(time.monotonic() - started, 3),
}
(output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
if not selected_rows:
    raise SystemExit("no reseeded training rows were produced")
PY

python -m blackbox_finetune_threeclass.train \
  --base-model "$BASE_MODEL" \
  --data-dir "$RESEEDED_DATA_DIR" \
  --checkpoint-eval-data-dir "$COMMON_EVAL_DATA_DIR" \
  --output-dir "$CONTINUED_OUTPUT_DIR" \
  --initial-binary-adapter-dir "$SOURCE_CHECKPOINT_DIR" \
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

python -m blackbox_finetune_threeclass.evaluate \
  --base-model "$BASE_MODEL" \
  --adapter-dir "$CONTINUED_OUTPUT_DIR/adapter" \
  --data-dir "$COMMON_EVAL_DATA_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --output "$EVAL_OUTPUT" \
  --cuda-device "$CUDA_DEVICE"

python - "$RESULTS_JSONL" "$EVAL_OUTPUT" "$RESEEDED_DATA_DIR" "$CONTINUED_OUTPUT_DIR" "$SOURCE_CHECKPOINT_DIR" "$TOP_POSITIVE_LIMIT" "$CONTINUE_EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
eval_path = Path(sys.argv[2])
dataset_dir = Path(sys.argv[3])
record = {
    "strategy": "top_positive_reseed_same_day",
    "top_positive_limit": int(sys.argv[6]),
    "continue_epochs": float(sys.argv[7]),
    "source_checkpoint": sys.argv[5],
    "dataset_dir": str(dataset_dir),
    "continued_output_dir": sys.argv[4],
    "dataset_stats": json.loads((dataset_dir / "stats.json").read_text(encoding="utf-8")),
    "evaluation_path": str(eval_path),
    "evaluation": json.loads(eval_path.read_text(encoding="utf-8")),
}
with results_path.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY

echo "Top-positive reseed evaluation appended to: $RESULTS_JSONL"
