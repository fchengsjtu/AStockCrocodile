#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash blackbox_finetune_threeclass/scripts/predict_update000300_top5.sh YYYYMMDD [output.csv]" >&2
  exit 1
fi

PREDICT_DATE="$1"
source "$SCRIPT_DIR/set_wsl_env.sh"

ADAPTER_DIR="${ADAPTER_DIR:-/mnt/d/Documents/StockInfoCrawler/blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-p1_n4_u11-lora/selected_group_epoch_runs/update-000200/top1_positive/checkpoints/update-000300}"
OUTPUT_CSV="${2:-data/threeclass_update000300_top5_${PREDICT_DATE}.csv}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-80}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing virtualenv: $VENV_DIR" >&2
  echo "Run the three-class setup first, for example: bash blackbox_finetune_threeclass/scripts/one_click_deploy.sh diagnose" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "Missing adapter checkpoint: $ADAPTER_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "==== Three-class top5 prediction ===="
echo "  date=$PREDICT_DATE"
echo "  adapter_dir=$ADAPTER_DIR"
echo "  output=$OUTPUT_CSV"
echo "  sample_mode=$SAMPLE_MODE"
echo "  max_seq_length=$MAX_SEQ_LENGTH"
echo "  negative_weight=$NEGATIVE_WEIGHT"
echo "  neutral_weight=$NEUTRAL_WEIGHT"
echo "====================================="

python -m blackbox_finetune_threeclass.gpu --cuda-device "$CUDA_DEVICE"
python -m blackbox_finetune_threeclass.predict_day \
  --base-model "$BASE_MODEL" \
  --adapter-dir "$ADAPTER_DIR" \
  --date "$PREDICT_DATE" \
  --sample-mode "$SAMPLE_MODE" \
  --batch-size "$PREDICT_BATCH_SIZE" \
  --limit 5 \
  --negative-weight "$NEGATIVE_WEIGHT" \
  --neutral-weight "$NEUTRAL_WEIGHT" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --output "$OUTPUT_CSV" \
  --cuda-device "$CUDA_DEVICE"
