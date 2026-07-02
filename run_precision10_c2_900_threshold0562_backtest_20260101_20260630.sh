#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-/mnt/d/Models/precision10@0.4-c2-900}"
START_DATE="${2:-20260101}"
END_DATE="${3:-20260630}"
BACKTEST_NAME="${BACKTEST_NAME:-precision10_c2_900_threshold0562_20260101_20260630_sl6_limit100000}"

if [[ ! "$START_DATE" =~ ^[0-9]{8}$ || ! "$END_DATE" =~ ^[0-9]{8}$ ]]; then
  echo "StartDate and EndDate must use yyyyMMdd format." >&2
  exit 2
fi

if [[ "$MODEL_DIR" =~ ^[A-Za-z]:[\\/] ]] && command -v wslpath >/dev/null 2>&1; then
  MODEL_PATH="$(wslpath -u "$MODEL_DIR" 2>/dev/null || printf '%s\n' "$MODEL_DIR")"
else
  MODEL_PATH="$MODEL_DIR"
fi
if [[ -d "$MODEL_PATH/adapter" && -f "$MODEL_PATH/adapter/adapter_config.json" ]]; then
  MODEL_PATH="$MODEL_PATH/adapter"
fi
if [[ ! -f "$MODEL_PATH/adapter_config.json" ]]; then
  echo "Cannot find adapter_config.json in '$MODEL_PATH' or '$MODEL_PATH/adapter'." >&2
  exit 1
fi

export ASTOCK_DISABLE_LOCAL_DEPS="${ASTOCK_DISABLE_LOCAL_DEPS:-1}"
export SAMPLE_MODE="${SAMPLE_MODE:-xlong}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CUDA_DEVICE}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-blackbox-finetune-recall60}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing WSL blackbox Python: $VENV_DIR/bin/python" >&2
  exit 1
fi

cat <<EOF
Running threshold blackbox portfolio backtest in WSL/Linux
  model=$MODEL_PATH
  start_date=$START_DATE
  end_date=$END_DATE
  backtest_name=$BACKTEST_NAME
  threshold=0.562
  trade_rule=stop_loss_6pct_take_profit_10_20_hold_3d
  buy_budget=100000
  sample_mode=$SAMPLE_MODE
  max_seq_length=$MAX_SEQ_LENGTH
EOF

"$VENV_DIR/bin/python" -m portfolio_backtest.run \
  --strategy-name blackbox_finetune_recall60 \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --blackbox-sample-mode "$SAMPLE_MODE" \
  --blackbox-threshold 0.562 \
  --blackbox-max-seq-length "$MAX_SEQ_LENGTH" \
  --blackbox-daily-window 21 \
  --blackbox-weekly-window 13 \
  --blackbox-monthly-window 8 \
  --blackbox-adapter-dir "$MODEL_PATH" \
  --blackbox-cuda-device "$CUDA_DEVICE" \
  --trade-rule stop_loss_6pct_take_profit_10_20_hold_3d \
  --buy-budget 100000 \
  --backtest-name "$BACKTEST_NAME"

echo "Portfolio backtest completed: $BACKTEST_NAME"
