#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/Documents/StockInfoCrawler}"
DEFAULT_CYCLE_DIR="/mnt/d/Models/precision10@0.4-3200/negative_reshuffle/cycle-01"
CYCLE_DIR="${1:-$DEFAULT_CYCLE_DIR}"
START_DATE="${2:-20260101}"
END_DATE="${3:-20260529}"
RESULT_FILE="${4:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/python}"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-0200 1000 1500 1600 2100 2900 3000 4200 4700 4800}"
STRATEGY_NAME="${STRATEGY_NAME:-blackbox_finetune_recall60}"
TRADE_RULE="${TRADE_RULE:-stop_loss_3pct_take_profit_10_20_hold_3d}"
SAMPLE_MODE="${SAMPLE_MODE:-xlong}"
THRESHOLD="${THRESHOLD:-0.45}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
DAILY_WINDOW="${DAILY_WINDOW:-21}"
WEEKLY_WINDOW="${WEEKLY_WINDOW:-13}"
MONTHLY_WINDOW="${MONTHLY_WINDOW:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LIMIT_PER_DAY="${LIMIT_PER_DAY:-5}"
BACKTEST_PREFIX="${BACKTEST_PREFIX:-negative_reshuffle_cycle01}"
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

CHECKPOINT_DIR="$CYCLE_DIR/training/checkpoints"
RESULT_ROOT="$CYCLE_DIR/checkpoint_backtests_${START_DATE}_${END_DATE}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RESULT_ROOT/runs/$RUN_ID"
RESULT_FILE="${RESULT_FILE:-$RESULT_ROOT/results.json}"
RECORD_DIR="$RUN_DIR/records"

if [[ ! "$START_DATE" =~ ^[0-9]{8}$ || ! "$END_DATE" =~ ^[0-9]{8}$ ]]; then
  echo "Start and end dates must use YYYYMMDD format." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$RECORD_DIR" "$(dirname "$RESULT_FILE")"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

write_result_file() {
  "$PYTHON_BIN" - "$RESULT_FILE" "$RECORD_DIR" "$CYCLE_DIR" "$START_DATE" "$END_DATE" "$RUN_ID" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

result_path = Path(sys.argv[1])
record_dir = Path(sys.argv[2])
records = []
for path in sorted(record_dir.glob("update-*.json")):
    records.append(json.loads(path.read_text(encoding="utf-8")))

payload = {
    "cycle_dir": sys.argv[3],
    "start_date": sys.argv[4],
    "end_date": sys.argv[5],
    "run_id": sys.argv[6],
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "checkpoint_count": len(records),
    "checkpoints": records,
}
temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
temporary_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
temporary_path.replace(result_path)
PY
}

export_backtest_record() {
  local checkpoint_path="$1"
  local checkpoint_update="$2"
  local backtest_name="$3"
  local exit_code="$4"
  local log_path="$5"
  local record_path="$6"
  local error_message="${7:-}"

  "$PYTHON_BIN" - "$checkpoint_path" "$checkpoint_update" "$backtest_name" "$exit_code" "$log_path" "$record_path" "$error_message" "$STRATEGY_NAME" "$TRADE_RULE" "$START_DATE" "$END_DATE" <<'PY'
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from portfolio_backtest.db import connect

checkpoint_path = sys.argv[1]
checkpoint_update = int(sys.argv[2])
backtest_name = sys.argv[3]
exit_code = int(sys.argv[4])
log_path = sys.argv[5]
record_path = Path(sys.argv[6])
error_message = sys.argv[7]
strategy_name = sys.argv[8]
trade_rule_name = sys.argv[9]
start_date = sys.argv[10]
end_date = sys.argv[11]

def json_value(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value

columns = [
    "trade_date",
    "total_market_value",
    "holding_market_value",
    "cash_amount",
    "actual_profit",
    "daily_buy_amount",
    "daily_sell_amount",
    "trading_fee",
    "position_count",
]
rows = []
query_error = None
try:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                TradeDate,
                TotalMarketValue,
                HoldingMarketValue,
                CashAmount,
                ActualProfit,
                DailyBuyAmount,
                DailySellAmount,
                TradingFee,
                PositionCount
            FROM portfolio_backtest_daily
            WHERE BacktestName = %s
              AND StrategyName = %s
              AND TradeRuleName = %s
              AND TradeDate BETWEEN %s AND %s
            ORDER BY TradeDate
            """,
            (backtest_name, strategy_name, trade_rule_name, start_date, end_date),
        )
        rows = [
            {column: json_value(value) for column, value in zip(columns, row)}
            for row in cur.fetchall()
        ]
except Exception as exc:
    query_error = f"{type(exc).__name__}: {exc}"

record = {
    "checkpoint_update": checkpoint_update,
    "checkpoint_path": checkpoint_path,
    "backtest_name": backtest_name,
    "strategy_name": strategy_name,
    "trade_rule_name": trade_rule_name,
    "start_date": start_date,
    "end_date": end_date,
    "exit_code": exit_code,
    "status": "completed" if exit_code == 0 and query_error is None else "failed",
    "log_path": log_path,
    "daily_count": len(rows),
    "daily": rows,
}
if error_message:
    record["error"] = error_message
if query_error:
    record["query_error"] = query_error
record_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
}

cat <<EOF
Negative reshuffle checkpoint portfolio backtests
  cycle_dir=$CYCLE_DIR
  checkpoint_dir=$CHECKPOINT_DIR
  dates=$START_DATE-$END_DATE
  checkpoints=$CHECKPOINT_UPDATES
  strategy=$STRATEGY_NAME
  trade_rule=$TRADE_RULE
  threshold=$THRESHOLD
  limit_per_day=$LIMIT_PER_DAY
  sample_mode=$SAMPLE_MODE
  max_seq_length=$MAX_SEQ_LENGTH
  result_file=$RESULT_FILE
  run_dir=$RUN_DIR
  dry_run=$DRY_RUN
EOF

completed=0
failed=0
for short_update in $CHECKPOINT_UPDATES; do
  update_number=$((10#$short_update))
  checkpoint_name="$(printf 'update-%06d' "$update_number")"
  adapter_dir="$CHECKPOINT_DIR/$checkpoint_name"
  backtest_name="${BACKTEST_PREFIX}_${checkpoint_name//-/_}_${START_DATE}_${END_DATE}_limit${LIMIT_PER_DAY}"
  log_path="$RUN_DIR/${checkpoint_name}.log"
  record_path="$RECORD_DIR/${checkpoint_name}.json"

  echo
  echo "==== Backtesting $checkpoint_name as $backtest_name ===="
  if [[ ! -f "$adapter_dir/adapter_config.json" ]]; then
    message="Checkpoint adapter not found: $adapter_dir"
    echo "$message" >&2
    if [[ "$DRY_RUN" != "1" ]]; then
      export_backtest_record "$adapter_dir" "$update_number" "$backtest_name" 2 "$log_path" "$record_path" "$message"
      write_result_file
    fi
    failed=$((failed + 1))
    continue
  fi

  command=(
    "$PYTHON_BIN" -m portfolio_backtest.run
    --strategy-name "$STRATEGY_NAME"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --trade-rule "$TRADE_RULE"
    --blackbox-sample-mode "$SAMPLE_MODE"
    --blackbox-threshold "$THRESHOLD"
    --blackbox-max-seq-length "$MAX_SEQ_LENGTH"
    --blackbox-daily-window "$DAILY_WINDOW"
    --blackbox-weekly-window "$WEEKLY_WINDOW"
    --blackbox-monthly-window "$MONTHLY_WINDOW"
    --blackbox-adapter-dir "$adapter_dir"
    --blackbox-cuda-device "$CUDA_DEVICE"
    --limit-per-day "$LIMIT_PER_DAY"
    --backtest-name "$backtest_name"
  )
  printf 'command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  (
    cd "$PROJECT_DIR"
    "${command[@]}"
  ) 2>&1 | tee "$log_path"
  exit_code=${PIPESTATUS[0]}

  message=""
  if [[ "$exit_code" -ne 0 ]]; then
    message="Backtest command failed with exit code $exit_code; partial database rows, if any, are exported."
    failed=$((failed + 1))
  else
    completed=$((completed + 1))
  fi
  export_backtest_record "$adapter_dir" "$update_number" "$backtest_name" "$exit_code" "$log_path" "$record_path" "$message"
  write_result_file
  echo "Updated result file: $RESULT_FILE"
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run completed. No backtests were executed and no result JSON was written."
else
  echo "Backtest series completed: completed=$completed failed=$failed result_file=$RESULT_FILE"
fi
