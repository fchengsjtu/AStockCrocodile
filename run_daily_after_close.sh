#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TRADE_DATE="$(date +%Y%m%d)"
MODEL_DIR="D:\\Models\\precision10@0.4-1400"
CRAWLER_WORKERS="${CRAWLER_WORKERS:-8}"
CRAWL_MODE="${CRAWL_MODE:-full}"
CRAWL_START_DATE="${CRAWL_START_DATE:-20100101}"

if [[ $# -gt 2 ]]; then
  echo "Usage: bash ./run_daily_after_close.sh [trade_date:yyyymmdd] [model_dir]" >&2
  echo '   or: bash ./run_daily_after_close.sh [model_dir]' >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  if [[ "$1" =~ ^[0-9]{8}$ ]]; then
    TRADE_DATE="$1"
  else
    MODEL_DIR="$1"
  fi
elif [[ $# -eq 2 ]]; then
  TRADE_DATE="$1"
  MODEL_DIR="$2"
fi

if [[ ! "$TRADE_DATE" =~ ^[0-9]{8}$ ]]; then
  echo "Trade date must use yyyyMMdd format, for example 20260529." >&2
  exit 2
fi
if [[ "$CRAWL_MODE" != "full" && "$CRAWL_MODE" != "incremental" ]]; then
  echo "CRAWL_MODE must be full or incremental." >&2
  exit 2
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_after_close_${TRADE_DATE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export ASTOCK_DISABLE_LOCAL_DEPS="${ASTOCK_DISABLE_LOCAL_DEPS:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

resolve_python() {
  local requested="${MAIN_PYTHON:-}"
  if [[ -n "$requested" && -x "$requested" ]]; then
    printf '%s\n' "$requested"
    return
  fi
  local crawler_venv="$HOME/.venvs/astock-crawler"
  if [[ ! -x "$crawler_venv/bin/python" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      echo "Creating crawler virtualenv: $crawler_venv" >&2
      python3 -m venv "$crawler_venv"
    fi
  fi
  local candidates=(
    "$crawler_venv/bin/python"
    "$HOME/.venvs/astock-blackbox-finetune-recall60/bin/python"
    "python3"
  )
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == */* ]]; then
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
      fi
    elif command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done
  echo "No Python executable found. Set MAIN_PYTHON=/path/to/python." >&2
  exit 1
}

PYTHON_EXE="$(resolve_python)"

ensure_crawler_deps() {
  if "$PYTHON_EXE" - <<'PY' >/dev/null 2>&1
import akshare  # noqa: F401
import apscheduler  # noqa: F401
import pandas  # noqa: F401
import pymysql  # noqa: F401
import tqdm  # noqa: F401
PY
  then
    return 0
  fi
  echo "Crawler Python is missing dependencies; installing requirements.txt into: $PYTHON_EXE"
  "$PYTHON_EXE" -m pip install --upgrade pip wheel "setuptools<82"
  "$PYTHON_EXE" -m pip install -r "$ROOT/requirements.txt"
  "$PYTHON_EXE" - <<'PY'
import akshare  # noqa: F401
import apscheduler  # noqa: F401
import pandas  # noqa: F401
import pymysql  # noqa: F401
import tqdm  # noqa: F401
print("Crawler dependencies are ready.")
PY
}

run_step() {
  local name="$1"
  shift
  echo
  echo "==== $name ===="
  echo "$*"
  "$@"
}

is_weekly_due() {
  local weekday
  weekday="$(date -d "${TRADE_DATE:0:4}-${TRADE_DATE:4:2}-${TRADE_DATE:6:2}" +%u)"
  [[ "$weekday" -ge 5 ]]
}

is_monthly_due() {
  "$PYTHON_EXE" - <<PY
from datetime import datetime
from a_share_crawler import is_last_trade_day
raise SystemExit(0 if is_last_trade_day(datetime.strptime("$TRADE_DATE", "%Y%m%d").date()) else 1)
PY
}

echo "ProjectDir=$ROOT"
echo "TradeDate=$TRADE_DATE"
echo "ModelDir=$MODEL_DIR"
echo "Python=$PYTHON_EXE"
echo "CrawlerWorkers=$CRAWLER_WORKERS"
echo "CrawlMode=$CRAWL_MODE"
echo "CrawlStartDate=$CRAWL_START_DATE"
echo "LogFile=$LOG_FILE"
echo "ASTOCK_DISABLE_LOCAL_DEPS=$ASTOCK_DISABLE_LOCAL_DEPS"

ensure_crawler_deps

run_step "crawl daily K-lines" "$PYTHON_EXE" ./a_share_crawler.py run \
  --mode "$CRAWL_MODE" \
  --period daily \
  --start-date "$CRAWL_START_DATE" \
  --end-date "$TRADE_DATE" \
  --workers "$CRAWLER_WORKERS"

if is_weekly_due; then
  run_step "generate weekly K-lines" "$PYTHON_EXE" ./a_share_crawler.py generate --period weekly
else
  echo "Skip weekly K-line generation: trade date is not weekend-ready (Friday/Saturday/Sunday)."
fi

if is_monthly_due; then
  run_step "generate monthly K-lines" "$PYTHON_EXE" ./a_share_crawler.py generate --period monthly
else
  echo "Skip monthly K-line generation: trade date is not the last trading day of the month."
fi

run_step "predict blackbox recall60" bash ./predict_blackbox_recall60.sh "$MODEL_DIR" "$TRADE_DATE"

echo "Daily after-close WSL workflow completed."
