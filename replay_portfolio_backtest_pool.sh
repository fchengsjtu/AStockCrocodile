#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SOURCE_BACKTEST_NAME="${1:-precision10_c2_900_threshold0562_20260101_20260630_sl6_limit100000}"
START_DATE="${2:-20260101}"
END_DATE="${3:-20260630}"
TRADE_RULE="${TRADE_RULE:-stop_loss_6pct_take_profit_10_20_hold_3d}"
INITIAL_CASH="${INITIAL_CASH:-1000000}"
BUY_BUDGET="${BUY_BUDGET:-100000}"
LIMIT_PER_DAY="${LIMIT_PER_DAY:-3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-blackbox-finetune-recall60}"
CRAWLER_VENV_DIR="${CRAWLER_VENV_DIR:-$HOME/.venvs/astock-crawler}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing Python: $VENV_DIR/bin/python" >&2
  exit 1
fi

EXTRA_PYTHONPATH="$ROOT"
if [[ -x "$CRAWLER_VENV_DIR/bin/python" ]]; then
  CRAWLER_SITE_PACKAGES="$("$CRAWLER_VENV_DIR/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  if [[ -d "$CRAWLER_SITE_PACKAGES" ]]; then
    EXTRA_PYTHONPATH="$EXTRA_PYTHONPATH:$CRAWLER_SITE_PACKAGES"
  fi
fi

export ASTOCK_DISABLE_LOCAL_DEPS="${ASTOCK_DISABLE_LOCAL_DEPS:-1}"
export PYTHONPATH="$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

cat <<EOF
Replaying portfolio backtest pool without database writes
  source_backtest_name=$SOURCE_BACKTEST_NAME
  start_date=$START_DATE
  end_date=$END_DATE
  trade_rule=$TRADE_RULE
  initial_cash=$INITIAL_CASH
  buy_budget=$BUY_BUDGET
  limit_per_day=$LIMIT_PER_DAY
EOF

"$VENV_DIR/bin/python" -m portfolio_backtest.replay_backtest_pool \
  --source-backtest-name "$SOURCE_BACKTEST_NAME" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --trade-rule "$TRADE_RULE" \
  --initial-cash "$INITIAL_CASH" \
  --buy-budget "$BUY_BUDGET" \
  --limit-per-day "$LIMIT_PER_DAY"
