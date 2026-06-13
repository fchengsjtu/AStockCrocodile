# Portfolio Backtest

This directory contains a portfolio-level T+1 backtester.

Default period: `20260101` to `20260430`.

Rules:

- Initial cash: `1000000`.
- For stocks selected on day T, buy on the next trading day using that day's weighted average price.
- Each buy targets `100000` cash and rounds shares to the nearest 100-share lot.
- Total new-buy cash outflow per day, including buy commissions, is capped at
  the smaller of one third of the account's pre-buy total market value and
  the available cash after that day's sells.
- Commission rate is `0.0002` (0.02%, two basis points) for both buys and sells.
- Stamp duty is `0.0005` (0.05%, five basis points) of gross sell amount and is charged only on sells.
- A position bought on day B can only be sold from the next trading day because of T+1.
- During the first three sellable trading days, the default 3% rule first
  applies an intraday stop at `cost * 0.95`. If the stock opens at or below
  that price, sell at the opening price; otherwise sell at `cost * 0.95` when
  the daily low reaches it. If the intraday stop is not triggered but the
  close is at or below `cost * 0.97`, sell at the closing price.
- Take profit half at `cost * 1.10`, then sell the rest at `cost * 1.20`.
- If neither stop loss nor take profit closes the position, sell remaining shares at the third sellable trading day's close.
- If candidate buys exceed available cash, the program shuffles candidates with a fixed seed and buys until cash is full.
- Stocks whose names contain `ST` or `PT` are excluded before the daily TopN limit is applied.
- After a stock is selected, it is excluded for the following 13 market trading days.

Run:

```powershell
python -m portfolio_backtest.run --strategy-name ma_bullish_v1
```

Override the selection cooldown when needed:

```powershell
python -m portfolio_backtest.run `
  --strategy-name ma_bullish_v1 `
  --selection-cooldown-trading-days 13
```

Weekly volume-drop strategy:

```powershell
python -m portfolio_backtest.run --strategy-name weekly_volume_drop_v1
```

Black-box recall strategy:

```powershell
python -m portfolio_backtest.run --strategy-name blackbox_finetune_recall30 --limit-per-day 5
```

Supported black-box strategy names:

- `blackbox_finetune_recall30`
- `blackbox_finetune_recall35`
- `blackbox_finetune_recall40`
- `blackbox_finetune_recall45`
- `blackbox_finetune_recall50`
- `blackbox_finetune_recall55`
- `blackbox_finetune_recall60`
- `blackbox_finetune_recall65`
- `blackbox_finetune_recall70`
- `blackbox_finetune_recall75`
- `blackbox_finetune_recall80`

Black-box options:

```powershell
python -m portfolio_backtest.run `
  --strategy-name blackbox_finetune_recall30 `
  --blackbox-threshold 0.50 `
  --blackbox-max-seq-length 512 `
  --blackbox-cuda-device 0 `
  --limit-per-day 5
```

The black-box portfolio backtest loads the corresponding LoRA adapter from that recall directory, scores each trading day in the backtest period, and uses positive predictions as same-day selection signals. It can be slow because every stock/date requires model scoring.

Useful options:

```powershell
python -m portfolio_backtest.run `
  --start-date 20260101 `
  --end-date 20260430 `
  --strategy-name ma_bullish_v1 `
  --initial-cash 1000000 `
  --buy-budget 100000 `
  --fee-rate 0.0002 `
  --stamp-duty-rate 0.0005 `
  --random-seed 20260519
```

Run the stop-loss rule series:

```powershell
python -m portfolio_backtest.run `
  --strategy-name ma_bullish_v1 `
  --trade-rule-series stop_loss
```

This runs the same selection and take-profit logic with these stop-loss levels.
The default `3%` rule uses the combined intraday 5% and closing 3% behavior
described above; the other rules retain their configured intraday stop level:

- `3%`
- `3.5%`
- `4%`
- `4.5%`
- `5%`
- `5.5%`
- `6%`

Each result row stores `TradeRuleName`, for example `stop_loss_3_5pct_take_profit_10_20_hold_3d`.

Run a single custom stop-loss rule:

```powershell
python -m portfolio_backtest.run `
  --strategy-name ma_bullish_v1 `
  --stop-loss-pct 0.05
```

Tables:

- `portfolio_backtest_daily`: daily account summary.
- `portfolio_backtest_holdings`: per-stock daily holdings.
- `portfolio_backtest_trades`: buy and sell records.

Rows store `TradeRuleName`, `SelectionRule`, and `ExitRule`.
