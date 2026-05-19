# Portfolio Backtest

This directory contains a portfolio-level T+1 backtester.

Default period: `20260101` to `20260430`.

Rules:

- Initial cash: `1000000`.
- For stocks selected on day T, buy on the next trading day using that day's weighted average price.
- Each buy targets `100000` cash and rounds shares to the nearest 100-share lot.
- Fee rate is `0.0005` for both buys and sells.
- A position bought on day B can only be sold from the next trading day because of T+1.
- During the first three sellable trading days, stop loss at `cost * 0.97`.
- Take profit half at `cost * 1.10`, then sell the rest at `cost * 1.20`.
- If neither stop loss nor take profit closes the position, sell remaining shares at the third sellable trading day's close.
- If candidate buys exceed available cash, the program shuffles candidates with a fixed seed and buys until cash is full.

Run:

```powershell
python -m portfolio_backtest.run --strategy-name ma_bullish_v1
```

Weekly volume-drop strategy:

```powershell
python -m portfolio_backtest.run --strategy-name weekly_volume_drop_v1
```

Useful options:

```powershell
python -m portfolio_backtest.run `
  --start-date 20260101 `
  --end-date 20260430 `
  --strategy-name ma_bullish_v1 `
  --initial-cash 1000000 `
  --buy-budget 100000 `
  --fee-rate 0.0005 `
  --random-seed 20260519
```

Tables:

- `portfolio_backtest_daily`: daily account summary.
- `portfolio_backtest_holdings`: per-stock daily holdings.
- `portfolio_backtest_trades`: buy and sell records.

The daily and holding rows store both `SelectionRule` and `ExitRule`.

