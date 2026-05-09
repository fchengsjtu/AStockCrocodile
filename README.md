# StockInfoCrawler

A-share daily K-line crawler. The current version fetches forward-adjusted (`qfq`) daily K-line data from Tencent via AkShare and writes directly to the cloud MySQL database `emstocks`.

It no longer writes CSV files during crawling, and only `daily` K-line import is supported.

## Environment

Create `env.txt` in the project root. The program reads this file automatically before connecting to MySQL.

```powershell
$env:MYSQL_HOST='your-mysql-host'
$env:MYSQL_USER='your-mysql-user'
$env:MYSQL_PASSWORD='your-mysql-password'
$env:MYSQL_DATABASE='your-database'
```

Optional values:

```powershell
$env:MYSQL_PORT='3306'
$env:DKANDLES_KTYPE='D'
```

`env.txt` is ignored by git because it contains credentials.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install PyMySQL
```

If PowerShell blocks activation:

```powershell
powershell -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## CentOS 7 Cloud Server Deployment

The verified cloud server project path is:

```bash
/home/fcheng/work/AStockCrocodile
```

CentOS 7 ships with an old system Python, and the server's existing `python3.12` may not have the `_ssl` module. Use a CentOS 7 compatible Miniconda Python 3.10 to create the project virtual environment.

### 1. Log in and enter the project

```bash
ssh fcheng@49.235.114.211
cd /home/fcheng/work/AStockCrocodile
```

### 2. Install a CentOS 7 compatible Python bootstrap

Run this once on the server:

```bash
cd /home/fcheng/work/AStockCrocodile
MINICONDA="$HOME/miniconda3-py310"
INSTALLER="/tmp/miniconda-py310-centos7.sh"
URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh"

if [ ! -x "$MINICONDA/bin/python" ]; then
  curl -L --retry 3 -o "$INSTALLER" "$URL"
  bash "$INSTALLER" -b -p "$MINICONDA"
fi
```

Verify that the bootstrap Python has SSL support:

```bash
$HOME/miniconda3-py310/bin/python - <<'PY'
import ssl, sys
print(sys.version.split()[0])
print(ssl.OPENSSL_VERSION)
PY
```

The verified server output used Python `3.10.12` with OpenSSL `3.0.9`.

### 3. Create the project virtual environment

```bash
cd /home/fcheng/work/AStockCrocodile
rm -rf .venv
$HOME/miniconda3-py310/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --prefer-binary -i https://pypi.org/simple --timeout 120 --retries 5
```

If `pandas` tries to build from source and the compiler is killed by memory limits, install binary wheels first and rerun requirements:

```bash
source /home/fcheng/work/AStockCrocodile/.venv/bin/activate
python -m pip install --only-binary=:all: "numpy>=1.26" "pandas>=2.2.0" -i https://pypi.org/simple --timeout 120 --retries 5
python -m pip install -r requirements.txt --prefer-binary -i https://pypi.org/simple --timeout 120 --retries 5
```

### 4. Configure MySQL connection

Create `env.txt` in the project root. Do not commit this file.

```bash
cd /home/fcheng/work/AStockCrocodile
cat > env.txt <<'EOF'
MYSQL_HOST='your-mysql-host'
MYSQL_USER='your-mysql-user'
MYSQL_PASSWORD='your-mysql-password'
MYSQL_DATABASE='your-database'
MYSQL_PORT='3306'
DKANDLES_KTYPE='D'
EOF
chmod 600 env.txt
```

The parser also accepts the PowerShell-style `$env:NAME='value'` lines used on Windows.

### 5. Verify the environment

```bash
cd /home/fcheng/work/AStockCrocodile
source .venv/bin/activate
python -m py_compile a_share_crawler.py
python a_share_crawler.py --help
python - <<'PY'
for name in ["akshare", "pandas", "pymysql", "requests", "apscheduler", "tqdm"]:
    module = __import__(name)
    print(name, getattr(module, "__version__", "ok"))
PY
```

If the `tests` directory exists on the server, run:

```bash
python -m unittest discover -s tests -v
```

The server environment was verified with `akshare`, `pandas`, `PyMySQL`, `requests`, `APScheduler`, and `tqdm` import checks, `py_compile`, command help, and unit tests.

### 6. Run crawler commands on the server

Incremental daily import:

```bash
cd /home/fcheng/work/AStockCrocodile
source .venv/bin/activate
python a_share_crawler.py run --mode incremental
```

Full resume import from `2010-01-01` for unfinished stocks:

```bash
python a_share_crawler.py run --mode full --start-date 20100101
```

Fetch ex-rights/dividend data and refresh affected qfq K-lines:

```bash
python a_share_crawler.py exrights
```

Generate weekly and monthly K-lines from existing daily rows:

```bash
python a_share_crawler.py generate --period all
```

Run the scheduler in the foreground:

```bash
python a_share_crawler.py schedule
```

Run the scheduler in the background and keep logs:

```bash
cd /home/fcheng/work/AStockCrocodile
source .venv/bin/activate
mkdir -p logs
nohup python a_share_crawler.py schedule >> logs/scheduler.log 2>&1 &
```

Check the background scheduler:

```bash
ps -ef | grep a_share_crawler.py | grep -v grep
tail -f logs/scheduler.log
```

## Full Resume

Full mode does not clear `dkandles`. It refreshes stock basic info, checks each stock's `stockinfo.LatestUpdateKandle`, skips stocks that are already up to date, and continues unfinished stocks from the next missing date. Stocks without `LatestUpdateKandle` start from `2010-01-01`.

```powershell
python .\a_share_crawler.py run --mode full --period daily
```

Equivalent minimal command, because `--period daily` is the only supported period:

```powershell
python .\a_share_crawler.py run --mode full
```

## Incremental Update

Incremental mode is the default. It refreshes stock basic info, reads each stock's `stockinfo.LatestUpdateKandle`, and only fetches dates after that value.

```powershell
python .\a_share_crawler.py run
```

Equivalent explicit command:

```powershell
python .\a_share_crawler.py run --mode incremental --period daily
```

After each stock is written successfully, `stockinfo.LatestUpdateKandle` is updated to that stock's newest imported daily K-line time, e.g. `2026-05-06 15:00:00`.

## Run Parameters

### `run`

```powershell
python .\a_share_crawler.py run [options]
```

Available options:

- `--period {daily}`: only `daily` is supported.
- `--mode {full,incremental}`: default `incremental`; both modes continue from `stockinfo.LatestUpdateKandle + 1 day`. `full` is intended for resuming unfinished all-stock runs without clearing existing daily data.
- `--start-date START_DATE`: start date for full mode; earliest effective date is `20100101`. Accepts `YYYYMMDD` or `YYYY-MM-DD`.
- `--end-date END_DATE`: end date, default is today's date, format `YYYYMMDD` or `YYYY-MM-DD`.
- `--adjust {qfq}`: adjustment mode. All K-line data is forward-adjusted; `qfq` is the only supported value.
- `--sleep SLEEP`: seconds to pause inside each fetch worker after a request, default `0.05`.
- `--retries RETRIES`: retry count for Tencent requests, default `3`.
- `--workers WORKERS`: number of concurrent fetch worker threads, default `8`. Database writes remain serialized in the main thread.
- `--ktype KTYPE`: value written to `dkandles.KType`, default `D` or `DKANDLES_KTYPE` from environment.
- `--use-env-proxy`: use proxy environment variables instead of forcing direct requests.

Examples:

```powershell
python .\a_share_crawler.py run --mode full --start-date 20100101 --end-date 20260506 --ktype D
python .\a_share_crawler.py run --mode incremental --sleep 0.1 --retries 5 --workers 12
python .\a_share_crawler.py run --mode full --adjust qfq
```



## Fetch Ex-rights And Dividends

Dividend, bonus-share, share-transfer, record-date, and ex-dividend/ex-right information is extracted from Tencent daily K-line ex-rights fields and written to `exrights`.

```powershell
python .\a_share_crawler.py exrights
```

Available options:

- `--sleep SLEEP`: seconds to pause inside each fetch worker after a request, default `0.05`.
- `--retries RETRIES`: retry count for Tencent requests, default `3`.
- `--workers WORKERS`: number of concurrent fetch worker threads, default `8`.
- `--truncate`: truncate `exrights` before importing.
- `--end-date END_DATE`: end date for K-line refresh when ex-rights rows change, default today.
- `--ktype KTYPE`: value written to refreshed daily rows, default `D`.
- `--no-refresh-klines`: only update `exrights`; do not refresh daily/weekly/monthly K-lines for changed stocks.
- `--use-env-proxy`: use proxy environment variables.

The program creates `exrights` automatically if it does not exist. Rows are upserted by an internal `SourceKey` built from stock code, report date, ex-dividend/ex-right date, and notice date. Tencent fields such as `FHcontent`, `djr`, `cqr`, and `fh_sh` are normalized into the table columns. A `ContentHash` is stored for each row; when a stock has new or changed ex-rights data, the crawler deletes that stock's daily, weekly, and monthly K-lines, refetches forward-adjusted daily data from `2010-01-01`, and rebuilds weekly/monthly rows for that stock only.

## Generate Weekly And Monthly K-lines

Weekly and monthly K-lines are generated from existing daily rows in `dkandles`; no market API is called for this step.

Manual generation:

```powershell
python .\a_share_crawler.py generate --period weekly
python .\a_share_crawler.py generate --period monthly
python .\a_share_crawler.py generate --period all
```

Generation targets:

- Weekly K-lines: `wkandles`, `KType='W'`, `KTime` is Friday `17:00:00` for the week.
- Monthly K-lines: `mkandles`, `KType='M'`, `KTime` is the month's last actual trading day at `18:00:00`.

The current implementation rebuilds the selected target table before inserting regenerated rows.
## Schedule Mode

Schedule mode registers four jobs in Asia/Shanghai: daily import at 15:05, ex-rights change check and qfq K-line refresh at 16:00, weekly generation every Friday at 17:00, and monthly generation at 18:00 on the month last trading day check.

```powershell
python .\a_share_crawler.py schedule
```

Default schedule mode is incremental. To schedule full resume runs:

```powershell
python .\a_share_crawler.py schedule --mode full
```

Available schedule options:

- `--mode {full,incremental}`: default `incremental`.
- `--start-date START_DATE`: start date for full mode, earliest effective date is `20100101`.
- `--adjust {qfq}`: adjustment mode; `qfq` only.
- `--sleep SLEEP`: seconds to pause inside each fetch worker after a request, default `0.05`.
- `--retries RETRIES`: retry count, default `3`.
- `--workers WORKERS`: number of concurrent fetch worker threads, default `8`.
- `--ktype KTYPE`: value written to `dkandles.KType`, default `D`.
- `--use-env-proxy`: use proxy environment variables.

PowerShell helper:

```powershell
.\run_scheduler.ps1
```


## Stock Selection And Backtesting

`stock_selector.py` runs the stock selection strategy and writes selected rows to MySQL table `stockselection`.

Default strategy:

- MA bullish alignment: `MA5 > MA8 > MA13 > MA34 > MA55`.
- Close is above `MA5`.
- The candle is bullish: `Close > Open`.
- Optional liquidity filter by 5-day average `Amount`.
- Near limit-up days are excluded by default.

Available strategies:

- `ma_bullish_v1`: the moving-average strategy above.
- `news_hot_v1`: news hotspot strategy. It ranks stocks by concept heat from `news.ConceptHeat`, news-to-stock relation strength, source credibility, news heat, and recent stock performance. It recommends 3 to 5 stocks by default.
- `weekly_volume_drop_v1`: weekly two-bar volume-drop strategy. It selects stocks whose latest two weekly K-lines fall consecutively, whose two-week average volume is at least 1.5 times the previous five-week average volume, and whose latest weekly close is at least 15% below the close before the two-week drop.

Run selection for the latest trading day:

```powershell
python .\stock_selector.py
```

Run selection for a specific day:

```powershell
python .\stock_selector.py --date 20260508 --min-turnover-amount 50000 --limit 50
```

Run the news hotspot strategy:

```powershell
python .\stock_selector.py --strategy news_hot_v1 --date 20260508
```

Run the weekly volume-drop strategy:

```powershell
python .\stock_selector.py --strategy weekly_volume_drop_v1 --date 20260508
```

`backtest_strategy.py` runs a historical backtest with these definitions:

- Success: from the 3rd to the 8th trading day after selection, the minimum `Low` is at least 2% above the selection day's `Close`.
- Failure: from the 3rd to the 8th trading day after selection, the weighted average price is at least 1% below the selection day's `Close`.
- Explosive: from the 3rd to the 8th trading day after selection, the minimum `Low` is at least 20% above the selection day's `Close`.

Weighted average price uses Tencent units stored in `dkandles`: `sum(Amount) * 100 / sum(Volume)`, because `Amount` is in 10k yuan and `Volume` is in lots.

Backtest summaries are saved to MySQL table `strategybacktestresults` by default. Each row is grouped by stock code, strategy name, and backtest date range, with `SuccessRate`, `AvgRiseRate`, `FailureRate`, `AvgDropRate`, and `ExplosiveRate`. `AvgRiseRate` uses the 3rd-to-8th trading-day minimum low versus the selection close; `AvgDropRate` uses the same window's weighted average price versus the selection close.

Run backtest:

```powershell
python .\backtest_strategy.py --start-date 20240101 --end-date 20241231 --limit-per-day 50 --output data\backtest-2024.csv
```

Run without saving the summary table:

```powershell
python .\backtest_strategy.py --start-date 20240101 --end-date 20241231 --no-save-db
```

The output summary includes total selections, success rate, failure rate, and explosive rate. The optional CSV stores detailed rows for each evaluated selection.

## K-line Statistics

`kline_statistics.py` computes statistics from existing daily rows in `dkandles` and writes matched rows to MySQL table `klinestatistics`.

The default statistic type is `short_term_surge_3d_20pct`: the close price on the 3rd trading day after the start-rise date is at least 20% above the start-rise date close.

Stored fields include:

- `SCode`
- `SName`
- `StartRiseDate`: 起涨点日期
- `PrevTradeDate`: 起涨点前一个交易日
- `GainRate`: 起涨点之后第 3 个交易日相对起涨点日期收盘价的涨幅
- `StatType`: 统计类型

Run K-line statistics:

```powershell
python .\kline_statistics.py --start-date 20240101 --end-date 20241231
```

Run without saving to MySQL:

```powershell
python .\kline_statistics.py --start-date 20240101 --end-date 20241231 --no-save-db
```

## News Crawler

`news_crawler.py` crawls stock-market news from AkShare-backed public sources and writes rows to MySQL table `news`.

Default sources:

- `eastmoney`
- `ths`
- `caixin`

The program creates `news` automatically. Rows are de-duplicated by `NewsLink`.

Main stored fields:

- `NewsLink`: news URL.
- `CredibilityLevel`: source credibility level from 1 to 10; 1 is highest and 10 is lowest.
- `Heat`: click count or heat value when the source exposes it; otherwise `0`.
- `RelatedConcepts`: JSON array, ordered from strongest to weakest relationship, with at most 10 concepts.
- `ConceptHeat`: JSON array with each related concept and its share of the current news batch. The crawler keeps up to 10000 news rows by default, so the concept heat is computed as concept news count divided by crawled news count.

Additional useful fields include `Title`, `Summary`, `SourceName`, `PublishTime`, and `ContentHash`.

Run news crawling:

```powershell
python .\news_crawler.py
```

Run selected sources and limit rows:

```powershell
python .\news_crawler.py --sources eastmoney,ths --limit 100
```

Run without saving to MySQL:

```powershell
python .\news_crawler.py --no-save-db --output data\news.csv
```

## Database Writes

The crawler writes to:

- `dkandles`: daily K-line rows.
- `stockinfo`: stock basic info and `LatestUpdateKandle`.
- `exrights`: dividend, bonus-share, share-transfer, record-date, and ex-dividend/ex-right rows.

`dkandles` fields written by the crawler:

- `SCode`
- `KType`
- `KTime` (`15:00:00` on the trading day)
- `Amount`
- `Volume`
- `MA5`
- `MA8`
- `MA13`
- `Open`
- `Close`
- `High`
- `Low`
- `CreatedOn`
- `UpdatedOn`
- `MA55`
- `MA34`

`stockinfo` fields updated by the crawler:

- `SCode`
- `SName`
- `IsIndex`
- `LatestUpdateKandle` (latest imported daily K-line time)

## Notes

- All K-line data is based on forward-adjusted (`qfq`) prices.
- Full mode does not truncate `dkandles`; it skips stocks whose `LatestUpdateKandle` is already at or after the requested end date.
- Incremental mode uses `stockinfo.LatestUpdateKandle + 1 day` as each stock's start date; full mode now uses the same missing-date continuation logic.
- Incremental mode reads the latest 54 historical closes from MySQL to keep MA5/MA8/MA13/MA34/MA55 continuous.
- Fetching uses a thread pool controlled by `--workers`; MySQL insert/update work is performed in the main thread to keep transactions stable.
- Requests are direct by default; proxy environment variables are ignored unless `--use-env-proxy` is passed.
## Table Mapping

- Daily K-lines: `dkandles`, `KType='D'`.
- Weekly K-lines: `wkandles`, `KType='W'`, generated from `dkandles`.
- Monthly K-lines: `mkandles`, `KType='M'`, generated from `dkandles`.
- Ex-rights and dividends: `exrights`, extracted from Tencent daily K-line ex-rights fields.
