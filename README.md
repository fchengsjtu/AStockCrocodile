# StockInfoCrawler

A-share daily K-line crawler. The current version fetches forward-adjusted (`qfq`) daily K-line data from Tencent via AkShare and writes directly to the cloud MySQL database `emstocks`.

It no longer writes CSV files during crawling, and only `daily` K-line import is supported.

## Environment

Create `env.txt` in the project root. The program reads this file automatically before connecting to MySQL.

```powershell
$env:MYSQL_HOST='127.0.0.1'
$env:MYSQL_PORT='3306'
$env:MYSQL_USER='your-mysql-user'
$env:MYSQL_PASSWORD='your-mysql-password'
$env:MYSQL_DATABASE='emstocks'
```

Optional values:

```powershell
$env:MYSQL_PORT='3306'
$env:DKANDLES_KTYPE='D'
$env:LOCAL_LLM_BASE_URL='http://127.0.0.1:1234/v1'
$env:LOCAL_LLM_MODEL='deepseek-r1-distill-qwen-14b'
$env:LOCAL_LLM_API_KEY='local'
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

Backtest results are saved to MySQL table `strategybacktestresults` by default. Each selected stock is stored as one row. `StartDate` and `EndDate` keep the backtest range, while `SelectionDate` stores the date when the strategy selected the stock. `SuccessRate`, `FailureRate`, and `ExplosiveRate` are `1` or `0` for each selected stock row. `AvgRiseRate` uses the 3rd-to-8th trading-day minimum low versus the selection close; `AvgDropRate` uses the same window's weighted average price versus the selection close.

Run backtest:

```powershell
python .\backtest_strategy.py --start-date 20240101 --end-date 20241231 --limit-per-day 50 --output data\backtest-2024.csv
```

Run backtest for a specific strategy:

```powershell
python .\backtest_strategy.py --strategy-name weekly_volume_drop_v1 --start-date 20240101 --end-date 20241231
python .\backtest_strategy.py --strategy-name news_hot_v1 --start-date 20240101 --end-date 20241231
```

Long-range backtests use a low-memory path for all strategies: symbols are scanned in batches and strategy signals are collected. Daily strategies load only selected symbols' daily rows for the forward-window evaluation. `weekly_volume_drop_v1` is a weekly-only strategy and does not load `dkandles` during backtest. The default batch size is `80`; use `--batch-size` to tune it for smaller servers.

```powershell
python .\backtest_strategy.py --strategy-name weekly_volume_drop_v1 --start-date 20100101 --end-date 20251231 --batch-size 50
```

Run without saving the summary table:

```powershell
python .\backtest_strategy.py --start-date 20240101 --end-date 20241231 --no-save-db
```

The output summary includes total selections, success rate, failure rate, and explosive rate. The optional CSV stores detailed rows for each evaluated selection.

## K-line Statistics

`kline_statistics.py` computes statistics from existing daily rows in `dkandles` and writes matched rows to MySQL table `klinestatistics`.

The default statistic type is `short_term_surge_3d_20pct`: the close price on the 3rd trading day after the start-rise date is at least 20% above the start-rise date close.

Long-range statistics use a low-memory path: symbols are loaded in batches, each batch is queried, computed, and written before the next batch starts. The default batch size is `80`; use `--batch-size` on small servers.

After candidate rows are generated, the script scans the `news` table for internet news around `SelectionDate`. If news within the window mentions the stock and contains message-driven keywords such as announcements, restructuring, contracts, orders, earnings, approvals, or policy catalysts, that candidate is excluded before it is written to `klinestatistics`. The default news window is 3 days before and after `SelectionDate`.

Stored fields include:

- `SCode`
- `SName`
- `StartRiseDate`: start-rise date used to measure the surge window
- `PrevTradeDate`: trade date immediately before `StartRiseDate`; this is treated as the statistic trade date
- `SelectionDate`: selection date, the trading day immediately before `PrevTradeDate`
- `GainRate`: gain from `StartRiseDate` close to the close after `--forward-days`
- `StatType`

Run K-line statistics:

```powershell
python .\kline_statistics.py --start-date 20240101 --end-date 20241231
```

Run with a smaller batch size:

```powershell
python .\kline_statistics.py --start-date 20200101 --end-date 20251231 --batch-size 50
```

Run without message-driven news exclusion:

```powershell
python .\kline_statistics.py --start-date 20200101 --end-date 20251231 --no-news-filter
```

Run without saving to MySQL:

```powershell
python .\kline_statistics.py --start-date 20240101 --end-date 20241231 --no-save-db
```

Mine reusable patterns before `short_term_surge_3d_20pct` events:

```powershell
python .\surge_pattern_miner.py --test-start-date 20260101 --test-end-date 20260430
```

`surge_pattern_miner.py` reads positive samples from `klinestatistics`, using `SelectionDate` as the selected date. Training defaults to `20100101` through the day before `--test-start-date`; the test set is controlled by `--test-start-date` and `--test-end-date`. For each positive sample it extracts features from `SelectionDate` plus the previous 55 daily bars and `SelectionDate` plus the previous 55 weekly bars. It then scans historical candidate dates in small stock batches and writes retained patterns to MySQL table `surgepatterns` unless `--no-save-db` is used.

By default the script saves separate threshold groups for success rates `25%`, `30%`, `35%`, `40%`, `45%`, and `50%`.

Useful options:

- `--train-start-date 20100101 --train-end-date 20251231`: override the training range.
- `--success-rates 0.25,0.30,0.35,0.40,0.45,0.50`: choose retained success-rate thresholds.
- `--min-sample-count 20`: require at least this many test-set occurrences.
- `--min-positive-support 5`: require at least this many positive occurrences before a pattern is evaluated.
- `--max-pattern-size 2`: combine up to this many feature clauses into one pattern.
- `--daily-window 56 --weekly-window 56`: adjust lookback windows.
- `--batch-size 40`: reduce this on small servers.
- `--output data\surge_patterns.csv`: also write retained patterns to CSV.

Mine local DeepSeek-proposed surge setup patterns and validate them on the same test set:

```powershell
python .\llm_surge_pattern_miner.py --test-start-date 20260101 --test-end-date 20260430
```

`llm_surge_pattern_miner.py` first summarizes the positive-sample feature distribution before `20260101`, asks the local OpenAI-compatible model to propose diverse candidate patterns using exact feature tokens, then validates those patterns on all test-set candidate dates from `20260101` through `20260430`. Retained patterns for success-rate thresholds `25%`, `30%`, `35%`, `40%`, `45%`, and `50%` are written to `surgepatterns`.

For the local model server, set these values in the environment or in `env.txt`:

```powershell
$env:LOCAL_LLM_BASE_URL='http://127.0.0.1:1234/v1'
$env:LOCAL_LLM_MODEL='deepseek-r1-distill-qwen-14b'
$env:LOCAL_LLM_API_KEY='local'
```

Useful options:

- `--model deepseek-r1-distill-qwen-14b`: choose the local model.
- `--training-mode summary`: keep the previous feature-summary training mode.
- `--training-mode raw-kline`: use the new raw K-line mode; DeepSeek receives selected positive samples containing only 55 daily bars and 55 weekly bars ending at `SelectionDate`.
- `--raw-sample-size 30`: number of positive raw K-line samples sent to DeepSeek in raw mode.
- `--min-pattern-size 3 --max-pattern-size 8`: require each LLM pattern to contain at least 3 and at most 8 feature clauses.
- `--candidate-count 80`: number of LLM candidate patterns to validate.
- `--llm-response-file data\llm_patterns.json`: validate a saved LLM JSON response without calling the API.
- `--api-base-url http://127.0.0.1:1234/v1`: override the local OpenAI-compatible endpoint.

Run the new raw K-line training mode:

```powershell
python .\llm_surge_pattern_miner.py --training-mode raw-kline --test-start-date 20260101 --test-end-date 20260430
```

Run black-box local LLM training with an automatic 80/20 split from `klinestatistics`:

```powershell
python .\llm_blackbox_pattern_trainer.py
```

`llm_blackbox_pattern_trainer.py` reads positive samples from `klinestatistics`, uses `PrevTradeDate` as the input anchor, and sends each training sample's 55 daily bars ending at `PrevTradeDate` plus 55 weekly bars ending on or before `PrevTradeDate` to the local OpenAI-compatible model in small batches. To fit local models with 4096-token contexts, each OHLCV bar is sent in a compact numeric matrix: price fields are basis points versus the window's last close, and volume is percent of the window's average volume. The split is deterministic: 80% training samples and 20% held-out test samples by default.

The local model is used as a black-box rule generator, not as a weight fine-tuning endpoint. It proposes executable feature-token patterns; the program then validates those patterns on the held-out date range and writes only patterns with actual success rate at least `40%` to `surgepatterns`.

Useful options:

- `--train-ratio 0.8`: change the sample split ratio.
- `--min-success-rate 0.40`: require at least this validated success rate before saving.
- `--daily-window 55 --weekly-window 55`: keep the required input windows.
- `--prompt-batch-size 3`: number of samples sent in one LLM request; reduce this to `1` if the local model reports context/request errors.
- `--candidate-count 12`: number of patterns requested from each LLM training batch.
- `--max-training-batches 10`: limit LLM calls for a trial run; `0` means use all training batches.
- `--output data\blackbox_patterns.csv`: also write retained patterns to CSV.

For slow local models, set these optional values in `env.txt`:

```powershell
$env:LOCAL_LLM_TIMEOUT='600'
$env:LOCAL_LLM_MAX_TOKENS='2048'
```

Run a true LoRA/QLoRA fine-tuning pipeline for `DeepSeek-R1-Distill-Qwen-7B`.

Core commands:

```powershell
python .\llm_finetune\build_dataset.py --positive-limit 2000 --negative-ratio 1.0
python .\llm_finetune\train_lora.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --data-dir llm_finetune\data --output-dir llm_finetune\runs\deepseek-r1-distill-qwen-7b-lora --max-seq-length 4096 --batch-size 1 --gradient-accumulation-steps 8 --epochs 1
python .\llm_finetune\evaluate_model.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --adapter-dir llm_finetune\runs\deepseek-r1-distill-qwen-7b-lora\adapter --data-dir llm_finetune\data --min-success-rate 0.40
```

This creates:

```text
llm_finetune\data\train.jsonl
llm_finetune\data\valid.jsonl
llm_finetune\data\allowed_features.json
```

For a quick smoke dataset:

```powershell
python .\llm_finetune\build_dataset.py --positive-limit 50 --negative-ratio 1.0
```

Install the optional fine-tuning dependencies in a GPU environment:

```powershell
pip install -r requirements-finetune.txt
```

For RTX 3060, QLoRA training is strongly recommended. Windows native `bitsandbytes` support can be fragile, so WSL2/Linux is usually the smoother path.

Recommended RTX 3060 + WSL2 steps:

1. Install or update the NVIDIA Windows driver. Use a recent Game Ready or Studio driver with WSL CUDA support.

2. Enable WSL2 and install Ubuntu from an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
wsl --shutdown
```

3. Open Ubuntu and verify the GPU is visible:

```bash
nvidia-smi
```

4. Clone the project or enter the mounted Windows project path. Mounted Windows paths work, but training is usually faster inside the Linux filesystem.

```bash
cd ~
git clone https://github.com/fchengsjtu/AStockCrocodile.git
cd AStockCrocodile
```

5. Create the WSL Python environment:

```bash
sudo apt update
sudo apt install -y python3.10-venv python3-pip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

6. Install PyTorch with CUDA, then project dependencies:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -r requirements-finetune.txt
```

7. Copy or create `env.txt` in the WSL project root so the scripts can read MySQL. If MySQL runs on Windows, set `MYSQL_HOST` to the Windows host IP reachable from WSL, not always `127.0.0.1`.

8. Verify CUDA from Python:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

9. Build the fine-tuning dataset:

```bash
python ./llm_finetune/build_dataset.py --positive-limit 2000 --negative-ratio 1.0
```

10. Train the LoRA adapter:

```bash
python ./llm_finetune/train_lora.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --data-dir llm_finetune/data --output-dir llm_finetune/runs/deepseek-r1-distill-qwen-7b-lora --max-seq-length 4096 --batch-size 1 --gradient-accumulation-steps 8 --epochs 1
```

11. Evaluate and save only validated rules with actual success rate at least `40%` to `surgepatterns`:

```bash
python ./llm_finetune/evaluate_model.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --adapter-dir llm_finetune/runs/deepseek-r1-distill-qwen-7b-lora/adapter --data-dir llm_finetune/data --min-success-rate 0.40
```

Windows equivalent evaluation command:

```powershell
python .\llm_finetune\evaluate_model.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --adapter-dir llm_finetune\runs\deepseek-r1-distill-qwen-7b-lora\adapter --data-dir llm_finetune\data --min-success-rate 0.40
```

The fine-tuning pipeline is:

1. `build_dataset.py` creates supervised chat JSONL samples from `klinestatistics`, `dkandles`, and `wkandles`.
2. `train_lora.py` performs QLoRA/LoRA training and writes a PEFT adapter.
3. `evaluate_model.py` asks the fine-tuned model for feature-token patterns, backtests them with the existing K-line engine, and writes retained rules to `surgepatterns`.
4. Future selection still uses `llm_pattern_selector.py`, so the deployed selector remains deterministic and backtest-gated.

How raw K-line training is used for future selection:

1. In `raw-kline` mode, DeepSeek sees raw historical windows: the selection date plus the previous 54 daily bars, and the latest weekly bar on or before the selection date plus the previous 54 weekly bars.
2. DeepSeek proposes candidate setup patterns from those raw K-line samples.
3. The program converts the candidates into executable feature-token rules, validates them on the `20260101` to `20260430` test set, and writes only retained patterns to `surgepatterns`.
4. Future stock selection does not call DeepSeek again. It only reads the saved rules from `surgepatterns`, computes the same daily/weekly features for the target date, and matches stocks against those rules.

So raw training changes how candidate patterns are discovered, but future selection still uses deterministic saved rules such as:

```text
D_CLOSE_GT_MA5 && W_MA5_GT_MA13
```

Use saved LLM/DeepSeek patterns for future stock selection:

```powershell
python .\llm_pattern_selector.py --date 20260508
```

The selector reads validated rows from `surgepatterns`, extracts the same daily/weekly features for the target date, and saves matched stocks to `stockselection` with strategy name `llm_surge_pattern_v1`. Terminal and CSV output include the matched best pattern, its success rate, and its failure rate. The database `Reason` field also includes those values, for example `success=40.00% failure=60.00%`.

Useful filters:

- `--min-success-rate 0.35`: require at least this actual test-set success rate.
- `--min-sample-count 20`: require enough test-set occurrences.
- `--min-positive-support 5`: require enough training positive support.
- `--test-start-date 20260101 --test-end-date 20260430`: use patterns validated on a specific test range.
- `--limit 20`: keep only the top-ranked matches.
- `--output data\llm_pattern_selection.csv`: also write results to CSV.

## FinGPT-Forecaster 4-bit QLoRA 微调

项目新增独立目录 `fingpt_forecaster_qlora/`，用于按“FinGPT-Forecaster + 4-bit 量化 + QLoRA 微调”路线训练 A 股短线预测 adapter。

WSL2/Linux 一键 smoke 流程：

```bash
bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh smoke
```

如果只想继续使用已经生成好的数据集手动训练：

```bash
source $HOME/.venvs/astock-fingpt/bin/activate
cd /mnt/d/Documents/StockInfoCrawler

python -m fingpt_forecaster_qlora.train_qlora \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --no-forecaster-adapter \
  --data-dir fingpt_forecaster_qlora/data \
  --output-dir fingpt_forecaster_qlora/runs/smoke-qwen-0.5b \
  --max-seq-length 2048 \
  --epochs 0.05
```

训练完成后评估：

```bash
python -m fingpt_forecaster_qlora.evaluate \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir fingpt_forecaster_qlora/runs/smoke-qwen-0.5b/adapter \
  --data-dir fingpt_forecaster_qlora/data \
  --threshold 0.40 \
  --max-samples 100
```

正式训练：

```bash
bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh full
```

Windows 原生可先生成数据集，但 4-bit QLoRA 建议切到 WSL2/Linux：

```powershell
.\fingpt_forecaster_qlora\scripts\one_click_deploy.ps1 smoke
```

完整手册见 `fingpt_forecaster_qlora/README.md`。

## News Crawler

`news_crawler.py` crawls stock-market news from AkShare-backed public sources and writes rows to MySQL table `news`.

Default sources:

- `eastmoney`
- `ths`
- `caixin`
- `yicai`: Yicai
- `eeo`: Economic Observer
- `21jingji`: 21st Century Business Herald
- `caijing`: Caijing
- `ce`: China Economic Net
- `jwview`: JWView
- `stcn`: Securities Times
- `cnstock`: China Securities Journal
- `sina`: Sina Finance
- `xueqiu`: Xueqiu
- `jiemian`: Jiemian News
- `hexun`: Hexun
- `stockstar`: Stockstar

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
python .\news_crawler.py --sources eastmoney,ths,yicai,stcn,sina --limit 1000
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
