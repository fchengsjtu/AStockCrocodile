# StockInfoCrawler

A-share daily K-line crawler. The current version fetches forward-adjusted (`qfq`) daily K-line data from Tencent via AkShare and writes directly to the cloud MySQL database `emstocks`.

It no longer writes CSV files during crawling, and only `daily` K-line import is supported.

## Environment Variables

Create `env.txt` in the project root. The program reads this file automatically before connecting to MySQL.

```powershell
$env:MYSQL_HOST='127.0.0.1'
$env:MYSQL_PORT='3306'
$env:MYSQL_USER='your-mysql-user'
$env:MYSQL_PASSWORD='your-mysql-password'
$env:MYSQL_DATABASE='emstocks'
```

`env.txt` is ignored by git because it contains credentials. The parser also accepts normal `NAME=value` lines and the PowerShell-style `$env:NAME='value'` lines used on Windows.

All project-level environment variables are collected here:

| Variable | Used by | Default | Purpose |
| --- | --- | --- | --- |
| `MYSQL_HOST` | crawler, importers, strategy/backtest tools | `127.0.0.1` in most tools | MySQL host. In WSL, use `WSL_MYSQL_HOST` if Windows MySQL is not reachable as `127.0.0.1`. |
| `MYSQL_PORT` | MySQL tools | `3306` | MySQL port. |
| `MYSQL_USER` | MySQL tools | varies by script | MySQL user. |
| `MYSQL_PASSWORD` | MySQL tools | empty | MySQL password. |
| `MYSQL_DATABASE` | MySQL tools | `emstocks` | MySQL database. |
| `WSL_MYSQL_HOST` | WSL/Linux MySQL helpers | auto-detected Windows host | Override MySQL host from WSL. |
| `DKANDLES_KTYPE` | `a_share_crawler.py` | `D` | Daily K-line `KType` written to `dkandles`. |
| `MKANDLES_KTYPE` | `mysql_importer/import_daily_to_mysql.py` | `M` | Importer `KType` when importing monthly rows. |
| `LOCAL_LLM_BASE_URL` | local LLM pattern tools | `http://127.0.0.1:1234/v1` | OpenAI-compatible local LLM base URL. |
| `LOCAL_LLM_MODEL` | local LLM pattern tools | local DeepSeek/Qwen default | Local model name. |
| `LOCAL_LLM_API_KEY` | local LLM pattern tools | `local` | API key placeholder for local OpenAI-compatible servers. |
| `LOCAL_LLM_TIMEOUT` | local LLM pattern tools | `600` | LLM request timeout in seconds. |
| `LOCAL_LLM_MAX_TOKENS` | local LLM pattern tools | `2048` | Max response tokens for LLM calls. |
| `LOCAL_LLM_RESPONSE_FORMAT` | local LLM pattern tools | off | Enable JSON response-format request when the local server supports it. |
| `DEEPSEEK_BASE_URL` | local LLM pattern tools | fallback only | Backward-compatible alias when `LOCAL_LLM_BASE_URL` is unset. |
| `DEEPSEEK_MODEL` | local LLM pattern tools | fallback only | Backward-compatible alias when `LOCAL_LLM_MODEL` is unset. |
| `PYTHON_BIN` | one-click scripts | `python`/`python3` or `.venv` Python | Python executable used to create or run a venv. |
| `VENV_DIR` | one-click scripts | script-specific `.venv-*` path | Virtual environment directory. |
| `BASE_MODEL` | fine-tuning one-click scripts | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace-format base model or local model directory. |
| `DATA_DIR` | fine-tuning one-click scripts | mode-specific path | Training dataset directory. For recallXX scripts, the default includes `SAMPLE_MODE`, such as `data_no_partial_week_long` or `data_no_partial_week_xlong`. |
| `VALIDATION_DATA_DIR` | blackbox one-click scripts | mode-specific path | Validation/evaluation dataset directory. For recallXX scripts, the default includes `SAMPLE_MODE`, such as `data_evaluation_no_partial_week_long`. |
| `OUTPUT_DIR` | fine-tuning one-click scripts | script-specific `runs/...` path | Adapter/checkpoint output directory. For recallXX scripts, default is split by `SAMPLE_MODE`: `...-short-lora`, `...-long-lora`, `...-xlong-lora`, or `...-xxlong-lora`. |
| `CUDA_DEVICE` | blackbox recallXX scripts | `0` | CUDA device id, normally the RTX 3060. |
| `CUDA_VISIBLE_DEVICES` | GPU tools | set from `CUDA_DEVICE` | CUDA visibility binding. Usually do not set directly. |
| `PYTORCH_CUDA_ALLOC_CONF` | GPU tools | `expandable_segments:True` on Linux scripts | PyTorch CUDA allocator tuning. |
| `TORCH_CUDA_INDEX` | Windows recallXX scripts | `https://download.pytorch.org/whl/cu121` | CUDA PyTorch wheel index if the script needs to install GPU PyTorch. |
| `SAMPLE_MODE` | `blackbox_finetune_recallXX` | `long` | `short`: 8日K+5周K, `MAX_SEQ_LENGTH=1024`; `long`: 13日K+8周K+5月K, `MAX_SEQ_LENGTH=2048`; `xlong`: 21日K+13周K+8月K, `MAX_SEQ_LENGTH=3072`; `xxlong`: 34日K+21周K+13月K, `MAX_SEQ_LENGTH=4096`. |
| `MAX_SEQ_LENGTH` | fine-tuning scripts | mode/script-specific | Override token length. For recallXX, omit unless intentionally overriding `SAMPLE_MODE` default. |
| `NEGATIVE_RATIO` | dataset builders | recallXX default `3.0`; older fine-tune default `1.0` | Negative samples per positive sample. |
| `SAMPLE_BOTTOM_BAND_RATIO` | recallXX dataset builders | `0.10` | Bottom-band filter ratio. `short`/`long` require anchor daily close in the bottom band of the weekly K-line range; `xlong`/`xxlong` use the monthly K-line range. |
| `TRAIN_START_DATE` | recallXX dataset builders and one-click scripts | target/mode-specific | Training sample start date, format `YYYYMMDD`. |
| `TRAIN_END_DATE` | recallXX dataset builders and one-click scripts | target/mode-specific | Training sample end date, format `YYYYMMDD`. |
| `VALIDATION_START_DATE` | recallXX validation builders and one-click scripts | target/mode-specific | Validation/test sample start date, format `YYYYMMDD`. Takes priority over `TEST_START_DATE`. |
| `VALIDATION_END_DATE` | recallXX validation builders and one-click scripts | target/mode-specific | Validation/test sample end date, format `YYYYMMDD`. Takes priority over `TEST_END_DATE`. |
| `TEST_START_DATE` | recallXX validation builders and one-click scripts | unset | Alias for validation/test sample start date when `VALIDATION_START_DATE` is unset. |
| `TEST_END_DATE` | recallXX validation builders and one-click scripts | unset | Alias for validation/test sample end date when `VALIDATION_END_DATE` is unset. |
| `POSITIVE_LIMIT` | one-click full mode | empty | Limit positive samples in full dataset builds. Empty means no limit. |
| `SMOKE_POSITIVE_LIMIT` | one-click smoke mode | script-specific, often `12` or `200` | Limit positive samples for smoke runs. |
| `DATA_BATCH_SIZE` | `llm_finetune` dataset script | `30` | Dataset materialization batch size. |
| `REBUILD_DATASET` | recallXX one-click scripts | off | Force rebuilding cached training datasets. |
| `REBUILD_VALIDATION_DATASET` | recallXX one-click scripts | falls back to `REBUILD_DATASET` | Force rebuilding cached validation datasets. |
| `REBUILD_TOKEN_CACHE` | recallXX one-click scripts | off | Force re-tokenization instead of using tokenized cache. |
| `NO_AUTO_RESUME` | recallXX one-click scripts | off | Disable automatic resume from latest checkpoint. |
| `RESUME_ADAPTER_DIR` | recallXX one-click scripts | empty | Explicit adapter checkpoint directory to resume from. |
| `CHECKPOINT_EVERY` | recallXX one-click scripts/trainers | `500` | Save adapter checkpoint every N optimizer updates. Applies to both `SAMPLE_MODE=short` and `SAMPLE_MODE=long` unless overridden. |
| `EPOCHS` | fine-tuning one-click scripts | smoke/full script-specific | Training epochs. |
| `BATCH_SIZE` | `llm_finetune` one-click scripts | `1` | Per-device batch size. |
| `GRADIENT_ACCUMULATION_STEPS` | fine-tuning one-click scripts | smoke/full script-specific | Gradient accumulation steps. |
| `LEARNING_RATE` | fine-tuning one-click scripts | recall60-style `5e-6`, recall80 `2e-5`, older tools vary | Training learning rate. |
| `TRAIN_SEED` | recallXX one-click scripts | `20260500 + recall target` | Target-specific training seed. |
| `MAX_GRAD_NORM` | recallXX one-click scripts | `0.5` | Gradient clipping norm. |
| `OOM_PATIENCE` | recallXX one-click scripts | `20` | Abort after this many consecutive CUDA OOM skips. |
| `MIN_SEQ_LENGTH_ON_OOM` | recall80 train script | `512` | Smallest sequence length allowed by automatic OOM shrinking. |
| `OOM_SHRINK_FACTOR` | recall80 train script | `0.5` | Sequence-length multiplier after repeated OOM. |
| `NONFINITE_SKIP_LIMIT` | recallXX one-click scripts | `100` | Abort after this many non-finite losses/gradients. |
| `NONFINITE_BACKOFF_EVERY` | recallXX one-click scripts | `10` | Reduce LR after every N non-finite skips. |
| `LR_BACKOFF_FACTOR` | recallXX one-click scripts | `0.5` | Learning-rate multiplier during non-finite backoff. |
| `MIN_LEARNING_RATE` | recallXX one-click scripts | `1e-6` | Lower bound for automatic LR backoff. |
| `MIN_POSITIVE_RECALL` | blackbox recall scripts | recall target, e.g. `0.60` | Required true-positive recall during evaluation. |
| `MIN_SUCCESS_RATE` | `llm_finetune` one-click scripts | `0.20` | Required selected-positive success rate. |
| `EVAL_MAX_SAMPLES` | `llm_finetune` one-click scripts | `200` | Max evaluation samples for one-click evaluation. |

```powershell
# Example overrides
$env:SAMPLE_MODE='short'
$env:NEGATIVE_RATIO='3.0'
$env:SAMPLE_BOTTOM_BAND_RATIO='0.10'
$env:CHECKPOINT_EVERY='500'
$env:LOCAL_LLM_BASE_URL='http://127.0.0.1:1234/v1'
```

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

## Qwen2.5 Stock Fine-Tuning

The current fine-tuning pipeline is in `llm_finetune/`. It uses `Qwen/Qwen2.5-0.5B-Instruct` as the base model. Samples come from `klinestatistics`: `PrevTradeDate` is the anchor date, and each input contains the 55 daily K-lines and 55 weekly K-lines ending at or before that date. The dataset is split into 80% training and 20% testing. Evaluation fails if model-selected positives have a success rate below `20%`.

One-click deploy, train, evaluate, and run tests in WSL2/Linux:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash llm_finetune/scripts/one_click_deploy.sh smoke
```

Formal run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash llm_finetune/scripts/one_click_deploy.sh full
```

Windows smoke run:

```powershell
cd D:\Documents\StockInfoCrawler
.\llm_finetune\scripts\one_click_deploy.ps1 smoke
```

Build the 80/20 dataset manually:

```bash
python -m llm_finetune.build_dataset \
  --output-dir llm_finetune/data \
  --stat-type short_term_surge_3d_20pct \
  --positive-limit 2000 \
  --negative-ratio 1.0 \
  --daily-window 55 \
  --weekly-window 55 \
  --batch-size 30
```

Train:

```bash
python -m llm_finetune.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir llm_finetune/data \
  --output-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora \
  --max-seq-length 2048 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4
```

Evaluate and enforce at least `20%` success rate:

```bash
python -m llm_finetune.evaluate \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora/adapter \
  --data-dir llm_finetune/data \
  --threshold 0.40 \
  --min-success-rate 0.20 \
  --max-samples 500
```

Predict one stock/date:

```bash
python -m llm_finetune.predict \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir llm_finetune/runs/qwen2.5-0.5b-stock-lora/adapter \
  --scode 000001 \
  --date 20260512
```

Run automated tests:

```bash
python -m unittest tests.test_llm_finetune -v
```

The old `fingpt_forecaster_qlora/` entry has been retired; its README points back to this pipeline.

## Black-Box Qwen Fine-Tuning

The black-box fine-tuning pipeline is in `blackbox_finetune/`. It treats `Qwen/Qwen2.5-0.5B-Instruct` as a trainable classifier rather than a rule generator. Positive samples come from `klinestatistics`. For each positive sample, the input is the `PrevTradeDate` plus the previous 55 daily K-lines and previous 55 weekly K-lines. Positive samples are deduplicated per stock with a 20-trading-day cooldown, keeping the earliest signal in each cluster. Negative candidates are trading days outside each positive sample's `PrevTradeDate +/- 20` trading-day window, using the same K-line input format. The default training period is `20110101-20241231`; the default validation period is `20260101-20260430`. Evaluation fails unless positive recall is at least `60%`.

The `blackbox_finetune_recallXX/` pipelines support two compact sample modes:

- `long` default: 13 daily K-lines, 8 weekly K-lines, 5 monthly K-lines, default `MAX_SEQ_LENGTH=2048`. Samples are kept only when all 5 monthly K-lines exist and the anchor daily close is in the bottom band of the weekly K-line price range.
- `short`: 8 daily K-lines and 5 weekly K-lines, default `MAX_SEQ_LENGTH=1024`. Samples are kept only when the latest 5 weekly K-lines exist, each has `ma13`, and the anchor daily close is in the bottom band of the weekly K-line price range.
- `xlong`: 21 daily K-lines, 13 weekly K-lines, 8 monthly K-lines, default `MAX_SEQ_LENGTH=3072`. Samples are kept only when all 8 monthly K-lines exist and the anchor daily close is in the bottom band of the monthly K-line price range.
- `xxlong`: 34 daily K-lines, 21 weekly K-lines, 13 monthly K-lines, default `MAX_SEQ_LENGTH=4096`. Samples are kept only when all 13 monthly K-lines exist and the anchor daily close is in the bottom band of the monthly K-line price range.

Use `SAMPLE_MODE=short`, `SAMPLE_MODE=long`, `SAMPLE_MODE=xlong`, or `SAMPLE_MODE=xxlong` with the recallXX one-click scripts. Set `MAX_SEQ_LENGTH` only when you intentionally want to override the mode default.
Set `SAMPLE_BOTTOM_BAND_RATIO` to control the bottom-band filter. The default is `0.10`, meaning bottom 10% of the relevant weekly or monthly price range.

The `xlong` encoding was checked with 1000 materialized samples from `20240101-20251231` using the Qwen2.5-0.5B tokenizer. Full chat prompt length averaged about `2180` tokens and maxed at `2242`, so the default `MAX_SEQ_LENGTH=3072` has enough room for the current compact CSV prompt format.

The `xxlong` encoding was checked with the same tokenizer and sample range. Full chat prompt length averaged about `3513` tokens and maxed at `3588`, so it needs the default `MAX_SEQ_LENGTH=4096`; `3072` is not enough for this window.

Set `NEGATIVE_RATIO` to control the negative-sample multiplier. The default is `3.0`, meaning three negative samples per positive sample. All recallXX environment overrides are listed in [Environment Variables](#environment-variables).

The recallXX one-click scripts print only project-related environment variables after the CUDA check. They do not print unrelated system variables such as `Path` or `APPDATA`. Dataset and token cache reuse follows these rules:

- If `train.jsonl` and `test.jsonl` already exist under `DATA_DIR`, the training dataset is reused unless `REBUILD_DATASET=1/true/yes` is set.
- If validation files already exist under `VALIDATION_DATA_DIR`, the validation dataset is reused unless `REBUILD_VALIDATION_DATASET=1/true/yes` or `REBUILD_DATASET=1/true/yes` is set.
- By default, recallXX datasets are separated by sample mode: `data_no_partial_week_short`, `data_no_partial_week_long`, `data_no_partial_week_xlong`, and `data_no_partial_week_xxlong`, with matching `data_evaluation_no_partial_week_*` validation directories.
- Tokenized samples are reused when `train.jsonl`, `BASE_MODEL`, and `MAX_SEQ_LENGTH` are unchanged. Rebuilding `train.jsonl` changes its timestamp/size fingerprint, so tokenization is rebuilt too.
- `REBUILD_TOKEN_CACHE=1/true/yes` forces tokenization rebuild even when the tokenized cache exists.

`predict_day` uses the same `SAMPLE_MODE` window and bottom-band filters before model scoring. Stocks with incomplete weekly/monthly windows or a close outside `SAMPLE_BOTTOM_BAND_RATIO` are skipped without running inference, and each batch prints `skipped_by_sample_rule`.

Windows one-click smoke run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune\scripts\one_click_deploy.ps1 smoke
```

Windows full run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune\scripts\one_click_deploy.ps1 full
```

WSL2/Linux one-click smoke run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune/scripts/one_click_deploy.sh smoke
```

Build the training dataset manually:

```powershell
python -m blackbox_finetune.build_dataset `
  --start-date 20110101 `
  --end-date 20241231 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune/data `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Build the validation dataset manually:

```powershell
python -m blackbox_finetune.build_validation_dataset `
  --start-date 20260101 `
  --end-date 20260430 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune/data_validation `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Train with LoRA/QLoRA:

```powershell
python -m blackbox_finetune.train `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --data-dir blackbox_finetune/data `
  --output-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora `
  --max-seq-length 2048 `
  --epochs 1 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-4
```

On native Windows CPU/GPU, add `--no-4bit` because `bitsandbytes` 4-bit training is best supported in Linux/WSL2:

```powershell
python -m blackbox_finetune.train `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --data-dir blackbox_finetune/data `
  --output-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora `
  --max-seq-length 2048 `
  --epochs 1 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-4 `
  --no-4bit
```

Evaluate on the `20260101-20260430` validation dataset and enforce `60%` positive recall:

```powershell
python -m blackbox_finetune.evaluate `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --adapter-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora/adapter `
  --data-dir blackbox_finetune/data_validation `
  --threshold 0.50 `
  --min-positive-recall 0.60
```

Predict all stocks for one trading day:

```powershell
python -m blackbox_finetune.predict_day `
  --date 20260514 `
  --adapter-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora/adapter `
  --threshold 0.50 `
  --limit 20 `
  --output data\blackbox_predictions_20260514.csv
```

Run automated tests:

```powershell
python -m unittest tests.test_blackbox_finetune -v
```

## Goal Pattern Search

Use `klinestatistics` samples from `20200101` to `20251231`, split them into 80% training and 20% internal evaluation, then validate the retained pattern on `20260101` to `20260430`. The script writes retained patterns to `surgepatterns`.

```powershell
python .\signature_pattern_search.py `
  --train-start-date 20200101 `
  --train-end-date 20251231 `
  --holdout-start-date 20260101 `
  --holdout-end-date 20260430 `
  --target-success-rate 0.40 `
  --min-eval-success-rate 0.25 `
  --min-eval-sample-count 5 `
  --min-holdout-sample-count 5 `
  --min-positive-supports 200,100,50,20,10,5,3,2,1 `
  --batch-size 500
```

Verified result on the current local database:

```text
SampleCount=12
SuccessCount=8
SuccessRate=66.67%
PositiveSupport=24
```

Select stocks for one trading day from all stocks using the current validated mode:

```powershell
python .\llm_pattern_selector.py `
  --date 20260105 `
  --min-success-rate 0.40 `
  --min-threshold 0.40 `
  --min-sample-count 5 `
  --min-positive-support 20 `
  --train-start-date 20200101 `
  --train-end-date 20251231 `
  --test-start-date 20260101 `
  --test-end-date 20260430 `
  --limit 20
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

See [Environment Variables](#environment-variables) for the complete `env.txt` format and supported values.

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

Daily incremental workflow with ex-rights/forward-adjustment check first:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_daily_incremental_with_exrights.ps1
```

This script first runs `python .\a_share_crawler.py exrights`. If new or changed ex-rights rows are found, the crawler refreshes that stock's forward-adjusted daily K-lines from `2010-01-01` and rebuilds weekly/monthly K-lines for the affected stock. It then runs `python .\a_share_crawler.py run --mode incremental --period daily`.

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

Run portfolio backtest with a trained black-box model. Use the black-box virtual environment because the main `.venv` does not install PyTorch:

```powershell
.\.venv-blackbox-finetune-recall30\Scripts\python.exe -m portfolio_backtest.run `
  --strategy-name blackbox_finetune_recall30 `
  --blackbox-threshold 0.50 `
  --blackbox-max-seq-length 512 `
  --blackbox-cuda-device 0 `
  --limit-per-day 5
```

Schedule recall80 prediction for the previous calendar day, running Tuesday through Saturday at 05:00 local time:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\install_recall80_weekday_prediction_task.ps1
```

The scheduled task runs `run_recall80_previous_day_prediction.ps1`. Because it runs after midnight, the prediction date defaults to the previous calendar day: Tuesday 08:00 predicts Monday, Wednesday 08:00 predicts Tuesday, and Saturday 08:00 predicts Friday. The default task uses `SAMPLE_MODE=long`, `MAX_SEQ_LENGTH=2048`, and `threshold=0.80`. Before prediction, it runs `blackbox_finetune_recall80.evaluate` against `blackbox_finetune_recall80\data_evaluation_no_partial_week_long` and requires `positive_recall >= 0.80`; if the validation gate fails, prediction is not written. When the gate passes, the script saves the top 20 predictions to MySQL and also writes a dated CSV under `data\`.

Black-box `predict_day` saves the top 5 ranked predictions to MySQL table `blackbox_predictions` by default, including the strategy name:

```powershell
.\.venv-blackbox-finetune-recall30\Scripts\python.exe -m blackbox_finetune_recall30.predict_day `
  --date 20260514 `
  --threshold 0.50 `
  --max-seq-length 512 `
  --cuda-device 0
```

Track a live/paper portfolio from the first saved prediction date for a strategy. It uses the same T+1, stop-loss, take-profit, and day-3 exit rules as `portfolio_backtest.run`, starts with `1000000` cash by default, and writes daily value, holdings, and trades to the portfolio tables with `BacktestName=blackbox_prediction_tracker_v1`:

```powershell
.\.venv-blackbox-finetune-recall30\Scripts\python.exe -m portfolio_backtest.track_blackbox `
  --strategy-name blackbox_finetune_recall30
```

Daily after-close automation runs the full daily workflow at 16:00: incremental daily K-line crawl, weekly K-line generation on Fridays, monthly K-line generation on the last trading day of the month, black-box recallXX top-5 predictions written to `blackbox_predictions`, and prediction portfolio tracking.

Run it manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_daily_after_close.ps1
```

Install the Windows scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_daily_after_close_task.ps1
```

Useful manual options:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_daily_after_close.ps1 `
  -TradeDate 20260522 `
  -BlackboxTopN 5 `
  -BlackboxThreshold 0.50 `
  -CrawlerWorkers 8
```

The script skips missing recallXX directories or strategies whose trained adapter is not present, so unfinished models do not block the daily K-line update.

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

For the local model server, configure `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, and `LOCAL_LLM_API_KEY` as described in [Environment Variables](#environment-variables).

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

For slow local models, tune `LOCAL_LLM_TIMEOUT` and `LOCAL_LLM_MAX_TOKENS`; both are listed in [Environment Variables](#environment-variables).

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
