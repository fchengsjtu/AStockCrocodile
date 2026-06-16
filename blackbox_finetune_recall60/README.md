# Black-Box Qwen Fine-Tuning, Recall 60

This directory contains an independent black-box fine-tuning task for A-share surge selection.

The model is `Qwen/Qwen2.5-0.5B-Instruct`. It is trained by LoRA/QLoRA parameter fine-tuning. It is not used as a rule miner and does not search explicit K-line features.

This target writes to its own output directory and uses its own training seed, so it produces independent adapter parameters from the other recall target directories.

- Positive samples come from `klinestatistics`.
- Positive anchor date is `PrevTradeDate`.
- Negative samples are trading days outside each positive sample's `PrevTradeDate +/- 3` trading-day window.
- Each positive sample input contains the anchor date plus the previous 55 daily K-lines and previous 55 weekly K-lines.
- Each negative sample input contains the negative trading day plus the previous 55 daily K-lines and previous 55 weekly K-lines.
- For Monday-Thursday anchor dates, the current week is represented by a temporary in-memory weekly K-line aggregated from Monday through the anchor date. This temporary K-line is used only for model input and is not written to MySQL.
- Training period: `20110101-20241231`.
- Validation period: `20260101-20260430`.
- Target metric: `precision@{PRECISION_TOP_K}`, meaning the positive hit rate among the model's top-k ranked validation samples.
- Required target: `precision@20 >= 30%` by default, configurable through `PRECISION_TOP_K` and `PRECISION_THRESHOLD`.
- Evaluation results are stored per target, for example `runs/qwen2.5-0.5b-blackbox-recall60-long-lora/evaluations/top20_precision030/evaluation.json`.

## One-Click Run

Windows smoke run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 smoke
```

Diagnose RTX3060/PyTorch CUDA before training:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 diagnose
```

On Windows the one-click script checks whether PyTorch can see CUDA. If the environment contains a CPU-only PyTorch build, it uninstalls it and installs CUDA-enabled PyTorch from `https://download.pytorch.org/whl/cu121` by default. To use another CUDA wheel index:

```powershell
$env:TORCH_CUDA_INDEX='https://download.pytorch.org/whl/cu124'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 diagnose
```

Windows full run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
```

The one-click scripts reuse existing `train.jsonl` and `test.jsonl` files in `blackbox_finetune_recall60/data_no_partial_week_recall60_long` and `blackbox_finetune_recall60/data_evaluation_no_partial_week_recall60_long`. After the first full dataset build, later full runs skip the expensive sample materialization step and go straight to training/evaluation.

Force a full dataset rebuild:

```powershell
$env:REBUILD_DATASET='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_DATASET
```

Force only the validation dataset rebuild:

```powershell
$env:REBUILD_VALIDATION_DATASET='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_VALIDATION_DATASET
```

Training also caches tokenized samples under `blackbox_finetune_recall60/data_no_partial_week_recall60_long/tokenized`. If `train.jsonl`, `BASE_MODEL`, and `MAX_SEQ_LENGTH` are unchanged, later training runs load the tokenized cache and skip the slow tokenization pass.

Force tokenization rebuild:

```powershell
$env:REBUILD_TOKEN_CACHE='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_TOKEN_CACHE
```

Training automatically resumes from the latest `blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-recall60-long-lora/checkpoints/update-*` checkpoint. `CHECKPOINT_EVERY` is the only in-training evaluation cadence: when a checkpoint is saved, training also immediately runs evaluation and writes a JSON file with `trigger=checkpoint` under the configured evaluation directory. If `EVAL_MAX_SAMPLES` is greater than 0, that checkpoint evaluation randomly samples that many rows from `test.jsonl` with a deterministic seed based on the checkpoint update. The old `EVAL_EVERY_EPOCH_FRACTION` setting is ignored. The log should show `resuming adapter from ...` and `start_update=N`. Disable automatic resume only when you intentionally want to restart from the base model:

WSL/Linux defaults to `ON_THE_FLY_TOKENIZE=1`, so training tokenizes each batch on demand instead of loading the large `tokenized/*.pkl` cache into RAM. This avoids Linux `Killed` exits when the tokenized cache is larger than available memory. Set `ON_THE_FLY_TOKENIZE=0` only when RAM is ample and you prefer faster cached token loading.

Optional high-scoring negative penalty is disabled by default. Enable it only when you want training to softly penalize negative samples whose current positive probability exceeds the dynamic false-positive cutoff. After every checkpoint evaluation, the cutoff is updated from `0.5 * (next_threshold + max_p)` with EMA smoothing and then clamped by the configured bounds:

```bash
export FP_DYNAMIC_PENALTY=1
export FP_PENALTY_WEIGHT=0.1
export FP_THRESHOLD_EMA_ALPHA=0.2
export FP_THRESHOLD_MIN=0.45
export FP_THRESHOLD_MAX=0.65
```

With the switch off, training loss is unchanged. The penalty requires on-the-fly raw samples, so keep `ON_THE_FLY_TOKENIZE=1` when using it.

```powershell
$env:NO_AUTO_RESUME='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
Remove-Item Env:\NO_AUTO_RESUME
```

If a few 2048-token batches hit CUDA OOM on Windows, training now clears the CUDA cache, skips that micro batch, and continues. Abort happens only after `OOM_PATIENCE` consecutive OOM batches. If OOM persists, reduce sequence length and reuse the same checkpoint:

```powershell
$env:MAX_SEQ_LENGTH='1024'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
Remove-Item Env:\MAX_SEQ_LENGTH
```

The default learning rate is `1e-5`. When non-finite loss or gradient appears, training counts total skipped batches and automatically halves the optimizer learning rate every 10 skips down to `1e-6`. If total non-finite skips reach `NONFINITE_SKIP_LIMIT` (default `100`), training stops so you can resume from an earlier checkpoint.

For a more conservative resume:

```powershell
$env:LEARNING_RATE='5e-6'
$env:RESUME_ADAPTER_DIR='blackbox_finetune_recall60\runs\qwen2.5-0.5b-blackbox-recall60-long-lora\checkpoints\update-012000'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall60\scripts\one_click_deploy.ps1 full
Remove-Item Env:\LEARNING_RATE
Remove-Item Env:\RESUME_ADAPTER_DIR
```

WSL2/Linux full run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune_recall60/scripts/one_click_deploy.sh full
```

### Use Existing Dataset Files

By default, the deployment script builds or reuses datasets under `DATA_DIR` and `VALIDATION_DATA_DIR`. To bypass dataset generation and use three existing JSONL files, set all three variables together:

```bash
export TRAIN_DATASET_PATH=/path/to/train.jsonl
export TEST_DATASET_PATH=/path/to/test.jsonl
export VALIDATION_DATASET_PATH=/path/to/validation.jsonl
bash "$BLACKBOX_RECALL_DIR/scripts/one_click_deploy.sh" full
```

Windows paths such as `D:\Models\dataset\train.jsonl` are accepted when the script runs inside WSL. The script creates lightweight symbolic links under `$OUTPUT_DIR/input_datasets`; it does not copy or modify the source files. `EXPLICIT_DATASET_WORK_DIR` can override that runtime link directory.

The three variables must be supplied together. If none is set, the original database sampling, dataset cache, and rebuild behavior remains unchanged.

## Manual Commands

Build the training dataset:

```powershell
python -m blackbox_finetune_recall60.build_dataset `
  --start-date 20110101 `
  --end-date 20241231 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune_recall60/data_no_partial_week_recall60_long `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Build the validation dataset:

```powershell
python -m blackbox_finetune_recall60.build_validation_dataset `
  --start-date 20260101 `
  --end-date 20260430 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune_recall60/data_evaluation_no_partial_week_recall60_long `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Train on WSL2/Linux with QLoRA:

```bash
python -m blackbox_finetune_recall60.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_recall60/data_no_partial_week_recall60_long \
  --output-dir blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-recall60-long-lora \
  --max-seq-length 2048 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --cuda-device 0
```

Train on native Windows with 4-bit loading disabled:

```powershell
python -m blackbox_finetune_recall60.train `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --data-dir blackbox_finetune_recall60/data_no_partial_week_recall60_long `
  --output-dir blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-recall60-long-lora `
  --max-seq-length 2048 `
  --epochs 1 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-4 `
  --cuda-device 0 `
  --no-4bit
```

Training, evaluation, and prediction now bind CUDA device `0` by default and verify that the visible CUDA device name contains `RTX3060` or `RTX 3060`. Set `CUDA_DEVICE=0` before the one-click scripts if the RTX3060 is not the first GPU. For non-RTX3060 development machines, add `--allow-non-rtx3060` to the manual Python commands.

If native Windows reports `os error 1455` or `妞ょ敻娼伴弬鍥︽婢额亜鐨敍灞炬￥濞夋洖鐣幋鎰惙娴ｆ竴 while loading Qwen, increase the Windows page file size or run the full training in WSL2/Linux. The dataset and validation builders are lightweight, but model loading can still require several GB of RAM plus page file space even for Qwen2.5-0.5B.

Evaluate and enforce the configurable precision target:

```powershell
python -m blackbox_finetune_recall60.evaluate `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --adapter-dir blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-recall60-long-lora/adapter `
  --data-dir blackbox_finetune_recall60/data_evaluation_no_partial_week_recall60_long `
  --threshold 0.50 `
  --precision-top-k 20 `
  --precision-threshold 0.30 `
  --output-dir blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-recall60-long-lora/evaluations/top20_precision030 `
  --max-seq-length 512 `
  --cuda-device 0
```

Predict all stocks for one trading day:

```powershell
python -m blackbox_finetune_recall60.predict_day `
  --date 20260514 `
  --adapter-dir blackbox_finetune_recall60/runs/qwen2.5-0.5b-blackbox-recall60-long-lora/adapter `
  --threshold 0.50 `
  --max-seq-length 512 `
  --cuda-device 0 `
  --limit 20 `
  --output data\blackbox_recall60_predictions_20260514.csv
```

## Tests

```powershell
python -m unittest tests.test_blackbox_finetune_recall60 -v
python -m unittest discover -s tests -v
```
