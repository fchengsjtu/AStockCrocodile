# Black-Box Qwen Fine-Tuning, Recall 75

This directory contains an independent black-box fine-tuning task for A-share surge selection.

The model is `Qwen/Qwen2.5-0.5B-Instruct`. It is trained by LoRA/QLoRA parameter fine-tuning. It is not used as a rule miner and does not search explicit K-line features.

This target writes to its own output directory and uses its own training seed, so it produces independent adapter parameters from the other recall target directories.

- Positive samples come from `klinestatistics`.
- Positive anchor date is `PrevTradeDate`.
- Negative samples are trading days outside each positive sample's `PrevTradeDate +/- 3` trading-day window.
- Each positive sample input contains the anchor date plus the previous 55 daily K-lines and previous 55 weekly K-lines.
- Each negative sample input contains the negative trading day plus the previous 55 daily K-lines and previous 55 weekly K-lines.
- Training period: `20110101-20251231`.
- Validation period: `20260101-20260430`.
- Target metric: positive recall, meaning the correctness rate when the sample is actually positive.
- Required target: `positive_recall >= 75%`.

## One-Click Run

Windows smoke run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 smoke
```

Diagnose RTX3060/PyTorch CUDA before training:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 diagnose
```

On Windows the one-click script checks whether PyTorch can see CUDA. If the environment contains a CPU-only PyTorch build, it uninstalls it and installs CUDA-enabled PyTorch from `https://download.pytorch.org/whl/cu121` by default. To use another CUDA wheel index:

```powershell
$env:TORCH_CUDA_INDEX='https://download.pytorch.org/whl/cu124'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 diagnose
```

Windows full run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
```

The one-click scripts reuse existing `train.jsonl` and `test.jsonl` files in `blackbox_finetune_recall75/data` and `blackbox_finetune_recall75/data_validation`. After the first full dataset build, later full runs skip the expensive sample materialization step and go straight to training/evaluation.

Force a full dataset rebuild:

```powershell
$env:REBUILD_DATASET='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_DATASET
```

Force only the validation dataset rebuild:

```powershell
$env:REBUILD_VALIDATION_DATASET='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_VALIDATION_DATASET
```

Training also caches tokenized samples under `blackbox_finetune_recall75/data/tokenized`. If `train.jsonl`, `BASE_MODEL`, and `MAX_SEQ_LENGTH` are unchanged, later training runs load the tokenized cache and skip the slow tokenization pass.

Force tokenization rebuild:

```powershell
$env:REBUILD_TOKEN_CACHE='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_TOKEN_CACHE
```

Training automatically resumes from the latest `blackbox_finetune_recall75/runs/qwen2.5-0.5b-blackbox-recall75-lora/checkpoints/update-*` checkpoint. The log should show `resuming adapter from ...` and `start_update=N`. Disable automatic resume only when you intentionally want to restart from the base model:

```powershell
$env:NO_AUTO_RESUME='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
Remove-Item Env:\NO_AUTO_RESUME
```

If a few 2048-token batches hit CUDA OOM on Windows, training now clears the CUDA cache, skips that micro batch, and continues. Abort happens only after `OOM_PATIENCE` consecutive OOM batches. If OOM persists, reduce sequence length and reuse the same checkpoint:

```powershell
$env:MAX_SEQ_LENGTH='1024'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
Remove-Item Env:\MAX_SEQ_LENGTH
```

The default learning rate is `1e-5`. When non-finite loss or gradient appears, training counts total skipped batches and automatically halves the optimizer learning rate every 10 skips down to `1e-6`. If total non-finite skips reach `NONFINITE_SKIP_LIMIT` (default `100`), training stops so you can resume from an earlier checkpoint.

For a more conservative resume:

```powershell
$env:LEARNING_RATE='5e-6'
$env:RESUME_ADAPTER_DIR='blackbox_finetune_recall75\runs\qwen2.5-0.5b-blackbox-recall75-lora\checkpoints\update-012000'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall75\scripts\one_click_deploy.ps1 full
Remove-Item Env:\LEARNING_RATE
Remove-Item Env:\RESUME_ADAPTER_DIR
```

WSL2/Linux full run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune_recall75/scripts/one_click_deploy.sh full
```

## Manual Commands

Build the training dataset:

```powershell
python -m blackbox_finetune_recall75.build_dataset `
  --start-date 20110101 `
  --end-date 20251231 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune_recall75/data `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Build the validation dataset:

```powershell
python -m blackbox_finetune_recall75.build_validation_dataset `
  --start-date 20260101 `
  --end-date 20260430 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune_recall75/data_validation `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Train on WSL2/Linux with QLoRA:

```bash
python -m blackbox_finetune_recall75.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_recall75/data \
  --output-dir blackbox_finetune_recall75/runs/qwen2.5-0.5b-blackbox-recall75-lora \
  --max-seq-length 2048 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --cuda-device 0
```

Train on native Windows with 4-bit loading disabled:

```powershell
python -m blackbox_finetune_recall75.train `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --data-dir blackbox_finetune_recall75/data `
  --output-dir blackbox_finetune_recall75/runs/qwen2.5-0.5b-blackbox-recall75-lora `
  --max-seq-length 2048 `
  --epochs 1 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-4 `
  --cuda-device 0 `
  --no-4bit
```

Training, evaluation, and prediction now bind CUDA device `0` by default and verify that the visible CUDA device name contains `RTX3060` or `RTX 3060`. Set `CUDA_DEVICE=0` before the one-click scripts if the RTX3060 is not the first GPU. For non-RTX3060 development machines, add `--allow-non-rtx3060` to the manual Python commands.

If native Windows reports `os error 1455` or `页面文件太小，无法完成操作` while loading Qwen, increase the Windows page file size or run the full training in WSL2/Linux. The dataset and validation builders are lightweight, but model loading can still require several GB of RAM plus page file space even for Qwen2.5-0.5B.

Evaluate and enforce the `75%` positive-recall target:

```powershell
python -m blackbox_finetune_recall75.evaluate `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --adapter-dir blackbox_finetune_recall75/runs/qwen2.5-0.5b-blackbox-recall75-lora/adapter `
  --data-dir blackbox_finetune_recall75/data_validation `
  --threshold 0.50 `
  --min-positive-recall 0.75 `
  --max-seq-length 512 `
  --cuda-device 0
```

Predict all stocks for one trading day:

```powershell
python -m blackbox_finetune_recall75.predict_day `
  --date 20260514 `
  --adapter-dir blackbox_finetune_recall75/runs/qwen2.5-0.5b-blackbox-recall75-lora/adapter `
  --threshold 0.50 `
  --max-seq-length 512 `
  --cuda-device 0 `
  --limit 20 `
  --output data\blackbox_recall75_predictions_20260514.csv
```

## Tests

```powershell
python -m unittest tests.test_blackbox_finetune_recall_targets -v
python -m unittest discover -s tests -v
```
