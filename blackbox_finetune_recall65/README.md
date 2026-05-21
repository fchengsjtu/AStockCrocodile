# Black-Box Qwen Fine-Tuning, Recall 65

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
- Required target: `positive_recall >= 65%`.

## One-Click Run

Windows smoke run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall65\scripts\one_click_deploy.ps1 smoke
```

Diagnose RTX3060/PyTorch CUDA before training:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall65\scripts\one_click_deploy.ps1 diagnose
```

On Windows the one-click script checks whether PyTorch can see CUDA. If the environment contains a CPU-only PyTorch build, it uninstalls it and installs CUDA-enabled PyTorch from `https://download.pytorch.org/whl/cu121` by default. To use another CUDA wheel index:

```powershell
$env:TORCH_CUDA_INDEX='https://download.pytorch.org/whl/cu124'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall65\scripts\one_click_deploy.ps1 diagnose
```

Windows full run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall65\scripts\one_click_deploy.ps1 full
```

The one-click scripts reuse existing `train.jsonl` and `test.jsonl` files in `blackbox_finetune_recall65/data` and `blackbox_finetune_recall65/data_validation`. After the first full dataset build, later full runs skip the expensive sample materialization step and go straight to training/evaluation.

Force a full dataset rebuild:

```powershell
$env:REBUILD_DATASET='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall65\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_DATASET
```

Force only the validation dataset rebuild:

```powershell
$env:REBUILD_VALIDATION_DATASET='1'
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune_recall65\scripts\one_click_deploy.ps1 full
Remove-Item Env:\REBUILD_VALIDATION_DATASET
```

WSL2/Linux full run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune_recall65/scripts/one_click_deploy.sh full
```

## Manual Commands

Build the training dataset:

```powershell
python -m blackbox_finetune_recall65.build_dataset `
  --start-date 20110101 `
  --end-date 20251231 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune_recall65/data `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Build the validation dataset:

```powershell
python -m blackbox_finetune_recall65.build_validation_dataset `
  --start-date 20260101 `
  --end-date 20260430 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune_recall65/data_validation `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 80
```

Train on WSL2/Linux with QLoRA:

```bash
python -m blackbox_finetune_recall65.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_recall65/data \
  --output-dir blackbox_finetune_recall65/runs/qwen2.5-0.5b-blackbox-recall65-lora \
  --max-seq-length 2048 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --cuda-device 0
```

Train on native Windows with 4-bit loading disabled:

```powershell
python -m blackbox_finetune_recall65.train `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --data-dir blackbox_finetune_recall65/data `
  --output-dir blackbox_finetune_recall65/runs/qwen2.5-0.5b-blackbox-recall65-lora `
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

Evaluate and enforce the `65%` positive-recall target:

```powershell
python -m blackbox_finetune_recall65.evaluate `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --adapter-dir blackbox_finetune_recall65/runs/qwen2.5-0.5b-blackbox-recall65-lora/adapter `
  --data-dir blackbox_finetune_recall65/data_validation `
  --threshold 0.50 `
  --min-positive-recall 0.65 `
  --max-seq-length 512 `
  --cuda-device 0
```

Predict all stocks for one trading day:

```powershell
python -m blackbox_finetune_recall65.predict_day `
  --date 20260514 `
  --adapter-dir blackbox_finetune_recall65/runs/qwen2.5-0.5b-blackbox-recall65-lora/adapter `
  --threshold 0.50 `
  --max-seq-length 512 `
  --cuda-device 0 `
  --limit 20 `
  --output data\blackbox_recall65_predictions_20260514.csv
```

## Tests

```powershell
python -m unittest tests.test_blackbox_finetune_recall_targets -v
python -m unittest discover -s tests -v
```
