# Black-Box Qwen Fine-Tuning

This directory contains the black-box fine-tuning pipeline for A-share surge selection.

The model is `Qwen/Qwen2.5-0.5B-Instruct`. It is trained as a classifier, not as a K-line rule generator.

- Positive samples come from `klinestatistics`.
- Positive anchor date is `PrevTradeDate`.
- Negative samples are trading days outside each positive sample's `PrevTradeDate +/- 3` trading-day window.
- Each sample input contains 55 daily K-lines and 55 weekly K-lines ending at the anchor date.
- Default training period: `20110101-20241231`.
- Default validation period: `20260101-20260430`.
- Evaluation fails unless positive recall is at least `60%`.

## One-Click Commands

Windows smoke run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune\scripts\one_click_deploy.ps1 smoke
```

Windows full run:

```powershell
cd D:\Documents\StockInfoCrawler
powershell -ExecutionPolicy Bypass -File .\blackbox_finetune\scripts\one_click_deploy.ps1 full
```

WSL2/Linux smoke run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune/scripts/one_click_deploy.sh smoke
```

WSL2/Linux full run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune/scripts/one_click_deploy.sh full
```

## Manual Dataset Build

Build training data:

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

Build validation data:

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

For a quick dataset smoke test:

```powershell
python -m blackbox_finetune.build_dataset `
  --start-date 20110101 `
  --end-date 20151231 `
  --positive-limit 20 `
  --negative-ratio 1.0 `
  --output-dir blackbox_finetune/data/smoke `
  --daily-window 55 `
  --weekly-window 55 `
  --batch-size 20
```

## Train

WSL2/Linux QLoRA training:

```bash
python -m blackbox_finetune.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune/data \
  --output-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora \
  --max-seq-length 2048 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4
```

Native Windows training should disable 4-bit loading:

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

## Evaluate

Evaluate against `20260101-20260430` validation data:

```powershell
python -m blackbox_finetune.evaluate `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --adapter-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora/adapter `
  --data-dir blackbox_finetune/data_validation `
  --threshold 0.50 `
  --min-positive-recall 0.60
```

Evaluate only a small sample:

```powershell
python -m blackbox_finetune.evaluate `
  --base-model Qwen/Qwen2.5-0.5B-Instruct `
  --adapter-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora/adapter `
  --data-dir blackbox_finetune/data_validation `
  --threshold 0.50 `
  --min-positive-recall 0.60 `
  --max-samples 100
```

## Predict

Predict all stocks for one trading day:

```powershell
python -m blackbox_finetune.predict_day `
  --date 20260514 `
  --adapter-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora/adapter `
  --threshold 0.50 `
  --limit 20 `
  --output data\blackbox_predictions_20260514.csv
```

Use a different threshold:

```powershell
python -m blackbox_finetune.predict_day `
  --date 20260514 `
  --adapter-dir blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora/adapter `
  --threshold 0.60 `
  --limit 50
```

## Tests

```powershell
python -m unittest tests.test_blackbox_finetune -v
python -m unittest discover -s tests -v
```
