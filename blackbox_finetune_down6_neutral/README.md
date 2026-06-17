# Black-Box Down6 vs Neutral Fine-Tuning

This directory is an independent binary fine-tuning task based on the current `blackbox_finetune_recall60` implementation.

The positive label (`{"p":1}`) means the stock drops at least 6% within the next 3 trading days. The negative label (`{"p":0}`) means neutral: not a 3-day 6% drop, not a 3-day 20% surge, and outside every 3-day 20% surge anchor's +/-20 trading-day exclusion window for the same stock.

Defaults:

- Training period: `20230101-20241231`
- Evaluation period: `20260101-20260529`
- No cooldown is applied to down6 samples.
- Sample encoding, LoRA/QLoRA training, checkpoint evaluation, and RTX3060 settings follow `blackbox_finetune_recall60`.
- `NEUTRAL_RATIO` controls neutral samples per down6 sample. If unset, the builder falls back to `NEGATIVE_RATIO`, then `9.0`.
- The training objective is risk recall: true down6 samples are weighted more heavily and can receive an extra low-score penalty so the model is less likely to assign low risk to future drawdown samples.

WSL/Linux full run:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_finetune_down6_neutral/scripts/one_click_deploy.sh full
```

Common overrides:

```bash
export SAMPLE_MODE=xlong
export NEUTRAL_RATIO=9
export EPOCHS=0.3
export MAX_SEQ_LENGTH=3072
export EVAL_PRECISION_TOP_K=10
export EVAL_PRECISION_THRESHOLD=0.40
export DOWN6_CE_WEIGHT=3.0
export NEUTRAL_CE_WEIGHT=1.0
export DOWN6_LOW_SCORE_PENALTY=1
export DOWN6_SCORE_FLOOR=0.45
export DOWN6_LOW_SCORE_WEIGHT=0.2
bash blackbox_finetune_down6_neutral/scripts/one_click_deploy.sh full
```

Build only the training dataset:

```bash
python -m blackbox_finetune_down6_neutral.build_dataset \
  --start-date 20230101 \
  --end-date 20241231 \
  --neutral-ratio 9 \
  --sample-mode xlong \
  --output-dir blackbox_finetune_down6_neutral/data_no_partial_week_down6_xlong_neutral9
```

Build only the evaluation dataset:

```bash
python -m blackbox_finetune_down6_neutral.build_validation_dataset \
  --start-date 20260101 \
  --end-date 20260529 \
  --neutral-ratio 9 \
  --sample-mode xlong \
  --output-dir blackbox_finetune_down6_neutral/data_evaluation_no_partial_week_down6_xlong_neutral9
```
