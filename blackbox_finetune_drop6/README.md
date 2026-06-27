# Black-Box Drop6 Binary Classifier

This directory is an independent binary fine-tuning task for downside-risk prediction.

The model still learns a JSON answer shaped as `{"p": 1}` or `{"p": 0}`, but in this package:

- `p=1` means the stock drops more than 6% within the next 3 trading days.
- The drop label is triggered when any of the next 3 trading days has `Low <= anchor Close * 0.94`.
- Positive samples do not use a cooldown period.
- Negative samples exclude the same stock's dates within `+/-10` trading days around any positive sample.
- The default negative ratio is `9.0`.
- Default training period is `20230101-20241231`.
- Default evaluation period is `20260101-20260530`.
- Default sample mode is `xlong`, with `MAX_SEQ_LENGTH=3072`.

## WSL Run

```bash
cd /mnt/d/Documents/StockInfoCrawler
source ./set_blackbox_drop6_wsl_env.sh
bash blackbox_finetune_drop6/scripts/one_click_deploy.sh full
```

Force dataset rebuild:

```bash
REBUILD_DATASET=1 bash blackbox_finetune_drop6/scripts/one_click_deploy.sh full
```

## Manual Dataset Build

```bash
python -m blackbox_finetune_drop6.build_dataset \
  --start-date 20230101 \
  --end-date 20241231 \
  --negative-ratio 9 \
  --sample-mode xlong \
  --output-dir blackbox_finetune_drop6/data_no_partial_week_drop6_xlong_neg9
```

```bash
python -m blackbox_finetune_drop6.build_validation_dataset \
  --start-date 20260101 \
  --end-date 20260530 \
  --negative-ratio 9 \
  --sample-mode xlong \
  --output-dir blackbox_finetune_drop6/data_evaluation_no_partial_week_drop6_xlong_neg9
```
