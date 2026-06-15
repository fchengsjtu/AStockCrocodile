# Three-class black-box fine-tuning

This directory is independent from `blackbox_finetune_recall60`. It uses the same Qwen2.5-0.5B-Instruct base model, compact K-line encoding, sample modes, RTX 3060 training flow, LoRA parameters, checkpointing, and date defaults.

## Labels and sampling

Each anchor uses only K-lines on or before the anchor date. The next three trading days are used only to create the target:

- `positive`: price first reaches `anchor close * 1.20`.
- `negative`: price first reaches `anchor close * 0.94`.
- `neutral`: neither threshold is reached.
- If both thresholds are reached on the same daily bar before either was previously reached, the sample is discarded because daily K-lines cannot reveal which happened first.

Positive anchors for the same stock use the same 20-trading-day cooldown as recall60. Negative and neutral samples are not excluded merely because they are close to a positive anchor; their labels are determined only by the future-three-trading-day path. After recall60 K-line completeness and bottom-band filters are applied, the final dataset is rebalanced to exactly:

```text
positive : negative : neutral = 1 : 2 : 10
```

Training and test files are stratified, so both retain approximately the same class ratio.

## Environment and one-click workflow

```bash
cd /mnt/d/Documents/StockInfoCrawler
source blackbox_finetune_threeclass/scripts/set_wsl_env.sh
bash blackbox_finetune_threeclass/scripts/one_click_deploy.sh full
```

Other modes:

```bash
bash blackbox_finetune_threeclass/scripts/one_click_deploy.sh smoke
bash blackbox_finetune_threeclass/scripts/one_click_deploy.sh dataset-only
bash blackbox_finetune_threeclass/scripts/one_click_deploy.sh diagnose
```

Set `REBUILD_DATASET=1` to rebuild data. Otherwise existing `train.jsonl` and `test.jsonl` files are reused.

## Manual commands

Build training and validation datasets:

```bash
python -m blackbox_finetune_threeclass.build_dataset \
  --output-dir blackbox_finetune_threeclass/data_xlong_p1_n2_u10 \
  --start-date 20230101 --end-date 20241231 \
  --sample-mode xlong

python -m blackbox_finetune_threeclass.build_validation_dataset \
  --output-dir blackbox_finetune_threeclass/data_evaluation_xlong_p1_n2_u10 \
  --start-date 20260101 --end-date 20260530 \
  --sample-mode xlong
```

Train:

```bash
python -m blackbox_finetune_threeclass.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_threeclass/data_xlong_p1_n2_u10 \
  --output-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-lora \
  --max-seq-length 3072 --epochs 0.3 \
  --gradient-accumulation-steps 16 --learning-rate 5e-6 \
  --checkpoint-every 100 --on-the-fly-tokenize \
  --cuda-device 0
```

To initialize from a good binary recall60 adapter while starting a new three-class run at update 0:

```bash
python -m blackbox_finetune_threeclass.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_threeclass/data_xlong_p1_n2_u10 \
  --output-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-from-binary-lora \
  --initial-binary-adapter-dir /mnt/d/Models/precision10@0.4-3200 \
  --max-seq-length 3072 --epochs 0.3 \
  --gradient-accumulation-steps 16 --learning-rate 5e-6 \
  --checkpoint-every 100 --on-the-fly-tokenize \
  --cuda-device 0
```

The source binary adapter is loaded as trainable LoRA initialization only. The optimizer and update counter start from zero, checkpoints are written under the new three-class output directory, and the binary model is not overwritten. Without `--initial-binary-adapter-dir`, training starts from a fresh LoRA adapter on the Qwen base model. Use `--resume-adapter-dir` only to continue an existing three-class checkpoint; the two options cannot be combined.

The same initialization can be supplied to the one-click script:

```bash
export INITIAL_BINARY_ADAPTER_DIR='D:\Models\precision10@0.4-3200'
bash blackbox_finetune_threeclass/scripts/one_click_deploy.sh full
```

Evaluate:

```bash
python -m blackbox_finetune_threeclass.evaluate \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-lora/adapter \
  --data-dir blackbox_finetune_threeclass/data_evaluation_xlong_p1_n2_u10 \
  --max-seq-length 3072 --cuda-device 0
```

The result contains overall accuracy, macro F1, a 3-by-3 confusion matrix, precision/recall/F1 for every class, and positive precision at 5/10/20/50.

Predict one day:

```bash
python -m blackbox_finetune_threeclass.predict_day \
  --date 20260612 \
  --adapter-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-lora/adapter \
  --sample-mode xlong --max-seq-length 3072 \
  --negative-weight 0.5 --neutral-weight 0 \
  --limit 20 \
  --output data/threeclass_predictions_20260612.csv \
  --cuda-device 0
```

Prediction selection and ranking use:

```text
SelectionScore =
    PositiveProbability
    - negative_weight * NegativeProbability
    - neutral_weight * NeutralProbability
```

Defaults are `negative_weight=0.5` and `neutral_weight=0`. Every stock that passes the K-line/sample validity filters receives a score; there is no probability or score threshold. Candidates are ranked by `SelectionScore`, then `PositiveProbability`, and the first `--limit` rows are returned. `--positive-threshold` is accepted only for compatibility with older commands and no longer filters candidates.

Every checkpoint evaluation writes and prints `positive_probability_top50`. Each row contains the stock code, anchor date, `PositiveProbability`, `NeutralProbability`, `NegativeProbability`, predicted class, and actual class.
