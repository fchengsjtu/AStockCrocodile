# Three-class black-box fine-tuning

This directory is independent from `blackbox_finetune_recall60`. It uses the same Qwen2.5-0.5B-Instruct base model, compact K-line encoding, sample modes, RTX 3060 training flow, LoRA parameters, checkpointing, and date defaults.

## Labels and sampling

Each anchor uses only K-lines on or before the anchor date. The next three trading days are used only to create the target:

- `positive`: price first reaches `anchor close * 1.20`.
- `negative`: price first reaches `anchor close * 0.94`.
- `neutral`: neither threshold is reached.
- If both thresholds are reached on the same daily bar before either was previously reached, the sample is discarded because daily K-lines cannot reveal which happened first.

Positive anchors for the same stock use the same 20-trading-day cooldown as recall60. Negative and neutral samples are not excluded merely because they are close to a positive anchor; their labels are determined only by the future-three-trading-day path. After recall60 K-line completeness and bottom-band filters are applied, the final dataset is rebalanced into strict same-day cycles:

```text
positive : negative : neutral = 1 : 4 : 11
```

For training data, the builder first selects positive rows, then pairs each kept positive only with negative and neutral rows from the same `anchor_date`. Positives without enough same-day support rows are skipped, and rows inside each same-day cycle are shuffled to avoid leaking a fixed class order. Validation/evaluation data is sampled deterministically but is not organized into same-day cycles.

All samples from `TRAIN_START_DATE` to `TRAIN_END_DATE` are written to the training dataset `train.jsonl`. All samples from `VALIDATION_START_DATE` to `VALIDATION_END_DATE` are written to the validation dataset `test.jsonl`, which is used for checkpoint and final evaluation.

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

Set `REBUILD_DATASET=1` to rebuild data. Otherwise the existing training `train.jsonl` and validation `test.jsonl` files are reused.

Candidate classification is queried from MySQL in symbol batches to avoid one full-market window query timing out. `CANDIDATE_BATCH_SIZE` defaults to `80`; lower it to `40` or `20` on a slow MySQL host. `MYSQL_QUERY_RETRIES` defaults to `3` and reconnects the current batch after MySQL errors 2006/2013/2055.

## Manual commands

Build training and validation datasets:

```bash
python -m blackbox_finetune_threeclass.build_dataset \
  --output-dir blackbox_finetune_threeclass/data_xlong_p1_n4_u11 \
  --start-date 20230101 --end-date 20241231 \
  --sample-mode xlong

python -m blackbox_finetune_threeclass.build_validation_dataset \
  --output-dir blackbox_finetune_threeclass/data_evaluation_xlong_p1_n4_u11 \
  --start-date 20260101 --end-date 20260530 \
  --sample-mode xlong
```

Train:

```bash
python -m blackbox_finetune_threeclass.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_threeclass/data_xlong_p1_n4_u11 \
  --checkpoint-eval-data-dir blackbox_finetune_threeclass/data_evaluation_xlong_p1_n4_u11 \
  --output-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-p1_n4_u11-lora \
  --max-seq-length 3072 --epochs 3.0 \
  --gradient-accumulation-steps 16 --learning-rate 5e-6 \
  --checkpoint-every 100 --eval-max-samples 1500 --on-the-fly-tokenize \
  --cuda-device 0
```

The training objective uses an asymmetric three-class loss:

```text
total_loss =
    weighted_CE
    + FP_LOSS_WEIGHT * negative_fp_loss
    + NEUTRAL_FP_LOSS_WEIGHT * neutral_fp_loss
    + top1_negative_penalty
    + top1_neutral_penalty
    - high_score_positive_reward
```

Defaults:

```text
POSITIVE_CE_WEIGHT=4.0
NEGATIVE_CE_WEIGHT=1.0
NEUTRAL_CE_WEIGHT=0.5
FP_LOSS_WEIGHT=1.0
NEUTRAL_FP_LOSS_WEIGHT=0.3
HIGH_SCORE_POSITIVE_BONUS=1.0
HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER=8.0
HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT=1.0
HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT=0.5
HIGH_SCORE_NEGATIVE_MARGIN=0.2
HIGH_SCORE_NEUTRAL_MARGIN=0.1
POSITIVE_PURIFICATION_ENABLED=1
POSITIVE_PURIFICATION_GROUP_SIZE=16
POSITIVE_PURIFICATION_BOTTOM_K=3
POSITIVE_PURIFICATION_DECAY=0.5
FP_DYNAMIC_PENALTY=1
```

`POSITIVE_CE_WEIGHT`, `NEGATIVE_CE_WEIGHT`, and `NEUTRAL_CE_WEIGHT` control the base CE contribution for each true class. For a true negative sample, `negative_fp_loss` compares the complete `{"c":"positive"}` and `{"c":"negative"}` answer NLL values, exactly matching the probability calculation used by inference. For a true neutral sample, `neutral_fp_loss` compares `{"c":"positive"}` with `{"c":"neutral"}` and applies a lower-weight penalty when neutral rows look positive. With the default 1:4:11 class cycle and `batch_size=1`, these false-positive losses are scaled so gradient accumulation behaves like a full effective batch: negative fp is averaged over the 4 negative rows and neutral fp is averaged over the 11 neutral rows, rather than being diluted by the other classes.

The high-score terms compare the Positive-answer scores once per optimizer update, after the full gradient-accumulation window is collected:

```text
positive_answer_score = -positive_nll

high_score_positive_reward =
    for every true positive row:
        (
            positive_row_score
            - average(top5_score, top6_score, top7_score, top8_score, top9_score, top10_score)
        )
        * HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER

top1_negative_penalty =
    if top1 is true negative:
        relu(
            top1_score
            - average(top2_score, top3_score, top4_score, top5_score)
            + HIGH_SCORE_NEGATIVE_MARGIN
        )
        * HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT

top1_neutral_penalty =
    if top1 is true neutral:
        relu(
            top1_score
            - average(top2_score, top3_score, top4_score, top5_score)
            + HIGH_SCORE_NEUTRAL_MARGIN
        )
        * HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT
```

For true positive samples, `positive_nll` is the normal CE target NLL. For true negative samples, it is the extra `{"c":"positive"}` answer NLL already computed for `negative_fp_loss`. For true neutral samples, it is the extra `{"c":"positive"}` answer NLL already computed for `neutral_fp_loss`. The explicit high-score reward is applied once per optimizer update to every true positive row in the gradient-accumulation window, using the top5-through-top10 average baseline above. Positive rows above that baseline reduce loss; positive rows below it produce a negative reward, which increases loss. The explicit high-score penalty is still applied only to the window's top-1 row when that row is negative or neutral. Training logs print `loss`, `ce`, `negative_fp`, `neutral_fp`, `high_score_negative`, `high_score_neutral`, `high_score_positive_reward`, and `high_score_positive_best_rank`. `high_score_positive_best_rank` is the best rank achieved by any true positive row in the update window, where `1` means a positive row is ranked first. Negative and neutral auxiliary penalties require extra positive-answer forward passes, so training is slower and uses more GPU memory than plain CE.

Positive purification adds a persistent `positive_weight` field to true positive rows. It starts at `1.0` and multiplies the positive CE contribution in addition to `POSITIVE_CE_WEIGHT`. After each checkpoint evaluation, the trainer scores the positive answer for each training row in groups of `POSITIVE_PURIFICATION_GROUP_SIZE` rows. If a true positive row is in the lowest `POSITIVE_PURIFICATION_BOTTOM_K` scores inside its group, its `positive_weight` is multiplied by `POSITIVE_PURIFICATION_DECAY`. The updated training set is written back to `train.jsonl`, and a checkpoint-specific snapshot named `train_positive_weights_update-XXXXXX.jsonl` is also written beside it.

To initialize from a good binary recall60 adapter while starting a new three-class run at update 0:

```bash
python -m blackbox_finetune_threeclass.train \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir blackbox_finetune_threeclass/data_xlong_p1_n4_u11 \
  --output-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-from-binary-lora \
  --initial-binary-adapter-dir /mnt/d/Models/precision10@0.4-3200 \
  --max-seq-length 3072 --epochs 3.0 \
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
  --adapter-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-p1_n4_u11-lora/adapter \
  --data-dir blackbox_finetune_threeclass/data_evaluation_xlong_p1_n4_u11 \
  --max-seq-length 3072 --cuda-device 0
```

The result contains overall accuracy, macro F1, a 3-by-3 confusion matrix, precision/recall/F1 for every class, and positive precision at 5/10/20/50.

Predict one day:

```bash
python -m blackbox_finetune_threeclass.predict_day \
  --date 20260612 \
  --adapter-dir blackbox_finetune_threeclass/runs/qwen2.5-0.5b-threeclass-xlong-p1_n4_u11-lora/adapter \
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

Every checkpoint evaluation writes and prints two independent Top 50 rankings:

- `selection_score_top50`: ranked by `SelectionScore`. Checkpoint `positive_precision@5/10/20/50` and the precision gate use this ranking, matching daily stock selection.
- `positive_probability_top50`: ranked only by `PositiveProbability`.

Both rankings include the stock code, anchor date, all three class probabilities, `SelectionScore`, predicted class, and actual class. Comparing the two lists shows how `NEGATIVE_WEIGHT` and `NEUTRAL_WEIGHT` change the selected stocks.

After each checkpoint evaluation, the next `eval_threshold` is calculated from all evaluated `SelectionScore` values:

```text
average = mean(SelectionScore)
maximum = max(SelectionScore)
next_eval_threshold = average + (1 - EVAL_THRESHOLD_TOP_RATIO) * (maximum - average)
```

`EVAL_THRESHOLD_TOP_RATIO` defaults to `0.2`, so the next threshold is at the Top 20% position between the average and maximum score. With the default value this is `average + 0.8 * (maximum - average)`. The evaluation JSON records the current threshold, average score, maximum score, position, and next threshold.
