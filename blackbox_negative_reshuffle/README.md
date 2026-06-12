# Black-box Negative Reshuffle

This standalone tool reshuffles negative samples with an already trained adapter. It does not modify `blackbox_finetune_recall60`.

## Model metadata

Place an evaluation JSON under the model directory, preferably:

```text
runs/evaluations/eval-xxxx.json
```

The JSON must contain:

```json
{
  "original_train_dataset_path": "/mnt/d/Documents/StockInfoCrawler/blackbox_finetune_recall60/data_no_partial_week_recall60_xlong_neg9",
  "original_eval_dataset_path": "/mnt/d/Documents/StockInfoCrawler/blackbox_finetune_recall60/data_evaluation_no_partial_week_recall60_xlong_neg9"
}
```

The training directory must contain `train.jsonl`, `test.jsonl`, and `all.jsonl`. The evaluation directory must contain at least one JSONL dataset.

For compatibility, if an old JSON contains `original_train_dataset_path` twice, the first value is treated as the training directory and the second as the evaluation directory.

## Run in WSL

```bash
source /home/fcheng/.venvs/astock-blackbox-finetune-recall60/bin/activate
cd /mnt/d/Documents/StockInfoCrawler

python -m blackbox_negative_reshuffle.run \
  --model-dir /mnt/d/Models/precision10@0.4-1500 \
  --keep-ratio 0.20 \
  --stat-type short_term_surge_3d_20pct \
  --sample-mode xlong \
  --max-seq-length 3072 \
  --database-max-attempts 20 \
  --cuda-device 0
```

Use `--keep-count N` instead of `--keep-ratio` to retain an exact number of highest-scoring negatives in each split.

The default database refill limit is 20 attempts. Each successful attempt saves
`database_replacement_pool.jsonl` under the selected output directory. If a run is interrupted or the database cannot provide enough valid rows, rerun the same command with the same `--output-name`; the saved replacement pool is loaded and only the missing rows are sampled. Increase `--database-max-attempts` when the selected sample mode filters out many candidates.

## Output

The default output is written below the source model:

```text
MODEL_DIR/negative_reshuffle/run-YYYYMMDD-HHMMSS/
  adapter/
  datasets/
    training/
      train.jsonl
      test.jsonl
      all.jsonl
    evaluation/
  negative_scores.jsonl
  database_replacement_pool.jsonl
  reshuffle_manifest.json
  runs/
    evaluations/
      eval-xxxx.json
```

The model scores only the current train/test negative samples. The highest-scoring configured portion is retained. All replacement negatives are sampled directly from MySQL over the original dataset date range, excluding every current negative sample, and are materialized with the selected sample mode. The adapter is copied unchanged as the starting model for subsequent training. The evaluation JSON is copied and enriched with the original and generated dataset/model paths. The evaluation dataset is copied unchanged. Positive rows remain in their original split.

## Evaluate selected reshuffle checkpoints

The checkpoint evaluation script defaults to cycle `cycle-01`, the full
`20260101` through `20260531` evaluation period, and updates
`0200, 1000, 1500, 1600, 2100, 2900, 3000, 4200, 4700, 4800`.

Run it in WSL:

```bash
cd /mnt/d/Documents/StockInfoCrawler
bash blackbox_negative_reshuffle/scripts/evaluate_checkpoints.sh
```

Optional positional arguments are cycle directory, start date, end date, and
the shared JSONL result path:

```bash
bash blackbox_negative_reshuffle/scripts/evaluate_checkpoints.sh \
  "/mnt/d/Models/precision10@0.4-3200/negative_reshuffle/cycle-01" \
  20260101 \
  20260531 \
  "/mnt/d/Models/precision10@0.4-3200/negative_reshuffle/cycle-01/checkpoint-results.jsonl"
```

The script filters the evaluation dataset by `anchor_date`, evaluates every
checkpoint in a separate Python process, retains each full `evaluation.json`
and log, and appends one complete compact JSON record per checkpoint to:

```text
cycle-01/checkpoint_full_evaluations_20260101_20260531/results.jsonl
```

Set `DRY_RUN=1` to validate paths and print every command without loading a
model. `CHECKPOINT_UPDATES`, `THRESHOLD`, `PRECISION_TOP_K`,
`PRECISION_THRESHOLD`, `MAX_SEQ_LENGTH`, `BASE_MODEL`, and `CUDA_DEVICE` can
be overridden through environment variables.
