# Three-Class WSL Parameter Guide

This file explains the important defaults in `scripts/set_wsl_env.sh`. Parameters are grouped by how they affect the run.

## Dataset And Run Paths

`SAMPLE_MODE=xlong`
: Uses the xlong encoding window. Keep `MAX_SEQ_LENGTH=3072` with this mode unless token length tests show a safe smaller value.

`TRAIN_START_DATE=20230101`, `TRAIN_END_DATE=20241231`
: Date range used to build the training dataset.

`VALIDATION_START_DATE=20260101`, `VALIDATION_END_DATE=20260530`
: Date range used for checkpoint and final evaluation.

`DATA_DIR`, `VALIDATION_DATA_DIR`, `OUTPUT_DIR`
: Training dataset, evaluation dataset, and model output locations. Change these together when comparing different experimental runs.

The default training class ratio is `positive:negative:neutral = 1:4:11`. One full class cycle has 16 rows, matching the default gradient accumulation window. Training dataset generation first selects positive rows, then only pairs them with negative and neutral rows from the same `anchor_date`; positives that cannot form a complete same-day cycle are skipped. The rows inside each cycle are shuffled so the model cannot exploit a fixed label order. Evaluation datasets use deterministic sampling and ordering, but do not require same-day `1:4:11` cycles.

`REBUILD_DATASET=0`
: Reuses cached datasets. Set to `1` only when date ranges, sample filters, or class-ratio logic changed.

## Base Training

`EPOCHS=1.0`
: Number of passes over the generated training rows. Increase only after evaluation improves without signs of overfitting.

`GRADIENT_ACCUMULATION_STEPS=16`
: Effective batch size is `BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. The default matches one complete `1:4:11` class cycle.

`LEARNING_RATE=5e-6`
: Main step size. Lower it if `grad_norm` repeatedly spikes or checkpoint precision swings violently.

`WEIGHT_DECAY=0.05`, `LORA_DROPOUT=0.3`
: Regularization knobs. Increase cautiously when evaluation looks overfit.

`MAX_GRAD_NORM=1.0`
: Gradient clipping threshold used by the trainer.

`CHECKPOINT_EVERY=100`
: Saves a checkpoint and runs evaluation every 100 optimizer updates.

## Class CE Weights

`POSITIVE_CE_WEIGHT=2.0`
: Makes true positive classification errors more important than default rows.

`NEGATIVE_CE_WEIGHT=1.0`
: Base CE weight for true negative samples.

`NEUTRAL_CE_WEIGHT=0.5`
: Neutral rows are useful background but should not dominate training.

## False-Positive Suppression

`FP_LOSS_WEIGHT=1.5`
: Penalizes true negative rows when the complete positive answer is preferred. Raise this if negative rows enter the top scored list too often.

`NEUTRAL_FP_LOSS_WEIGHT=0.8`
: Similar to `FP_LOSS_WEIGHT`, but for true neutral rows. It is lower because neutral rows are less harmful than downside samples.

`RANK_LOSS_WEIGHT=1.5`, `RANK_MARGIN=0.25`
: Requires true negative rows to prefer the negative answer over the positive answer by a margin.

`NEUTRAL_RANK_LOSS_WEIGHT=0.5`, `NEUTRAL_RANK_MARGIN=0.08`
: Requires true neutral rows to prefer the neutral answer over the positive answer by a smaller margin. Increase gently if top scores contain too many neutral rows.

## High-Score Region

`HIGH_SCORE_EMA=1`, `HIGH_SCORE_EMA_ALPHA=0.02`
: Maintains a stable EMA cutoff for the high-score region. Smaller alpha makes the cutoff less noisy.

`HIGH_SCORE_CUTOFF_POSITION=0.75`
: Raw cutoff position between batch average score and batch max score. Higher values define a stricter high-score region.

`POSITIVE_HIGH_SCORE_LOSS_WEIGHT=50.0`, `POSITIVE_HIGH_SCORE_MARGIN=0.05`
: Push true positive rows above the high-score cutoff plus a margin. Raise the margin only if positives do not reach the high-score region.

`HIGH_SCORE_POSITIVE_BONUS=2.0`
: Explicit reward subtracted from loss when true positive rows enter the high-score region.

`HIGH_SCORE_POSITIVE_BONUS_SCALE=0.02`
: Controls how quickly the reward increases as positive score exceeds the cutoff. Smaller values make the reward more aggressive.

`HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER=60.0`
: Caps the positive high-score reward. Lower it if `grad_norm` spikes after positive reward hits.

`HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT=20.0`
: Penalizes true negative rows that enter the high-score region.

`HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT=15`
: Penalizes true neutral rows that enter the high-score region.

## Evaluation And Prediction Ranking

`EVAL_SAMPLE_METHOD=random`, `EVAL_MAX_SAMPLES=750`
: Checkpoint evaluation samples 750 validation rows. Use larger samples for more stable checkpoint comparisons.

`EVAL_PRECISION_TOP_K=10`, `EVAL_PRECISION_THRESHOLD=0.40`
: Checkpoint pass condition is based on positive precision at top K.

`NEGATIVE_WEIGHT=0.5`, `NEUTRAL_WEIGHT=0.0`
: Prediction ranking uses `SelectionScore = PositiveProbability - NEGATIVE_WEIGHT * NegativeProbability - NEUTRAL_WEIGHT * NeutralProbability`.
Increase these for stricter filtering at prediction time without changing training.

`EVAL_THRESHOLD_TOP_RATIO=0.2`
: Evaluation threshold update position between average and max selection score.

Each checkpoint evaluation JSON records the active `training_parameters` and `evaluation_parameters`. Use those fields to compare runs because they preserve the CE weights, false-positive penalties, rank margins, high-score settings, and selection weights that produced the checkpoint result.

## Runtime And Offline Mode

`ON_THE_FLY_TOKENIZE=1`
: Avoids storing a large tokenized cache in memory.

`CUDA_DEVICE=0`
: GPU index used for training and inference.

`HF_LOCAL_FILES_ONLY=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TRUST_REMOTE_CODE=0`
: Keeps WSL runs offline and avoids HuggingFace network access.

`CANDIDATE_BATCH_SIZE=80`, `MYSQL_QUERY_RETRIES=3`
: Database build controls. Increase retries if MySQL occasionally times out.
