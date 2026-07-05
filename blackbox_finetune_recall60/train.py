from __future__ import annotations

import argparse
import json
import os
import hashlib
import math
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall60.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_TRAIN_SEED,
    compact_messages_from_sample,
    default_max_seq_length,
    default_output_dir,
    label_answer,
    normalize_precision_threshold,
    normalize_precision_top_k,
    precision_at_k,
    REPORTED_PRECISION_KS,
)
from blackbox_finetune_recall60.gpu import prepare_rtx3060
from blackbox_finetune_recall60.inference import score_prediction


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


from llm_finetune.common import read_jsonl


def default_checkpoint_every() -> int:
    return _env_int("CHECKPOINT_EVERY", _env_int("CHECKOUT_EVERY", 100))

TOKEN_CACHE_VERSION = "v7_csv11_21d13w_no_partial_2020_2025"
TRAINING_STATE_FILE = "training_state.pt"
NEGATIVE_KIND_DROP6 = "drop6"
NEGATIVE_KIND_NEUTRAL = "neutral"


def _missing_dataset_error(data_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(f"missing dataset files under {data_dir}; run blackbox_finetune_recall60.build_dataset first")


def _checkpoint_eval_path(data_dir: Path, checkpoint_eval_data_dir: Path | None) -> Path:
    return (checkpoint_eval_data_dir or data_dir) / "test.jsonl"


def _extra_training_state() -> dict:
    return {}


def _load_extra_training_state(state: dict) -> None:
    return None


def _after_checkpoint_evaluation(**_kwargs) -> dict:
    return {}


def _after_train_items_prepared(**_kwargs) -> dict:
    return {}


def _build_train_order(train_items: list[dict], seed: int, rng) -> list[int]:
    train_order = list(range(len(train_items)))
    rng.shuffle(train_order)
    return train_order


def _reshuffle_train_order(train_order: list[int], train_items: list[dict], rng) -> None:
    rng.shuffle(train_order)


def _training_run_summary(**kwargs) -> str:
    return (
        f"manual RTX3060 LoRA train rows={kwargs['train_rows']} valid={kwargs['valid_rows']} "
        f"checkpoint_eval_data_dir={kwargs['checkpoint_eval_data_dir']} "
        f"updates={kwargs['total_updates']} start_update={kwargs['start_update']} batch_size={kwargs['batch_size']} grad_accum={kwargs['gradient_accumulation_steps']} "
        f"train_seed={kwargs['train_seed']} lr={kwargs['learning_rate']} weight_decay={kwargs['weight_decay']} max_grad_norm={kwargs['max_grad_norm']} lora_rank={kwargs['lora_rank']} lora_dropout={kwargs['lora_dropout']} "
        f"max_seq_length={kwargs['max_seq_length']} on_the_fly_tokenize={kwargs['on_the_fly_tokenize']} "
        f"positive_loss_weight={kwargs['positive_loss_weight']} negative_loss_weight={kwargs['negative_loss_weight']} "
        f"drop6_negative_loss_weight={kwargs['drop6_negative_loss_weight']} neutral_negative_loss_weight={kwargs['neutral_negative_loss_weight']} "
        f"high_score_positive_bonus={kwargs['high_score_positive_bonus']} high_score_positive_position={kwargs['high_score_positive_position']} "
        f"high_score_positive_cutoff={kwargs['high_score_positive_cutoff']} "
        f"checkpoint_every={kwargs['checkpoint_every']} checkpoint_evaluate=True eval_threshold={kwargs['evaluation_threshold']} "
        f"eval_threshold_position={kwargs['evaluation_threshold_position']} "
        f"eval_precision_top_k={kwargs['evaluation_precision_top_k']} "
        f"eval_precision_threshold={kwargs['evaluation_precision_threshold']} eval_max_samples={kwargs['evaluation_max_samples']} "
        f"fp_dynamic_penalty={kwargs['fp_dynamic_penalty']} fp_penalty_weight={kwargs['fp_penalty_weight']} "
        f"fp_threshold_ema_alpha={kwargs['fp_threshold_ema_alpha']} fp_threshold_min={kwargs['fp_threshold_min']} "
        f"fp_threshold_max={kwargs['fp_threshold_max']} fp_penalty_cutoff={kwargs['fp_penalty_cutoff']}"
    )


def _torch_state_to_device(value, device: str):
    if isinstance(value, dict):
        return {key: _torch_state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_torch_state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_torch_state_to_device(item, device) for item in value)
    try:
        import torch
    except Exception:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _training_state_path(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / TRAINING_STATE_FILE


def _load_training_state(checkpoint_dir: Path | None, device: str) -> dict | None:
    state_path = _training_state_path(checkpoint_dir)
    if state_path is None or not state_path.exists():
        return None
    try:
        import torch
    except Exception:
        return None
    return torch.load(state_path, map_location=device, weights_only=False)


def _save_training_state(
    checkpoint_dir: Path,
    *,
    optimizer,
    scheduler,
    update: int,
    next_micro_step: int,
    train_order: list[int],
    rng,
    evaluation_threshold: float,
    high_score_positive_cutoff: float,
    fp_penalty_cutoff: float,
    total_nonfinite_skips: int,
) -> None:
    import torch

    state = {
        "version": 1,
        "update": int(update),
        "next_micro_step": int(next_micro_step),
        "train_order": list(train_order),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "python_random_state": random.getstate(),
        "train_rng_state": rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "evaluation_threshold": float(evaluation_threshold),
        "next_threshold": float(evaluation_threshold),
        "high_score_positive_cutoff": float(high_score_positive_cutoff),
        "fp_penalty_cutoff": float(fp_penalty_cutoff),
        "total_nonfinite_skips": int(total_nonfinite_skips),
        "extra": _extra_training_state(),
    }
    state_path = _training_state_path(checkpoint_dir)
    assert state_path is not None
    torch.save(state, state_path)
    print(f"training state saved: {state_path}", flush=True)


def _tokenize_row(tokenizer, row: dict, max_seq_length: int) -> dict:
    messages = compact_messages_from_sample(row)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = messages[-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(answer_ids) >= max_seq_length:
        answer_ids = answer_ids[: max_seq_length - 1] + [tokenizer.eos_token_id]
    prompt_budget = max(1, max_seq_length - len(answer_ids))
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = (prompt_ids + answer_ids)[:max_seq_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _collate(tokenizer, batch: list[dict], device: str) -> dict:
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _label_from_training_row(row: dict) -> int | None:
    metadata = row.get("metadata") if isinstance(row, dict) else None
    if isinstance(metadata, dict) and "label" in metadata:
        try:
            return int(metadata["label"])
        except (TypeError, ValueError):
            return None
    if "label" in row:
        try:
            return int(row["label"])
        except (TypeError, ValueError):
            return None
    return None


def _negative_kind_from_training_row(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row, dict) else None
    value = ""
    if isinstance(metadata, dict):
        value = str(metadata.get("negative_kind", "")).strip().lower()
    return value if value in {NEGATIVE_KIND_DROP6, NEGATIVE_KIND_NEUTRAL} else NEGATIVE_KIND_NEUTRAL


def _loss_weight_for_training_row(
    row: dict,
    label: int,
    *,
    positive_loss_weight: float,
    negative_loss_weight: float,
    drop6_negative_loss_weight: float,
    neutral_negative_loss_weight: float,
) -> tuple[float, str]:
    if int(label) == 1:
        return float(positive_loss_weight), "positive"
    negative_kind = _negative_kind_from_training_row(row)
    if negative_kind == NEGATIVE_KIND_DROP6:
        return float(drop6_negative_loss_weight), NEGATIVE_KIND_DROP6
    if negative_kind == NEGATIVE_KIND_NEUTRAL:
        return float(neutral_negative_loss_weight), NEGATIVE_KIND_NEUTRAL
    return float(negative_loss_weight), "negative"


def _differentiable_answer_loss(model, tokenizer, row: dict, label: int, max_seq_length: int, device: str):
    import torch

    messages = compact_messages_from_sample(row)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = label_answer(int(label))
    answer = answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.shape[1] >= max_seq_length:
        answer_ids = answer_ids[:, : max_seq_length - 1]
        answer_ids[0, -1] = tokenizer.eos_token_id
    prompt_budget = max(1, max_seq_length - answer_ids.shape[1])
    if prompt_ids.shape[1] > prompt_budget:
        prompt_ids = prompt_ids[:, -prompt_budget:]
    prompt_ids = prompt_ids.to(device)
    answer_ids = answer_ids.to(device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.shape[1] :] = answer_ids
    return model(input_ids=input_ids, labels=labels).loss


def _clamp_fp_penalty_cutoff(value: float, minimum: float, maximum: float) -> float:
    low = min(max(float(minimum), 0.0), 1.0)
    high = min(max(float(maximum), 0.0), 1.0)
    if high < low:
        low, high = high, low
    return min(max(float(value), low), high)


def _update_fp_penalty_cutoff(
    current_cutoff: float,
    next_threshold: float,
    max_probability: float,
    ema_alpha: float,
    minimum: float,
    maximum: float,
) -> float:
    alpha = min(max(float(ema_alpha), 0.0), 1.0)
    raw_cutoff = 0.5 * (float(next_threshold) + float(max_probability))
    smoothed = (1.0 - alpha) * float(current_cutoff) + alpha * raw_cutoff
    return _clamp_fp_penalty_cutoff(smoothed, minimum, maximum)


def _score_between_average_and_max(average_probability: float, max_probability: float, position: float) -> float:
    position = min(max(float(position), 0.0), 1.0)
    average_probability = float(average_probability)
    max_probability = float(max_probability)
    return min(max(average_probability + position * (max_probability - average_probability), 0.0), 1.0)


def _compute_training_loss(
    model,
    tokenizer,
    tensors: dict,
    batch: list[dict],
    raw_batch: list[dict] | None,
    max_seq_length: int,
    fp_dynamic_penalty: bool = False,
    fp_penalty_weight: float = 0.0,
    fp_penalty_cutoff: float = 0.5,
    positive_loss_weight: float = 1.0,
    negative_loss_weight: float = 1.0,
    drop6_negative_loss_weight: float = 1.2,
    neutral_negative_loss_weight: float = 1.0,
    high_score_positive_bonus: float = 0.0,
    high_score_positive_cutoff: float = 1.0,
    micro_step: int | None = None,
    gradient_accumulation_steps: int | None = None,
):
    output = model(**tensors)
    metrics = {}
    base_loss = output.loss
    positive_bonus = max(0.0, float(high_score_positive_bonus))
    use_weighted_ce = (
        float(positive_loss_weight) != 1.0
        or float(negative_loss_weight) != 1.0
        or float(drop6_negative_loss_weight) != 1.0
        or float(neutral_negative_loss_weight) != 1.0
        or positive_bonus > 0.0
    )
    if raw_batch and use_weighted_ce:
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None:
            weighted_losses = []
            weight_sums = {"positive": 0.0, NEGATIVE_KIND_DROP6: 0.0, NEGATIVE_KIND_NEUTRAL: 0.0}
            weight_counts = {"positive": 0, NEGATIVE_KIND_DROP6: 0, NEGATIVE_KIND_NEUTRAL: 0}
            high_score_positive_count = 0
            high_score_positive_probabilities = []
            positive_cutoff = min(max(float(high_score_positive_cutoff), 0.0), 1.0)
            for row in raw_batch:
                label = _label_from_training_row(row)
                if label is None:
                    continue
                label = 1 if int(label) == 1 else 0
                weight, weight_key = _loss_weight_for_training_row(
                    row,
                    label,
                    positive_loss_weight=positive_loss_weight,
                    negative_loss_weight=negative_loss_weight,
                    drop6_negative_loss_weight=drop6_negative_loss_weight,
                    neutral_negative_loss_weight=neutral_negative_loss_weight,
                )
                label_loss = _differentiable_answer_loss(
                    model,
                    tokenizer,
                    row,
                    label,
                    max_seq_length,
                    tensors["input_ids"].device,
                )
                if label == 1 and positive_bonus > 0.0:
                    negative_loss = _differentiable_answer_loss(
                        model,
                        tokenizer,
                        row,
                        0,
                        max_seq_length,
                        tensors["input_ids"].device,
                    )
                    positive_probability = torch.sigmoid(negative_loss - label_loss)
                    high_score_positive_probabilities.append(positive_probability.detach())
                    if float(positive_probability.detach().cpu()) >= positive_cutoff:
                        high_score_positive_count += 1
                        weight *= 1.0 + positive_bonus
                if weight_key in weight_sums:
                    weight_sums[weight_key] += max(0.0, weight)
                    weight_counts[weight_key] += 1
                weighted_losses.append(label_loss * max(0.0, weight))
            if weighted_losses:
                base_loss = torch.stack(weighted_losses).mean()
                metrics["weighted_ce"] = float(base_loss.detach().cpu())
                for key, count in weight_counts.items():
                    if count:
                        metrics[f"{key}_loss_weight"] = weight_sums[key] / count
                if positive_bonus > 0.0:
                    metrics["high_pos_cutoff"] = positive_cutoff
                    metrics["high_pos_hits"] = float(high_score_positive_count)
                    if high_score_positive_probabilities:
                        metrics["high_pos_p"] = float(torch.stack(high_score_positive_probabilities).mean().cpu())
    if not fp_dynamic_penalty or fp_penalty_weight <= 0 or not raw_batch:
        return base_loss, metrics
    try:
        import torch
    except Exception:
        return base_loss, metrics

    penalties = []
    probabilities = []
    cutoff = min(max(float(fp_penalty_cutoff), 0.0), 1.0)
    for row in raw_batch:
        if _label_from_training_row(row) != 0:
            continue
        positive_loss = _differentiable_answer_loss(model, tokenizer, row, 1, max_seq_length, tensors["input_ids"].device)
        negative_loss = _differentiable_answer_loss(model, tokenizer, row, 0, max_seq_length, tensors["input_ids"].device)
        positive_probability = torch.sigmoid(negative_loss - positive_loss)
        probabilities.append(positive_probability.detach())
        penalties.append(_high_scoring_negative_penalty(positive_probability, cutoff))
    if not penalties:
        return base_loss, metrics
    fp_penalty = torch.stack(penalties).mean()
    loss = base_loss + float(fp_penalty_weight) * fp_penalty
    metrics["fp_penalty"] = float(fp_penalty.detach().cpu())
    metrics["fp_cutoff"] = cutoff
    if probabilities:
        metrics["fp_neg_p"] = float(torch.stack(probabilities).mean().cpu())
    return loss, metrics


def _high_scoring_negative_penalty(positive_probability, cutoff: float):
    return positive_probability.sub(float(cutoff)).relu()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _params_are_finite(params) -> bool:
    for param in params:
        if not param.data.isfinite().all().item():
            return False
    return True


def _sample_eval_rows(
    rows: list[dict],
    max_samples: int,
    update: int,
    method: str | None = None,
) -> tuple[list[dict], str, int | None]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows, "all", None
    method = (method or os.environ.get("EVAL_SAMPLE_METHOD", "fixed")).strip().lower()
    if method == "random":
        seed = _env_int("EVAL_RANDOM_SEED", 20260530) + int(update)
        return random.Random(seed).sample(rows, max_samples), "random", seed
    if method == "stride":
        step = len(rows) / max_samples
        return [rows[min(int(index * step), len(rows) - 1)] for index in range(max_samples)], "stride", None
    return rows[:max_samples], "fixed", None


def _next_evaluation_threshold(
    probabilities: list[float],
    current_threshold: float,
    threshold_position: float = 0.2,
) -> tuple[float, float, float]:
    if not probabilities:
        value = min(max(float(current_threshold), 0.0), 1.0)
        return value, value, value
    average_probability = sum(probabilities) / len(probabilities)
    max_probability = max(probabilities)
    position = min(max(float(threshold_position), 0.0), 1.0)
    next_threshold = average_probability + position * (max_probability - average_probability)
    return average_probability, max_probability, min(max(next_threshold, 0.0), 1.0)



def _evaluate_training_checkpoint(
    model,
    tokenizer,
    rows: list[dict],
    output_dir: Path,
    update: int,
    total_updates: int,
    progress: float,
    trained_epochs: float,
    threshold: float,
    threshold_position: float,
    max_samples: int,
    max_seq_length: int,
    precision_top_k: int,
    precision_threshold: float,
) -> dict:
    eval_rows, sample_method, sample_seed = _sample_eval_rows(rows, max_samples, update)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    tp = fp = tn = fn = positives = 0
    scored_rows: list[dict] = []
    eval_total = len(eval_rows)
    eval_start = time.monotonic()
    print(
        f"evaluation start update={update}/{total_updates} "
        f"progress={progress * 100:.2f}% samples={eval_total} "
        f"sample_method={sample_method} sample_seed={sample_seed} threshold={threshold}",
        flush=True,
    )
    try:
        for index, row in enumerate(eval_rows, start=1):
            messages = compact_messages_from_sample(row)
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            pred = score_prediction(model, tokenizer, prompt, max_seq_length, threshold)
            predicted_positive = pred["label"] == "positive"
            actual_positive = int(row["metadata"]["label"]) == 1
            scored_rows.append(
                {
                    "actual_label": 1 if actual_positive else 0,
                    "positive_probability": pred["positive_probability"],
                }
            )
            if actual_positive:
                positives += 1
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
            elif not predicted_positive and actual_positive:
                fn += 1
            else:
                tn += 1
            if index == eval_total or index % 100 == 0:
                elapsed = time.monotonic() - eval_start
                eval_progress = index / eval_total if eval_total else 1.0
                remaining = elapsed * (1.0 - eval_progress) / eval_progress if eval_progress > 0 else 0.0
                running_recall = tp / positives if positives else 0.0
                running_precision = tp / (tp + fp) if tp + fp else 0.0
                print(
                    f"evaluation progress {index}/{eval_total} "
                    f"({eval_progress * 100:.2f}%) "
                    f"elapsed={_format_duration(elapsed)} "
                    f"remaining={_format_duration(remaining)} "
                    f"positive_recall={running_recall:.4f} precision={running_precision:.4f}",
                    flush=True,
                )
    finally:
        if was_training:
            model.train()
    positive_recall = tp / positives if positives else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    precision_top_k = normalize_precision_top_k(precision_top_k)
    precision_threshold = normalize_precision_threshold(precision_threshold)
    precision_values = precision_at_k(scored_rows, sorted({*REPORTED_PRECISION_KS, precision_top_k}))
    target_key = f"precision@{precision_top_k}"
    probabilities = [float(row["positive_probability"]) for row in scored_rows]
    threshold_position = min(max(float(threshold_position), 0.0), 1.0)
    average_probability, max_probability, next_threshold = _next_evaluation_threshold(
        probabilities,
        threshold,
        threshold_position,
    )
    result = {
        "update": update,
        "total_updates": total_updates,
        "trigger": "checkpoint",
        "progress": progress,
        "trained_epochs": trained_epochs,
        "samples": len(eval_rows),
        "source_samples": len(rows),
        "sample_method": sample_method,
        "sample_seed": sample_seed,
        "positive_samples": positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_recall": positive_recall,
        "precision": precision,
        **precision_values,
        "threshold": threshold,
        "average_positive_probability": average_probability,
        "max_positive_probability": max_probability,
        "threshold_position": threshold_position,
        "next_threshold": next_threshold,
        "precision_top_k": precision_top_k,
        "precision_threshold": precision_threshold,
        "passed": precision_values[target_key] >= precision_threshold,
        "max_seq_length": max_seq_length,
    }
    output_path = output_dir / f"eval-update-{update:06d}-progress-{int(round(progress * 1000)):04d}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(
        f"evaluation saved: {output_path} positive_recall={positive_recall:.4f} "
        f"precision={precision:.4f} precision@5={precision_values['precision@5']:.4f} "
        f"precision@10={precision_values['precision@10']:.4f} "
        f"precision@20={precision_values['precision@20']:.4f} "
        f"precision@50={precision_values['precision@50']:.4f} "
        f"avg_p={average_probability:.4f} max_p={max_probability:.4f} "
        f"next_threshold={next_threshold:.4f} "
        f"{target_key}={precision_values[target_key]:.4f} target={precision_threshold:.4f} "
        f"passed={result['passed']} samples={len(eval_rows)}",
        flush=True,
    )
    return result
def _checkpoint_update(checkpoint_dir: Path | None) -> int:
    if checkpoint_dir is None:
        return 0
    match = re.fullmatch(r"update-(\d+)", checkpoint_dir.name)
    if not match:
        return 0
    return int(match.group(1))


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None
    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and (path / "adapter_config.json").exists() and _checkpoint_update(path) > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_update)


def _token_cache_path(data_dir: Path, train_path: Path, base_model: str, max_seq_length: int) -> Path:
    stat = train_path.stat()
    fingerprint = hashlib.sha256(
        f"{TOKEN_CACHE_VERSION}|{train_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{base_model}|{max_seq_length}".encode("utf-8")
    ).hexdigest()[:16]
    model_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_model).strip("_") or "model"
    return data_dir / "tokenized" / f"{model_key}_seq{max_seq_length}_{fingerprint}.pkl"


def _load_or_build_tokenized(tokenizer, rows: list[dict], data_dir: Path, train_path: Path, base_model: str, max_seq_length: int, rebuild: bool) -> list[dict]:
    cache_path = _token_cache_path(data_dir, train_path, base_model, max_seq_length)
    if cache_path.exists() and not rebuild:
        load_start = time.monotonic()
        with cache_path.open("rb") as file:
            tokenized = pickle.load(file)
        print(
            f"loaded tokenized train cache rows={len(tokenized)} path={cache_path} "
            f"elapsed={_format_duration(time.monotonic() - load_start)}",
            flush=True,
        )
        return tokenized

    tokenized = []
    tokenize_start = time.monotonic()
    tokenize_total = len(rows)
    print(f"tokenizing train rows={tokenize_total} max_seq_length={max_seq_length}", flush=True)
    for index, row in enumerate(rows, start=1):
        tokenized.append(_tokenize_row(tokenizer, row, max_seq_length))
        if index % 5000 == 0 or index == tokenize_total:
            elapsed = time.monotonic() - tokenize_start
            progress = index / tokenize_total if tokenize_total else 1.0
            remaining = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            print(
                f"tokenize progress {index}/{tokenize_total} "
                f"({progress * 100:.2f}%) "
                f"elapsed={_format_duration(elapsed)} "
                f"remaining={_format_duration(remaining)}",
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(tokenized, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"tokenized train cache saved rows={len(tokenized)} path={cache_path}", flush=True)
    return tokenized


def train_recall60_lora(
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    checkpoint_eval_data_dir: Path | None,
    max_seq_length: int,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    train_seed: int,
    max_grad_norm: float,
    lora_rank: int,
    lora_dropout: float,
    checkpoint_every: int,
    resume_adapter_dir: Path | None,
    initial_adapter_dir: Path | None,
    nonfinite_patience: int,
    rebuild_token_cache: bool,
    auto_resume: bool,
    oom_patience: int,
    nonfinite_skip_limit: int,
    nonfinite_backoff_every: int,
    lr_backoff_factor: float,
    min_learning_rate: float,
    evaluation_threshold: float,
    evaluation_threshold_position: float,
    evaluation_max_samples: int,
    evaluation_output_dir: Path | None,
    evaluation_precision_top_k: int,
    evaluation_precision_threshold: float,
    on_the_fly_tokenize: bool,
    positive_loss_weight: float,
    negative_loss_weight: float,
    high_score_positive_bonus: float,
    high_score_positive_position: float,
    fp_dynamic_penalty: bool,
    fp_penalty_weight: float,
    fp_threshold_ema_alpha: float,
    fp_threshold_min: float,
    fp_threshold_max: float,
    drop6_negative_loss_weight: float = 1.2,
    neutral_negative_loss_weight: float = 1.0,
) -> None:
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise _missing_dataset_error(data_dir)
    checkpoint_eval_path = _checkpoint_eval_path(data_dir, checkpoint_eval_data_dir)
    if not checkpoint_eval_path.exists():
        raise FileNotFoundError(f"missing checkpoint evaluation dataset: {checkpoint_eval_path}")
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("missing training dependencies; run one_click_deploy.ps1 first") from exc

    rows = read_jsonl(train_path)
    valid_rows = read_jsonl(checkpoint_eval_path)
    if not rows or not valid_rows:
        raise RuntimeError(f"dataset is empty or incomplete: train={len(rows)} test={len(valid_rows)}")

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora_rank = max(1, int(lora_rank))
    lora_dropout = min(max(float(lora_dropout), 0.0), 1.0)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if resume_adapter_dir is None and auto_resume:
        resume_adapter_dir = _latest_checkpoint(output_dir)
    if resume_adapter_dir is not None and (resume_adapter_dir / "adapter_config.json").exists():
        print(f"resuming adapter from {resume_adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(resume_adapter_dir), is_trainable=True)
    elif initial_adapter_dir is not None and (initial_adapter_dir / "adapter_config.json").exists():
        print(f"initializing adapter from {initial_adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(initial_adapter_dir), is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    for param in trainable_params:
        param.data = param.data.float()
    model.to(device)
    model.train()

    max_seq_length = max(64, max_seq_length)
    if on_the_fly_tokenize:
        train_items = rows
        print(
            f"on-the-fly tokenization enabled; skipping tokenized train cache to reduce RAM rows={len(rows)}",
            flush=True,
        )
    else:
        train_items = _load_or_build_tokenized(
            tokenizer,
            rows,
            data_dir,
            train_path,
            base_model,
            max_seq_length,
            rebuild_token_cache,
        )
    prepared_hook_result = _after_train_items_prepared(
        rows=rows,
        train_items=train_items,
        train_path=train_path,
        on_the_fly_tokenize=on_the_fly_tokenize,
    )
    if prepared_hook_result:
        hook_text = " ".join(f"{key}={value}" for key, value in sorted(prepared_hook_result.items()))
        print(f"train items prepared hook: {hook_text}", flush=True)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=max(0.0, float(weight_decay)))
    scheduler = None
    total_updates = max(1, math.ceil((len(train_items) * max(epochs, 0.001)) / max(1, batch_size * gradient_accumulation_steps)))
    total_micro_steps = total_updates * max(1, gradient_accumulation_steps)
    start_update = min(_checkpoint_update(resume_adapter_dir), total_updates)
    start_micro_step = start_update * max(1, gradient_accumulation_steps)
    evaluation_output_dir = evaluation_output_dir or (output_dir / "evaluations")
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    rng = random.Random(train_seed)
    fp_penalty_cutoff = _clamp_fp_penalty_cutoff(evaluation_threshold, fp_threshold_min, fp_threshold_max)
    high_score_positive_cutoff = min(max(float(evaluation_threshold), 0.0), 1.0)
    train_order = list(range(len(train_items)))
    restored_total_nonfinite_skips = 0
    restored_training_state = False
    state = _load_training_state(resume_adapter_dir, device) if resume_adapter_dir is not None else None
    if state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
            optimizer.state = _torch_state_to_device(optimizer.state, device)
            if scheduler is not None and state.get("scheduler_state") is not None:
                scheduler.load_state_dict(state["scheduler_state"])
            start_micro_step = min(int(state.get("next_micro_step", start_micro_step)), total_micro_steps)
            start_update = min(int(state.get("update", start_update)), total_updates)
            saved_order = state.get("train_order")
            if isinstance(saved_order, list) and len(saved_order) == len(train_items):
                train_order = [int(index) for index in saved_order]
            if state.get("python_random_state") is not None:
                random.setstate(state["python_random_state"])
            if state.get("train_rng_state") is not None:
                rng.setstate(state["train_rng_state"])
            if state.get("torch_rng_state") is not None:
                torch.random.set_rng_state(state["torch_rng_state"])
            if torch.cuda.is_available() and state.get("torch_cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
            evaluation_threshold = float(state.get("evaluation_threshold", evaluation_threshold))
            high_score_positive_cutoff = float(state.get("high_score_positive_cutoff", high_score_positive_cutoff))
            fp_penalty_cutoff = float(state.get("fp_penalty_cutoff", fp_penalty_cutoff))
            restored_total_nonfinite_skips = int(state.get("total_nonfinite_skips", 0))
            _load_extra_training_state(state.get("extra") or {})
            restored_training_state = True
            print(
                f"training state restored: {_training_state_path(resume_adapter_dir)} "
                f"next_micro_step={start_micro_step} update={start_update} "
                f"eval_threshold={evaluation_threshold:.6f} "
                f"high_score_positive_cutoff={high_score_positive_cutoff:.6f} "
                f"fp_penalty_cutoff={fp_penalty_cutoff:.6f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"WARNING failed to restore full training state from {_training_state_path(resume_adapter_dir)}: {exc}. "
                f"Continuing from adapter weights only.",
                flush=True,
            )
    elif resume_adapter_dir is not None and _checkpoint_update(resume_adapter_dir) > 0:
        print(
            f"WARNING no {TRAINING_STATE_FILE} found in {resume_adapter_dir}; "
            f"optimizer/RNG/dynamic-threshold state will restart from configured defaults.",
            flush=True,
        )
    if not restored_training_state:
        train_order = _build_train_order(train_items, train_seed, rng)
    print(
        _training_run_summary(
            train_rows=len(train_items),
            valid_rows=len(valid_rows),
            checkpoint_eval_data_dir=checkpoint_eval_data_dir or data_dir,
            total_updates=total_updates,
            start_update=start_update,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_seed=train_seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            max_seq_length=max_seq_length,
            on_the_fly_tokenize=on_the_fly_tokenize,
            positive_loss_weight=positive_loss_weight,
            negative_loss_weight=negative_loss_weight,
            drop6_negative_loss_weight=drop6_negative_loss_weight,
            neutral_negative_loss_weight=neutral_negative_loss_weight,
            high_score_positive_bonus=high_score_positive_bonus,
            high_score_positive_position=high_score_positive_position,
            high_score_positive_cutoff=high_score_positive_cutoff,
            checkpoint_every=checkpoint_every,
            evaluation_threshold=evaluation_threshold,
            evaluation_threshold_position=evaluation_threshold_position,
            evaluation_precision_top_k=evaluation_precision_top_k,
            evaluation_precision_threshold=evaluation_precision_threshold,
            evaluation_max_samples=evaluation_max_samples,
            fp_dynamic_penalty=fp_dynamic_penalty,
            fp_penalty_weight=fp_penalty_weight,
            fp_threshold_ema_alpha=fp_threshold_ema_alpha,
            fp_threshold_min=fp_threshold_min,
            fp_threshold_max=fp_threshold_max,
            fp_penalty_cutoff=fp_penalty_cutoff,
        ),
        flush=True,
    )
    if start_update >= total_updates:
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"checkpoint already reached target updates; adapter saved: {adapter_dir}", flush=True)
        return
    optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    consecutive_nonfinite = 0
    total_nonfinite_skips = restored_total_nonfinite_skips
    consecutive_oom = 0
    accumulated_loss = 0.0
    accumulated_loss_count = 0
    accumulated_metric_sums: dict[str, float] = {}
    accumulated_metric_counts: dict[str, int] = {}
    for micro_step in range(start_micro_step, total_micro_steps):
        update = (micro_step + 1) // max(1, gradient_accumulation_steps)
        try:
            if micro_step > 0 and micro_step % len(train_order) == 0:
                _reshuffle_train_order(train_order, train_items, rng)
            raw_batch = [
                train_items[train_order[(micro_step * batch_size + offset) % len(train_order)]]
                for offset in range(batch_size)
            ]
            if on_the_fly_tokenize:
                batch = [_tokenize_row(tokenizer, row, max_seq_length) for row in raw_batch]
                penalty_rows = raw_batch
            else:
                batch = raw_batch
                penalty_rows = None
            tensors = _collate(tokenizer, batch, device)
            raw_loss, loss_metrics = _compute_training_loss(
                model,
                tokenizer,
                tensors,
                batch,
                penalty_rows,
                max_seq_length,
                fp_dynamic_penalty=fp_dynamic_penalty,
                fp_penalty_weight=fp_penalty_weight,
                fp_penalty_cutoff=fp_penalty_cutoff,
                positive_loss_weight=positive_loss_weight,
                negative_loss_weight=negative_loss_weight,
                drop6_negative_loss_weight=drop6_negative_loss_weight,
                neutral_negative_loss_weight=neutral_negative_loss_weight,
                high_score_positive_bonus=high_score_positive_bonus,
                high_score_positive_cutoff=high_score_positive_cutoff,
                micro_step=micro_step,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if not torch.isfinite(raw_loss.detach()):
                consecutive_nonfinite += 1
                total_nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"WARNING skipped non-finite loss before update {max(1, update)}: "
                    f"{float(raw_loss.detach().cpu())} consecutive={consecutive_nonfinite}/{nonfinite_patience} "
                    f"total={total_nonfinite_skips}/{nonfinite_skip_limit}",
                    flush=True,
                )
                if nonfinite_backoff_every > 0 and total_nonfinite_skips % nonfinite_backoff_every == 0:
                    for group in optimizer.param_groups:
                        group["lr"] = max(min_learning_rate, float(group["lr"]) * lr_backoff_factor)
                    print(f"WARNING reduced learning rate after non-finite skips: lr={optimizer.param_groups[0]['lr']}", flush=True)
                if total_nonfinite_skips >= max(1, nonfinite_skip_limit):
                    raise RuntimeError(
                        f"Training aborted after {total_nonfinite_skips} total non-finite skips near update {max(1, update)}. "
                        f"Resume from an earlier checkpoint and/or lower --learning-rate."
                    )
                if consecutive_nonfinite >= max(1, nonfinite_patience):
                    raise RuntimeError(
                        f"Training aborted after {consecutive_nonfinite} consecutive non-finite losses near update {max(1, update)}. "
                        f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                    )
                continue
            consecutive_nonfinite = 0
            consecutive_oom = 0
            accumulated_loss += float(raw_loss.detach().cpu())
            accumulated_loss_count += 1
            for metric_name, metric_value in loss_metrics.items():
                accumulated_metric_sums[metric_name] = accumulated_metric_sums.get(metric_name, 0.0) + float(metric_value)
                accumulated_metric_counts[metric_name] = accumulated_metric_counts.get(metric_name, 0) + 1
            loss = raw_loss / max(1, gradient_accumulation_steps)
            loss.backward()
        except torch.OutOfMemoryError as exc:
            consecutive_oom += 1
            optimizer.zero_grad(set_to_none=True)
            tensors = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for group in optimizer.param_groups:
                group["lr"] = max(min_learning_rate, float(group["lr"]) * 0.5)
            print(
                f"WARNING skipped CUDA OOM before update {max(1, update)} "
                f"micro_step={micro_step + 1}/{total_micro_steps} "
                f"consecutive={consecutive_oom}/{oom_patience} "
                f"max_seq_length={max_seq_length} lr={optimizer.param_groups[0]['lr']}: {exc}",
                flush=True,
            )
            if consecutive_oom >= max(1, oom_patience):
                raise RuntimeError(
                    f"Training aborted after {consecutive_oom} consecutive CUDA OOM errors near update {max(1, update)}. "
                    f"MAX_SEQ_LENGTH was not changed automatically; set --max-seq-length explicitly if you want a smaller value."
                )
            continue
        if (micro_step + 1) % max(1, gradient_accumulation_steps) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            if not torch.isfinite(grad_norm.detach()):
                consecutive_nonfinite += 1
                total_nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"WARNING skipped update {update}/{total_updates} because grad_norm is non-finite: "
                    f"{float(grad_norm.detach().cpu())} consecutive={consecutive_nonfinite}/{nonfinite_patience} "
                    f"total={total_nonfinite_skips}/{nonfinite_skip_limit}",
                    flush=True,
                )
                if nonfinite_backoff_every > 0 and total_nonfinite_skips % nonfinite_backoff_every == 0:
                    for group in optimizer.param_groups:
                        group["lr"] = max(min_learning_rate, float(group["lr"]) * lr_backoff_factor)
                    print(f"WARNING reduced learning rate after non-finite skips: lr={optimizer.param_groups[0]['lr']}", flush=True)
                if total_nonfinite_skips >= max(1, nonfinite_skip_limit):
                    raise RuntimeError(
                        f"Training aborted after {total_nonfinite_skips} total non-finite skips near update {update}. "
                        f"Resume from an earlier checkpoint and/or lower --learning-rate."
                    )
                if consecutive_nonfinite >= max(1, nonfinite_patience):
                    raise RuntimeError(
                        f"Training aborted after {consecutive_nonfinite} consecutive non-finite gradients near update {update}. "
                        f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                    )
                continue
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if not _params_are_finite(trainable_params):
                raise RuntimeError(
                    f"Training aborted because trainable LoRA parameters became non-finite after update {update}. "
                    f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                )
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - start_time
            progress = update / total_updates
            remaining = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            eta_epoch = time.localtime(time.time() + remaining)
            update_loss = accumulated_loss / max(1, accumulated_loss_count)
            update_metrics = {
                name: accumulated_metric_sums[name] / max(1, accumulated_metric_counts.get(name, 0))
                for name in sorted(accumulated_metric_sums)
            }
            loss_metric_text = "".join(f"{name}={value:.4f} " for name, value in update_metrics.items())
            print(
                f"train update {update}/{total_updates} "
                f"({progress * 100:.2f}%) "
                f"loss={update_loss:.4f} "
                f"{loss_metric_text}"
                f"grad_norm={float(grad_norm.detach().cpu()):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"elapsed={_format_duration(elapsed)} "
                f"remaining={_format_duration(remaining)} "
                f"eta={time.strftime('%Y-%m-%d %H:%M:%S', eta_epoch)}",
                flush=True,
            )
            accumulated_loss = 0.0
            accumulated_loss_count = 0
            accumulated_metric_sums.clear()
            accumulated_metric_counts.clear()
            if checkpoint_every > 0 and update % checkpoint_every == 0:
                checkpoint_dir = output_dir / "checkpoints" / f"update-{update:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"checkpoint saved: {checkpoint_dir}", flush=True)
                _save_training_state(
                    checkpoint_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=update,
                    next_micro_step=micro_step + 1,
                    train_order=train_order,
                    rng=rng,
                    evaluation_threshold=evaluation_threshold,
                    high_score_positive_cutoff=high_score_positive_cutoff,
                    fp_penalty_cutoff=fp_penalty_cutoff,
                    total_nonfinite_skips=total_nonfinite_skips,
                )
                evaluation_result = _evaluate_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    rows=valid_rows,
                    output_dir=evaluation_output_dir,
                    update=update,
                    total_updates=total_updates,
                    progress=min(progress, 1.0),
                    trained_epochs=epochs * min(progress, 1.0),
                    threshold=evaluation_threshold,
                    threshold_position=evaluation_threshold_position,
                    max_samples=evaluation_max_samples,
                    max_seq_length=max_seq_length,
                    precision_top_k=evaluation_precision_top_k,
                    precision_threshold=evaluation_precision_threshold,
                )
                evaluation_threshold = float(evaluation_result["next_threshold"])
                high_score_positive_cutoff = _score_between_average_and_max(
                    evaluation_result["average_positive_probability"],
                    evaluation_result["max_positive_probability"],
                    high_score_positive_position,
                )
                if fp_dynamic_penalty:
                    fp_penalty_cutoff = _update_fp_penalty_cutoff(
                        current_cutoff=fp_penalty_cutoff,
                        next_threshold=float(evaluation_result["next_threshold"]),
                        max_probability=float(evaluation_result["max_positive_probability"]),
                        ema_alpha=fp_threshold_ema_alpha,
                        minimum=fp_threshold_min,
                        maximum=fp_threshold_max,
                    )
                checkpoint_hook_result = _after_checkpoint_evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    rows=rows,
                    train_items=train_items,
                    train_path=train_path,
                    train_order=train_order,
                    checkpoint_dir=checkpoint_dir,
                    update=update,
                    max_seq_length=max_seq_length,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    on_the_fly_tokenize=on_the_fly_tokenize,
                )
                if checkpoint_hook_result:
                    hook_text = " ".join(
                        f"{key}={value}" for key, value in sorted(checkpoint_hook_result.items())
                    )
                    print(f"checkpoint post-evaluation hook: {hook_text}", flush=True)
                print(
                    f"evaluation threshold updated for next checkpoint: {evaluation_threshold:.6f} "
                    f"high_score_positive_cutoff={high_score_positive_cutoff:.6f} "
                    f"fp_penalty_cutoff={fp_penalty_cutoff:.6f}",
                    flush=True,
                )
                _save_training_state(
                    checkpoint_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=update,
                    next_micro_step=micro_step + 1,
                    train_order=train_order,
                    rng=rng,
                    evaluation_threshold=evaluation_threshold,
                    high_score_positive_cutoff=high_score_positive_cutoff,
                    fp_penalty_cutoff=fp_penalty_cutoff,
                    total_nonfinite_skips=total_nonfinite_skips,
                )

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adapter saved: {adapter_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for recall60 black-box stock classification")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--checkpoint-eval-data-dir", type=Path, default=None, help="Dataset directory whose test.jsonl is used for checkpoint evaluation. Defaults to --data-dir.")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Override token length; default follows sample mode")
    parser.add_argument("--epochs", type=float, default=_env_float("EPOCHS", 0.3))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=_env_int("GRADIENT_ACCUMULATION_STEPS", 16))
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=_env_float("WEIGHT_DECAY", 0.0), help="AdamW weight decay. Values like 0.01 can reduce overfitting; 0 keeps prior behavior.")
    parser.add_argument(
        "--train-seed",
        type=int,
        default=_env_int("TRAIN_SEED", _env_int("RANDOM_SEED", DEFAULT_TRAIN_SEED)),
        help="Training shuffle and PyTorch seed. TRAIN_SEED takes precedence over RANDOM_SEED.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Clip LoRA gradients and skip non-finite updates.")
    parser.add_argument("--lora-rank", type=int, default=_env_int("LORA_RANK", 16), help="LoRA rank. Lower values reduce trainable capacity and can reduce overfitting.")
    parser.add_argument("--lora-dropout", type=float, default=_env_float("LORA_DROPOUT", 0.05), help="LoRA dropout in [0, 1]. Higher values can reduce overfitting.")
    parser.add_argument("--checkpoint-every", type=int, default=default_checkpoint_every(), help="Save adapter checkpoint every N optimizer updates; 0 disables checkpoints.")
    parser.add_argument("--eval-every-epoch-fraction", type=float, default=0.0, help="Deprecated compatibility option. Evaluation now runs immediately after every checkpoint.")
    parser.add_argument("--eval-threshold", type=float, default=_env_float("EVAL_THRESHOLD", 0.48), help="Threshold used by in-training evaluation.")
    parser.add_argument("--eval-threshold-position", type=float, default=_env_float("EVAL_THRESHOLD_POSITION", 0.2), help="Position between average positive probability and max probability used for the next dynamic evaluation threshold.")
    parser.add_argument("--eval-precision-top-k", type=int, default=_env_int("EVAL_PRECISION_TOP_K", 10), help="K used by the in-training precision@K gate.")
    parser.add_argument("--eval-precision-threshold", type=float, default=_env_float("EVAL_PRECISION_THRESHOLD", 0.40), help="Required precision@K value for an evaluation checkpoint to pass.")
    parser.add_argument("--eval-max-samples", type=int, default=_env_int("EVAL_MAX_SAMPLES", 0), help="Max test samples for in-training evaluation; 0 means all.")
    parser.add_argument("--eval-output-dir", type=Path, default=None, help="Directory for in-training evaluation JSON files; default is output-dir/evaluations.")
    parser.add_argument("--positive-loss-weight", type=float, default=_env_float("POSITIVE_LOSS_WEIGHT", 1.2), help="Multiplier applied to positive-sample CE loss during on-the-fly training.")
    parser.add_argument("--negative-loss-weight", type=float, default=_env_float("NEGATIVE_LOSS_WEIGHT", 1.0), help="Compatibility fallback multiplier for negative-sample CE loss during on-the-fly training.")
    parser.add_argument("--drop6-negative-loss-weight", type=float, default=_env_float("DROP6_NEGATIVE_LOSS_WEIGHT", 1.2), help="Multiplier applied to drop6 negative-sample CE loss when metadata.negative_kind=drop6.")
    parser.add_argument("--neutral-negative-loss-weight", type=float, default=_env_float("NEUTRAL_NEGATIVE_LOSS_WEIGHT", 1.0), help="Multiplier applied to neutral or untagged negative-sample CE loss.")
    parser.add_argument("--high-score-positive-bonus", type=float, default=_env_float("HIGH_SCORE_POSITIVE_BONUS", 0.0), help="Extra bonus for positive samples whose score is at or above the high-score positive cutoff. 1.0 doubles the positive-sample weight.")
    parser.add_argument("--high-score-positive-position", type=float, default=_env_float("HIGH_SCORE_POSITIVE_POSITION", 0.8), help="Position between last avg_p and max_p used as the high-score positive cutoff after checkpoint evaluation.")
    parser.add_argument("--fp-dynamic-penalty", action="store_true", default=_env_bool("FP_DYNAMIC_PENALTY", False), help="Enable extra loss for negative samples whose positive probability exceeds the dynamic FP cutoff.")
    parser.add_argument("--fp-penalty-weight", type=float, default=_env_float("FP_PENALTY_WEIGHT", 1.0), help="Weight of the high-scoring negative-sample penalty.")
    parser.add_argument("--fp-threshold-ema-alpha", type=float, default=_env_float("FP_THRESHOLD_EMA_ALPHA", 0.2), help="EMA alpha used when updating the dynamic FP cutoff after checkpoint evaluation.")
    parser.add_argument("--fp-threshold-min", type=float, default=_env_float("FP_THRESHOLD_MIN", 0.40), help="Lower bound for the dynamic FP penalty cutoff.")
    parser.add_argument("--fp-threshold-max", type=float, default=_env_float("FP_THRESHOLD_MAX", 0.65), help="Upper bound for the dynamic FP penalty cutoff.")
    parser.add_argument("--resume-adapter-dir", type=Path, default=None, help="Resume LoRA training from an adapter checkpoint directory.")
    parser.add_argument("--initial-adapter-dir", type=Path, default=None, help="Initialize LoRA weights from an adapter directory but start training at update 0.")
    parser.add_argument("--nonfinite-patience", type=int, default=20, help="Abort after this many consecutive non-finite losses.")
    parser.add_argument("--nonfinite-skip-limit", type=int, default=100, help="Abort after this many total non-finite losses or gradients.")
    parser.add_argument("--nonfinite-backoff-every", type=int, default=10, help="Reduce optimizer LR after every N total non-finite skips; 0 disables.")
    parser.add_argument("--lr-backoff-factor", type=float, default=0.5, help="Multiplier used when non-finite LR backoff is triggered.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Smallest LR allowed by automatic non-finite backoff.")
    parser.add_argument("--oom-patience", type=int, default=20, help="Abort after this many consecutive CUDA OOM batches.")
    parser.add_argument("--rebuild-token-cache", action="store_true", help="Re-tokenize train.jsonl even when a tokenized cache exists.")
    parser.add_argument("--on-the-fly-tokenize", action="store_true", default=_env_bool("ON_THE_FLY_TOKENIZE", False), help="Tokenize each training row when it is used instead of loading or building the token cache.")
    parser.add_argument("--no-auto-resume", action="store_true", help="Do not automatically resume from the latest output-dir checkpoint.")
    parser.add_argument("--no-4bit", action="store_true", help="Accepted for script compatibility; recall60 Windows training uses fp16 LoRA on RTX3060.")
    parser.add_argument("--cuda-device", default="0", help="CUDA device id. Default binds the RTX3060 as cuda:0.")
    parser.add_argument("--allow-non-rtx3060", action="store_true", help="Allow CUDA devices whose name is not RTX 3060.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    train_recall60_lora(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_eval_data_dir=args.checkpoint_eval_data_dir,
        max_seq_length=max(64, args.max_seq_length or default_max_seq_length()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_seed=args.train_seed,
        max_grad_norm=args.max_grad_norm,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        checkpoint_every=args.checkpoint_every,
        resume_adapter_dir=args.resume_adapter_dir,
        initial_adapter_dir=args.initial_adapter_dir,
        nonfinite_patience=args.nonfinite_patience,
        rebuild_token_cache=args.rebuild_token_cache,
        auto_resume=not args.no_auto_resume,
        oom_patience=args.oom_patience,
        nonfinite_skip_limit=args.nonfinite_skip_limit,
        nonfinite_backoff_every=args.nonfinite_backoff_every,
        lr_backoff_factor=args.lr_backoff_factor,
        min_learning_rate=args.min_learning_rate,
        evaluation_threshold=args.eval_threshold,
        evaluation_threshold_position=args.eval_threshold_position,
        evaluation_max_samples=args.eval_max_samples,
        evaluation_output_dir=args.eval_output_dir,
        evaluation_precision_top_k=args.eval_precision_top_k,
        evaluation_precision_threshold=args.eval_precision_threshold,
        on_the_fly_tokenize=args.on_the_fly_tokenize,
        positive_loss_weight=args.positive_loss_weight,
        negative_loss_weight=args.negative_loss_weight,
        drop6_negative_loss_weight=args.drop6_negative_loss_weight,
        neutral_negative_loss_weight=args.neutral_negative_loss_weight,
        high_score_positive_bonus=args.high_score_positive_bonus,
        high_score_positive_position=args.high_score_positive_position,
        fp_dynamic_penalty=args.fp_dynamic_penalty,
        fp_penalty_weight=args.fp_penalty_weight,
        fp_threshold_ema_alpha=args.fp_threshold_ema_alpha,
        fp_threshold_min=args.fp_threshold_min,
        fp_threshold_max=args.fp_threshold_max,
    )


if __name__ == "__main__":
    main()
