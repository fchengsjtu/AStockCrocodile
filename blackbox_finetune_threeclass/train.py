from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

from blackbox_finetune_recall60 import train as base_train
from blackbox_finetune_threeclass.common import (
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    CLASS_NAMES,
    CLASS_POSITIVE,
    DEFAULT_CLASS_RATIO,
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OUTPUT_DIR,
    compact_messages_from_sample,
    label_answer,
)
from blackbox_finetune_threeclass.gpu import prepare_rtx3060
from blackbox_finetune_threeclass.inference import score_prediction, selection_score
from blackbox_finetune_threeclass.metrics import (
    positive_probability_top_rows,
    selection_score_top_rows,
    summarize_scored_rows,
)

_POSITIVE_CE_WEIGHT = 2.0
_NEGATIVE_CE_WEIGHT = 1.0
_NEUTRAL_CE_WEIGHT = 0.5
_FP_LOSS_WEIGHT = 1.0
_NEUTRAL_FP_LOSS_WEIGHT = 0.3
_HIGH_SCORE_POSITIVE_BONUS = 1.0
_HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER = 8.0
_HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT = 1.0
_HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT = 0.5
_NEGATIVE_PER_POSITIVE = DEFAULT_CLASS_RATIO[1]
_NEUTRAL_PER_POSITIVE = DEFAULT_CLASS_RATIO[2]
_TOP_SCORE_WINDOW: list[tuple[float, int]] = []


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _configure_asymmetric_loss(
    positive_ce_weight: float,
    negative_ce_weight: float,
    neutral_ce_weight: float,
    fp_loss_weight: float,
    neutral_fp_loss_weight: float,
    high_score_positive_bonus: float,
    high_score_positive_bonus_max_multiplier: float,
    high_score_negative_penalty_weight: float,
    high_score_neutral_penalty_weight: float,
) -> None:
    global _POSITIVE_CE_WEIGHT, _NEGATIVE_CE_WEIGHT, _NEUTRAL_CE_WEIGHT
    global _FP_LOSS_WEIGHT, _NEUTRAL_FP_LOSS_WEIGHT
    global _HIGH_SCORE_POSITIVE_BONUS
    global _HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER, _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT
    global _HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT
    _POSITIVE_CE_WEIGHT = max(0.0, float(positive_ce_weight))
    _NEGATIVE_CE_WEIGHT = max(0.0, float(negative_ce_weight))
    _NEUTRAL_CE_WEIGHT = max(0.0, float(neutral_ce_weight))
    _FP_LOSS_WEIGHT = max(0.0, float(fp_loss_weight))
    _NEUTRAL_FP_LOSS_WEIGHT = max(0.0, float(neutral_fp_loss_weight))
    _HIGH_SCORE_POSITIVE_BONUS = max(0.0, float(high_score_positive_bonus))
    _HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER = max(0.0, float(high_score_positive_bonus_max_multiplier))
    _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT = max(0.0, float(high_score_negative_penalty_weight))
    _HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT = max(0.0, float(high_score_neutral_penalty_weight))


def _extra_training_state() -> dict:
    return {}


def _load_extra_training_state(state: dict) -> None:
    return


def _per_sample_answer_nll(logits, labels):
    import torch
    import torch.nn.functional as functional

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_losses = functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _negative_auxiliary_losses(negative_nll, positive_nll):
    import torch.nn.functional as functional

    # Inference assigns probability proportional to exp(-answer_nll).
    # A positive delta therefore means the positive answer is preferred.
    score_delta = negative_nll.float() - positive_nll.float()
    return functional.softplus(score_delta).mean()


def _neutral_false_positive_loss(neutral_nll, positive_nll):
    import torch.nn.functional as functional

    # Penalize neutral samples when the complete positive answer is preferred
    # over the complete neutral answer. This is intentionally separate from
    # negative_fp because neutral contamination is less severe than true
    # downside samples entering the selected top-N pool.
    score_delta = neutral_nll.float() - positive_nll.float()
    return functional.softplus(score_delta).mean()


def _answer_tensors_for_indices(
    tokenizer,
    tensors: dict,
    indices: list[int],
    answer_label: int,
    max_seq_length: int,
) -> dict:
    answer_ids = tokenizer(
        label_answer(answer_label) + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    if len(answer_ids) >= max_seq_length:
        answer_ids = answer_ids[: max_seq_length - 1] + [tokenizer.eos_token_id]
    items = []
    for index in indices:
        prompt_mask = tensors["attention_mask"][index].bool() & tensors["labels"][index].eq(-100)
        prompt_ids = tensors["input_ids"][index][prompt_mask].detach().cpu().tolist()
        prompt_budget = max(1, max_seq_length - len(answer_ids))
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = (prompt_ids + answer_ids)[:max_seq_length]
        items.append(
            {
                "input_ids": input_ids,
                "labels": ([-100] * len(prompt_ids) + answer_ids)[:max_seq_length],
                "attention_mask": [1] * len(input_ids),
            }
        )
    return _ORIGINAL_COLLATE(tokenizer, items, str(tensors["input_ids"].device))


def _positive_answer_tensors(tokenizer, tensors: dict, negative_indices: list[int], max_seq_length: int) -> dict:
    return _answer_tensors_for_indices(tokenizer, tensors, negative_indices, CLASS_POSITIVE, max_seq_length)


def _positive_high_score_reward(positive_answer_scores, monitored_labels, current_mask=None):
    if positive_answer_scores.numel() < 2 or _HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER <= 0:
        return positive_answer_scores.new_zeros(()), 0.0
    top_values, top_indices = positive_answer_scores.topk(2)
    top_label = int(monitored_labels[top_indices[0]].detach().cpu())
    if current_mask is not None and not bool(current_mask[top_indices[0]].detach().cpu()):
        return positive_answer_scores.new_zeros(()), 0.0
    if top_label != CLASS_POSITIVE:
        return positive_answer_scores.new_zeros(()), 0.0
    margin = top_values[0] - top_values[1]
    reward = margin * _HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER
    return reward, 1.0


def _top_nonpositive_high_score_penalties(positive_answer_scores, monitored_labels, current_mask=None):
    if positive_answer_scores.numel() < 4:
        zero = positive_answer_scores.new_zeros(())
        return zero, zero
    top_values, top_indices = positive_answer_scores.topk(4)
    baseline = top_values[3].detach()
    negative_penalty = positive_answer_scores.new_zeros(())
    neutral_penalty = positive_answer_scores.new_zeros(())
    for rank_index in range(3):
        if current_mask is not None and not bool(current_mask[top_indices[rank_index]].detach().cpu()):
            continue
        label = int(monitored_labels[top_indices[rank_index]].detach().cpu())
        if label == CLASS_POSITIVE:
            continue
        margin = top_values[rank_index] - baseline
        if label == CLASS_NEGATIVE:
            negative_penalty = negative_penalty + margin * _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT
        elif label == CLASS_NEUTRAL:
            neutral_penalty = neutral_penalty + margin * _HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT
    return negative_penalty, neutral_penalty


def _windowed_positive_scores(current_scores, current_labels, micro_step, gradient_accumulation_steps):
    import torch

    global _TOP_SCORE_WINDOW
    accum_steps = max(1, int(gradient_accumulation_steps or 1))
    if micro_step is not None and int(micro_step) % accum_steps == 0:
        _TOP_SCORE_WINDOW = []
    if accum_steps <= 1 or not _TOP_SCORE_WINDOW:
        current_mask = torch.ones_like(current_labels, dtype=torch.bool)
        return current_scores, current_labels, current_mask
    cached_scores = current_scores.new_tensor([score for score, _label in _TOP_SCORE_WINDOW])
    cached_labels = current_labels.new_tensor([label for _score, label in _TOP_SCORE_WINDOW])
    scores = torch.cat([cached_scores, current_scores])
    labels = torch.cat([cached_labels, current_labels])
    current_mask = torch.cat(
        [
            torch.zeros(cached_labels.shape, dtype=torch.bool, device=current_scores.device),
            torch.ones(current_labels.shape, dtype=torch.bool, device=current_scores.device),
        ]
    )
    return scores, labels, current_mask


def _remember_positive_scores(current_scores, current_labels) -> None:
    global _TOP_SCORE_WINDOW
    detached_scores = current_scores.detach().float().cpu().tolist()
    detached_labels = [int(label) for label in current_labels.detach().cpu().tolist()]
    _TOP_SCORE_WINDOW.extend(zip(detached_scores, detached_labels))


def _compute_asymmetric_training_loss(
    model,
    tokenizer,
    tensors: dict,
    batch: list[dict],
    penalty_rows=None,
    max_seq_length: int | None = None,
    micro_step: int | None = None,
    gradient_accumulation_steps: int | None = None,
    **_ignored_kwargs,
):
    import torch

    effective_max_seq_length = max(64, int(max_seq_length or DEFAULT_MAX_SEQ_LENGTH))
    class_labels = tensors.pop("class_labels")
    output = model(**tensors)
    correct_nll = _per_sample_answer_nll(output.logits, tensors["labels"])
    class_weights = torch.ones_like(correct_nll)
    class_weights = torch.where(
        class_labels.eq(CLASS_POSITIVE),
        torch.full_like(correct_nll, _POSITIVE_CE_WEIGHT),
        class_weights,
    )
    class_weights = torch.where(
        class_labels.eq(CLASS_NEGATIVE),
        torch.full_like(correct_nll, _NEGATIVE_CE_WEIGHT),
        class_weights,
    )
    class_weights = torch.where(
        class_labels.eq(CLASS_NEUTRAL),
        torch.full_like(correct_nll, _NEUTRAL_CE_WEIGHT),
        class_weights,
    )
    negative_indices = class_labels.eq(CLASS_NEGATIVE).nonzero(as_tuple=False).flatten().tolist()
    if negative_indices:
        positive_tensors = _positive_answer_tensors(
            tokenizer,
            tensors,
            negative_indices,
            effective_max_seq_length,
        )
        positive_output = model(**positive_tensors)
        positive_nll = _per_sample_answer_nll(
            positive_output.logits,
            positive_tensors["labels"],
        )
        negative_nll = correct_nll[negative_indices]
        negative_fp_loss = _negative_auxiliary_losses(
            negative_nll,
            positive_nll,
        )
    else:
        positive_nll = None
        negative_fp_loss = correct_nll.new_zeros(())
    neutral_indices = class_labels.eq(CLASS_NEUTRAL).nonzero(as_tuple=False).flatten().tolist()
    if neutral_indices and (
        _NEUTRAL_FP_LOSS_WEIGHT > 0
        or _HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT > 0
    ):
        neutral_positive_tensors = _answer_tensors_for_indices(
            tokenizer,
            tensors,
            neutral_indices,
            CLASS_POSITIVE,
            effective_max_seq_length,
        )
        neutral_positive_output = model(**neutral_positive_tensors)
        neutral_positive_nll = _per_sample_answer_nll(
            neutral_positive_output.logits,
            neutral_positive_tensors["labels"],
        )
        if _NEUTRAL_FP_LOSS_WEIGHT > 0:
            neutral_nll = correct_nll[neutral_indices]
            neutral_fp_loss = _neutral_false_positive_loss(neutral_nll, neutral_positive_nll)
        else:
            neutral_fp_loss = correct_nll.new_zeros(())
    else:
        neutral_positive_nll = None
        neutral_fp_loss = correct_nll.new_zeros(())

    high_score_positive_hits = 0
    high_score_positive_reward = correct_nll.new_zeros(())
    high_score_negative_penalty = correct_nll.new_zeros(())
    high_score_neutral_penalty = correct_nll.new_zeros(())
    positive_indices = class_labels.eq(CLASS_POSITIVE).nonzero(as_tuple=False).flatten().tolist()
    high_score_parts = []
    high_score_labels = []
    high_score_indices: list[int] = []
    if positive_indices:
        high_score_parts.append(-correct_nll[positive_indices])
        high_score_labels.extend([CLASS_POSITIVE] * len(positive_indices))
        high_score_indices.extend(positive_indices)
    if negative_indices and positive_nll is not None:
        high_score_parts.append(-positive_nll)
        high_score_labels.extend([CLASS_NEGATIVE] * len(negative_indices))
        high_score_indices.extend(negative_indices)
    if neutral_indices and neutral_positive_nll is not None:
        high_score_parts.append(-neutral_positive_nll)
        high_score_labels.extend([CLASS_NEUTRAL] * len(neutral_indices))
        high_score_indices.extend(neutral_indices)
    if high_score_parts:
        current_positive_answer_scores = torch.cat(high_score_parts)
        current_monitored_labels = class_labels.new_tensor(high_score_labels)
        positive_answer_scores, monitored_labels, current_mask = _windowed_positive_scores(
            current_positive_answer_scores,
            current_monitored_labels,
            micro_step,
            gradient_accumulation_steps,
        )
        if _HIGH_SCORE_POSITIVE_BONUS > 0:
            high_score_positive_reward, high_score_positive_hits = _positive_high_score_reward(
                positive_answer_scores,
                monitored_labels,
                current_mask,
            )
        if _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT > 0 or _HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT > 0:
            high_score_negative_penalty, high_score_neutral_penalty = _top_nonpositive_high_score_penalties(
                positive_answer_scores,
                monitored_labels,
                current_mask,
            )
        _remember_positive_scores(current_positive_answer_scores, current_monitored_labels)

    weighted_ce = (correct_nll * class_weights).mean()

    total_loss = (
        weighted_ce
        + _FP_LOSS_WEIGHT * negative_fp_loss
        + _NEUTRAL_FP_LOSS_WEIGHT * neutral_fp_loss
        + high_score_negative_penalty
        + high_score_neutral_penalty
        - high_score_positive_reward
    )
    metrics = {
        "ce": float(weighted_ce.detach().cpu()),
        "negative_fp": float(negative_fp_loss.detach().cpu()),
        "neutral_fp": float(neutral_fp_loss.detach().cpu()),
        "high_score_negative": float(high_score_negative_penalty.detach().cpu()),
        "high_score_neutral": float(high_score_neutral_penalty.detach().cpu()),
        "high_score_positive_reward": float(high_score_positive_reward.detach().cpu()),
        "high_score_positive_bonus_multiplier": float(high_score_positive_reward.detach().cpu()),
        "high_score_positive_hits": float(high_score_positive_hits),
    }
    return total_loss, metrics


def _tokenize_threeclass_row(tokenizer, row: dict, max_seq_length: int) -> dict:
    item = _ORIGINAL_TOKENIZE_ROW(tokenizer, row, max_seq_length)
    item["class_label"] = int(row["metadata"]["label"])
    return item


def _collate_threeclass(tokenizer, batch: list[dict], device: str) -> dict:
    import torch

    tensors = _ORIGINAL_COLLATE(tokenizer, batch, device)
    tensors["class_labels"] = torch.tensor(
        [int(item["class_label"]) for item in batch],
        dtype=torch.long,
        device=device,
    )
    return tensors


def _item_class_label(item: dict) -> int | None:
    if "class_label" in item:
        return int(item["class_label"])
    metadata = item.get("metadata") or {}
    if "label" in metadata:
        return int(metadata["label"])
    return None


def _item_anchor_date(item: dict) -> str:
    return str((item.get("metadata") or {}).get("anchor_date", ""))


def _class_pattern() -> list[int]:
    return [CLASS_POSITIVE] + [CLASS_NEGATIVE] * _NEGATIVE_PER_POSITIVE + [CLASS_NEUTRAL] * _NEUTRAL_PER_POSITIVE


def _build_balanced_train_order(train_items: list[dict], seed: int, rng) -> list[int]:
    grouped = {label: [] for label in (CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL)}
    fallback: list[int] = []
    by_date: dict[str, dict[int, list[int]]] = {}
    for index, item in enumerate(train_items):
        label = _item_class_label(item)
        if label in grouped:
            grouped[label].append(index)
            by_date.setdefault(
                _item_anchor_date(item),
                {class_label: [] for class_label in (CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL)},
            )[label].append(index)
        else:
            fallback.append(index)
    if not all(grouped[label] for label in grouped):
        raise RuntimeError(
            f"Unable to build a strict same-date 1:{_NEGATIVE_PER_POSITIVE}:{_NEUTRAL_PER_POSITIVE} train order: "
            "one or more classes are missing. Rebuild the three-class dataset."
        )
    if fallback:
        raise RuntimeError(
            "Unable to build a strict same-date train order: some rows are missing class labels. "
            "Rebuild the three-class dataset."
        )
    train_order: list[int] = []
    date_keys = list(by_date)
    rng.shuffle(date_keys)
    for anchor_date in date_keys:
        date_grouped = by_date[anchor_date]
        for indices in date_grouped.values():
            rng.shuffle(indices)
        positions = {label: 0 for label in date_grouped}
        while (
            positions[CLASS_POSITIVE] < len(date_grouped[CLASS_POSITIVE])
            and positions[CLASS_NEGATIVE] + _NEGATIVE_PER_POSITIVE <= len(date_grouped[CLASS_NEGATIVE])
            and positions[CLASS_NEUTRAL] + _NEUTRAL_PER_POSITIVE <= len(date_grouped[CLASS_NEUTRAL])
        ):
            cycle: list[int] = []
            for label in _class_pattern():
                cycle.append(date_grouped[label][positions[label]])
                positions[label] += 1
            rng.shuffle(cycle)
            train_order.extend(cycle)
    if not train_order:
        raise RuntimeError(
            f"Unable to build a strict same-date 1:{_NEGATIVE_PER_POSITIVE}:{_NEUTRAL_PER_POSITIVE} train order. "
            "Rebuild the three-class dataset so each positive row has enough same-day negative and neutral rows."
        )
    return train_order


def _reshuffle_balanced_train_order(train_order: list[int], train_items: list[dict], rng) -> None:
    train_order[:] = _build_balanced_train_order(train_items, 0, rng)


_ORIGINAL_TOKENIZE_ROW = base_train._tokenize_row
_ORIGINAL_COLLATE = base_train._collate


def _next_selection_score_threshold(
    scores: list[float],
    current_threshold: float,
    top_ratio: float = 0.2,
) -> tuple[float, float, float, float]:
    normalized_top_ratio = min(max(float(top_ratio), 0.0), 1.0)
    threshold_position = 1.0 - normalized_top_ratio
    if not scores:
        value = float(current_threshold)
        return value, value, threshold_position, value
    average_score = sum(scores) / len(scores)
    max_score = max(scores)
    next_threshold = average_score + threshold_position * (max_score - average_score)
    return average_score, max_score, threshold_position, next_threshold


def _current_training_parameters() -> dict:
    return {
        "positive_ce_weight": _POSITIVE_CE_WEIGHT,
        "negative_ce_weight": _NEGATIVE_CE_WEIGHT,
        "neutral_ce_weight": _NEUTRAL_CE_WEIGHT,
        "fp_loss_weight": _FP_LOSS_WEIGHT,
        "neutral_fp_loss_weight": _NEUTRAL_FP_LOSS_WEIGHT,
        "high_score_positive_bonus": _HIGH_SCORE_POSITIVE_BONUS,
        "high_score_positive_bonus_max_multiplier": _HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER,
        "high_score_negative_penalty_weight": _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT,
        "high_score_neutral_penalty_weight": _HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT,
    }


def _evaluation_parameters(
    negative_weight: float,
    neutral_weight: float,
    threshold_top_ratio: float,
    max_samples: int,
    precision_top_k: int,
    precision_threshold: float,
) -> dict:
    return {
        "negative_weight": negative_weight,
        "neutral_weight": neutral_weight,
        "eval_threshold_top_ratio": threshold_top_ratio,
        "eval_max_samples": max_samples,
        "eval_precision_top_k": max(1, precision_top_k),
        "eval_precision_threshold": precision_threshold,
    }


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
    threshold_position: float | None = None,
    max_samples: int = 0,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    precision_top_k: int = 10,
    precision_threshold: float = 0.4,
    **_ignored_kwargs,
) -> dict:
    eval_rows, sample_method, sample_seed = base_train._sample_eval_rows(rows, max_samples, update)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    scored: list[dict] = []
    negative_weight = max(0.0, _env_float("NEGATIVE_WEIGHT", 0.5))
    neutral_weight = max(0.0, _env_float("NEUTRAL_WEIGHT", 0.0))
    threshold_top_ratio = min(max(_env_float("EVAL_THRESHOLD_TOP_RATIO", 0.2), 0.0), 1.0)
    started = time.monotonic()
    try:
        for index, row in enumerate(eval_rows, start=1):
            messages = compact_messages_from_sample(row)
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            prediction = score_prediction(model, tokenizer, prompt, max_seq_length)
            score = selection_score(
                prediction["positive_probability"],
                prediction["negative_probability"],
                prediction["neutral_probability"],
                negative_weight,
                neutral_weight,
            )
            scored.append(
                {
                    "scode": row.get("metadata", {}).get("scode"),
                    "anchor_date": row.get("metadata", {}).get("anchor_date"),
                    "actual_label": int(row["metadata"]["label"]),
                    "actual_class": CLASS_NAMES[int(row["metadata"]["label"])],
                    "predicted_label": int(prediction["label_id"]),
                    "predicted_class": prediction["label"],
                    "positive_probability": prediction["positive_probability"],
                    "neutral_probability": prediction["neutral_probability"],
                    "negative_probability": prediction["negative_probability"],
                    "selection_score": score,
                }
            )
            if index % 100 == 0 or index == len(eval_rows):
                running = summarize_scored_rows(scored)
                elapsed = time.monotonic() - started
                fraction = index / len(eval_rows) if eval_rows else 1.0
                remaining = elapsed * (1.0 - fraction) / fraction if fraction else 0.0
                print(
                    f"evaluation progress {index}/{len(eval_rows)} ({fraction * 100:.2f}%) "
                    f"elapsed={base_train._format_duration(elapsed)} "
                    f"remaining={base_train._format_duration(remaining)} "
                    f"accuracy={running['accuracy']:.4f} macro_f1={running['macro_f1']:.4f} "
                    f"positive_precision@10={running['positive_precision@10']:.4f}",
                    flush=True,
                )
    finally:
        if was_training:
            model.train()
    summary = summarize_scored_rows(scored, tuple(sorted({5, 10, 20, 50, max(1, precision_top_k)})))
    selection_score_top50 = selection_score_top_rows(scored, 50)
    positive_probability_top50 = positive_probability_top_rows(scored, 50)
    average_selection_score, max_selection_score, threshold_position, next_threshold = (
        _next_selection_score_threshold(
            [float(row["selection_score"]) for row in scored],
            threshold,
            threshold_top_ratio,
        )
    )
    target_key = f"positive_precision@{max(1, precision_top_k)}"
    result = {
        "update": update,
        "total_updates": total_updates,
        "trigger": "checkpoint",
        "progress": progress,
        "trained_epochs": trained_epochs,
        "source_samples": len(rows),
        "sample_method": sample_method,
        "sample_seed": sample_seed,
        **summary,
        "selection_score_top50": selection_score_top50,
        "positive_probability_top50": positive_probability_top50,
        "precision_top_k": max(1, precision_top_k),
        "precision_threshold": precision_threshold,
        "passed": summary[target_key] >= precision_threshold,
        "max_seq_length": max_seq_length,
        "threshold": threshold,
        "negative_weight": negative_weight,
        "neutral_weight": neutral_weight,
        "average_selection_score": average_selection_score,
        "max_selection_score": max_selection_score,
        "average_positive_probability": average_selection_score,
        "max_positive_probability": max_selection_score,
        "eval_threshold_top_ratio": threshold_top_ratio,
        "threshold_position": threshold_position,
        "next_threshold": next_threshold,
        "training_parameters": _current_training_parameters(),
        "evaluation_parameters": _evaluation_parameters(
            negative_weight,
            neutral_weight,
            threshold_top_ratio,
            max_samples,
            precision_top_k,
            precision_threshold,
        ),
    }
    output_path = output_dir / f"eval-update-{update:06d}-progress-{int(round(progress * 1000)):04d}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("checkpoint SelectionScore top50:", flush=True)
    for row in selection_score_top50:
        print(
            f"  rank={row['rank']} scode={row.get('scode')} anchor_date={row.get('anchor_date')} "
            f"SelectionScore={row['selection_score']:.6f} "
            f"PositiveProbability={row['positive_probability']:.6f} "
            f"NeutralProbability={row['neutral_probability']:.6f} "
            f"NegativeProbability={row['negative_probability']:.6f} "
            f"actual_class={row['actual_class']}",
            flush=True,
        )
    print("checkpoint PositiveProbability top50:", flush=True)
    for row in positive_probability_top50:
        print(
            f"  rank={row['rank']} scode={row.get('scode')} anchor_date={row.get('anchor_date')} "
            f"PositiveProbability={row['positive_probability']:.6f} "
            f"SelectionScore={row['selection_score']:.6f} "
            f"NeutralProbability={row['neutral_probability']:.6f} "
            f"NegativeProbability={row['negative_probability']:.6f} "
            f"actual_class={row['actual_class']}",
            flush=True,
        )
    print(
        f"evaluation saved: {output_path} accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f} "
        f"positive_precision@5={summary['positive_precision@5']:.4f} "
        f"positive_precision@10={summary['positive_precision@10']:.4f} "
        f"positive_precision@20={summary['positive_precision@20']:.4f} "
        f"positive_precision@50={summary['positive_precision@50']:.4f} "
        f"avg_selection_score={average_selection_score:.6f} "
        f"max_selection_score={max_selection_score:.6f} "
        f"threshold_position={threshold_position:.4f} "
        f"next_threshold={next_threshold:.6f} "
        f"{target_key}={summary[target_key]:.4f} target={precision_threshold:.4f} "
        f"passed={result['passed']}",
        flush=True,
    )
    params = result["training_parameters"]
    eval_params = result["evaluation_parameters"]
    print(
        "evaluation parameters: "
        f"ce=({params['positive_ce_weight']:.4g},{params['negative_ce_weight']:.4g},{params['neutral_ce_weight']:.4g}) "
        f"fp=({params['fp_loss_weight']:.4g},{params['neutral_fp_loss_weight']:.4g}) "
        f"high_score_positive_reward=(enabled={params['high_score_positive_bonus']:.4g},top_margin_multiplier={params['high_score_positive_bonus_max_multiplier']:.4g}) "
        f"high_score_penalty=({params['high_score_negative_penalty_weight']:.4g},{params['high_score_neutral_penalty_weight']:.4g}) "
        f"selection_weights=({eval_params['negative_weight']:.4g},{eval_params['neutral_weight']:.4g})",
        flush=True,
    )
    return result


def _validate_initial_binary_adapter(adapter_dir: Path) -> Path:
    resolved = adapter_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"initial binary adapter directory does not exist: {resolved}")
    if not (resolved / "adapter_config.json").is_file():
        raise FileNotFoundError(f"initial binary adapter is missing adapter_config.json: {resolved}")
    return resolved


def _checkpoint_update_with_initial(
    checkpoint_dir: Path | None,
    initial_binary_adapter_dir: Path,
    original_checkpoint_update,
) -> int:
    if checkpoint_dir is not None and checkpoint_dir.expanduser().resolve() == initial_binary_adapter_dir:
        return 0
    return original_checkpoint_update(checkpoint_dir)


def _patch_base_trainer(initial_binary_adapter_dir: Path | None = None) -> None:
    base_train.compact_messages_from_sample = compact_messages_from_sample
    base_train._evaluate_training_checkpoint = _evaluate_training_checkpoint
    base_train._tokenize_row = _tokenize_threeclass_row
    base_train._collate = _collate_threeclass
    base_train._compute_training_loss = _compute_asymmetric_training_loss
    base_train._extra_training_state = _extra_training_state
    base_train._load_extra_training_state = _load_extra_training_state
    base_train._build_train_order = _build_balanced_train_order
    base_train._reshuffle_train_order = _reshuffle_balanced_train_order
    base_train._training_run_summary = _threeclass_training_run_summary
    base_train.TOKEN_CACHE_VERSION = f"{base_train.TOKEN_CACHE_VERSION}_threeclass_asymmetric_v1"
    if initial_binary_adapter_dir is None:
        return
    initial_path = _validate_initial_binary_adapter(initial_binary_adapter_dir)
    original_checkpoint_update = base_train._checkpoint_update

    def checkpoint_update(checkpoint_dir: Path | None) -> int:
        return _checkpoint_update_with_initial(checkpoint_dir, initial_path, original_checkpoint_update)

    base_train._checkpoint_update = checkpoint_update


def _threeclass_training_run_summary(**kwargs) -> str:
    params = _current_training_parameters()
    return (
        f"manual RTX3060 three-class LoRA train rows={kwargs['train_rows']} valid={kwargs['valid_rows']} "
        f"checkpoint_eval_data_dir={kwargs['checkpoint_eval_data_dir']} "
        f"updates={kwargs['total_updates']} start_update={kwargs['start_update']} "
        f"batch_size={kwargs['batch_size']} grad_accum={kwargs['gradient_accumulation_steps']} "
        f"train_seed={kwargs['train_seed']} lr={kwargs['learning_rate']} weight_decay={kwargs['weight_decay']} "
        f"max_grad_norm={kwargs['max_grad_norm']} lora_rank={kwargs['lora_rank']} lora_dropout={kwargs['lora_dropout']} "
        f"max_seq_length={kwargs['max_seq_length']} on_the_fly_tokenize={kwargs['on_the_fly_tokenize']} "
        f"checkpoint_every={kwargs['checkpoint_every']} checkpoint_evaluate=True "
        f"eval_threshold={kwargs['evaluation_threshold']} eval_threshold_position={kwargs['evaluation_threshold_position']} "
        f"eval_precision_top_k={kwargs['evaluation_precision_top_k']} "
        f"eval_precision_threshold={kwargs['evaluation_precision_threshold']} eval_max_samples={kwargs['evaluation_max_samples']} "
        f"ce_weights=({params['positive_ce_weight']},{params['negative_ce_weight']},{params['neutral_ce_weight']}) "
        f"fp_weights=({params['fp_loss_weight']},{params['neutral_fp_loss_weight']}) "
        f"high_score_positive_reward=(enabled={params['high_score_positive_bonus']},top_margin_multiplier={params['high_score_positive_bonus_max_multiplier']}) "
        f"high_score_penalties=({params['high_score_negative_penalty_weight']},{params['high_score_neutral_penalty_weight']})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_train.build_parser()
    parser.description = "Fine-tune Qwen2.5 for positive/negative/neutral stock classification"
    parser.set_defaults(
        base_model=DEFAULT_BASE_MODEL,
        data_dir=DEFAULT_DATA_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        learning_rate=_env_float("LEARNING_RATE", 5e-6),
        eval_max_samples=_env_int("EVAL_MAX_SAMPLES", 1500),
        high_score_positive_bonus=_env_float("HIGH_SCORE_POSITIVE_BONUS", 1.0),
        fp_dynamic_penalty=_env_bool("FP_DYNAMIC_PENALTY", True),
    )
    parser.add_argument(
        "--high-score-positive-bonus-max-multiplier",
        type=float,
        default=_env_float("HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER", 8.0),
        help="Maximum CE multiplier for positive high-score bonus; use 0 to disable the cap.",
    )
    parser.add_argument(
        "--initial-binary-adapter-dir",
        dest="initial_binary_adapter_dir",
        type=Path,
        default=Path(os.environ["INITIAL_BINARY_ADAPTER_DIR"]) if os.environ.get("INITIAL_BINARY_ADAPTER_DIR") else None,
        help=(
            "Initialize trainable LoRA weights from an already trained binary-classification adapter, "
            "but start three-class optimizer updates at step 0. This does not overwrite the source adapter."
        ),
    )
    parser.add_argument(
        "--positive-ce-weight",
        type=float,
        default=_env_float("POSITIVE_CE_WEIGHT", 2.0),
        help="CE multiplier for true positive samples.",
    )
    parser.add_argument(
        "--negative-ce-weight",
        type=float,
        default=_env_float("NEGATIVE_CE_WEIGHT", 1.0),
        help="CE multiplier for true negative samples.",
    )
    parser.add_argument(
        "--neutral-ce-weight",
        type=float,
        default=_env_float("NEUTRAL_CE_WEIGHT", 0.5),
        help="CE multiplier for true neutral samples.",
    )
    parser.add_argument(
        "--fp-loss-weight",
        type=float,
        default=_env_float("FP_LOSS_WEIGHT", 1.0),
        help="Weight for penalizing a positive answer on true negative samples.",
    )
    parser.add_argument(
        "--neutral-fp-loss-weight",
        type=float,
        default=_env_float("NEUTRAL_FP_LOSS_WEIGHT", 0.3),
        help="Weight for penalizing a positive answer on true neutral samples.",
    )
    parser.add_argument(
        "--high-score-negative-penalty-weight",
        type=float,
        default=_env_float("HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT", 1.0),
        help="Weight for penalizing true negative samples that appear in the batch top-3 positive-answer scores.",
    )
    parser.add_argument(
        "--high-score-neutral-penalty-weight",
        type=float,
        default=_env_float("HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT", 0.5),
        help="Weight for penalizing true neutral samples that appear in the batch top-3 positive-answer scores.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.initial_binary_adapter_dir is not None and args.resume_adapter_dir is not None:
        raise SystemExit("--initial-binary-adapter-dir and --resume-adapter-dir cannot be used together")
    initial_binary_adapter_dir = (
        _validate_initial_binary_adapter(args.initial_binary_adapter_dir)
        if args.initial_binary_adapter_dir is not None
        else None
    )
    _configure_asymmetric_loss(
        args.positive_ce_weight,
        args.negative_ce_weight,
        args.neutral_ce_weight,
        args.fp_loss_weight,
        args.neutral_fp_loss_weight,
        args.high_score_positive_bonus,
        args.high_score_positive_bonus_max_multiplier,
        args.high_score_negative_penalty_weight,
        args.high_score_neutral_penalty_weight,
    )
    _patch_base_trainer(initial_binary_adapter_dir)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    if initial_binary_adapter_dir is not None:
        print(
            f"initializing three-class LoRA from binary adapter: {initial_binary_adapter_dir}; "
            "optimizer and update counter start from 0",
            flush=True,
        )
    print(
        "three-class asymmetric loss: "
        f"total=weighted_ce+{_FP_LOSS_WEIGHT}*negative_fp+{_NEUTRAL_FP_LOSS_WEIGHT}*neutral_fp "
        f"+top3_negative_penalty "
        f"+top3_neutral_penalty "
        f"-high_score_positive_reward "
        f"positive_ce_weight={_POSITIVE_CE_WEIGHT} negative_ce_weight={_NEGATIVE_CE_WEIGHT} "
        f"neutral_ce_weight={_NEUTRAL_CE_WEIGHT} "
        f"high_score_positive_bonus={_HIGH_SCORE_POSITIVE_BONUS} "
        f"high_score_positive_bonus_max_multiplier={_HIGH_SCORE_POSITIVE_BONUS_MAX_MULTIPLIER} "
        f"high_score_negative_penalty_weight={_HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT} "
        f"high_score_neutral_penalty_weight={_HIGH_SCORE_NEUTRAL_PENALTY_WEIGHT}",
        flush=True,
    )
    base_train.train_recall60_lora(
        base_model=args.base_model,
        data_dir=args.data_dir,
        checkpoint_eval_data_dir=args.checkpoint_eval_data_dir,
        output_dir=args.output_dir,
        max_seq_length=max(64, args.max_seq_length),
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
        resume_adapter_dir=initial_binary_adapter_dir or args.resume_adapter_dir,
        initial_adapter_dir=None,
        nonfinite_patience=args.nonfinite_patience,
        rebuild_token_cache=args.rebuild_token_cache,
        auto_resume=not args.no_auto_resume and initial_binary_adapter_dir is None,
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
        positive_loss_weight=1.0,
        negative_loss_weight=1.0,
        high_score_positive_bonus=0.0,
        high_score_positive_position=0.0,
        fp_dynamic_penalty=False,
        fp_penalty_weight=0.0,
        fp_threshold_ema_alpha=0.0,
        fp_threshold_min=0.0,
        fp_threshold_max=1.0,
    )


if __name__ == "__main__":
    main()
