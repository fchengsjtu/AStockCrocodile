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
_RANK_LOSS_WEIGHT = 0.5
_RANK_MARGIN = 0.2
_HIGH_SCORE_EMA_ENABLED = True
_HIGH_SCORE_EMA_ALPHA = 0.02
_HIGH_SCORE_CUTOFF_POSITION = 0.6
_HIGH_SCORE_POSITIVE_BONUS = 1.0
_HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT = 1.0
_HIGH_SCORE_CUTOFF: float | None = None


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
    rank_loss_weight: float,
    rank_margin: float,
    high_score_ema: bool,
    high_score_ema_alpha: float,
    high_score_cutoff_position: float,
    high_score_positive_bonus: float,
    high_score_negative_penalty_weight: float,
) -> None:
    global _POSITIVE_CE_WEIGHT, _NEGATIVE_CE_WEIGHT, _NEUTRAL_CE_WEIGHT
    global _FP_LOSS_WEIGHT, _RANK_LOSS_WEIGHT, _RANK_MARGIN
    global _HIGH_SCORE_EMA_ENABLED, _HIGH_SCORE_EMA_ALPHA, _HIGH_SCORE_CUTOFF_POSITION
    global _HIGH_SCORE_POSITIVE_BONUS, _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT, _HIGH_SCORE_CUTOFF
    _POSITIVE_CE_WEIGHT = max(0.0, float(positive_ce_weight))
    _NEGATIVE_CE_WEIGHT = max(0.0, float(negative_ce_weight))
    _NEUTRAL_CE_WEIGHT = max(0.0, float(neutral_ce_weight))
    _FP_LOSS_WEIGHT = max(0.0, float(fp_loss_weight))
    _RANK_LOSS_WEIGHT = max(0.0, float(rank_loss_weight))
    _RANK_MARGIN = max(0.0, float(rank_margin))
    _HIGH_SCORE_EMA_ENABLED = bool(high_score_ema)
    _HIGH_SCORE_EMA_ALPHA = min(max(float(high_score_ema_alpha), 0.0), 1.0)
    _HIGH_SCORE_CUTOFF_POSITION = min(max(float(high_score_cutoff_position), 0.0), 1.0)
    _HIGH_SCORE_POSITIVE_BONUS = max(0.0, float(high_score_positive_bonus))
    _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT = max(0.0, float(high_score_negative_penalty_weight))
    _HIGH_SCORE_CUTOFF = None


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
    return functional.softplus(score_delta).mean(), functional.relu(_RANK_MARGIN + score_delta).mean()


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


def _update_high_score_cutoff(scores) -> tuple[float, float]:
    global _HIGH_SCORE_CUTOFF
    detached = scores.detach().float()
    average_score = detached.mean()
    max_score = detached.max()
    raw_cutoff = average_score + _HIGH_SCORE_CUTOFF_POSITION * (max_score - average_score)
    raw_value = float(raw_cutoff.cpu())
    if _HIGH_SCORE_CUTOFF is None:
        _HIGH_SCORE_CUTOFF = raw_value
    else:
        _HIGH_SCORE_CUTOFF = (
            _HIGH_SCORE_EMA_ALPHA * raw_value
            + (1.0 - _HIGH_SCORE_EMA_ALPHA) * _HIGH_SCORE_CUTOFF
        )
    return raw_value, float(_HIGH_SCORE_CUTOFF)


def _class_probabilities_from_nll(nll_by_label: dict[int, object]):
    import torch

    labels = sorted(nll_by_label)
    losses = torch.stack([nll_by_label[label].float() for label in labels], dim=1)
    probabilities = torch.softmax(-losses, dim=1)
    return {label: probabilities[:, index] for index, label in enumerate(labels)}


def _selection_scores_from_probabilities(probabilities: dict[int, object]):
    negative_weight = max(0.0, _env_float("NEGATIVE_WEIGHT", 0.5))
    neutral_weight = max(0.0, _env_float("NEUTRAL_WEIGHT", 0.0))
    return (
        probabilities[CLASS_POSITIVE]
        - negative_weight * probabilities[CLASS_NEGATIVE]
        - neutral_weight * probabilities[CLASS_NEUTRAL]
    )


def _score_monitored_rows(model, tokenizer, tensors: dict, correct_nll, indices: list[int], max_seq_length: int):
    if not indices:
        return None, None
    nll_by_label: dict[int, object] = {}
    class_labels = tensors["class_labels"]
    for label in (CLASS_NEGATIVE, CLASS_NEUTRAL, CLASS_POSITIVE):
        label_mask = class_labels[indices].eq(label)
        nll_values = correct_nll[indices].clone()
        missing_positions = label_mask.logical_not().nonzero(as_tuple=False).flatten().tolist()
        if missing_positions:
            missing_indices = [indices[position] for position in missing_positions]
            answer_tensors = _answer_tensors_for_indices(
                tokenizer,
                tensors,
                missing_indices,
                label,
                max_seq_length,
            )
            answer_output = model(**answer_tensors)
            answer_nll = _per_sample_answer_nll(answer_output.logits, answer_tensors["labels"])
            nll_values[missing_positions] = answer_nll
        nll_by_label[label] = nll_values
    probabilities = _class_probabilities_from_nll(nll_by_label)
    return _selection_scores_from_probabilities(probabilities), probabilities


def _compute_asymmetric_training_loss(
    model,
    tokenizer,
    tensors: dict,
    batch: list[dict],
    penalty_rows=None,
    max_seq_length: int | None = None,
    **_ignored_kwargs,
):
    import torch
    import torch.nn.functional as functional

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
    tensors["class_labels"] = class_labels

    monitored_indices = (
        class_labels.eq(CLASS_POSITIVE).logical_or(class_labels.eq(CLASS_NEGATIVE))
    ).nonzero(as_tuple=False).flatten().tolist()
    high_score_raw_cutoff = None
    high_score_cutoff = None
    high_score_positive_hits = 0
    high_score_negative_penalty = correct_nll.new_zeros(())
    if _HIGH_SCORE_EMA_ENABLED and monitored_indices:
        selection_scores, _ = _score_monitored_rows(
            model,
            tokenizer,
            tensors,
            correct_nll,
            monitored_indices,
            effective_max_seq_length,
        )
        high_score_raw_cutoff, high_score_cutoff = _update_high_score_cutoff(selection_scores)
        cutoff_tensor = selection_scores.new_tensor(high_score_cutoff)
        monitored_labels = class_labels[monitored_indices]
        positive_mask = monitored_labels.eq(CLASS_POSITIVE)
        negative_mask = monitored_labels.eq(CLASS_NEGATIVE)
        high_score_mask = selection_scores.detach().ge(cutoff_tensor)
        positive_hits_mask = positive_mask.logical_and(high_score_mask)
        high_score_positive_hits = int(positive_hits_mask.sum().detach().cpu())
        if high_score_positive_hits and _HIGH_SCORE_POSITIVE_BONUS > 0:
            positive_indices = [
                monitored_indices[position]
                for position in positive_hits_mask.nonzero(as_tuple=False).flatten().tolist()
            ]
            class_weights[positive_indices] = class_weights[positive_indices] * (1.0 + _HIGH_SCORE_POSITIVE_BONUS)
        if _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT > 0 and bool(negative_mask.any()):
            negative_scores = selection_scores[negative_mask]
            high_score_negative_penalty = functional.relu(negative_scores - cutoff_tensor).mean()

    weighted_ce = (correct_nll * class_weights).mean()
    tensors.pop("class_labels")

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
        negative_fp_loss, ranking_loss = _negative_auxiliary_losses(
            negative_nll,
            positive_nll,
        )
    else:
        negative_fp_loss = weighted_ce.new_zeros(())
        ranking_loss = weighted_ce.new_zeros(())

    total_loss = (
        weighted_ce
        + _FP_LOSS_WEIGHT * negative_fp_loss
        + _RANK_LOSS_WEIGHT * ranking_loss
        + _HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT * high_score_negative_penalty
    )
    metrics = {
        "ce": float(weighted_ce.detach().cpu()),
        "negative_fp": float(negative_fp_loss.detach().cpu()),
        "rank": float(ranking_loss.detach().cpu()),
        "high_score_negative": float(high_score_negative_penalty.detach().cpu()),
        "high_score_positive_hits": float(high_score_positive_hits),
    }
    if high_score_raw_cutoff is not None and high_score_cutoff is not None:
        metrics["high_score_raw_cutoff"] = float(high_score_raw_cutoff)
        metrics["high_score_cutoff"] = float(high_score_cutoff)
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
    max_samples: int,
    max_seq_length: int,
    precision_top_k: int,
    precision_threshold: float,
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
        "eval_threshold_top_ratio": threshold_top_ratio,
        "threshold_position": threshold_position,
        "next_threshold": next_threshold,
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
    base_train.TOKEN_CACHE_VERSION = f"{base_train.TOKEN_CACHE_VERSION}_threeclass_asymmetric_v1"
    if initial_binary_adapter_dir is None:
        return
    initial_path = _validate_initial_binary_adapter(initial_binary_adapter_dir)
    original_checkpoint_update = base_train._checkpoint_update

    def checkpoint_update(checkpoint_dir: Path | None) -> int:
        return _checkpoint_update_with_initial(checkpoint_dir, initial_path, original_checkpoint_update)

    base_train._checkpoint_update = checkpoint_update


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
        "--rank-loss-weight",
        type=float,
        default=_env_float("RANK_LOSS_WEIGHT", 0.5),
        help="Weight for the negative-versus-positive margin ranking loss.",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=_env_float("RANK_MARGIN", 0.2),
        help="Required score margin between the negative and positive answers.",
    )
    parser.add_argument(
        "--high-score-ema",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HIGH_SCORE_EMA", True),
        help="Update the high-score cutoff with EMA during training and use it for positive bonus/negative suppression.",
    )
    parser.add_argument(
        "--high-score-ema-alpha",
        type=float,
        default=_env_float("HIGH_SCORE_EMA_ALPHA", 0.02),
        help="EMA alpha for the in-training high-score cutoff. Smaller values make the cutoff more stable.",
    )
    parser.add_argument(
        "--high-score-cutoff-position",
        type=float,
        default=_env_float("HIGH_SCORE_CUTOFF_POSITION", 0.6),
        help="Position between batch average SelectionScore and max SelectionScore used as the raw high-score cutoff.",
    )
    parser.add_argument(
        "--high-score-negative-penalty-weight",
        type=float,
        default=_env_float("HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT", 1.0),
        help="Weight for suppressing true negative samples whose SelectionScore is above the EMA cutoff.",
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
        args.rank_loss_weight,
        args.rank_margin,
        args.high_score_ema,
        args.high_score_ema_alpha,
        args.high_score_cutoff_position,
        args.high_score_positive_bonus,
        args.high_score_negative_penalty_weight,
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
        f"total=weighted_ce+{_FP_LOSS_WEIGHT}*negative_fp+{_RANK_LOSS_WEIGHT}*ranking "
        f"+{_HIGH_SCORE_NEGATIVE_PENALTY_WEIGHT}*high_score_negative "
        f"positive_ce_weight={_POSITIVE_CE_WEIGHT} negative_ce_weight={_NEGATIVE_CE_WEIGHT} "
        f"neutral_ce_weight={_NEUTRAL_CE_WEIGHT} "
        f"rank_margin={_RANK_MARGIN} high_score_ema={_HIGH_SCORE_EMA_ENABLED} "
        f"high_score_ema_alpha={_HIGH_SCORE_EMA_ALPHA} "
        f"high_score_cutoff_position={_HIGH_SCORE_CUTOFF_POSITION} "
        f"high_score_positive_bonus={_HIGH_SCORE_POSITIVE_BONUS}",
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
        high_score_positive_position=args.high_score_positive_position,
        fp_dynamic_penalty=False,
        fp_penalty_weight=0.0,
        fp_threshold_ema_alpha=args.fp_threshold_ema_alpha,
        fp_threshold_min=args.fp_threshold_min,
        fp_threshold_max=args.fp_threshold_max,
    )


if __name__ == "__main__":
    main()
