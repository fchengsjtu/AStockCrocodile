from __future__ import annotations

import math

from blackbox_finetune_recall60.inference import answer_loss, load_model
from blackbox_finetune_threeclass.common import CLASS_NAMES, label_answer


def selection_score(
    positive_probability: float,
    negative_probability: float,
    neutral_probability: float,
    negative_weight: float = 0.5,
    neutral_weight: float = 0.0,
) -> float:
    return (
        float(positive_probability)
        - float(negative_weight) * float(negative_probability)
        - float(neutral_weight) * float(neutral_probability)
    )


def probabilities_from_losses(losses: dict[int, float]) -> dict[int, float]:
    minimum = min(losses.values())
    weights = {label: math.exp(-(loss - minimum)) for label, loss in losses.items()}
    total = sum(weights.values())
    return {label: (weight / total if total else 0.0) for label, weight in weights.items()}


def score_prediction(model, tokenizer, prompt: str, max_seq_length: int) -> dict:
    losses = {
        label: answer_loss(model, tokenizer, prompt, label_answer(label), max_seq_length)
        for label in sorted(CLASS_NAMES)
    }
    probabilities = probabilities_from_losses(losses)
    predicted_label = max(probabilities, key=probabilities.get)
    result = {
        "label": CLASS_NAMES[predicted_label],
        "label_id": predicted_label,
    }
    for label, name in CLASS_NAMES.items():
        result[f"{name}_probability"] = probabilities[label]
        result[f"{name}_loss"] = losses[label]
    return result
