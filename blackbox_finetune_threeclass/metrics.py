from __future__ import annotations

from blackbox_finetune_threeclass.common import CLASS_NAMES


def _ranked_rows(rows: list[dict], score_key: str, limit: int) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: float(row.get(score_key, 0.0)),
        reverse=True,
    )[: max(0, limit)]
    return [{**row, "rank": rank} for rank, row in enumerate(ordered, start=1)]


def positive_probability_top_rows(rows: list[dict], limit: int = 50) -> list[dict]:
    return _ranked_rows(rows, "positive_probability", limit)


def selection_score_top_rows(rows: list[dict], limit: int = 50) -> list[dict]:
    return _ranked_rows(rows, "selection_score", limit)


def summarize_scored_rows(rows: list[dict], top_ks: tuple[int, ...] = (5, 10, 20, 50)) -> dict:
    confusion = {
        CLASS_NAMES[actual]: {CLASS_NAMES[predicted]: 0 for predicted in CLASS_NAMES}
        for actual in CLASS_NAMES
    }
    for row in rows:
        confusion[CLASS_NAMES[int(row["actual_label"])]][CLASS_NAMES[int(row["predicted_label"])]] += 1
    per_class = {}
    for label, name in CLASS_NAMES.items():
        tp = confusion[name][name]
        fp = sum(confusion[other][name] for other in confusion if other != name)
        fn = sum(value for predicted, value in confusion[name].items() if predicted != name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    accuracy = sum(confusion[name][name] for name in confusion) / len(rows) if rows else 0.0
    ranking_key = "selection_score" if any("selection_score" in row for row in rows) else "positive_probability"
    ordered = sorted(rows, key=lambda row: float(row.get(ranking_key, 0.0)), reverse=True)
    top_metrics = {}
    for k in top_ks:
        top = ordered[:k]
        top_metrics[f"positive_precision@{k}"] = (
            sum(1 for row in top if int(row["actual_label"]) == 2) / len(top) if top else 0.0
        )
    return {
        "samples": len(rows),
        "accuracy": accuracy,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(per_class),
        "confusion_matrix": confusion,
        "per_class": per_class,
        **top_metrics,
    }
