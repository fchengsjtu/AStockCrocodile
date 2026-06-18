from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class SourceMetadata:
    evaluation_json: Path
    training_dataset_dir: Path
    evaluation_dataset_dir: Path


def row_key(row: dict) -> tuple[str, str, int]:
    metadata = row.get("metadata", {})
    return (
        str(metadata.get("scode", "")),
        str(metadata.get("anchor_date", "")),
        int(metadata.get("label", 0)),
    )


def row_label(row: dict) -> int:
    return int(row.get("metadata", {}).get("label", 0))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def resolve_cross_platform_path(value: str, base_dir: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("dataset path is empty")
    if os.name == "nt":
        match = re.fullmatch(r"/mnt/([A-Za-z])/(.*)", text)
        if match:
            text = f"{match.group(1).upper()}:/{match.group(2)}"
    else:
        match = re.fullmatch(r"([A-Za-z]):[\\/](.*)", text)
        if match:
            text = f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}"
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _json_pairs(path: Path) -> list[tuple[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file, object_pairs_hook=lambda pairs: pairs)


def find_evaluation_json(model_dir: Path, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        path = explicit_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evaluation JSON does not exist: {path}")
        return path
    candidates = sorted(
        (
            path
            for path in model_dir.rglob("eval-*.json")
            if "negative_reshuffle" not in path.parts
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no eval-*.json found under {model_dir}; add runs/evaluations/eval-xxx.json"
        )
    return candidates[0]


def load_source_metadata(model_dir: Path, evaluation_json: Path | None = None) -> SourceMetadata:
    eval_path = find_evaluation_json(model_dir, evaluation_json)
    pairs = _json_pairs(eval_path)
    train_values = [str(value) for key, value in pairs if key == "original_train_dataset_path"]
    eval_values = [str(value) for key, value in pairs if key == "original_eval_dataset_path"]
    if not eval_values and len(train_values) >= 2:
        eval_values = [train_values[-1]]
        train_values = [train_values[0]]
    if not train_values or not eval_values:
        raise ValueError(
            f"{eval_path} must contain original_train_dataset_path and original_eval_dataset_path"
        )
    training_dir = resolve_cross_platform_path(train_values[0], eval_path.parent)
    evaluation_dir = resolve_cross_platform_path(eval_values[0], eval_path.parent)
    validate_training_dataset(training_dir)
    validate_evaluation_dataset(evaluation_dir)
    return SourceMetadata(eval_path, training_dir, evaluation_dir)


def validate_training_dataset(path: Path) -> None:
    missing = [name for name in ("train.jsonl", "test.jsonl", "all.jsonl") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"training dataset directory {path} is missing: {', '.join(missing)}")


def validate_evaluation_dataset(path: Path) -> None:
    if not any((path / name).is_file() for name in ("test.jsonl", "all.jsonl", "train.jsonl")):
        raise FileNotFoundError(f"evaluation dataset directory has no JSONL dataset: {path}")


def score_negative_rows(
    negative_rows: list[dict],
    scorer: Callable[[dict], float],
    progress_every: int = 100,
) -> list[tuple[float, dict]]:
    scored: list[tuple[float, dict]] = []
    total = len(negative_rows)
    started = datetime.now()
    for index, row in enumerate(negative_rows, start=1):
        scored.append((float(scorer(row)), row))
        if index == total or index % max(1, progress_every) == 0:
            elapsed = (datetime.now() - started).total_seconds()
            remaining = elapsed * (total - index) / index if index else 0.0
            print(
                f"negative scoring {index}/{total} ({index / total * 100:.2f}%) "
                f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
                flush=True,
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def reshuffle_split(
    source_rows: list[dict],
    scored_current_negatives: list[tuple[float, dict]],
    replacement_pool: list[dict],
    keep_count: int,
    rng: random.Random,
    excluded_keys: set[tuple[str, str, int]] | None = None,
    target_negative_count: int | None = None,
) -> tuple[list[dict], set[tuple[str, str, int]], dict]:
    positives = [row for row in source_rows if row_label(row) == 1]
    source_negatives = [row for row in source_rows if row_label(row) == 0]
    desired_negatives = len(source_negatives) if target_negative_count is None else max(0, int(target_negative_count))
    excluded = set(excluded_keys or ())
    source_keys = {row_key(row) for row in source_negatives}
    ranked_source = [
        (score, row)
        for score, row in scored_current_negatives
        if row_key(row) in source_keys and row_key(row) not in excluded
    ]
    retained = [row for _, row in ranked_source[: min(max(0, keep_count), desired_negatives)]]
    selected_keys = {row_key(row) for row in retained}
    refill_candidates = [
        row
        for row in replacement_pool
        if row_key(row) not in excluded and row_key(row) not in selected_keys
    ]
    refill_count = desired_negatives - len(retained)
    if len(refill_candidates) < refill_count:
        raise RuntimeError(
            f"negative pool is too small: need {refill_count} replacements, have {len(refill_candidates)}"
        )
    replacements = rng.sample(refill_candidates, refill_count)
    selected_negatives = retained + replacements
    rng.shuffle(selected_negatives)
    output_rows = positives + selected_negatives
    rng.shuffle(output_rows)
    output_keys = {row_key(row) for row in selected_negatives}
    stats = {
        "positive_count": len(positives),
        "source_negative_count": len(source_negatives),
        "negative_count": desired_negatives,
        "retained_hard_negatives": len(retained),
        "random_replacements": len(replacements),
    }
    return output_rows, output_keys, stats


def copy_model_files(model_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in model_dir.iterdir():
        if not source.is_file():
            continue
        if source.name.startswith("eval-") and source.suffix == ".json":
            continue
        target = target_dir / source.name
        shutil.copy2(source, target)
        copied.append(source.name)
    if not (target_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"model directory does not contain adapter_config.json: {model_dir}")
    return copied


def copy_dataset(source_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in source_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)
            copied.append(source.name)
    return copied
