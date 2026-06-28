from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_drop6.common import compact_messages_from_sample
from blackbox_finetune_drop6.gpu import prepare_rtx3060
from blackbox_finetune_drop6.inference import load_model, score_prediction

SPLIT_FILES = ("train.jsonl", "test.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
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


def row_label(row: dict) -> int:
    return int(row.get("metadata", {}).get("label", 0))


def row_key(row: dict) -> tuple[str, str, int]:
    metadata = row.get("metadata", {})
    return (
        str(metadata.get("scode", "")),
        str(metadata.get("anchor_date", "")),
        int(metadata.get("label", 0)),
    )


def load_dataset_splits(dataset_dir: Path) -> dict[str, list[dict]]:
    splits = {name: read_jsonl(dataset_dir / name) for name in SPLIT_FILES}
    if not any(splits.values()) and (dataset_dir / "all.jsonl").is_file():
        splits["test.jsonl"] = read_jsonl(dataset_dir / "all.jsonl")
    return splits


def infer_base_model(adapter_dir: Path, base_model: str | None) -> str:
    if base_model:
        return base_model
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing adapter_config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = config.get("base_model_name_or_path")
    if not resolved:
        raise ValueError(f"base model is missing in {config_path}; pass --base-model")
    return str(resolved)


def build_positive_scorer(adapter_dir: Path, base_model: str | None, max_seq_length: int) -> Callable[[dict], float]:
    resolved_base_model = infer_base_model(adapter_dir, base_model)
    model, tokenizer = load_model(resolved_base_model, adapter_dir)

    def scorer(row: dict) -> float:
        messages = compact_messages_from_sample(row)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        return float(score_prediction(model, tokenizer, prompt, max_seq_length, threshold=0.5)["positive_probability"])

    return scorer


def score_positive_rows(
    rows: list[dict],
    scorer: Callable[[dict], float],
    progress_every: int,
) -> list[tuple[float, dict]]:
    positives = [row for row in rows if row_label(row) == 1]
    scored: list[tuple[float, dict]] = []
    total = len(positives)
    started = datetime.now()
    for index, row in enumerate(positives, start=1):
        scored.append((scorer(row), row))
        if index == total or index % max(1, progress_every) == 0:
            elapsed = (datetime.now() - started).total_seconds()
            remaining = elapsed * (total - index) / index if index else 0.0
            print(
                f"positive scoring {index}/{total} ({index / total * 100:.2f}%) "
                f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
                flush=True,
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def select_reshuffled_rows(
    split_rows: dict[str, list[dict]],
    scored_positives: list[tuple[float, dict]],
    *,
    top_positive_ratio: float,
    keep_ratio_within_top: float,
    negative_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    positive_count = len(scored_positives)
    top_count = min(positive_count, max(0, math.ceil(positive_count * top_positive_ratio)))
    top_positive_pool = [row for _, row in scored_positives[:top_count]]
    tail_positive_rows = [row for _, row in scored_positives[top_count:]]
    keep_positive_count = min(len(top_positive_pool), max(1 if top_positive_pool else 0, math.ceil(top_count * keep_ratio_within_top)))
    retained_top_positives = rng.sample(top_positive_pool, keep_positive_count) if keep_positive_count else []
    selected_positives = tail_positive_rows + retained_top_positives
    selected_positive_keys = {row_key(row) for row in selected_positives}

    source_negatives = [row for rows in split_rows.values() for row in rows if row_label(row) == 0]
    negative_keep_count = min(len(source_negatives), max(0, round(len(selected_positives) * negative_ratio)))
    selected_negatives = rng.sample(source_negatives, negative_keep_count) if negative_keep_count else []
    selected_negative_keys = {row_key(row) for row in selected_negatives}

    output: dict[str, list[dict]] = {}
    for split_name, rows in split_rows.items():
        selected = [
            row
            for row in rows
            if row_key(row) in selected_positive_keys or row_key(row) in selected_negative_keys
        ]
        rng.shuffle(selected)
        output[split_name] = selected

    stats = {
        "source_rows": sum(len(rows) for rows in split_rows.values()),
        "source_positive_rows": positive_count,
        "source_negative_rows": len(source_negatives),
        "top_positive_ratio": top_positive_ratio,
        "top_positive_pool_rows": len(top_positive_pool),
        "keep_ratio_within_top": keep_ratio_within_top,
        "retained_top_positive_rows": len(retained_top_positives),
        "retained_tail_positive_rows": len(tail_positive_rows),
        "selected_positive_rows": len(selected_positives),
        "negative_ratio": negative_ratio,
        "selected_negative_rows": len(selected_negatives),
        "output_rows": sum(len(rows) for rows in output.values()),
        "split_rows": {name: len(rows) for name, rows in output.items()},
    }
    return output, stats


def write_dataset(output_dir: Path, split_rows: dict[str, list[dict]]) -> dict[str, int]:
    counts = {name: write_jsonl(output_dir / name, split_rows.get(name, [])) for name in SPLIT_FILES}
    all_rows = []
    for name in SPLIT_FILES:
        all_rows.extend(split_rows.get(name, []))
    counts["all.jsonl"] = write_jsonl(output_dir / "all.jsonl", all_rows)
    return counts


def reshuffle_dataset(
    dataset_dir: Path,
    output_dir: Path,
    scorer: Callable[[dict], float],
    *,
    top_positive_ratio: float,
    keep_ratio_within_top: float,
    negative_ratio: float,
    seed: int,
    progress_every: int,
) -> dict:
    split_rows = load_dataset_splits(dataset_dir)
    all_rows = [row for rows in split_rows.values() for row in rows]
    scored_positives = score_positive_rows(all_rows, scorer, progress_every)
    output_rows, stats = select_reshuffled_rows(
        split_rows,
        scored_positives,
        top_positive_ratio=top_positive_ratio,
        keep_ratio_within_top=keep_ratio_within_top,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    counts = write_dataset(output_dir, output_rows)
    score_path = output_dir / "positive_scores.jsonl"
    write_jsonl(
        score_path,
        (
            {
                "positive_probability": score,
                "scode": row.get("metadata", {}).get("scode"),
                "anchor_date": row.get("metadata", {}).get("anchor_date"),
                "source": row.get("metadata", {}).get("source"),
            }
            for score, row in scored_positives
        ),
    )
    stats.update(
        {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "output_file_rows": counts,
            "positive_scores_path": str(score_path),
        }
    )
    with (output_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    return stats


def to_path(value: str) -> Path:
    text = str(value)
    if os.name != "nt" and len(text) >= 3 and text[1:3] in {":\\", ":/"}:
        text = f"/mnt/{text[0].lower()}/{text[3:].replace(chr(92), '/')}"
    return Path(text).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build drop6 reshuffled datasets by ranking only positive samples."
    )
    parser.add_argument("--adapter-dir", type=to_path, required=True, help="Trained drop6 adapter/checkpoint directory.")
    parser.add_argument("--train-data-dir", type=to_path, required=True, help="Source training dataset directory.")
    parser.add_argument("--eval-data-dir", type=to_path, required=True, help="Source evaluation dataset directory.")
    parser.add_argument("--output-dir", type=to_path, required=True, help="Output root directory.")
    parser.add_argument("--base-model", help="Base model override. Defaults to adapter_config.json.")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    parser.add_argument("--top-positive-ratio", type=float, default=0.30)
    parser.add_argument("--keep-ratio-within-top", type=float, default=0.10)
    parser.add_argument("--negative-ratio", type=float, default=9.0, help="Random negatives kept per selected positive.")
    parser.add_argument("--seed", type=int, default=937498347)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    scorer = build_positive_scorer(args.adapter_dir, args.base_model, max(64, args.max_seq_length))
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    train_stats = reshuffle_dataset(
        args.train_data_dir,
        output_root / "training",
        scorer,
        top_positive_ratio=min(max(args.top_positive_ratio, 0.0), 1.0),
        keep_ratio_within_top=min(max(args.keep_ratio_within_top, 0.0), 1.0),
        negative_ratio=max(0.0, args.negative_ratio),
        seed=args.seed,
        progress_every=args.progress_every,
    )
    eval_stats = reshuffle_dataset(
        args.eval_data_dir,
        output_root / "evaluation",
        scorer,
        top_positive_ratio=min(max(args.top_positive_ratio, 0.0), 1.0),
        keep_ratio_within_top=min(max(args.keep_ratio_within_top, 0.0), 1.0),
        negative_ratio=max(0.0, args.negative_ratio),
        seed=args.seed + 100003,
        progress_every=args.progress_every,
    )
    summary = {
        "adapter_dir": str(args.adapter_dir),
        "base_model": args.base_model,
        "max_seq_length": args.max_seq_length,
        "top_positive_ratio": args.top_positive_ratio,
        "keep_ratio_within_top": args.keep_ratio_within_top,
        "effective_positive_keep_ratio": args.top_positive_ratio * args.keep_ratio_within_top,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "training": train_stats,
        "evaluation": eval_stats,
    }
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
