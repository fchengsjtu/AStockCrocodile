from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_drop6 import common as drop_common
from blackbox_finetune_drop6.inference import load_model as load_drop_model
from blackbox_finetune_drop6.inference import score_prediction as score_drop_prediction
from blackbox_finetune_recall60 import common as up_common
from blackbox_finetune_recall60.gpu import prepare_rtx3060
from blackbox_finetune_recall60.inference import load_model as load_up_model
from blackbox_finetune_recall60.inference import score_prediction as score_up_prediction


def to_path(value: str) -> Path:
    text = str(value)
    if os.name != "nt" and len(text) >= 3 and text[1:3] in {":\\", ":/"}:
        text = f"/mnt/{text[0].lower()}/{text[3:].replace(chr(92), '/')}"
    return Path(text).expanduser()


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if max_rows is not None and len(rows) >= max_rows:
                break
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


def row_key(row: dict, index: int) -> str:
    metadata = row.get("metadata", {})
    return f"{index}:{metadata.get('scode', '')}:{metadata.get('anchor_date', '')}:{metadata.get('label', '')}"


def actual_label(row: dict) -> int:
    return int(row.get("metadata", {}).get("label", 0))


def build_prompt(row: dict, tokenizer, system_prompt: str) -> str:
    messages = up_common.compact_messages_from_sample(row)
    prompt_messages = [dict(message) for message in messages[:-1]]
    if prompt_messages:
        prompt_messages[0]["content"] = system_prompt
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def precision_at_k(rows: list[dict], ks: tuple[int, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in ks:
        top_rows = rows[:k]
        result[f"precision@{k}"] = (
            sum(1 for row in top_rows if int(row.get("actual_label", 0)) == 1) / len(top_rows)
            if top_rows
            else 0.0
        )
    return result


def combined_score(row: dict, drop_weight: float) -> float:
    return float(row["up_probability"]) - drop_weight * float(row["drop_probability"])


def summarize_combined(rows: list[dict], ks: tuple[int, ...], drop_weight: float) -> dict:
    ordered = sorted(
        (
            {
                **row,
                "combined_score": combined_score(row, drop_weight),
            }
            for row in rows
        ),
        key=lambda item: item["combined_score"],
        reverse=True,
    )
    summary = {
        "candidate_count": len(ordered),
        **precision_at_k(ordered, ks),
    }
    return summary


def score_drop_for_candidates(
    candidates: list[dict],
    *,
    raw_rows: list[dict],
    drop_model,
    drop_tokenizer,
    max_seq_length: int,
    drop_cache: dict[str, float],
) -> list[dict]:
    result: list[dict] = []
    for candidate in candidates:
        key = str(candidate["key"])
        if key not in drop_cache:
            row = raw_rows[int(candidate["row_index"])]
            prompt = build_prompt(row, drop_tokenizer, drop_common.SYSTEM_PROMPT)
            pred = score_drop_prediction(drop_model, drop_tokenizer, prompt, max_seq_length, threshold=0.5)
            drop_cache[key] = float(pred["positive_probability"])
        result.append({**candidate, "drop_probability": drop_cache[key]})
    return result


def evaluate_joint(
    *,
    up_adapter_dir: Path,
    drop_adapter_dir: Path,
    data_dir: Path,
    output_dir: Path,
    base_model: str | None,
    drop_weight: float,
    max_seq_length: int,
    progress_every: int,
    interim_top_n: int,
    final_top_n: int,
    max_samples: int | None,
) -> dict:
    test_path = data_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"missing evaluation dataset: {test_path}")
    rows = read_jsonl(test_path, max_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded evaluation rows={len(rows)} data={test_path}", flush=True)

    resolved_base_model = infer_base_model(up_adapter_dir, base_model)
    print(f"loading up model adapter={up_adapter_dir}", flush=True)
    up_model, up_tokenizer = load_up_model(resolved_base_model, up_adapter_dir)
    print(f"loading drop model adapter={drop_adapter_dir}", flush=True)
    drop_model, drop_tokenizer = load_drop_model(resolved_base_model, drop_adapter_dir)

    up_scored: list[dict] = []
    drop_cache: dict[str, float] = {}
    progress_records: list[dict] = []
    started = datetime.now()
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    for index, row in enumerate(rows, start=1):
        prompt = build_prompt(row, up_tokenizer, up_common.SYSTEM_PROMPT)
        pred = score_up_prediction(up_model, up_tokenizer, prompt, max_seq_length, threshold=0.5)
        metadata = row.get("metadata", {})
        scored = {
            "key": row_key(row, index - 1),
            "row_index": index - 1,
            "scode": metadata.get("scode"),
            "anchor_date": metadata.get("anchor_date"),
            "actual_label": actual_label(row),
            "up_probability": float(pred["positive_probability"]),
        }
        up_scored.append(scored)

        if index % progress_every == 0 or index == len(rows):
            top_up = sorted(up_scored, key=lambda item: item["up_probability"], reverse=True)[:interim_top_n]
            combined_candidates = score_drop_for_candidates(
                top_up,
                raw_rows=rows,
                drop_model=drop_model,
                drop_tokenizer=drop_tokenizer,
                max_seq_length=max_seq_length,
                drop_cache=drop_cache,
            )
            summary = summarize_combined(combined_candidates, (5, 10, 20, 50), drop_weight)
            elapsed = (datetime.now() - started).total_seconds()
            record = {
                "scored_rows": index,
                "total_rows": len(rows),
                "progress": index / len(rows) if rows else 1.0,
                "elapsed_seconds": elapsed,
                "drop_scored_unique": len(drop_cache),
                "interim_top_n": interim_top_n,
                "drop_weight": drop_weight,
                **summary,
            }
            progress_records.append(record)
            write_jsonl(progress_path, [record])
            print(json.dumps(record, ensure_ascii=False), flush=True)

    final_top_up = sorted(up_scored, key=lambda item: item["up_probability"], reverse=True)[:final_top_n]
    final_candidates = score_drop_for_candidates(
        final_top_up,
        raw_rows=rows,
        drop_model=drop_model,
        drop_tokenizer=drop_tokenizer,
        max_seq_length=max_seq_length,
        drop_cache=drop_cache,
    )
    final_ordered = sorted(
        (
            {
                **row,
                "combined_score": combined_score(row, drop_weight),
            }
            for row in final_candidates
        ),
        key=lambda item: item["combined_score"],
        reverse=True,
    )
    final_summary = {
        "samples": len(rows),
        "positive_samples": sum(actual_label(row) == 1 for row in rows),
        "final_top_n": final_top_n,
        "drop_scored_unique": len(drop_cache),
        "drop_weight": drop_weight,
        **precision_at_k(final_ordered, (5, 10, 20, 50, 100, 200)),
    }
    write_jsonl(output_dir / f"final_top{final_top_n}.jsonl", final_ordered)
    write_jsonl(output_dir / "up_scores.jsonl", up_scored)
    summary = {
        "up_adapter_dir": str(up_adapter_dir),
        "drop_adapter_dir": str(drop_adapter_dir),
        "data_dir": str(data_dir),
        "base_model": resolved_base_model,
        "max_seq_length": max_seq_length,
        "progress_every": progress_every,
        "interim_top_n": interim_top_n,
        "final_top_n": final_top_n,
        "final": final_summary,
        "progress_records": progress_records,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jointly evaluate up and drop binary classifiers.")
    parser.add_argument("--up-adapter-dir", type=to_path, required=True)
    parser.add_argument("--drop-adapter-dir", type=to_path, required=True)
    parser.add_argument("--data-dir", type=to_path, required=True)
    parser.add_argument("--output-dir", type=to_path, required=True)
    parser.add_argument("--base-model", help="Base model override. Defaults to up adapter_config.json.")
    parser.add_argument("--drop-weight", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--interim-top-n", type=int, default=50)
    parser.add_argument("--final-top-n", type=int, default=500)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    evaluate_joint(
        up_adapter_dir=args.up_adapter_dir,
        drop_adapter_dir=args.drop_adapter_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        drop_weight=args.drop_weight,
        max_seq_length=max(64, args.max_seq_length),
        progress_every=max(1, args.progress_every),
        interim_top_n=max(1, args.interim_top_n),
        final_top_n=max(1, args.final_top_n),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
