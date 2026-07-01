from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall60.gpu import prepare_rtx3060
from blackbox_finetune_recall60.scripts.joint_evaluate_up_drop import (
    actual_label,
    build_prompt,
    combined_score,
    drop_common,
    infer_base_model,
    load_drop_model,
    load_up_model,
    precision_at_k,
    read_jsonl,
    row_key,
    score_drop_for_candidates,
    score_up_prediction,
    to_path,
    up_common,
    write_jsonl,
)


PRECISION_KS = (5, 10, 20, 50, 100, 200, 500)


def parse_drop_weights(value: str | None) -> list[float] | None:
    if not value:
        return None
    weights: list[float] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        weights.append(float(text))
    if not weights:
        raise ValueError("--drop-weights did not contain any numeric weights")
    return weights


def build_weight_values(
    *,
    weight_start: float,
    weight_end: float,
    weight_step: float,
    drop_weights: str | None = None,
) -> list[float]:
    explicit = parse_drop_weights(drop_weights)
    if explicit is not None:
        return explicit
    if weight_step <= 0:
        raise ValueError("--weight-step must be positive")

    start = Decimal(str(weight_start))
    end = Decimal(str(weight_end))
    step = Decimal(str(weight_step))
    values: list[float] = []
    current = start
    while current <= end + Decimal("0.000000001"):
        values.append(float(current))
        current += step
    return values


def weight_label(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "_")


def order_candidates_for_weight(candidates: list[dict], weight: float) -> list[dict]:
    return sorted(
        (
            {
                **row,
                "drop_weight": weight,
                "combined_score": combined_score(row, weight),
            }
            for row in candidates
        ),
        key=lambda item: item["combined_score"],
        reverse=True,
    )


def summarize_weight(
    *,
    ordered_rows: list[dict],
    drop_weight: float,
    samples: int,
    positive_samples: int,
    final_top_n: int,
) -> dict:
    return {
        "drop_weight": drop_weight,
        "samples": samples,
        "positive_samples": positive_samples,
        "final_top_n": final_top_n,
        "candidate_count": len(ordered_rows),
        **precision_at_k(ordered_rows, PRECISION_KS),
    }


def evaluate_weight_sweep(
    *,
    up_adapter_dir: Path,
    drop_adapter_dir: Path,
    data_dir: Path,
    output_dir: Path,
    base_model: str | None,
    drop_weights: list[float],
    max_seq_length: int,
    final_top_n: int,
    max_samples: int | None,
) -> dict:
    if final_top_n < max(PRECISION_KS):
        raise ValueError(f"--final-top-n must be at least {max(PRECISION_KS)}")

    test_path = data_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"missing evaluation dataset: {test_path}")
    rows = read_jsonl(test_path, max_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded evaluation rows={len(rows)} data={test_path}", flush=True)

    resolved_base_model = infer_base_model(up_adapter_dir, base_model)
    print(f"loading up model adapter={up_adapter_dir}", flush=True)
    up_model, up_tokenizer = load_up_model(resolved_base_model, up_adapter_dir)

    up_scored: list[dict] = []
    started = datetime.now()
    for index, row in enumerate(rows, start=1):
        prompt = build_prompt(row, up_tokenizer, up_common.SYSTEM_PROMPT)
        pred = score_up_prediction(up_model, up_tokenizer, prompt, max_seq_length, threshold=0.5)
        metadata = row.get("metadata", {})
        up_scored.append(
            {
                "key": row_key(row, index - 1),
                "row_index": index - 1,
                "scode": metadata.get("scode"),
                "anchor_date": metadata.get("anchor_date"),
                "actual_label": actual_label(row),
                "up_probability": float(pred["positive_probability"]),
            }
        )
        if index % 1000 == 0 or index == len(rows):
            elapsed = (datetime.now() - started).total_seconds()
            print(
                json.dumps(
                    {
                        "stage": "up_scoring",
                        "scored_rows": index,
                        "total_rows": len(rows),
                        "progress": index / len(rows) if rows else 1.0,
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final_top_up = sorted(up_scored, key=lambda item: item["up_probability"], reverse=True)[:final_top_n]
    print(f"loading drop model adapter={drop_adapter_dir}", flush=True)
    drop_model, drop_tokenizer = load_drop_model(resolved_base_model, drop_adapter_dir)
    final_candidates = score_drop_for_candidates(
        final_top_up,
        raw_rows=rows,
        drop_model=drop_model,
        drop_tokenizer=drop_tokenizer,
        max_seq_length=max_seq_length,
        drop_cache={},
    )

    positive_samples = sum(actual_label(row) == 1 for row in rows)
    summary_records: list[dict] = []
    for weight in drop_weights:
        ordered = order_candidates_for_weight(final_candidates, weight)
        record = summarize_weight(
            ordered_rows=ordered,
            drop_weight=weight,
            samples=len(rows),
            positive_samples=positive_samples,
            final_top_n=final_top_n,
        )
        summary_records.append(record)
        write_jsonl(output_dir / f"top{final_top_n}_drop_weight_{weight_label(weight)}.jsonl", ordered)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    write_jsonl(output_dir / f"final_candidates_top{final_top_n}.jsonl", final_candidates)
    write_jsonl(output_dir / "up_scores.jsonl", up_scored)
    write_jsonl(output_dir / "weight_sweep_summary.jsonl", summary_records)

    summary = {
        "up_adapter_dir": str(up_adapter_dir),
        "drop_adapter_dir": str(drop_adapter_dir),
        "data_dir": str(data_dir),
        "base_model": resolved_base_model,
        "max_seq_length": max_seq_length,
        "final_top_n": final_top_n,
        "drop_weights": drop_weights,
        "results": summary_records,
    }
    with (output_dir / "weight_sweep_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep drop weights for the final up/drop joint evaluation ranking."
    )
    parser.add_argument("--up-adapter-dir", type=to_path, required=True)
    parser.add_argument("--drop-adapter-dir", type=to_path, required=True)
    parser.add_argument("--data-dir", type=to_path, required=True)
    parser.add_argument("--output-dir", type=to_path, required=True)
    parser.add_argument("--base-model", help="Base model override. Defaults to up adapter_config.json.")
    parser.add_argument("--weight-start", type=float, default=0.20)
    parser.add_argument("--weight-end", type=float, default=0.50)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--drop-weights", help="Comma-separated explicit weights; overrides start/end/step.")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--final-top-n", type=int, default=500)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    weights = build_weight_values(
        weight_start=args.weight_start,
        weight_end=args.weight_end,
        weight_step=args.weight_step,
        drop_weights=args.drop_weights,
    )
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    evaluate_weight_sweep(
        up_adapter_dir=args.up_adapter_dir,
        drop_adapter_dir=args.drop_adapter_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        drop_weights=weights,
        max_seq_length=max(64, args.max_seq_length),
        final_top_n=max(1, args.final_top_n),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
