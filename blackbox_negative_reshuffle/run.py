from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_negative_reshuffle.core import (
    copy_dataset,
    copy_model_files,
    load_source_metadata,
    read_jsonl,
    reshuffle_split,
    row_key,
    row_label,
    score_negative_rows,
    write_jsonl,
)

NEGATIVE_EXCLUSION_TRADING_DAYS = 20
TRADING_DATE_SYMBOL_BATCH_SIZE = 500


def infer_dataset_settings(rows: list[dict], sample_mode: str | None) -> tuple[str, object, object]:
    from blackbox_finetune_recall60.common import parse_date

    if not rows:
        raise ValueError("original training dataset is empty")
    metadata_rows = [row.get("metadata", {}) for row in rows]
    resolved_mode = sample_mode or next(
        (str(metadata.get("sample_mode")) for metadata in metadata_rows if metadata.get("sample_mode")),
        None,
    )
    if not resolved_mode:
        raise ValueError("sample mode is missing; pass --sample-mode")
    dates = [
        parse_date(metadata["anchor_date"])
        for metadata in metadata_rows
        if metadata.get("anchor_date")
    ]
    if not dates:
        raise ValueError("original dataset contains no anchor_date values")
    return resolved_mode, min(dates), max(dates)


def load_excluded_positive_windows(conn, stat_type: str, start_date: date, end_date: date, padding_days: int = 10) -> dict[str, set[date]]:
    from blackbox_finetune_recall60.common import parse_date

    sql = """
        SELECT SCode, PrevTradeDate
        FROM klinestatistics
        WHERE StatType = %s
          AND PrevTradeDate >= %s
          AND PrevTradeDate <= %s
          AND PrevTradeDate IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (stat_type, start_date - timedelta(days=padding_days), end_date + timedelta(days=padding_days)))
        rows = cur.fetchall()
    raw_dates: dict[str, set[date]] = {}
    for scode, prev_trade_date in rows:
        raw_dates.setdefault(str(scode), set()).add(parse_date(prev_trade_date))
    return raw_dates


def load_trading_dates_for_symbols(conn, symbols: list[str], start_date: date, end_date: date) -> dict[str, list[date]]:
    from blackbox_finetune_recall60.common import parse_date

    if not symbols:
        return {}
    placeholders = ",".join(["%s"] * len(symbols))
    sql = f"""
        SELECT SCode, DATE(KTime)
        FROM dkandles
        WHERE KType = 'D'
          AND SCode IN ({placeholders})
          AND KTime >= %s
          AND KTime < %s
        ORDER BY SCode, KTime
    """
    with conn.cursor() as cur:
        cur.execute(sql, [*symbols, start_date, end_date + timedelta(days=1)])
        rows = cur.fetchall()
    dates: dict[str, list[date]] = {}
    for scode, trade_date in rows:
        dates.setdefault(str(scode), []).append(parse_date(trade_date))
    return dates


def load_trading_dates_for_symbols_batched(
    conn,
    symbols: list[str],
    start_date: date,
    end_date: date,
    batch_size: int = TRADING_DATE_SYMBOL_BATCH_SIZE,
) -> dict[str, list[date]]:
    result: dict[str, list[date]] = {}
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        batch_dates = load_trading_dates_for_symbols(conn, batch, start_date, end_date)
        for scode, dates in batch_dates.items():
            result.setdefault(scode, []).extend(dates)
    return result


def excluded_dates_by_symbol(
    conn,
    positive_windows: dict[str, set[date]],
    start_date: date,
    end_date: date,
    buffer: int,
    batch_size: int,
) -> dict[str, set[date]]:
    result: dict[str, set[date]] = {scode: set() for scode in positive_windows}
    symbols = sorted(positive_windows)
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        padding_days = max(20, buffer * 3 + 10)
        date_map = load_trading_dates_for_symbols_batched(
            conn,
            batch,
            start_date - timedelta(days=padding_days),
            end_date + timedelta(days=padding_days),
            batch_size,
        )
        for scode in batch:
            dates = date_map.get(scode, [])
            index = {trade_date: idx for idx, trade_date in enumerate(dates)}
            for pos_date in positive_windows.get(scode, set()):
                idx = index.get(pos_date)
                if idx is None:
                    continue
                lo = max(0, idx - buffer)
                hi = min(len(dates), idx + buffer + 1)
                result.setdefault(scode, set()).update(dates[lo:hi])
    return result


def load_random_negative_events_without_akshare_dependency(
    conn,
    stat_type: str,
    start_date: date,
    end_date: date,
    limit: int,
    seed: int,
    batch_size: int,
):
    from blackbox_finetune_recall60.common import SampleEvent, parse_date

    positive_windows = load_excluded_positive_windows(
        conn,
        stat_type,
        start_date,
        end_date,
        NEGATIVE_EXCLUSION_TRADING_DAYS * 3 + 10,
    )
    excluded = excluded_dates_by_symbol(
        conn,
        positive_windows,
        start_date,
        end_date,
        NEGATIVE_EXCLUSION_TRADING_DAYS,
        batch_size,
    )
    candidates: list[SampleEvent] = []
    seen: set[tuple[str, date]] = set()
    scan_limit = max(limit * 3, limit + 1000, 5000)
    attempt = 0
    while len(candidates) < limit and attempt < 5:
        sql = """
            SELECT SCode, DATE(KTime)
            FROM dkandles
            WHERE KType = 'D'
              AND KTime >= %s
              AND KTime < %s
            ORDER BY RAND(%s)
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (start_date, end_date + timedelta(days=1), seed + attempt, scan_limit))
            rows = cur.fetchall()
        for scode, trade_date_value in rows:
            scode = str(scode)
            trade_date = parse_date(trade_date_value)
            key = (scode, trade_date)
            if key in seen or trade_date in excluded.get(scode, set()):
                continue
            seen.add(key)
            candidates.append(SampleEvent(scode, trade_date, 0, "negative", None))
            if len(candidates) >= limit:
                break
        attempt += 1
        scan_limit *= 2
    return candidates[:limit]


def load_database_replacement_pool(
    current_negative_keys: set[tuple[str, str, int]],
    required_count: int,
    stat_type: str,
    start_date,
    end_date,
    sample_mode: str,
    seed: int,
    batch_size: int,
    max_attempts: int,
    cache_path: Path | None = None,
) -> list[dict]:
    from blackbox_finetune_recall60.common import materialize_events, mysql_connect

    replacements: dict[tuple[str, str, int], dict] = {}
    if cache_path is not None and cache_path.is_file():
        for row in read_jsonl(cache_path):
            key = row_key(row)
            if row_label(row) == 0 and key not in current_negative_keys:
                replacements[key] = row
        print(
            f"loaded cached database replacement pool usable={len(replacements)} "
            f"path={cache_path}",
            flush=True,
        )
    with mysql_connect() as conn:
        for attempt in range(max(1, max_attempts)):
            remaining = required_count - len(replacements)
            if remaining <= 0:
                break
            prior_usable = len(replacements)
            growth_factor = min(8, 3 + attempt // 2)
            requested_events = max(remaining * growth_factor, remaining + 5000)
            events = load_random_negative_events_without_akshare_dependency(
                conn,
                stat_type,
                start_date,
                end_date,
                requested_events,
                seed + attempt * 100003,
                batch_size,
            )
            filtered_events = [
                event
                for event in events
                if (event.scode, event.anchor_date.isoformat(), 0) not in current_negative_keys
                and (event.scode, event.anchor_date.isoformat(), 0) not in replacements
            ]
            if not filtered_events:
                continue
            materialized = materialize_events(
                conn,
                filtered_events,
                daily_window=None,
                weekly_window=None,
                batch_size=batch_size,
                sample_mode=sample_mode,
                monthly_window=None,
            )
            for row in materialized:
                key = row_key(row)
                if key in current_negative_keys or key in replacements:
                    continue
                replacements[key] = row
                if len(replacements) >= required_count:
                    break
            if cache_path is not None:
                write_jsonl(cache_path, replacements.values())
            gained = len(replacements) - prior_usable
            print(
                f"database replacement pool attempt={attempt + 1}/{max_attempts} "
                f"requested_events={requested_events} gained={gained} "
                f"required={required_count} usable={len(replacements)} "
                f"remaining={max(0, required_count - len(replacements))}",
                flush=True,
            )
    if len(replacements) < required_count:
        raise RuntimeError(
            f"database produced only {len(replacements)} usable replacement negatives; "
            f"{required_count} are required"
        )
    return list(replacements.values())


def build_model_scorer(model_dir: Path, base_model: str | None, max_seq_length: int, cuda_device: str):
    from blackbox_finetune_recall60.common import compact_messages_from_sample
    from blackbox_finetune_recall60.gpu import prepare_rtx3060
    from blackbox_finetune_recall60.inference import load_model, score_prediction

    prepare_rtx3060(cuda_device, require_device=True)
    adapter_config = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
    resolved_base_model = base_model or adapter_config.get("base_model_name_or_path")
    if not resolved_base_model:
        raise ValueError("base model is missing; pass --base-model")
    model, tokenizer = load_model(resolved_base_model, model_dir)

    def scorer(row: dict) -> float:
        messages = compact_messages_from_sample(row)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        return float(
            score_prediction(model, tokenizer, prompt, max_seq_length, threshold=0.5)[
                "positive_probability"
            ]
        )

    return scorer, resolved_base_model


def run_reshuffle(
    model_dir: Path,
    evaluation_json: Path | None,
    output_name: str,
    keep_ratio: float,
    keep_count: int | None,
    seed: int,
    base_model: str | None,
    max_seq_length: int,
    cuda_device: str,
    progress_every: int,
    stat_type: str,
    sample_mode: str | None,
    database_batch_size: int,
    database_max_attempts: int,
) -> Path:
    model_dir = model_dir.resolve()
    run_dir = model_dir / "negative_reshuffle" / output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_source_metadata(model_dir, evaluation_json)
    train_rows = read_jsonl(metadata.training_dataset_dir / "train.jsonl")
    test_rows = read_jsonl(metadata.training_dataset_dir / "test.jsonl")
    all_source_rows = train_rows + test_rows
    current_negative_by_key = {
        row_key(row): row
        for row in all_source_rows
        if row_label(row) == 0
    }
    current_negatives = list(current_negative_by_key.values())
    if not current_negatives:
        raise RuntimeError("original training dataset contains no negative samples")
    scorer, resolved_base_model = build_model_scorer(
        model_dir,
        base_model,
        max_seq_length,
        cuda_device,
    )
    scored_current_negatives = score_negative_rows(current_negatives, scorer, progress_every)
    rng = random.Random(seed)
    train_negative_count = sum(row_label(row) == 0 for row in train_rows)
    test_negative_count = sum(row_label(row) == 0 for row in test_rows)
    train_keep_count = (
        min(max(0, keep_count), train_negative_count)
        if keep_count is not None
        else round(train_negative_count * min(max(keep_ratio, 0.0), 1.0))
    )
    test_keep_count = (
        min(max(0, keep_count), test_negative_count)
        if keep_count is not None
        else round(test_negative_count * min(max(keep_ratio, 0.0), 1.0))
    )
    replacement_count = (
        train_negative_count
        - train_keep_count
        + test_negative_count
        - test_keep_count
    )
    resolved_sample_mode, start_date, end_date = infer_dataset_settings(
        all_source_rows,
        sample_mode,
    )
    replacement_pool = load_database_replacement_pool(
        current_negative_keys=set(current_negative_by_key),
        required_count=replacement_count,
        stat_type=stat_type,
        start_date=start_date,
        end_date=end_date,
        sample_mode=resolved_sample_mode,
        seed=seed,
        batch_size=database_batch_size,
        max_attempts=database_max_attempts,
        cache_path=run_dir / "database_replacement_pool.jsonl",
    )
    new_train_rows, train_keys, train_stats = reshuffle_split(
        train_rows,
        scored_current_negatives,
        replacement_pool,
        train_keep_count,
        rng,
    )
    new_test_rows, test_keys, test_stats = reshuffle_split(
        test_rows,
        scored_current_negatives,
        replacement_pool,
        test_keep_count,
        rng,
        excluded_keys=train_keys,
    )
    training_output = run_dir / "datasets" / "training"
    evaluation_output = run_dir / "datasets" / "evaluation"
    adapter_output = run_dir / "adapter"
    write_jsonl(training_output / "train.jsonl", new_train_rows)
    write_jsonl(training_output / "test.jsonl", new_test_rows)
    write_jsonl(training_output / "all.jsonl", new_train_rows + new_test_rows)
    copy_dataset(metadata.evaluation_dataset_dir, evaluation_output)
    copied_model_files = copy_model_files(model_dir, adapter_output)
    scores_output = run_dir / "negative_scores.jsonl"
    write_jsonl(
        scores_output,
        (
            {
                "scode": row.get("metadata", {}).get("scode"),
                "anchor_date": row.get("metadata", {}).get("anchor_date"),
                "score": score,
            }
            for score, row in scored_current_negatives
        ),
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_model_path": str(model_dir),
        "source_evaluation_json": str(metadata.evaluation_json),
        "original_train_dataset_path": str(metadata.training_dataset_dir),
        "original_eval_dataset_path": str(metadata.evaluation_dataset_dir),
        "generated_train_dataset_path": str(training_output),
        "generated_eval_dataset_path": str(evaluation_output),
        "generated_model_path": str(adapter_output),
        "base_model": resolved_base_model,
        "max_seq_length": max_seq_length,
        "stat_type": stat_type,
        "sample_mode": resolved_sample_mode,
        "database_start_date": start_date.isoformat(),
        "database_end_date": end_date.isoformat(),
        "seed": seed,
        "keep_ratio": keep_ratio,
        "keep_count": keep_count,
        "scored_current_negative_count": len(current_negatives),
        "database_replacement_pool_count": len(replacement_pool),
        "train": train_stats,
        "test": test_stats,
        "train_test_negative_overlap": len(train_keys & test_keys),
        "copied_model_files": copied_model_files,
    }
    manifest_path = run_dir / "reshuffle_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation_record = {}
    try:
        evaluation_record = json.loads(metadata.evaluation_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        evaluation_record = {}
    evaluation_record.update(
        {
            "original_train_dataset_path": str(metadata.training_dataset_dir),
            "original_eval_dataset_path": str(metadata.evaluation_dataset_dir),
            "generated_train_dataset_path": str(training_output),
            "generated_eval_dataset_path": str(evaluation_output),
            "generated_model_path": str(adapter_output),
            "negative_reshuffle_manifest": str(manifest_path),
        }
    )
    evaluation_output_path = run_dir / "runs" / "evaluations" / metadata.evaluation_json.name
    evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output_path.write_text(
        json.dumps(evaluation_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"negative reshuffle completed: {run_dir}", flush=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reshuffle negative samples using a trained black-box adapter."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--output-name", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--keep-ratio", type=float, default=0.20)
    parser.add_argument("--keep-count", type=int)
    parser.add_argument("--seed", type=int, default=937498347)
    parser.add_argument("--base-model")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--stat-type", default="short_term_surge_3d_20pct")
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"])
    parser.add_argument("--database-batch-size", type=int, default=80)
    parser.add_argument(
        "--database-max-attempts",
        type=int,
        default=20,
        help="Maximum database refill rounds. Reusing output-name resumes the cached replacement pool.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_reshuffle(
        model_dir=args.model_dir,
        evaluation_json=args.evaluation_json,
        output_name=args.output_name,
        keep_ratio=args.keep_ratio,
        keep_count=args.keep_count,
        seed=args.seed,
        base_model=args.base_model,
        max_seq_length=max(64, args.max_seq_length),
        cuda_device=args.cuda_device,
        progress_every=max(1, args.progress_every),
        stat_type=args.stat_type,
        sample_mode=args.sample_mode,
        database_batch_size=max(1, args.database_batch_size),
        database_max_attempts=max(1, args.database_max_attempts),
    )


if __name__ == "__main__":
    main()
