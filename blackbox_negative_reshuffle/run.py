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
    NEGATIVE_KIND_DROP6,
    NEGATIVE_KIND_NEUTRAL,
    copy_dataset,
    copy_model_files,
    load_source_metadata_from_paths,
    load_source_metadata,
    plan_negative_kind_counts,
    read_jsonl,
    reshuffle_split,
    reshuffle_split_by_negative_kind,
    row_key,
    row_label,
    score_negative_rows,
    write_jsonl,
)

NEGATIVE_EXCLUSION_TRADING_DAYS = 20
TRADING_DATE_SYMBOL_BATCH_SIZE = 500
DROP6_LOOKAHEAD_TRADING_DAYS = 3
DROP6_THRESHOLD = 0.06


def parse_date_value(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return date.fromisoformat(text)


def infer_dataset_settings(rows: list[dict], sample_mode: str | None) -> tuple[str, object, object]:

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
        parse_date_value(metadata["anchor_date"])
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


def row_symbol_anchor(row: dict) -> tuple[str, date] | None:
    metadata = row.get("metadata", {})
    scode = metadata.get("scode")
    anchor_date = metadata.get("anchor_date")
    if not scode or not anchor_date:
        return None
    return str(scode), parse_date_value(anchor_date)


def _to_positive_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def classify_drop6_anchors(
    conn,
    anchors: list[tuple[str, date]],
    *,
    drop_threshold: float = DROP6_THRESHOLD,
    lookahead_days: int = DROP6_LOOKAHEAD_TRADING_DAYS,
    batch_size: int = TRADING_DATE_SYMBOL_BATCH_SIZE,
) -> dict[tuple[str, str], bool]:
    from blackbox_finetune_recall60.common import parse_date

    if not anchors:
        return {}
    anchors_by_symbol: dict[str, set[date]] = {}
    for scode, anchor_date in anchors:
        anchors_by_symbol.setdefault(str(scode), set()).add(anchor_date)
    min_date = min(anchor_date for _, anchor_date in anchors)
    max_date = max(anchor_date for _, anchor_date in anchors)
    result: dict[tuple[str, str], bool] = {
        (str(scode), anchor_date.isoformat()): False for scode, anchor_date in anchors
    }
    symbols = sorted(anchors_by_symbol)
    query_end = max_date + timedelta(days=max(20, lookahead_days * 7 + 7))
    for start in range(0, len(symbols), max(1, batch_size)):
        batch = symbols[start : start + max(1, batch_size)]
        placeholders = ",".join(["%s"] * len(batch))
        sql = f"""
            SELECT SCode, DATE(KTime), Close, Low
            FROM dkandles
            WHERE KType = 'D'
              AND SCode IN ({placeholders})
              AND KTime >= %s
              AND KTime < %s
              AND Close IS NOT NULL
              AND Low IS NOT NULL
            ORDER BY SCode, KTime
        """
        with conn.cursor() as cur:
            cur.execute(sql, [*batch, min_date, query_end + timedelta(days=1)])
            rows = cur.fetchall()
        rows_by_symbol: dict[str, list[tuple[date, float, float]]] = {}
        for scode, trade_date, close_value, low_value in rows:
            close = _to_positive_float(close_value)
            low = _to_positive_float(low_value)
            if close is None or low is None:
                continue
            rows_by_symbol.setdefault(str(scode), []).append((parse_date(trade_date), close, low))
        for scode, bars in rows_by_symbol.items():
            index_by_date = {trade_date: index for index, (trade_date, _, _) in enumerate(bars)}
            for anchor_date in anchors_by_symbol.get(scode, set()):
                index = index_by_date.get(anchor_date)
                if index is None:
                    continue
                future = bars[index + 1 : index + 1 + lookahead_days]
                if len(future) < lookahead_days:
                    continue
                anchor_close = bars[index][1]
                min_future_low = min(low for _, _, low in future)
                result[(scode, anchor_date.isoformat())] = min_future_low <= anchor_close * (1.0 - drop_threshold)
    return result


def classify_negative_rows_by_kind(
    conn,
    rows: list[dict],
    *,
    drop_threshold: float = DROP6_THRESHOLD,
    lookahead_days: int = DROP6_LOOKAHEAD_TRADING_DAYS,
    batch_size: int = TRADING_DATE_SYMBOL_BATCH_SIZE,
) -> dict[tuple[str, str, int], str]:
    anchors = [anchor for row in rows if (anchor := row_symbol_anchor(row)) is not None]
    flags = classify_drop6_anchors(
        conn,
        anchors,
        drop_threshold=drop_threshold,
        lookahead_days=lookahead_days,
        batch_size=batch_size,
    )
    return {
        row_key(row): NEGATIVE_KIND_DROP6
        if (anchor := row_symbol_anchor(row)) is not None
        and flags.get((anchor[0], anchor[1].isoformat()), False)
        else NEGATIVE_KIND_NEUTRAL
        for row in rows
    }


def filter_events_by_negative_kind(
    conn,
    events,
    negative_kind: str,
    *,
    drop_threshold: float,
    lookahead_days: int,
    batch_size: int,
):
    if negative_kind not in {NEGATIVE_KIND_DROP6, NEGATIVE_KIND_NEUTRAL}:
        return list(events)
    anchors = [(str(event.scode), event.anchor_date) for event in events]
    flags = classify_drop6_anchors(
        conn,
        anchors,
        drop_threshold=drop_threshold,
        lookahead_days=lookahead_days,
        batch_size=batch_size,
    )
    want_drop6 = negative_kind == NEGATIVE_KIND_DROP6
    return [
        event
        for event in events
        if flags.get((str(event.scode), event.anchor_date.isoformat()), False) == want_drop6
    ]


def load_random_negative_events_without_akshare_dependency(
    conn,
    stat_type: str,
    start_date: date,
    end_date: date,
    limit: int,
    seed: int,
    batch_size: int,
    negative_kind: str = "any",
    drop_threshold: float = DROP6_THRESHOLD,
    lookahead_days: int = DROP6_LOOKAHEAD_TRADING_DAYS,
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
        batch_candidates = []
        for scode, trade_date_value in rows:
            scode = str(scode)
            trade_date = parse_date(trade_date_value)
            key = (scode, trade_date)
            if key in seen or trade_date in excluded.get(scode, set()):
                continue
            seen.add(key)
            batch_candidates.append(SampleEvent(scode, trade_date, 0, "negative", None))
        batch_candidates = filter_events_by_negative_kind(
            conn,
            batch_candidates,
            negative_kind,
            drop_threshold=drop_threshold,
            lookahead_days=lookahead_days,
            batch_size=batch_size,
        )
        candidates.extend(batch_candidates[: max(0, limit - len(candidates))])
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
    negative_kind: str = "any",
    drop_threshold: float = DROP6_THRESHOLD,
    lookahead_days: int = DROP6_LOOKAHEAD_TRADING_DAYS,
) -> list[dict]:
    from blackbox_finetune_recall60.common import materialize_events, mysql_connect

    replacements: dict[tuple[str, str, int], dict] = {}
    if cache_path is not None and cache_path.is_file():
        for row in read_jsonl(cache_path):
            key = row_key(row)
            row_kind = row.get("metadata", {}).get("negative_kind")
            kind_matches = negative_kind == "any" or row_kind in (None, negative_kind)
            if row_label(row) == 0 and key not in current_negative_keys and kind_matches:
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
                negative_kind=negative_kind,
                drop_threshold=drop_threshold,
                lookahead_days=lookahead_days,
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
                if negative_kind != "any":
                    row.setdefault("metadata", {})["negative_kind"] = negative_kind
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


def load_cached_negative_scores(score_path: Path, current_negatives: list[dict]) -> list[tuple[float, dict]]:
    score_by_anchor: dict[tuple[str, str], float] = {}
    for row in read_jsonl(score_path):
        scode = str(row.get("scode", ""))
        anchor_date = str(row.get("anchor_date", ""))
        if not scode or not anchor_date:
            continue
        score_by_anchor[(scode, anchor_date)] = float(row.get("score", float("-inf")))
    scored = []
    for row in current_negatives:
        metadata = row.get("metadata", {})
        key = (str(metadata.get("scode", "")), str(metadata.get("anchor_date", "")))
        scored.append((score_by_anchor.get(key, float("-inf")), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def split_scored_negatives_by_kind(
    scored_current_negatives: list[tuple[float, dict]],
    kind_by_key: dict[tuple[str, str, int], str],
) -> tuple[list[tuple[float, dict]], list[tuple[float, dict]]]:
    drop6 = []
    neutral = []
    for score, row in scored_current_negatives:
        if kind_by_key.get(row_key(row)) == NEGATIVE_KIND_DROP6:
            drop6.append((score, row))
        else:
            neutral.append((score, row))
    return drop6, neutral


def count_split_kind_available(
    rows: list[dict],
    kind_by_key: dict[tuple[str, str, int], str],
    kind: str,
) -> int:
    return sum(1 for row in rows if row_label(row) == 0 and kind_by_key.get(row_key(row)) == kind)


def run_reshuffle(
    model_dir: Path,
    evaluation_json: Path | None,
    train_dataset_dir: Path | None,
    eval_dataset_dir: Path | None,
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
    target_negative_ratio: float,
    drop6_target_ratio: float,
    drop6_keep_ratio: float,
    neutral_keep_ratio: float,
    drop6_threshold: float,
    drop6_lookahead_days: int,
) -> Path:
    model_dir = model_dir.resolve()
    run_dir = model_dir / "negative_reshuffle" / output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (train_dataset_dir is None) != (eval_dataset_dir is None):
        raise ValueError("--train-dataset-dir and --eval-dataset-dir must be provided together")
    if train_dataset_dir is not None and eval_dataset_dir is not None:
        metadata = load_source_metadata_from_paths(
            model_dir,
            train_dataset_dir,
            eval_dataset_dir,
            evaluation_json,
        )
    else:
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
    from blackbox_finetune_recall60.common import mysql_connect

    with mysql_connect() as conn:
        negative_kind_by_key = classify_negative_rows_by_kind(
            conn,
            current_negatives,
            drop_threshold=drop6_threshold,
            lookahead_days=drop6_lookahead_days,
            batch_size=database_batch_size,
        )
    current_drop6_count = sum(1 for kind in negative_kind_by_key.values() if kind == NEGATIVE_KIND_DROP6)
    current_neutral_count = len(current_negatives) - current_drop6_count
    print(
        f"classified current negatives drop6={current_drop6_count} neutral={current_neutral_count} "
        f"drop6_threshold={drop6_threshold} lookahead_days={drop6_lookahead_days}",
        flush=True,
    )
    scorer, resolved_base_model = build_model_scorer(
        model_dir,
        base_model,
        max_seq_length,
        cuda_device,
    )
    scored_current_negatives = score_negative_rows(current_negatives, scorer, progress_every)
    scored_drop6_negatives, scored_neutral_negatives = split_scored_negatives_by_kind(
        scored_current_negatives,
        negative_kind_by_key,
    )
    rng = random.Random(seed)
    train_positive_count = sum(row_label(row) == 1 for row in train_rows)
    test_positive_count = sum(row_label(row) == 1 for row in test_rows)
    train_target_negative_count = round(train_positive_count * max(0.0, target_negative_ratio))
    test_target_negative_count = round(test_positive_count * max(0.0, target_negative_ratio))
    train_plan = plan_negative_kind_counts(
        train_positive_count,
        train_target_negative_count,
        drop6_target_ratio=drop6_target_ratio,
        drop6_keep_ratio=drop6_keep_ratio,
        neutral_keep_ratio=neutral_keep_ratio,
    )
    test_plan = plan_negative_kind_counts(
        test_positive_count,
        test_target_negative_count,
        drop6_target_ratio=drop6_target_ratio,
        drop6_keep_ratio=drop6_keep_ratio,
        neutral_keep_ratio=neutral_keep_ratio,
    )
    train_available_drop6 = count_split_kind_available(train_rows, negative_kind_by_key, NEGATIVE_KIND_DROP6)
    test_available_drop6 = count_split_kind_available(test_rows, negative_kind_by_key, NEGATIVE_KIND_DROP6)
    train_available_neutral = count_split_kind_available(train_rows, negative_kind_by_key, NEGATIVE_KIND_NEUTRAL)
    test_available_neutral = count_split_kind_available(test_rows, negative_kind_by_key, NEGATIVE_KIND_NEUTRAL)
    drop6_replacement_count = (
        max(0, train_plan.target_drop6_count - min(train_plan.keep_drop6_count, train_available_drop6))
        + max(0, test_plan.target_drop6_count - min(test_plan.keep_drop6_count, test_available_drop6))
    )
    neutral_replacement_count = (
        max(0, train_plan.target_neutral_count - min(train_plan.keep_neutral_count, train_available_neutral))
        + max(0, test_plan.target_neutral_count - min(test_plan.keep_neutral_count, test_available_neutral))
    )
    resolved_sample_mode, start_date, end_date = infer_dataset_settings(
        all_source_rows,
        sample_mode,
    )
    drop6_replacement_pool = load_database_replacement_pool(
        current_negative_keys=set(current_negative_by_key),
        required_count=drop6_replacement_count,
        stat_type=stat_type,
        start_date=start_date,
        end_date=end_date,
        sample_mode=resolved_sample_mode,
        seed=seed,
        batch_size=database_batch_size,
        max_attempts=database_max_attempts,
        cache_path=run_dir / "database_replacement_pool_drop6.jsonl",
        negative_kind=NEGATIVE_KIND_DROP6,
        drop_threshold=drop6_threshold,
        lookahead_days=drop6_lookahead_days,
    )
    neutral_excluded_keys = set(current_negative_by_key) | {row_key(row) for row in drop6_replacement_pool}
    neutral_replacement_pool = load_database_replacement_pool(
        current_negative_keys=neutral_excluded_keys,
        required_count=neutral_replacement_count,
        stat_type=stat_type,
        start_date=start_date,
        end_date=end_date,
        sample_mode=resolved_sample_mode,
        seed=seed + 700001,
        batch_size=database_batch_size,
        max_attempts=database_max_attempts,
        cache_path=run_dir / "database_replacement_pool_neutral.jsonl",
        negative_kind=NEGATIVE_KIND_NEUTRAL,
        drop_threshold=drop6_threshold,
        lookahead_days=drop6_lookahead_days,
    )
    new_train_rows, train_keys, train_stats = reshuffle_split_by_negative_kind(
        train_rows,
        scored_drop6_negatives,
        scored_neutral_negatives,
        drop6_replacement_pool,
        neutral_replacement_pool,
        train_plan,
        rng,
    )
    new_test_rows, test_keys, test_stats = reshuffle_split_by_negative_kind(
        test_rows,
        scored_drop6_negatives,
        scored_neutral_negatives,
        drop6_replacement_pool,
        neutral_replacement_pool,
        test_plan,
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
                "negative_kind": negative_kind_by_key.get(row_key(row), NEGATIVE_KIND_NEUTRAL),
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
        "target_negative_ratio": target_negative_ratio,
        "drop6_target_ratio": drop6_target_ratio,
        "drop6_keep_ratio": drop6_keep_ratio,
        "neutral_keep_ratio": neutral_keep_ratio,
        "drop6_threshold": drop6_threshold,
        "drop6_lookahead_days": drop6_lookahead_days,
        "scored_current_negative_count": len(current_negatives),
        "current_drop6_negative_count": current_drop6_count,
        "current_neutral_negative_count": current_neutral_count,
        "database_drop6_replacement_pool_count": len(drop6_replacement_pool),
        "database_neutral_replacement_pool_count": len(neutral_replacement_pool),
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
    parser.add_argument("--train-dataset-dir", type=Path, help="Training dataset directory containing train.jsonl/test.jsonl/all.jsonl. Overrides eval JSON dataset paths when paired with --eval-dataset-dir.")
    parser.add_argument("--eval-dataset-dir", type=Path, help="Evaluation dataset directory copied into the generated reshuffle run. Overrides eval JSON dataset paths when paired with --train-dataset-dir.")
    parser.add_argument("--output-name", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--keep-ratio", type=float, default=0.30)
    parser.add_argument("--keep-count", type=int)
    parser.add_argument("--seed", type=int, default=937498347)
    parser.add_argument("--base-model")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--stat-type", default="short_term_surge_3d_20pct")
    parser.add_argument("--sample-mode", choices=["short", "long", "xlong", "xxlong"])
    parser.add_argument("--database-batch-size", type=int, default=80)
    parser.add_argument("--target-negative-ratio", type=float, default=9.0)
    parser.add_argument("--drop6-target-ratio", type=float, default=3.0)
    parser.add_argument("--drop6-keep-ratio", type=float, default=1.0)
    parser.add_argument("--neutral-keep-ratio", type=float, default=1.0)
    parser.add_argument("--drop6-threshold", type=float, default=DROP6_THRESHOLD)
    parser.add_argument("--drop6-lookahead-days", type=int, default=DROP6_LOOKAHEAD_TRADING_DAYS)
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
        train_dataset_dir=args.train_dataset_dir,
        eval_dataset_dir=args.eval_dataset_dir,
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
        target_negative_ratio=max(0.0, args.target_negative_ratio),
        drop6_target_ratio=min(max(0.0, args.drop6_target_ratio), max(0.0, args.target_negative_ratio)),
        drop6_keep_ratio=max(0.0, args.drop6_keep_ratio),
        neutral_keep_ratio=max(0.0, args.neutral_keep_ratio),
        drop6_threshold=min(max(0.0, args.drop6_threshold), 1.0),
        drop6_lookahead_days=max(1, args.drop6_lookahead_days),
    )


if __name__ == "__main__":
    main()
