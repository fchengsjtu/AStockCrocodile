from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

from blackbox_finetune_recall60 import train as base_train
from blackbox_finetune_threeclass.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OUTPUT_DIR,
    compact_messages_from_sample,
)
from blackbox_finetune_threeclass.gpu import prepare_rtx3060
from blackbox_finetune_threeclass.inference import score_prediction
from blackbox_finetune_threeclass.metrics import summarize_scored_rows


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
    started = time.monotonic()
    try:
        for index, row in enumerate(eval_rows, start=1):
            messages = compact_messages_from_sample(row)
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            prediction = score_prediction(model, tokenizer, prompt, max_seq_length)
            scored.append(
                {
                    "actual_label": int(row["metadata"]["label"]),
                    "predicted_label": int(prediction["label_id"]),
                    "positive_probability": prediction["positive_probability"],
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
        "precision_top_k": max(1, precision_top_k),
        "precision_threshold": precision_threshold,
        "passed": summary[target_key] >= precision_threshold,
        "max_seq_length": max_seq_length,
        "next_threshold": threshold,
    }
    output_path = output_dir / f"eval-update-{update:06d}-progress-{int(round(progress * 1000)):04d}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"evaluation saved: {output_path} accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f} "
        f"positive_precision@5={summary['positive_precision@5']:.4f} "
        f"positive_precision@10={summary['positive_precision@10']:.4f} "
        f"positive_precision@20={summary['positive_precision@20']:.4f} "
        f"positive_precision@50={summary['positive_precision@50']:.4f} "
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
    )
    parser.add_argument(
        "--initial-binary-adapter-dir",
        "--initial-adapter-dir",
        dest="initial_binary_adapter_dir",
        type=Path,
        default=Path(os.environ["INITIAL_BINARY_ADAPTER_DIR"]) if os.environ.get("INITIAL_BINARY_ADAPTER_DIR") else None,
        help=(
            "Initialize trainable LoRA weights from an already trained binary-classification adapter, "
            "but start three-class optimizer updates at step 0. This does not overwrite the source adapter."
        ),
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
    _patch_base_trainer(initial_binary_adapter_dir)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    if initial_binary_adapter_dir is not None:
        print(
            f"initializing three-class LoRA from binary adapter: {initial_binary_adapter_dir}; "
            "optimizer and update counter start from 0",
            flush=True,
        )
    base_train.train_recall60_lora(
        base_model=args.base_model,
        data_dir=args.data_dir,
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
        nonfinite_patience=args.nonfinite_patience,
        rebuild_token_cache=args.rebuild_token_cache,
        auto_resume=not args.no_auto_resume and initial_binary_adapter_dir is None,
        oom_patience=args.oom_patience,
        nonfinite_skip_limit=args.nonfinite_skip_limit,
        nonfinite_backoff_every=args.nonfinite_backoff_every,
        lr_backoff_factor=args.lr_backoff_factor,
        min_learning_rate=args.min_learning_rate,
        evaluation_threshold=args.eval_threshold,
        evaluation_max_samples=args.eval_max_samples,
        evaluation_output_dir=args.eval_output_dir,
        evaluation_precision_top_k=args.eval_precision_top_k,
        evaluation_precision_threshold=args.eval_precision_threshold,
        on_the_fly_tokenize=args.on_the_fly_tokenize,
    )


if __name__ == "__main__":
    main()
