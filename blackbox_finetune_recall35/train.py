from __future__ import annotations

import argparse
import json
import os
import hashlib
import math
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall35.common import DEFAULT_BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_TRAIN_SEED, compact_messages_from_sample, default_max_seq_length, default_output_dir
from blackbox_finetune_recall35.gpu import prepare_rtx3060
from blackbox_finetune_recall35.inference import score_prediction


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


from llm_finetune.common import read_jsonl


def default_checkpoint_every() -> int:
    return 500

TOKEN_CACHE_VERSION = "v7_csv11_21d13w_no_partial_2020_2025"


def _missing_dataset_error(data_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(f"missing dataset files under {data_dir}; run blackbox_finetune_recall35.build_dataset first")


def _tokenize_row(tokenizer, row: dict, max_seq_length: int) -> dict:
    messages = compact_messages_from_sample(row)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = messages[-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(answer_ids) >= max_seq_length:
        answer_ids = answer_ids[: max_seq_length - 1] + [tokenizer.eos_token_id]
    prompt_budget = max(1, max_seq_length - len(answer_ids))
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = (prompt_ids + answer_ids)[:max_seq_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _collate(tokenizer, batch: list[dict], device: str) -> dict:
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _params_are_finite(params) -> bool:
    for param in params:
        if not param.data.isfinite().all().item():
            return False
    return True



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
) -> dict:
    eval_rows = rows[:max_samples] if max_samples and max_samples > 0 else rows
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    tp = fp = tn = fn = positives = 0
    try:
        for row in eval_rows:
            messages = compact_messages_from_sample(row)
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            pred = score_prediction(model, tokenizer, prompt, max_seq_length, threshold)
            predicted_positive = pred["label"] == "positive"
            actual_positive = int(row["metadata"]["label"]) == 1
            if actual_positive:
                positives += 1
            if predicted_positive and actual_positive:
                tp += 1
            elif predicted_positive and not actual_positive:
                fp += 1
            elif not predicted_positive and actual_positive:
                fn += 1
            else:
                tn += 1
    finally:
        if was_training:
            model.train()
    positive_recall = tp / positives if positives else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    result = {
        "update": update,
        "total_updates": total_updates,
        "progress": progress,
        "trained_epochs": trained_epochs,
        "samples": len(eval_rows),
        "positive_samples": positives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_recall": positive_recall,
        "precision": precision,
        "threshold": threshold,
        "max_seq_length": max_seq_length,
    }
    output_path = output_dir / f"eval-update-{update:06d}-progress-{int(round(progress * 1000)):04d}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"evaluation saved: {output_path} positive_recall={positive_recall:.4f} precision={precision:.4f} samples={len(eval_rows)}", flush=True)
    return result
def _checkpoint_update(checkpoint_dir: Path | None) -> int:
    if checkpoint_dir is None:
        return 0
    match = re.fullmatch(r"update-(\d+)", checkpoint_dir.name)
    if not match:
        return 0
    return int(match.group(1))


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None
    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and (path / "adapter_config.json").exists() and _checkpoint_update(path) > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_update)


def _token_cache_path(data_dir: Path, train_path: Path, base_model: str, max_seq_length: int) -> Path:
    stat = train_path.stat()
    fingerprint = hashlib.sha256(
        f"{TOKEN_CACHE_VERSION}|{train_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{base_model}|{max_seq_length}".encode("utf-8")
    ).hexdigest()[:16]
    model_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_model).strip("_") or "model"
    return data_dir / "tokenized" / f"{model_key}_seq{max_seq_length}_{fingerprint}.pkl"


def _load_or_build_tokenized(tokenizer, rows: list[dict], data_dir: Path, train_path: Path, base_model: str, max_seq_length: int, rebuild: bool) -> list[dict]:
    cache_path = _token_cache_path(data_dir, train_path, base_model, max_seq_length)
    if cache_path.exists() and not rebuild:
        load_start = time.monotonic()
        with cache_path.open("rb") as file:
            tokenized = pickle.load(file)
        print(
            f"loaded tokenized train cache rows={len(tokenized)} path={cache_path} "
            f"elapsed={_format_duration(time.monotonic() - load_start)}",
            flush=True,
        )
        return tokenized

    tokenized = []
    tokenize_start = time.monotonic()
    tokenize_total = len(rows)
    print(f"tokenizing train rows={tokenize_total} max_seq_length={max_seq_length}", flush=True)
    for index, row in enumerate(rows, start=1):
        tokenized.append(_tokenize_row(tokenizer, row, max_seq_length))
        if index % 5000 == 0 or index == tokenize_total:
            elapsed = time.monotonic() - tokenize_start
            progress = index / tokenize_total if tokenize_total else 1.0
            remaining = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            print(
                f"tokenize progress {index}/{tokenize_total} "
                f"({progress * 100:.2f}%) "
                f"elapsed={_format_duration(elapsed)} "
                f"remaining={_format_duration(remaining)}",
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(tokenized, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"tokenized train cache saved rows={len(tokenized)} path={cache_path}", flush=True)
    return tokenized


def train_recall35_lora(
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    max_seq_length: int,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    train_seed: int,
    max_grad_norm: float,
    lora_rank: int,
    lora_dropout: float,
    checkpoint_every: int,
    resume_adapter_dir: Path | None,
    nonfinite_patience: int,
    rebuild_token_cache: bool,
    auto_resume: bool,
    oom_patience: int,
    nonfinite_skip_limit: int,
    nonfinite_backoff_every: int,
    lr_backoff_factor: float,
    min_learning_rate: float,
    evaluation_interval_fraction: float,
    evaluation_threshold: float,
    evaluation_max_samples: int,
    evaluation_output_dir: Path | None,
) -> None:
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise _missing_dataset_error(data_dir)
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("missing training dependencies; run one_click_deploy.ps1 first") from exc

    rows = read_jsonl(train_path)
    valid_rows = read_jsonl(test_path)
    if not rows or not valid_rows:
        raise RuntimeError(f"dataset is empty or incomplete: train={len(rows)} test={len(valid_rows)}")

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora_rank = max(1, int(lora_rank))
    lora_dropout = min(max(float(lora_dropout), 0.0), 1.0)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if resume_adapter_dir is None and auto_resume:
        resume_adapter_dir = _latest_checkpoint(output_dir)
    if resume_adapter_dir is not None and (resume_adapter_dir / "adapter_config.json").exists():
        print(f"resuming adapter from {resume_adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(resume_adapter_dir), is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    for param in trainable_params:
        param.data = param.data.float()
    model.to(device)
    model.train()

    max_seq_length = max(64, max_seq_length)
    tokenized = _load_or_build_tokenized(tokenizer, rows, data_dir, train_path, base_model, max_seq_length, rebuild_token_cache)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=max(0.0, float(weight_decay)))
    total_updates = max(1, math.ceil((len(tokenized) * max(epochs, 0.001)) / max(1, batch_size * gradient_accumulation_steps)))
    total_micro_steps = total_updates * max(1, gradient_accumulation_steps)
    start_update = min(_checkpoint_update(resume_adapter_dir), total_updates)
    start_micro_step = start_update * max(1, gradient_accumulation_steps)
    evaluation_interval_fraction = min(max(float(evaluation_interval_fraction), 0.0), 1.0)
    evaluation_output_dir = evaluation_output_dir or (output_dir / "evaluations")
    if evaluation_interval_fraction > 0:
        start_progress = start_update / total_updates if total_updates else 1.0
        next_evaluation_progress = (math.floor(start_progress / evaluation_interval_fraction) + 1) * evaluation_interval_fraction
    else:
        next_evaluation_progress = 0.0
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    rng = random.Random(train_seed)
    print(
        f"manual RTX3060 LoRA train rows={len(tokenized)} valid={len(valid_rows)} "
        f"updates={total_updates} start_update={start_update} batch_size={batch_size} grad_accum={gradient_accumulation_steps} "
        f"train_seed={train_seed} lr={learning_rate} weight_decay={weight_decay} max_grad_norm={max_grad_norm} lora_rank={lora_rank} lora_dropout={lora_dropout} "
        f"max_seq_length={max_seq_length} eval_every={evaluation_interval_fraction} eval_threshold={evaluation_threshold} eval_max_samples={evaluation_max_samples}",
        flush=True,
    )
    if start_update >= total_updates:
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"checkpoint already reached target updates; adapter saved: {adapter_dir}", flush=True)
        return
    optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    consecutive_nonfinite = 0
    total_nonfinite_skips = 0
    consecutive_oom = 0
    for micro_step in range(start_micro_step, total_micro_steps):
        update = (micro_step + 1) // max(1, gradient_accumulation_steps)
        try:
            batch = [tokenized[(micro_step * batch_size + offset) % len(tokenized)] for offset in range(batch_size)]
            if micro_step % len(tokenized) == 0:
                rng.shuffle(tokenized)
            tensors = _collate(tokenizer, batch, device)
            output = model(**tensors)
            raw_loss = output.loss
            if not torch.isfinite(raw_loss.detach()):
                consecutive_nonfinite += 1
                total_nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"WARNING skipped non-finite loss before update {max(1, update)}: "
                    f"{float(raw_loss.detach().cpu())} consecutive={consecutive_nonfinite}/{nonfinite_patience} "
                    f"total={total_nonfinite_skips}/{nonfinite_skip_limit}",
                    flush=True,
                )
                if nonfinite_backoff_every > 0 and total_nonfinite_skips % nonfinite_backoff_every == 0:
                    for group in optimizer.param_groups:
                        group["lr"] = max(min_learning_rate, float(group["lr"]) * lr_backoff_factor)
                    print(f"WARNING reduced learning rate after non-finite skips: lr={optimizer.param_groups[0]['lr']}", flush=True)
                if total_nonfinite_skips >= max(1, nonfinite_skip_limit):
                    raise RuntimeError(
                        f"Training aborted after {total_nonfinite_skips} total non-finite skips near update {max(1, update)}. "
                        f"Resume from an earlier checkpoint and/or lower --learning-rate."
                    )
                if consecutive_nonfinite >= max(1, nonfinite_patience):
                    raise RuntimeError(
                        f"Training aborted after {consecutive_nonfinite} consecutive non-finite losses near update {max(1, update)}. "
                        f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                    )
                continue
            consecutive_nonfinite = 0
            consecutive_oom = 0
            loss = raw_loss / max(1, gradient_accumulation_steps)
            loss.backward()
        except torch.OutOfMemoryError as exc:
            consecutive_oom += 1
            optimizer.zero_grad(set_to_none=True)
            tensors = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for group in optimizer.param_groups:
                group["lr"] = max(min_learning_rate, float(group["lr"]) * 0.5)
            print(
                f"WARNING skipped CUDA OOM before update {max(1, update)} "
                f"micro_step={micro_step + 1}/{total_micro_steps} "
                f"consecutive={consecutive_oom}/{oom_patience} "
                f"max_seq_length={max_seq_length} lr={optimizer.param_groups[0]['lr']}: {exc}",
                flush=True,
            )
            if consecutive_oom >= max(1, oom_patience):
                raise RuntimeError(
                    f"Training aborted after {consecutive_oom} consecutive CUDA OOM errors near update {max(1, update)}. "
                    f"MAX_SEQ_LENGTH was not changed automatically; set --max-seq-length explicitly if you want a smaller value."
                )
            continue
        if (micro_step + 1) % max(1, gradient_accumulation_steps) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            if not torch.isfinite(grad_norm.detach()):
                consecutive_nonfinite += 1
                total_nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"WARNING skipped update {update}/{total_updates} because grad_norm is non-finite: "
                    f"{float(grad_norm.detach().cpu())} consecutive={consecutive_nonfinite}/{nonfinite_patience} "
                    f"total={total_nonfinite_skips}/{nonfinite_skip_limit}",
                    flush=True,
                )
                if nonfinite_backoff_every > 0 and total_nonfinite_skips % nonfinite_backoff_every == 0:
                    for group in optimizer.param_groups:
                        group["lr"] = max(min_learning_rate, float(group["lr"]) * lr_backoff_factor)
                    print(f"WARNING reduced learning rate after non-finite skips: lr={optimizer.param_groups[0]['lr']}", flush=True)
                if total_nonfinite_skips >= max(1, nonfinite_skip_limit):
                    raise RuntimeError(
                        f"Training aborted after {total_nonfinite_skips} total non-finite skips near update {update}. "
                        f"Resume from an earlier checkpoint and/or lower --learning-rate."
                    )
                if consecutive_nonfinite >= max(1, nonfinite_patience):
                    raise RuntimeError(
                        f"Training aborted after {consecutive_nonfinite} consecutive non-finite gradients near update {update}. "
                        f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                    )
                continue
            optimizer.step()
            if not _params_are_finite(trainable_params):
                raise RuntimeError(
                    f"Training aborted because trainable LoRA parameters became non-finite after update {update}. "
                    f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                )
            optimizer.zero_grad(set_to_none=True)
            elapsed = time.monotonic() - start_time
            progress = update / total_updates
            remaining = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            eta_epoch = time.localtime(time.time() + remaining)
            print(
                f"train update {update}/{total_updates} "
                f"({progress * 100:.2f}%) "
                f"loss={float(raw_loss.detach().cpu()):.4f} "
                f"grad_norm={float(grad_norm.detach().cpu()):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"elapsed={_format_duration(elapsed)} "
                f"remaining={_format_duration(remaining)} "
                f"eta={time.strftime('%Y-%m-%d %H:%M:%S', eta_epoch)}",
                flush=True,
            )
            while evaluation_interval_fraction > 0 and progress + 1e-12 >= next_evaluation_progress:
                _evaluate_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    rows=valid_rows,
                    output_dir=evaluation_output_dir,
                    update=update,
                    total_updates=total_updates,
                    progress=min(progress, 1.0),
                    trained_epochs=epochs * min(progress, 1.0),
                    threshold=evaluation_threshold,
                    max_samples=evaluation_max_samples,
                    max_seq_length=active_max_seq_length if "active_max_seq_length" in locals() else max_seq_length,
                )
                next_evaluation_progress += evaluation_interval_fraction
            if checkpoint_every > 0 and update % checkpoint_every == 0:
                checkpoint_dir = output_dir / "checkpoints" / f"update-{update:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"checkpoint saved: {checkpoint_dir}", flush=True)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adapter saved: {adapter_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for recall35 black-box stock classification")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Override token length; default follows sample mode")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=_env_float("WEIGHT_DECAY", 0.0), help="AdamW weight decay. Values like 0.01 can reduce overfitting; 0 keeps prior behavior.")
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED, help="Target-specific seed for independent LoRA parameters.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Clip LoRA gradients and skip non-finite updates.")
    parser.add_argument("--lora-rank", type=int, default=_env_int("LORA_RANK", 16), help="LoRA rank. Lower values reduce trainable capacity and can reduce overfitting.")
    parser.add_argument("--lora-dropout", type=float, default=_env_float("LORA_DROPOUT", 0.05), help="LoRA dropout in [0, 1]. Higher values can reduce overfitting.")
    parser.add_argument("--checkpoint-every", type=int, default=default_checkpoint_every(), help="Save adapter checkpoint every N optimizer updates; 0 disables checkpoints.")
    parser.add_argument("--eval-every-epoch-fraction", type=float, default=_env_float("EVAL_EVERY_EPOCH_FRACTION", 0.1), help="Run in-training evaluation at this fraction of total requested epochs; 0 disables.")
    parser.add_argument("--eval-threshold", type=float, default=_env_float("EVAL_THRESHOLD", 0.50), help="Threshold used by in-training evaluation.")
    parser.add_argument("--eval-max-samples", type=int, default=_env_int("EVAL_MAX_SAMPLES", 0), help="Max test samples for in-training evaluation; 0 means all.")
    parser.add_argument("--eval-output-dir", type=Path, default=None, help="Directory for in-training evaluation JSON files; default is output-dir/evaluations.")
    parser.add_argument("--resume-adapter-dir", type=Path, default=None, help="Resume LoRA training from an adapter checkpoint directory.")
    parser.add_argument("--nonfinite-patience", type=int, default=20, help="Abort after this many consecutive non-finite losses.")
    parser.add_argument("--nonfinite-skip-limit", type=int, default=100, help="Abort after this many total non-finite losses or gradients.")
    parser.add_argument("--nonfinite-backoff-every", type=int, default=10, help="Reduce optimizer LR after every N total non-finite skips; 0 disables.")
    parser.add_argument("--lr-backoff-factor", type=float, default=0.5, help="Multiplier used when non-finite LR backoff is triggered.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Smallest LR allowed by automatic non-finite backoff.")
    parser.add_argument("--oom-patience", type=int, default=20, help="Abort after this many consecutive CUDA OOM batches.")
    parser.add_argument("--rebuild-token-cache", action="store_true", help="Re-tokenize train.jsonl even when a tokenized cache exists.")
    parser.add_argument("--no-auto-resume", action="store_true", help="Do not automatically resume from the latest output-dir checkpoint.")
    parser.add_argument("--no-4bit", action="store_true", help="Accepted for script compatibility; recall35 Windows training uses fp16 LoRA on RTX3060.")
    parser.add_argument("--cuda-device", default="0", help="CUDA device id. Default binds the RTX3060 as cuda:0.")
    parser.add_argument("--allow-non-rtx3060", action="store_true", help="Allow CUDA devices whose name is not RTX 3060.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    train_recall35_lora(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=max(64, args.max_seq_length or default_max_seq_length()),
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
        resume_adapter_dir=args.resume_adapter_dir,
        nonfinite_patience=args.nonfinite_patience,
        rebuild_token_cache=args.rebuild_token_cache,
        auto_resume=not args.no_auto_resume,
        oom_patience=args.oom_patience,
        nonfinite_skip_limit=args.nonfinite_skip_limit,
        nonfinite_backoff_every=args.nonfinite_backoff_every,
        lr_backoff_factor=args.lr_backoff_factor,
        min_learning_rate=args.min_learning_rate,
        evaluation_interval_fraction=args.eval_every_epoch_fraction,
        evaluation_threshold=args.eval_threshold,
        evaluation_max_samples=args.eval_max_samples,
        evaluation_output_dir=args.eval_output_dir,
    )


if __name__ == "__main__":
    main()
