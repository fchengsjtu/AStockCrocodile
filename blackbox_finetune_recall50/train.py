from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackbox_finetune_recall50.common import DEFAULT_BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_TRAIN_SEED
from blackbox_finetune_recall50.gpu import prepare_rtx3060
from llm_finetune.common import read_jsonl


def _missing_dataset_error(data_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(f"missing dataset files under {data_dir}; run blackbox_finetune_recall50.build_dataset first")


def _tokenize_row(tokenizer, row: dict, max_seq_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    answer = row["messages"][-1]["content"] + tokenizer.eos_token
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

def train_recall50_lora(
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    max_seq_length: int,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    train_seed: int,
    max_grad_norm: float,
    checkpoint_every: int,
    resume_adapter_dir: Path | None,
    nonfinite_patience: int,
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
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
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
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    total_updates = max(1, math.ceil((len(tokenized) * max(epochs, 0.001)) / max(1, batch_size * gradient_accumulation_steps)))
    total_micro_steps = total_updates * max(1, gradient_accumulation_steps)
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    rng = random.Random(train_seed)
    print(
        f"manual RTX3060 LoRA train rows={len(tokenized)} valid={len(valid_rows)} "
        f"updates={total_updates} batch_size={batch_size} grad_accum={gradient_accumulation_steps} "
        f"train_seed={train_seed} lr={learning_rate} max_grad_norm={max_grad_norm}",
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)
    start_time = time.monotonic()
    consecutive_nonfinite = 0
    for micro_step in range(total_micro_steps):
        batch = [tokenized[(micro_step * batch_size + offset) % len(tokenized)] for offset in range(batch_size)]
        if micro_step % len(tokenized) == 0:
            rng.shuffle(tokenized)
        tensors = _collate(tokenizer, batch, device)
        output = model(**tensors)
        raw_loss = output.loss
        update = (micro_step + 1) // max(1, gradient_accumulation_steps)
        if not torch.isfinite(raw_loss.detach()):
            consecutive_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"WARNING skipped non-finite loss before update {max(1, update)}: "
                f"{float(raw_loss.detach().cpu())} consecutive={consecutive_nonfinite}/{nonfinite_patience}",
                flush=True,
            )
            if consecutive_nonfinite >= max(1, nonfinite_patience):
                raise RuntimeError(
                    f"Training aborted after {consecutive_nonfinite} consecutive non-finite losses near update {max(1, update)}. "
                    f"Resume from the latest checkpoint with --resume-adapter-dir and a lower --learning-rate."
                )
            continue
        consecutive_nonfinite = 0
        loss = raw_loss / max(1, gradient_accumulation_steps)
        loss.backward()
        if (micro_step + 1) % max(1, gradient_accumulation_steps) == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            if not torch.isfinite(grad_norm.detach()):
                consecutive_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"WARNING skipped update {update}/{total_updates} because grad_norm is non-finite: "
                    f"{float(grad_norm.detach().cpu())} consecutive={consecutive_nonfinite}/{nonfinite_patience}",
                    flush=True,
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
                f"elapsed={_format_duration(elapsed)} "
                f"remaining={_format_duration(remaining)} "
                f"eta={time.strftime('%Y-%m-%d %H:%M:%S', eta_epoch)}",
                flush=True,
            )
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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for recall50 black-box stock classification")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED, help="Target-specific seed for independent LoRA parameters.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Clip LoRA gradients and skip non-finite updates.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Save adapter checkpoint every N optimizer updates; 0 disables checkpoints.")
    parser.add_argument("--resume-adapter-dir", type=Path, default=None, help="Resume LoRA training from an adapter checkpoint directory.")
    parser.add_argument("--nonfinite-patience", type=int, default=20, help="Abort after this many consecutive non-finite losses.")
    parser.add_argument("--no-4bit", action="store_true", help="Accepted for script compatibility; recall50 Windows training uses fp16 LoRA on RTX3060.")
    parser.add_argument("--cuda-device", default="0", help="CUDA device id. Default binds the RTX3060 as cuda:0.")
    parser.add_argument("--allow-non-rtx3060", action="store_true", help="Allow CUDA devices whose name is not RTX 3060.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)
    train_recall50_lora(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        train_seed=args.train_seed,
        max_grad_norm=args.max_grad_norm,
        checkpoint_every=args.checkpoint_every,
        resume_adapter_dir=args.resume_adapter_dir,
        nonfinite_patience=args.nonfinite_patience,
    )


if __name__ == "__main__":
    main()
