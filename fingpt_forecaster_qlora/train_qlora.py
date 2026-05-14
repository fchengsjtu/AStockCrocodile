from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fingpt_forecaster_qlora.common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_DIR,
    DEFAULT_FINGPT_FORECASTER_ADAPTER,
    DEFAULT_OUTPUT_DIR,
)


def missing_deps_error(exc: Exception) -> RuntimeError:
    error = RuntimeError(
        "Missing QLoRA dependencies. In WSL2/Linux install CUDA PyTorch first, then run: "
        "pip install -r fingpt_forecaster_qlora/requirements.txt"
    )
    error.__cause__ = exc
    return error


def model_load_error(model_ref: str, exc: Exception) -> RuntimeError:
    message = (
        f"Cannot load HuggingFace model or adapter: {model_ref}\n\n"
        f"Original error: {exc}\n\n"
        "Fix options:\n"
        "1. If the network cannot reach huggingface.co, set a reachable mirror before running:\n"
        "   export HF_ENDPOINT=https://hf-mirror.com\n"
        "2. Download the model in advance and set BASE_MODEL to the local HuggingFace-format directory in "
        "fingpt_forecaster_qlora/config.env.\n"
        "3. To only generate training data for now, run:\n"
        "   bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh dataset-only\n"
        "4. If only the FinGPT adapter is unreachable, set NO_FORECASTER_ADAPTER=1 in config.env."
    )
    error = RuntimeError(message)
    error.__cause__ = exc
    return error


def train_qlora(
    base_model: str,
    forecaster_adapter: str | None,
    data_dir: Path,
    output_dir: Path,
    max_seq_length: int,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    use_4bit: bool,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except Exception as exc:
        raise missing_deps_error(exc)

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"missing dataset files: {train_path} and {valid_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    except (OSError, RuntimeError) as exc:
        raise model_load_error(base_model, exc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )
    except (OSError, RuntimeError) as exc:
        raise model_load_error(base_model, exc)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if forecaster_adapter:
        print(f"loading FinGPT-Forecaster adapter: {forecaster_adapter}", flush=True)
        try:
            model = PeftModel.from_pretrained(model, forecaster_adapter, is_trainable=True)
        except (OSError, RuntimeError) as exc:
            raise model_load_error(forecaster_adapter, exc)
    else:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(valid_path)})

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to=[],
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        args=training_args,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adapter saved to {adapter_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune FinGPT-Forecaster with 4-bit QLoRA")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--forecaster-adapter", default=DEFAULT_FINGPT_FORECASTER_ADAPTER)
    parser.add_argument("--no-forecaster-adapter", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-4bit", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_qlora(
        base_model=args.base_model,
        forecaster_adapter=None if args.no_forecaster_adapter else args.forecaster_adapter,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        lora_r=max(1, args.lora_r),
        lora_alpha=max(1, args.lora_alpha),
        lora_dropout=args.lora_dropout,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
