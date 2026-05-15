from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_finetune.common import BASE_MODEL, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, read_jsonl


def missing_dataset_error(data_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"missing dataset files under {data_dir}; run:\n"
        "  python -m llm_finetune.build_dataset --output-dir llm_finetune/data --positive-limit 2000"
    )


def normalize_training_args(kwargs: dict, supported: set[str]) -> dict:
    normalized = dict(kwargs)
    if "evaluation_strategy" in normalized and "evaluation_strategy" not in supported and "eval_strategy" in supported:
        normalized["eval_strategy"] = normalized.pop("evaluation_strategy")
    return {key: value for key, value in normalized.items() if key in supported}


def train(
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    max_seq_length: int,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    use_4bit: bool,
) -> None:
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise missing_dataset_error(data_dir)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError("missing training dependencies; run scripts/one_click_deploy first") from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    rows = read_jsonl(train_path)
    valid_rows = read_jsonl(test_path)
    if not rows or not valid_rows:
        raise RuntimeError(f"dataset is empty or incomplete: train={len(rows)} test={len(valid_rows)}")

    def tokenize(row: dict) -> dict:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        answer = row["messages"][-1]["content"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= max_seq_length:
            answer_ids = answer_ids[: max_seq_length - 1] + [tokenizer.eos_token_id]
        prompt_budget = max_seq_length - len(answer_ids)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = (prompt_ids + answer_ids)[:max_seq_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    train_ds = Dataset.from_list(rows).map(tokenize, remove_columns=list(rows[0].keys()))
    valid_ds = Dataset.from_list(valid_rows).map(tokenize, remove_columns=list(valid_rows[0].keys()))

    def collate(batch: list[dict]) -> dict:
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
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    args = normalize_training_args(
        {
            "output_dir": str(output_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "fp16": torch.cuda.is_available(),
            "report_to": [],
            "remove_unused_columns": False,
        },
        supported,
    )
    trainer = Trainer(model=model, args=TrainingArguments(**args), train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collate)
    trainer.train()
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adapter saved: {adapter_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B-Instruct with LoRA/QLoRA")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true", help="Disable bitsandbytes 4-bit loading")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
