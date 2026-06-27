from __future__ import annotations

import math
import os
from pathlib import Path

from blackbox_finetune_drop6.common import label_answer, resolve_pretrained_source
from llm_finetune.evaluate import missing_adapter_error


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_model(base_model: str, adapter_dir: Path):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("missing inference dependencies; run one_click_deploy.ps1 first") from exc
    if not (adapter_dir / "adapter_config.json").exists():
        raise missing_adapter_error(adapter_dir)
    local_files_only = _env_bool("HF_LOCAL_FILES_ONLY") or _env_bool("TRANSFORMERS_OFFLINE") or _env_bool("HF_HUB_OFFLINE")
    trust_remote_code = _env_bool("TRUST_REMOTE_CODE", False)
    try:
        pretrained_source = resolve_pretrained_source(base_model)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model
    if tokenizer_source == base_model:
        tokenizer_source = pretrained_source
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code, use_fast=True, local_files_only=local_files_only)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load tokenizer for {str(tokenizer_source)!r}. If WSL/Linux cannot reach HuggingFace, "
            "set BASE_MODEL to a local HuggingFace-format model directory or pre-download the model cache. "
            "Set HF_LOCAL_FILES_ONLY=0 only when network access is available."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_source,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load model for {base_model!r}. If WSL/Linux cannot reach HuggingFace, "
            "set BASE_MODEL to a local HuggingFace-format model directory or pre-download the model cache. "
            "Set HF_LOCAL_FILES_ONLY=0 only when network access is available."
        ) from exc
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to("cuda")
    model.eval()
    return model, tokenizer


def answer_loss(model, tokenizer, prompt: str, answer: str, max_seq_length: int) -> float:
    import torch

    answer = answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.shape[1] >= max_seq_length:
        answer_ids = answer_ids[:, : max_seq_length - 1]
        answer_ids[0, -1] = tokenizer.eos_token_id
    prompt_budget = max(1, max_seq_length - answer_ids.shape[1])
    if prompt_ids.shape[1] > prompt_budget:
        prompt_ids = prompt_ids[:, -prompt_budget:]
    prompt_ids = prompt_ids.to(model.device)
    answer_ids = answer_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.shape[1] :] = answer_ids
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)
    return float(output.loss.detach().cpu())


def score_prediction(model, tokenizer, prompt: str, max_seq_length: int, threshold: float) -> dict:
    positive_loss = answer_loss(model, tokenizer, prompt, label_answer(1), max_seq_length)
    negative_loss = answer_loss(model, tokenizer, prompt, label_answer(0), max_seq_length)
    positive_weight = math.exp(-positive_loss)
    negative_weight = math.exp(-negative_loss)
    probability = positive_weight / (positive_weight + negative_weight) if positive_weight + negative_weight else 0.0
    return {
        "label": "positive" if probability >= threshold else "negative",
        "positive_probability": probability,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
    }
