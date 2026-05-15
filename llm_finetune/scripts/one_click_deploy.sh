#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-qwen-finetune}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<82"

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Installing CUDA PyTorch wheels. Adjust the index if your driver needs a different CUDA build."
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
fi
python -m pip install -r llm_finetune/requirements.txt

if [[ -f llm_finetune/config.env ]]; then
  set -a
  source llm_finetune/config.env
  set +a
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-llm_finetune/data}"
OUTPUT_DIR="${OUTPUT_DIR:-llm_finetune/runs/qwen2.5-0.5b-stock-lora}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.20}"
if [[ "$MODE" == "smoke" ]]; then
  DEFAULT_MAX_SEQ_LENGTH="512"
  DEFAULT_EPOCHS="5"
  DEFAULT_GRADIENT_ACCUMULATION_STEPS="1"
else
  DEFAULT_MAX_SEQ_LENGTH="2048"
  DEFAULT_EPOCHS="1"
  DEFAULT_GRADIENT_ACCUMULATION_STEPS="8"
fi
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

DATA_ARGS=(--output-dir "$DATA_DIR" --negative-ratio "${NEGATIVE_RATIO:-1.0}" --batch-size "${DATA_BATCH_SIZE:-30}")
if [[ "$MODE" == "smoke" ]]; then
  DATA_ARGS+=(--positive-limit "${SMOKE_POSITIVE_LIMIT:-200}")
fi

python -m llm_finetune.build_dataset "${DATA_ARGS[@]}"
python -m llm_finetune.train \
  --base-model "$BASE_MODEL" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE"
python -m llm_finetune.evaluate \
  --base-model "$BASE_MODEL" \
  --adapter-dir "$OUTPUT_DIR/adapter" \
  --data-dir "$DATA_DIR" \
  --min-success-rate "$MIN_SUCCESS_RATE" \
  --max-samples "${EVAL_MAX_SAMPLES:-200}"
python -m unittest tests.test_llm_finetune -v
