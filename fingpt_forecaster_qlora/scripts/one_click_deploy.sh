#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-fingpt-linux}"

if [[ ! -f "fingpt_forecaster_qlora/config.env" ]]; then
  cp fingpt_forecaster_qlora/config.example.env fingpt_forecaster_qlora/config.env
fi

set -a
source fingpt_forecaster_qlora/config.env
set +a

if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Existing $VENV_DIR is not a Linux virtualenv. Set VENV_DIR to another path or remove it."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if ! python - <<'PY'
import torch
print(torch.__version__)
print("cuda", torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Installing CUDA PyTorch for Linux/WSL2. Edit this script if your CUDA driver needs a different index."
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
fi

python -m pip install -r fingpt_forecaster_qlora/requirements.txt

DATA_ARGS=(
  --output-dir fingpt_forecaster_qlora/data
  --start-date "${TRAIN_START_DATE:-20100101}"
  --end-date "${TRAIN_END_DATE:-20251231}"
  --negative-ratio "${NEGATIVE_RATIO:-1.0}"
  --valid-ratio "${VALID_RATIO:-0.2}"
  --daily-window "${DAILY_WINDOW:-55}"
  --weekly-window "${WEEKLY_WINDOW:-55}"
  --min-success-rate "${MIN_SUCCESS_RATE:-0.40}"
)

TRAIN_ARGS=(
  --base-model "${BASE_MODEL:-NousResearch/Llama-2-7b-chat-hf}"
  --forecaster-adapter "${FINGPT_FORECASTER_ADAPTER:-FinGPT/fingpt-forecaster_dow30_llama2-7b_lora}"
  --data-dir "${DATA_DIR:-fingpt_forecaster_qlora/data}"
  --output-dir "${OUTPUT_DIR:-fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora}"
  --max-seq-length "${MAX_SEQ_LENGTH:-4096}"
  --epochs "${EPOCHS:-1}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
)

if [[ "$MODE" == "smoke" ]]; then
  DATA_ARGS+=(--positive-limit "${SMOKE_POSITIVE_LIMIT:-200}")
  TRAIN_ARGS+=(--epochs "${SMOKE_EPOCHS:-0.05}")
fi

python -m fingpt_forecaster_qlora.build_dataset "${DATA_ARGS[@]}"
python -m fingpt_forecaster_qlora.train_qlora "${TRAIN_ARGS[@]}"
python -m fingpt_forecaster_qlora.evaluate \
  --base-model "${BASE_MODEL:-NousResearch/Llama-2-7b-chat-hf}" \
  --adapter-dir "${OUTPUT_DIR:-fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora}/adapter" \
  --data-dir "${DATA_DIR:-fingpt_forecaster_qlora/data}" \
  --threshold "${MIN_SUCCESS_RATE:-0.40}" \
  --max-samples "${EVAL_MAX_SAMPLES:-100}"
