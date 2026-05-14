#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/astock-fingpt}"

print_venv_help() {
  local py_version
  py_version="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  cat <<EOF
Python venv/ensurepip is not available for $PYTHON_BIN.

On Ubuntu/WSL install the matching venv package, then rerun this script:

  sudo apt update
  sudo apt install -y python${py_version}-venv
  rm -rf ${VENV_DIR}
  bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh ${MODE}

If apt cannot find python${py_version}-venv, install the generic package instead:

  sudo apt install -y python3-venv
EOF
}

bootstrap_pip_with_get_pip() {
  local get_pip
  get_pip="$(mktemp)"
  echo "Bootstrapping pip with get-pip.py because ensurepip failed."
  if ! "$VENV_DIR/bin/python" - <<PY
from pathlib import Path
from urllib.request import urlopen

url = "https://bootstrap.pypa.io/get-pip.py"
target = Path("${get_pip}")
with urlopen(url, timeout=120) as response:
    target.write_bytes(response.read())
PY
  then
    rm -f "$get_pip"
    return 1
  fi
  if ! "$VENV_DIR/bin/python" "$get_pip"; then
    rm -f "$get_pip"
    return 1
  fi
  rm -f "$get_pip"
}

create_linux_venv() {
  mkdir -p "$(dirname "$VENV_DIR")"
  if "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    return
  fi

  echo "Standard venv creation failed; retrying with --without-pip."
  rm -rf "$VENV_DIR"
  if ! "$PYTHON_BIN" -m venv --without-pip "$VENV_DIR"; then
    return 1
  fi
  if ! bootstrap_pip_with_get_pip; then
    return 1
  fi
}

if [[ ! -f "fingpt_forecaster_qlora/config.env" ]]; then
  cp fingpt_forecaster_qlora/config.example.env fingpt_forecaster_qlora/config.env
fi

set -a
source fingpt_forecaster_qlora/config.env
set +a

if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Existing $VENV_DIR is not a complete Linux virtualenv; removing the broken directory."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  if ! create_linux_venv; then
    rm -rf "$VENV_DIR"
    print_venv_help
    exit 1
  fi
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
