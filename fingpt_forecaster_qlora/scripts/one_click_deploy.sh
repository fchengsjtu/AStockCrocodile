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

configure_wsl_mysql_host() {
  local host_value
  host_value="${MYSQL_HOST:-}"
  if [[ "$host_value" != "127.0.0.1" && "$host_value" != "localhost" && "$host_value" != "::1" && -n "$host_value" ]]; then
    return
  fi
  if ! grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    return
  fi
  local windows_host
  windows_host="$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"
  if [[ -n "$windows_host" ]]; then
    export WSL_MYSQL_HOST="$windows_host"
    echo "WSL detected: MYSQL_HOST=${host_value:-127.0.0.1} will be resolved to Windows host ${WSL_MYSQL_HOST}."
    echo "If MySQL still refuses the connection, set WSL_MYSQL_HOST manually or allow MySQL to listen on the Windows host IP."
  fi
}

is_local_model_dir() {
  [[ -d "$1" && -f "$1/$2" ]]
}

check_hf_model_access() {
  local model_id="$1"
  local label="$2"
  local required_file="$3"
  if [[ -z "$model_id" || "$model_id" == "none" ]]; then
    return
  fi
  if is_local_model_dir "$model_id" "$required_file"; then
    echo "$label uses local path: $model_id"
    return
  fi
  python - "$model_id" "$label" "$required_file" <<'PY'
import os
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

model_id, label, required_file = sys.argv[1], sys.argv[2], sys.argv[3]
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
url = f"{endpoint}/{model_id}/resolve/main/{required_file}"
try:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=8):
        pass
except Exception as exc:
    raise SystemExit(
        f"Cannot reach {label} on HuggingFace endpoint: {url}\n"
        f"Reason: {exc}\n\n"
        "Fix options:\n"
        "  1. Put the model in a local HuggingFace-format directory and set BASE_MODEL=/path/to/model in fingpt_forecaster_qlora/config.env.\n"
        "  2. Use a reachable mirror, for example: export HF_ENDPOINT=https://hf-mirror.com\n"
        "  3. Run only dataset generation now: bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh dataset-only\n"
    )
PY
}

if [[ ! -f "fingpt_forecaster_qlora/config.env" ]]; then
  cp fingpt_forecaster_qlora/config.example.env fingpt_forecaster_qlora/config.env
fi

set -a
source fingpt_forecaster_qlora/config.env
set +a

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

configure_wsl_mysql_host

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

EFFECTIVE_BASE_MODEL="${BASE_MODEL:-NousResearch/Llama-2-7b-chat-hf}"
EFFECTIVE_FORECASTER_ADAPTER="${FINGPT_FORECASTER_ADAPTER:-FinGPT/fingpt-forecaster_dow30_llama2-7b_lora}"
EFFECTIVE_NO_FORECASTER_ADAPTER="${NO_FORECASTER_ADAPTER:-0}"
EFFECTIVE_MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
EFFECTIVE_EPOCHS="${EPOCHS:-1}"
EFFECTIVE_DATA_DIR="${DATA_DIR:-fingpt_forecaster_qlora/data}"
EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR:-fingpt_forecaster_qlora/runs/astock-fingpt-forecaster-qlora}"

if [[ "$MODE" == "smoke" && "${SMOKE_USE_SMALL_MODEL:-1}" == "1" ]]; then
  EFFECTIVE_BASE_MODEL="${SMOKE_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  EFFECTIVE_NO_FORECASTER_ADAPTER="${SMOKE_NO_FORECASTER_ADAPTER:-1}"
  EFFECTIVE_MAX_SEQ_LENGTH="${SMOKE_MAX_SEQ_LENGTH:-2048}"
  EFFECTIVE_EPOCHS="${SMOKE_EPOCHS:-0.05}"
  EFFECTIVE_DATA_DIR="${SMOKE_DATA_DIR:-fingpt_forecaster_qlora/data/smoke}"
  EFFECTIVE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-fingpt_forecaster_qlora/runs/smoke-qwen-0.5b}"
  echo "Smoke mode uses small open model: ${EFFECTIVE_BASE_MODEL}"
  echo "Smoke mode skips FinGPT adapter unless SMOKE_NO_FORECASTER_ADAPTER=0 is set."
fi

DATA_ARGS=(
  --output-dir "${EFFECTIVE_DATA_DIR}"
  --start-date "${TRAIN_START_DATE:-20100101}"
  --end-date "${TRAIN_END_DATE:-20251231}"
  --negative-ratio "${NEGATIVE_RATIO:-1.0}"
  --valid-ratio "${VALID_RATIO:-0.2}"
  --daily-window "${DAILY_WINDOW:-55}"
  --weekly-window "${WEEKLY_WINDOW:-55}"
  --min-success-rate "${MIN_SUCCESS_RATE:-0.40}"
)

TRAIN_ARGS=(
  --base-model "${EFFECTIVE_BASE_MODEL}"
  --forecaster-adapter "${EFFECTIVE_FORECASTER_ADAPTER}"
  --data-dir "${EFFECTIVE_DATA_DIR}"
  --output-dir "${EFFECTIVE_OUTPUT_DIR}"
  --max-seq-length "${EFFECTIVE_MAX_SEQ_LENGTH}"
  --epochs "${EFFECTIVE_EPOCHS}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
)

if [[ "$MODE" == "smoke" ]]; then
  DATA_ARGS+=(--positive-limit "${SMOKE_POSITIVE_LIMIT:-200}")
fi

python -m fingpt_forecaster_qlora.build_dataset "${DATA_ARGS[@]}"

if [[ "$MODE" == "dataset-only" ]]; then
  echo "Dataset-only mode complete. Training was skipped."
  exit 0
fi

check_hf_model_access "${EFFECTIVE_BASE_MODEL}" "base model" "config.json"
if [[ "${EFFECTIVE_NO_FORECASTER_ADAPTER}" == "1" ]]; then
  TRAIN_ARGS+=(--no-forecaster-adapter)
else
  check_hf_model_access "${EFFECTIVE_FORECASTER_ADAPTER}" "FinGPT-Forecaster adapter" "adapter_config.json"
fi

python -m fingpt_forecaster_qlora.train_qlora "${TRAIN_ARGS[@]}"
python -m fingpt_forecaster_qlora.evaluate \
  --base-model "${EFFECTIVE_BASE_MODEL}" \
  --adapter-dir "${EFFECTIVE_OUTPUT_DIR}/adapter" \
  --data-dir "${EFFECTIVE_DATA_DIR}" \
  --threshold "${MIN_SUCCESS_RATE:-0.40}" \
  --max-samples "${EVAL_MAX_SAMPLES:-100}"
