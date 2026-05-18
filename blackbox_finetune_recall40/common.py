from __future__ import annotations

from pathlib import Path

from blackbox_finetune.common import *  # noqa: F401,F403
from blackbox_finetune.common import DEFAULT_BASE_MODEL, DEFAULT_STAT_TYPE, DEFAULT_WINDOW

DEFAULT_DATA_DIR = Path("blackbox_finetune_recall40") / "data"
DEFAULT_VALIDATION_DIR = Path("blackbox_finetune_recall40") / "data_validation"
DEFAULT_OUTPUT_DIR = Path("blackbox_finetune_recall40") / "runs" / "qwen2.5-0.5b-blackbox-recall40-lora"
DEFAULT_TRAIN_START_DATE = "20110101"
DEFAULT_TRAIN_END_DATE = "20251231"
DEFAULT_VALIDATION_START_DATE = "20260101"
DEFAULT_VALIDATION_END_DATE = "20260430"
DEFAULT_MIN_POSITIVE_RECALL = 0.40
DEFAULT_TRAIN_SEED = 20260540
