from __future__ import annotations

from pathlib import Path

from blackbox_finetune.common import SampleEvent
from blackbox_finetune_recall60 import common as recall_common

PROJECT_DIR = Path("blackbox_finetune_threeclass")
CLASS_NEGATIVE = 0
CLASS_NEUTRAL = 1
CLASS_POSITIVE = 2
CLASS_NAMES = {
    CLASS_NEGATIVE: "negative",
    CLASS_NEUTRAL: "neutral",
    CLASS_POSITIVE: "positive",
}
CLASS_IDS = {name: label for label, name in CLASS_NAMES.items()}
DEFAULT_CLASS_RATIO = (1, 4, 11)
SYSTEM_PROMPT = (
    "Classify the supplied A-share K-line history into positive, negative, or neutral. "
    "Positive means the future three trading days first reach +20%; negative means they "
    "first reach -6%; neutral means neither event occurs. Return strict JSON only."
)

DEFAULT_BASE_MODEL = recall_common.DEFAULT_BASE_MODEL
DEFAULT_SAMPLE_MODE = "xlong"
DEFAULT_MAX_SEQ_LENGTH = recall_common.default_max_seq_length(DEFAULT_SAMPLE_MODE)
DEFAULT_TRAIN_START_DATE = "20230101"
DEFAULT_TRAIN_END_DATE = "20241231"
DEFAULT_VALIDATION_START_DATE = "20260101"
DEFAULT_VALIDATION_END_DATE = "20260530"
DEFAULT_TRAIN_SEED = recall_common.DEFAULT_TRAIN_SEED
DEFAULT_DATA_DIR = PROJECT_DIR / f"data_{DEFAULT_SAMPLE_MODE}_p1_n4_u11"
DEFAULT_VALIDATION_DIR = PROJECT_DIR / f"data_evaluation_{DEFAULT_SAMPLE_MODE}_p1_n4_u11"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / f"qwen2.5-0.5b-threeclass-{DEFAULT_SAMPLE_MODE}-p1_n4_u11-lora"

parse_date = recall_common.parse_date
mysql_connect = recall_common.mysql_connect
read_jsonl = recall_common.read_jsonl
write_jsonl = recall_common.write_jsonl
sample_mode_config = recall_common.sample_mode_config
normalize_sample_mode = recall_common.normalize_sample_mode
default_max_seq_length = recall_common.default_max_seq_length
iter_batches = recall_common.iter_batches
load_kline_map = recall_common.load_kline_map
pick_window = recall_common.pick_window
pick_weekly_window = recall_common.pick_weekly_window
pick_monthly_window = recall_common.pick_monthly_window
load_abnormal_symbols = recall_common.load_abnormal_symbols
resolve_pretrained_source = recall_common.resolve_pretrained_source


def label_answer(label: int) -> str:
    try:
        class_name = CLASS_NAMES[int(label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported three-class label: {label}") from exc
    return f'{{"c":"{class_name}"}}'


def build_messages(*args, **kwargs) -> list[dict]:
    messages = recall_common.build_messages(*args, **kwargs)
    if messages:
        messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
    return messages


def compact_messages_from_sample(row: dict) -> list[dict]:
    messages = [dict(message) for message in recall_common.compact_messages_from_sample(row)]
    if messages:
        messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
    if len(messages) > 2:
        messages[-1] = {
            "role": "assistant",
            "content": label_answer(int(row["metadata"]["label"])),
        }
    return messages


def materialize_events(*args, **kwargs) -> list[dict]:
    rows = recall_common.materialize_events(*args, **kwargs)
    for row in rows:
        metadata = row["metadata"]
        label = int(metadata["label"])
        metadata["class_name"] = CLASS_NAMES[label]
        row["messages"][0] = {"role": "system", "content": SYSTEM_PROMPT}
        row["messages"][-1] = {"role": "assistant", "content": label_answer(label)}
    return rows
