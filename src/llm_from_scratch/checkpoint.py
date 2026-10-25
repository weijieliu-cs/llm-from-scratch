"""保存纯配置和张量，使用同目录临时文件原子替换 Checkpoint。"""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .tokenizer import ByteTokenizer

FORMAT_VERSION = 1


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("不支持的 Checkpoint 格式版本")
    if state.get("tokenizer") != ByteTokenizer.name:
        raise ValueError("Checkpoint Tokenizer 与当前字节编码不一致")
    required = {
        "config",
        "model",
        "optimizer",
        "step",
        "best_val_loss",
        "data_sha256",
        "torch_rng_state",
        "batch_rng_state",
        "cuda_rng_states",
    }
    if required - state.keys():
        raise ValueError(f"Checkpoint 缺少字段：{', '.join(sorted(required - state.keys()))}")
    return state
