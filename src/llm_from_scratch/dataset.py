"""先划分字节序列，再随机采样 next-token prediction 批次。"""

from importlib.resources import files
from pathlib import Path

import torch
from torch import Tensor

from .tokenizer import ByteTokenizer


def read_corpus(path: str | Path | None = None) -> bytes:
    """未指定路径时读取随 Python 包分发的示例语料。"""
    if path is None:
        return files("llm_from_scratch").joinpath("resources/sample.txt").read_bytes()
    return Path(path).read_bytes()


def load_data(
    path: str | Path | None = None, *, train_fraction: float = 0.9
) -> tuple[Tensor, Tensor]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction 必须在 (0, 1) 内")
    text = read_corpus(path).decode("utf-8")
    tokens = torch.tensor(ByteTokenizer().encode(text), dtype=torch.long)
    split = int(train_fraction * len(tokens))
    return tokens[:split], tokens[split:]


def get_batch(
    tokens: Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """输入与目标错开一位；generator 控制 CPU 上的窗口采样。"""
    if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.device.type != "cpu":
        raise ValueError("tokens 必须是一维 CPU torch.long 张量")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size 和 context_length 必须大于 0")
    if len(tokens) <= context_length:
        raise ValueError(
            f"每份数据至少需要 {context_length + 1} 个字节，实际为 {len(tokens)}；"
            "请增加文本或减小 context_length"
        )
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return inputs.to(device), targets.to(device)
