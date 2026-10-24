"""将文本转成 UTF-8 字节，并采样 next-token prediction 所需的训练批次。"""

from pathlib import Path

import torch


def load_data(path):
    text = Path(path).read_text(encoding="utf-8")
    # 每个 UTF-8 字节对应一个 token，因此词表大小固定为 256。
    tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    split = int(0.9 * len(tokens))
    # 先划分再采样，避免训练窗口跨入验证集。
    return tokens[:split], tokens[split:]


def get_batch(tokens, batch_size, context_length, device):
    if len(tokens) <= context_length:
        raise ValueError(
            f"每份数据至少需要 {context_length + 1} 个字节，请增加文本或减小上下文长度"
        )
    starts = torch.randint(len(tokens) - context_length, (batch_size,))
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    # inputs=[t0, t1, ...]，targets=[t1, t2, ...]：在每个位置预测下一个 token。
    targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return inputs.to(device), targets.to(device)
