"""小型单设备训练循环：可重复采样、验证、指标与完整状态恢复。"""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .checkpoint import FORMAT_VERSION, load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .dataset import get_batch, load_data, read_corpus
from .model import GPT
from .tokenizer import ByteTokenizer


def resolve_device(requested: str) -> torch.device:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定了 CUDA，但当前环境没有可用的 CUDA 设备")
    return torch.device(requested)


def create_optimizer(model: GPT, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    # named_parameters 会去重共享参数；仅矩阵权重衰减，偏置和归一化参数不衰减。
    groups = [
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    num_batches: int,
    *,
    seed: int = 0,
) -> tuple[float, float]:
    """固定评估窗口；不消耗训练的采样或 Dropout 随机数状态。"""
    if num_batches <= 0:
        raise ValueError("num_batches 必须大于 0")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    losses = []
    try:
        for tokens in (train_data, val_data):
            total = 0.0
            for _ in range(num_batches):
                inputs, targets = get_batch(
                    tokens, batch_size, model.config.context_length, device, generator=generator
                )
                _, loss = model(inputs, targets)
                total += loss.item()
            losses.append(total / num_batches)
    finally:
        model.train(was_training)
    return losses[0], losses[1]


def _check_resume(config: ExperimentConfig, state: dict[str, Any], digest: str) -> None:
    previous = ExperimentConfig.from_dict(state["config"])
    if previous.model != config.model:
        raise ValueError("恢复训练时不能更改模型结构")
    if state["data_sha256"] != digest:
        raise ValueError("训练语料校验值变化，不能连续恢复训练")
    mutable = {
        "max_steps",
        "output_dir",
        "data_path",
        "device",
        "num_threads",
        "eval_interval",
        "eval_batches",
    }
    for name, value in config.to_dict()["training"].items():
        if name not in mutable and value != getattr(previous.training, name):
            raise ValueError(f"恢复训练时不能更改 {name}")
    if config.training.max_steps <= state["step"]:
        raise ValueError(f"max_steps 必须大于 Checkpoint 已完成步数 {state['step']}")


def train(config: ExperimentConfig, *, resume: str | Path | None = None) -> Path:
    """训练到 max_steps，返回最后一次完整更新对应的 Checkpoint 路径。"""
    device = resolve_device(config.training.device)
    config = replace(
        config,
        training=replace(
            config.training,
            device=str(device),
            output_dir=str(Path(config.training.output_dir).resolve()),
            data_path=(
                str(Path(config.training.data_path).resolve())
                if config.training.data_path
                else None
            ),
        ),
    )
    settings = config.training
    torch.set_num_threads(settings.num_threads)
    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    digest = hashlib.sha256(read_corpus(settings.data_path)).hexdigest()
    train_data, val_data = load_data(settings.data_path, train_fraction=settings.train_fraction)
    for name, tokens in (("训练集", train_data), ("验证集", val_data)):
        if len(tokens) <= config.model.context_length:
            raise ValueError(
                f"{name}有 {len(tokens)} 字节，至少需要 {config.model.context_length + 1}"
            )

    output_dir = Path(settings.output_dir)
    has_artifacts = any(
        (output_dir / name).exists()
        for name in ("config.json", "metrics.jsonl", "last.pt", "best.pt")
    )
    if resume is None and has_artifacts:
        raise FileExistsError(f"{output_dir} 已有实验结果，请使用 --resume 或新的 --output-dir")
    if resume is not None and has_artifacts and (Path(resume).resolve() != output_dir / "last.pt"):
        raise ValueError(
            "恢复已有实验目录必须使用该目录的 last.pt；其他快照请指定新的 --output-dir"
        )

    state = load_checkpoint(resume) if resume else None
    if state is not None:
        _check_resume(config, state, digest)
    model = GPT(config.model).to(device)
    optimizer = create_optimizer(model, settings.learning_rate, settings.weight_decay)
    start_step, best_val_loss = 0, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_val_loss = state["step"], state["best_val_loss"]
        torch.set_rng_state(state["torch_rng_state"])
        generator.set_state(state["batch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, "
        f"train_bytes={len(train_data)}, val_bytes={len(val_data)}"
    )
    model.train()
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        for step in range(start_step + 1, settings.max_steps + 1):
            inputs, targets = get_batch(
                train_data,
                settings.batch_size,
                config.model.context_length,
                device,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"step={step} 出现非有限训练 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.grad_clip or float("inf"), error_if_nonfinite=True
            )
            optimizer.step()

            if step == 1 or step % settings.eval_interval == 0 or step == settings.max_steps:
                train_loss, val_loss = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    settings.batch_size,
                    settings.eval_batches,
                    seed=settings.seed,
                )
                if not all(math.isfinite(value) for value in (train_loss, val_loss)):
                    raise RuntimeError(f"step={step} 出现非有限评估 loss")
                improved = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                record = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "tokens_seen": step * settings.batch_size * config.model.context_length,
                }
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                print(f"step={step:4d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
                state = {
                    "format_version": FORMAT_VERSION,
                    "tokenizer": ByteTokenizer.name,
                    "config": config.to_dict(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "data_sha256": digest,
                    "torch_rng_state": torch.get_rng_state(),
                    "batch_rng_state": generator.get_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else [],
                    "torch_version": str(torch.__version__),
                }
                save_checkpoint(output_dir / "last.pt", state)
                if improved:
                    save_checkpoint(output_dir / "best.pt", state)
    return output_dir / "last.pt"
