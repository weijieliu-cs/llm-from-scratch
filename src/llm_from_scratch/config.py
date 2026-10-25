"""可序列化的实验配置；在分配模型和读取数据之前检查参数。"""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须为正整数，实际为 {value!r}")


def _finite_number(name: str, value: float, minimum: float = 0.0) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的有限数值，实际为 {value!r}")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "embedding_dim", "num_heads", "num_layers"):
            _positive_integer(name, getattr(self, name))
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim 必须能被 num_heads 整除")
        _finite_number("dropout", self.dropout)
        if self.dropout >= 1:
            raise ValueError("dropout 必须在 [0, 1) 内")


@dataclass(frozen=True)
class TrainingConfig:
    data_path: str | None = None
    output_dir: str = "outputs/tiny"
    train_fraction: float = 0.9
    batch_size: int = 8
    max_steps: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    seed: int = 1337
    device: str = "auto"
    num_threads: int = 1

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_steps", "eval_interval", "eval_batches", "num_threads"):
            _positive_integer(name, getattr(self, name))
        _finite_number("train_fraction", self.train_fraction)
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction 必须在 (0, 1) 内")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            _finite_number(name, getattr(self, name))
        if self.learning_rate == 0:
            raise ValueError("learning_rate 必须大于 0")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed 必须为 [0, 2**63) 内的整数")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 必须为 auto、cpu 或 cuda")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir 必须为非空路径字符串")
        if self.data_path is not None and (
            not isinstance(self.data_path, str) or not self.data_path.strip()
        ):
            raise ValueError("data_path 必须为非空路径字符串或 null")


def _from_dict(cls: type, values: dict[str, Any]) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} 配置必须为 JSON 对象")
    unknown = values.keys() - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__} 存在未知配置项：{', '.join(sorted(unknown))}")
    return cls(**values)


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.model.vocab_size != 256:
            raise ValueError("当前 ByteTokenizer 要求 vocab_size=256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(values, dict) or values.keys() - {"model", "training"}:
            raise ValueError("实验配置仅接受 model 和 training 两个 JSON 对象")
        return cls(
            model=_from_dict(ModelConfig, values.get("model", {})),
            training=_from_dict(TrainingConfig, values.get("training", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
