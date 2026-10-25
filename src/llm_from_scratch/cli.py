"""命令行只负责参数解析，模型与训练接口也可从 Python 直接调用。"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .model import GPT
from .tokenizer import ByteTokenizer
from .training import resolve_device, train


def generate_text(
    checkpoint: str | Path,
    prompt: str = "The workshop",
    *,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "auto",
    seed: int = 1337,
    num_threads: int = 1,
) -> str:
    if not prompt:
        raise ValueError("prompt 不能为空；字节词表没有隐式 BOS token")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed 必须为 [0, 2**63) 内的整数")
    if type(num_threads) is not int or num_threads <= 0:
        raise ValueError("num_threads 必须为正整数")
    target_device = resolve_device(device)
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    state = load_checkpoint(checkpoint)
    config = ExperimentConfig.from_dict(state["config"])
    model = GPT(config.model).to(target_device)
    model.load_state_dict(state["model"])
    model.eval()
    tokenizer = ByteTokenizer()
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=target_device)
    generated = model.generate(
        tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    return tokenizer.decode(generated[0].tolist())


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="训练一个字节级 GPT，并保存配置、指标与 Checkpoint。"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON 配置文件；未指定时使用默认小模型")
    source.add_argument("--resume", type=Path, help="从 Checkpoint 中的配置和完整状态恢复训练")
    parser.add_argument("--data", type=str, help="UTF-8 语料路径")
    parser.add_argument("--output-dir", type=str, help="实验输出目录")
    parser.add_argument("--steps", type=int, help="目标总训练步数，恢复时也按总步数解释")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    try:
        if args.resume:
            config = ExperimentConfig.from_dict(load_checkpoint(args.resume)["config"])
            config = replace(
                config,
                training=replace(config.training, output_dir=str(args.resume.resolve().parent)),
            )
        else:
            config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        overrides = {
            key: value
            for key, value in {
                "data_path": args.data,
                "output_dir": args.output_dir,
                "max_steps": args.steps,
                "device": args.device,
            }.items()
            if value is not None
        }
        config = replace(config, training=replace(config.training, **overrides))
        checkpoint = train(config, resume=args.resume)
    except (OSError, ValueError) as error:
        parser.exit(2, f"错误：{error}\n")
    print(f"checkpoint={checkpoint}")


def generate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="从已保存的 GPT Checkpoint 生成文本。")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The workshop")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 表示贪心解码")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        text = generate_text(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(2, f"错误：{error}\n")
    print(text)
