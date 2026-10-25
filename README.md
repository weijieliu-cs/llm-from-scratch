# LLM from Scratch

用 PyTorch 基础模块从零实现一个可训练的字节级 GPT，学习从文本编码、因果注意力到训练与自回归生成的完整过程。

项目保持小规模和可读性：注意力、LayerNorm 与 GELU 的计算显式展开；训练支持配置文件、固定窗口评估、Checkpoint 和状态恢复。默认模型约 12 万参数，可以在 CPU 上运行。

## 快速开始

需要 **Python 3.10+**。在仓库根目录执行，`python3` 应指向符合要求的版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 默认训练 200 步，自动选择 CUDA 或 CPU
llm-train --config configs/tiny.json

# 从已保存的模型生成文本
llm-generate --checkpoint outputs/tiny/best.pt \
  --prompt "The workshop" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。PyTorch 的安装选项可查阅[官方安装指南](https://pytorch.org/get-started/locally/)。

原入口仍可使用：`python train.py` 和 `python generate.py --checkpoint outputs/tiny/last.pt`；
`python -m pip install -r requirements.txt` 也会安装当前包。训练与生成现在使用独立命令。

## 实验配置与恢复

模型和训练参数集中在 [`configs/tiny.json`](configs/tiny.json)。配置项拼写错误、非法头数、无效采样参数等会直接报错。
所有相对路径均相对于**命令执行目录**；`data_path: null` 读取包内示例语料。

```bash
# 用独立目录快速验证训练链路
llm-train --config configs/tiny.json --device cpu --steps 20 --output-dir outputs/quick

# 恢复到总计 100 步，沿用 Checkpoint 中保存的配置
llm-train --resume outputs/quick/last.pt --steps 100

# 换用自己的 UTF-8 语料，作为新实验训练
llm-train --data data/raw/corpus.txt --output-dir outputs/custom --steps 1000

# temperature=0 使用贪心解码
llm-generate --checkpoint outputs/quick/last.pt --prompt "The workshop" --temperature 0
```

`--steps` 始终表示目标**总步数**。已有实验目录不会被新训练覆盖；恢复已有目录使用其中的 `last.pt`。
若从其他 Checkpoint 开始继续训练，请用 `--output-dir` 指定新的目录。
通过 `--help` 查看每个命令支持的参数；其余参数通过 JSON 配置调整。

每个实验目录包含：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 实际使用的完整配置，包括解析后的绝对路径与设备 |
| `metrics.jsonl` | 每次评估的步数、训练/验证 loss 和累计训练 token 数 |
| `last.pt` | 最近一次评估或训练结束时的完整训练状态 |
| `best.pt` | 验证 loss 创新低时保存的完整状态 |

Checkpoint 包含模型、优化器、步数、最佳验证 loss、CPU/CUDA 随机数状态、批次采样器状态、Tokenizer 标识、语料 SHA-256 和 PyTorch 版本。
保存时先写同目录临时文件，再原子替换。恢复时检查模型结构、语料和影响更新的训练参数。
评估使用固定种子采样窗口，与训练随机数流独立；相同 CPU 环境下，恢复后的参数与连续训练逐项一致，已有测试覆盖。
跨设备、PyTorch 版本或 CUDA 算法不保证逐位一致。程序中断后从最近一次保存的评估步数恢复。

生成只需要 Checkpoint，无需原训练语料。`temperature` 调整采样分布，`top-k` 限制候选数量，`--seed` 控制生成随机性。

## 代码导航

```text
configs/tiny.json                 默认小模型实验
src/llm_from_scratch/
  config.py                      配置、序列化与校验
  tokenizer.py                   UTF-8 字节 encode/decode
  dataset.py                     先划分数据，再构造错位一位的批次
  model.py                       LayerNorm、GELU、注意力、Transformer 和 GPT
  training.py                    AdamW 参数分组、梯度裁剪、评估和训练
  checkpoint.py                  完整状态的原子保存与加载
  cli.py                         训练与生成命令行
  resources/sample.txt           随包分发的示例语料
tests/                           组件正确性、过拟合、恢复一致性和命令行测试
docs/development.md              开发约定与参考仓库
LLM_COMPONENTS.md                 LLM 组件全景与后续学习路线
train.py / generate.py            仓库内的脚本入口
model.py / dataset.py             原导入路径的兼容接口
```

建议依次阅读 `tokenizer.py` → `dataset.py` → `model.py` → `training.py` → `cli.py`。
完整组件与实现路线见 [`LLM_COMPONENTS.md`](LLM_COMPONENTS.md)，本地数据说明见 [`data/README.md`](data/README.md)。

## 开发与验证

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m pytest
```

测试覆盖未来 token 不影响此前预测、归一化和激活与 PyTorch 对齐、梯度和共享权重、微型数据过拟合、采样边界、评估随机性、完整训练恢复、语料变化拒绝恢复及命令行入口。
GitHub Actions 在 Python 3.10 和 3.12 上执行代码检查、CPU 测试和 wheel 构建。

## 范围与参考

每个 token 对应一个 UTF-8 字节，词表固定为 256；中文字符通常占多个 token，上下文长度也按字节计算。
原始数据前 90% 用于训练、后 10% 用于验证；默认语料很小，验证 loss 用于观察流程，不代表通用语言能力。
生成可能包含不完整的 UTF-8 序列，解码时以替换字符显示。模型用于文本续写，尚未实现对话微调、BPE、KV Cache、混合精度或分布式训练。

工程整理参考了 [Sebastian Raschka 的 LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
的学习顺序、[minGPT](https://github.com/karpathy/minGPT) 的模型/训练职责划分，以及
[nanoGPT](https://github.com/karpathy/nanoGPT) 的实验配置、优化器分组和 Checkpoint 设计。
具体取舍见[开发说明](docs/development.md#参考与取舍)。

本项目使用 [MIT License](LICENSE)。
