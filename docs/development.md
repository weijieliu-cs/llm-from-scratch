# 开发说明

## 环境与质量检查

运行时依赖和开发依赖统一定义在 [`pyproject.toml`](../pyproject.toml)，
`requirements.txt` 仅提供兼容安装入口。开发使用可编辑安装：

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m pytest
```

自动整理代码可运行 `ruff check . --fix` 和 `ruff format .`。
新增的计算、采样和状态管理行为需要相应正确性测试；测试使用 CPU 和小模型，无需下载模型或语料。
CI 同时构建 wheel，并在仓库外使用已安装的 wheel 进行训练和生成，检查内置语料是否正确分发。

构建安装包：

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

## 模块约定

- **配置**：使用不可变 dataclass，默认值与 `configs/tiny.json` 一致。JSON 不执行 Python 代码；未知字段直接报错。
- **数据**：Tokenizer 负责文本与 ID 互转，数据模块负责先划分后采样。标签只在 `get_batch()` 中移位一次。
- **模型**：输入为 `(batch, sequence)` 的整数 ID；输出为 `(batch, sequence, vocab)` logits 和可选的标量 loss。
- **生成**：输入至少一个 token，允许超过上下文长度的提示词，模型仅关注最近的上下文。自动关闭 Dropout，结束时恢复原模式。
- **训练**：批次采样使用独立 CPU Generator。评估固定窗口，不推进训练采样器或 Dropout 随机数状态。
- **优化器**：AdamW 仅对二维及以上参数使用权重衰减；偏置和归一化参数不衰减，共享嵌入权重只出现一次。
- **Checkpoint**：只序列化配置、张量和基础 Python 数据，通过 `weights_only=True` 加载。改变格式时更新版本及恢复测试。
- **实验文件**：权重、日志、本地原始数据不提交到 Git。新实验使用新目录，连续恢复使用 `last.pt`。

`grad_clip=0` 关闭裁剪，但仍检查梯度是否为有限数值。训练保留固定学习率和单次更新对应单个批次的语义。
增加学习率调度、累积或混合精度时，应同时保存新增状态，并验证恢复后更新序列。

`best.pt` 是训练过程中验证 loss 创新低时的快照；恢复到新目录时，历史最佳值仍保留，只有产生新的最佳结果才会在新目录写入 `best.pt`。
更改评估窗口数量后，验证 loss 与历史最佳值的可比性会变化；对比实验应固定评估配置。

## 参考与取舍

以下为 2026-10-25 查阅的原始仓库代码。参考其职责划分和训练工程思路，在本项目原有实现上整理，未引入它们的代码作为运行依赖。

| 参考 | 采用的思路 | 本项目的取舍 |
| --- | --- | --- |
| [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 从数据、注意力到完整模型和训练逐步阅读 | 提供中文代码导航与组件路线，核心实现仍为普通 Python 模块 |
| [minGPT/model.py](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)、[trainer.py](https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py) | 模型、数据、训练职责分离 | 使用少量模块，注意力、归一化和激活公式保持可见 |
| [nanoGPT/model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) | 结构配置、共享权重、按维度分组的 AdamW 和可控采样 | 保留字节词表及现有小模型，不加载 GPT-2 权重 |
| [nanoGPT/train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py) | 配置持久化、验证 loss、优化器状态和训练步数恢复 | 使用 JSON 与 argparse；额外保存采样/随机状态和语料校验值，原子写入 Checkpoint |

当前规模没有引入 Trainer 继承框架、插件注册器、配置框架或服务层。先保证训练和生成闭环可运行、可检查，再按具体学习目标扩展。
