# LLM from Scratch

用于学习 LLM 相关技术的代码仓库。当前保留一个小型 GPT 示例，方便阅读实现、修改参数和观察结果。后续随学习进度逐步增加功能。

## 文件结构

```text
model.py          GPT 模型及模型参数
dataset.py        文本编码、数据划分和批次采样
train.py          训练、验证 loss 和简单文本生成
data/sample.txt   示例训练语料
requirements.txt  依赖
```

## 运行

使用 Python 3.10 或更新版本，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

训练默认运行 200 步，优先使用 CUDA，否则使用 CPU。脚本会打印训练和验证 loss，最后从 `The workshop` 开始生成一段文本。

## 阅读和修改

建议按以下顺序阅读：

1. `dataset.py`：将 UTF-8 文本转成字节 token，理解输入和目标如何错开一位。
2. `model.py`：从多头因果注意力、前馈网络和残差连接读到完整 GPT。
3. `train.py`：理解前向计算、交叉熵 loss、反向传播、参数更新和自回归生成。

模型参数在 `model.py` 的 `ModelConfig` 中；训练参数在 `train.py` 的 `main()` 开头。可以先修改上下文长度、注意力头数、模型层数或训练步数，观察 loss 和生成结果的变化。语料可直接替换 `data/sample.txt`。

当前模型使用可学习的位置嵌入、LayerNorm、GELU 和 dropout。注意力、归一化和 GELU 的计算在代码中显式展开，便于对照公式学习。

每个 token 对应一个 UTF-8 字节，词表大小固定为 256，上下文长度按字节计算。数据前 90% 用于训练，后 10% 用于验证；loss 通过随机采样的批次估计。生成的字节可能包含不完整的 UTF-8 序列，显示时会替换无法解码的部分。示例语料和模型都很小，主要用于观察学习过程。
