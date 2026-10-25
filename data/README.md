# 本地训练数据

将自己的 UTF-8 文本放在 `data/raw/`，通过 `llm-train --data data/raw/corpus.txt`
指定。`data/raw/` 和 `data/processed/` 已被 Git 忽略。

内置示例语料现位于
[`src/llm_from_scratch/resources/sample.txt`](../src/llm_from_scratch/resources/sample.txt)，
随 Python 包安装。默认配置的 `data_path: null` 使用这份语料，无需额外下载。

数据按字节顺序先划分训练集和验证集，再采样窗口。每个分区至少需要
`context_length + 1` 个字节。多篇文档的去重和独立文档划分留待后续实现。
