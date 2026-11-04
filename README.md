# LLM 101：从零训练一个小语言模型

这是一个面向会 Python 的读者的手把手课程：从公开短故事语料出发，逐步实现 tokenizer、decoder-only Transformer、预训练与推理，最终得到一个能续写短句的小模型。

## 教学目标

- 用 PyTorch 张量和矩阵运算，但不依赖 autograd；手写有代表性的反向传播与参数更新。
- 以清晰、可读、CPU 可运行优先，不以性能优化为目标。
- 分别训练英文、中文和中英混合三个版本；双语版不加入语言控制标记。
- 用验证集 loss/perplexity 和固定提示词生成结果评估；按时长实验选出最短达标版本。
- 教程内容使用中文，代码和练习在本地运行；网站由 MkDocs 生成，可部署到 GitHub Pages。

## 当前状态

已实现教学版 Unicode-codepoint BPE tokenizer、手写梯度与 SGD/AdamW、因果多头自注意力、Pre-LN decoder-only Transformer，以及完整语料预处理、显式梯度训练、验证、断点续训和自回归生成脚本。有限差分测试覆盖了注意力与小型完整模型的参数梯度。英文、中文、双语三个 5 小时基线，以及双语 1/3/5 小时累计对照已完成，结果见[实验结果](docs/course/results.md)；模型权重再发布许可复核仍待完成。

## 本地开发

```bash
uv sync --extra dev --extra docs
uv run pytest
uv run ruff check src tests scripts
uv run pyright --project pyrightconfig.json
uv run mkdocs serve
```

## 数据准备与训练

首次运行需要下载约 2.4 GB 原始语料，再生成本地 token 流：

```bash
uv run python scripts/download_datasets.py
uv run python scripts/prepare_training_data.py
uv run python scripts/train_model.py --corpus english --hours 5
```

`--corpus` 可选 `english`、`chinese` 或 `bilingual`。训练默认 CPU、batch size 4、上下文 128、AdamW 学习率 `3e-4`，每 5 分钟记录验证指标并保存断点。将实验延长到 12 小时可在原运行目录加 `--resume`；具体评估流程见[实验方案](docs/course/experiment-protocol.md)。生成示例：

```bash
uv run python scripts/generate.py \
  --model artifacts/runs/english-5h-seed20261027/model-5h.pt \
  --tokenizer data/processed/tokenizers/english.json \
  --prompt "Once upon a time, there was a little girl named"
```

原始数据、token 流和训练检查点都留在本地，不提交到 Git。

## 目录约定

- `docs/`：中文课程和数据来源说明。
- `src/llm101/`：逐步构建的可复用 Python 实现。
- `tests/`：单元测试、数值梯度检查和训练/推理回归测试。
- `data/raw/`、`data/processed/`：本地数据，忽略提交到 Git。
- `artifacts/`：本地检查点；最终选定的权重计划作为 GitHub Release 附件发布，不放进 Git 历史。

训练语料的上游许可证与模型权重许可分开记录。发布数据或权重前会复核来源、署名和许可条件。
