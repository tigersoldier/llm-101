# 最终版本：完整实现快照

本目录是课程完成时的完整代码快照，可以脱离仓库其余部分独立运行。后续每章的章节目录采用同样的形式：包含该章结束时的代码、测试和该章产生的关键结果。

## 内容

```text
final/
├── src/llm101/     # 11 个模块：BPE、手写梯度、注意力、Transformer、
│                   #   语料管道、训练、checkpoint、推理
├── scripts/        # 下载语料、预处理、限时训练、生成
├── tests/          # 11 个测试文件，含手写梯度的有限差分校验
├── results/        # 固定提示词生成样例（JSON）
├── pyproject.toml  # 可独立 uv sync 的项目定义
├── uv.lock         # 锁定的依赖版本
└── LICENSE
```

## 独立运行

```bash
cd final
uv sync --locked --extra dev
uv run pytest -q
```

训练前需要先准备语料，数据与权重都不进入 Git：

```bash
uv run python scripts/download_datasets.py
uv run python scripts/prepare_training_data.py
uv run python scripts/train_model.py --corpus bilingual --hours 1
```

数据写入仓库根目录的 `data/`，训练产物写入 `artifacts/runs/`；两者都在 `.gitignore` 中。

## 模型权重

权重约 21 MB（`model-*.pt`），续训 checkpoint 约 64 MB，均不提交到 Git，保留在训练机上的 `artifacts/runs/<corpus>-<hours>h-seed<seed>/`。上游 TinyStories 语料的再发布许可尚未复核，因此在许可结论明确前不公开发布权重。

## 已完成的训练

|模型|训练时长|呈现 token 数|验证 PPL|
|---|---|---:|---:|
|英文|5 小时|48,089,088|18.08|
|中文|5 小时|39,990,784|21.74|
|双语|5 小时|38,290,944|18.61（混合）|

双语模型另有一条 1h → 3h → 5h 的累计训练轨迹，在同一验证窗口上的 PPL 为 38.66 → 21.24 → 18.21。指标含义、样例和局限见 `docs/course/results.md`。

## 与主仓库的关系

本目录是快照，不随主仓库代码演进自动更新。修改实现时应同时更新本目录，否则两者会不一致；`results/` 中的样例来自 `bilingual-progressive-seed20261027` 等运行，命名中的时长是累计训练时长。

这里的模型是 next-token prediction 预训练的故事续写器，不是指令微调的助手：给定故事开头能继续写，但不会执行“讲个笑话”这类请求。
