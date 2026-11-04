# 限时训练实验方案

本页定义中英混合模型的训练时长、记录字段和评估顺序。目标是在现有 CPU 上找到满足课程用途的最短训练时长，而不是把小模型包装成生产级助手。

## 实验顺序

一次只跑一个训练进程，避免在 4 核 CPU 上互相争抢资源。模型使用联合词表和完整中英训练切分，不添加语言控制 token。

1. 先训练 5 小时。
2. 教学用途的达标标准是：固定提示词能让读者感受到模型在根据上下文生成新的故事续写，且中英文提示都能看到对应语言的生成。语法、连贯性、重复和局部错误可以存在，不要求完美，也不以某个 perplexity 数值作为硬门槛。
3. 若 5 小时样例达到这个演示标准，则用相同配置和 seed 从头训练 1 小时，再在同一运行目录依次 `--resume` 到总计 3 小时、5 小时。每个目标时长都会保存权重与样例；它们属于同一条训练轨迹，不是独立随机重跑。若 5 小时仍不能展示生成效果，则从 checkpoint 延长到 12、最多 24 小时。
4. 每次运行保存最终权重和验证集最佳权重；只考虑发布通过许可复核且达到教学标准的最短版本。

验证 loss/perplexity 用来观察学习进展和比较同一词表和验证窗口下的不同时长，不单独决定是否达标。`samples-<hours>h.json` 是固定提示词的教学演示样例，不是统计显著的能力评测。已完成运行的指标和样例见[实验结果](results.md)。

## 运行命令

首次数据准备后：

```bash
uv run python scripts/train_model.py --hours 5
```

默认输出目录包含目标时长和 seed，即 `artifacts/runs/bilingual-<hours>h-seed<seed>/`。若 5 小时版本未达标，将目标总时长延长到 12 小时并指向原目录：

```bash
uv run python scripts/train_model.py \
  --hours 12 \
  --run-dir artifacts/runs/bilingual-5h-seed20261027 --resume
```

短时长比较从头运行 1 小时，然后将同一个 run directory 续训到总计 3 小时、再到 5 小时。例如：

```bash
uv run python scripts/train_model.py --hours 1 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027
uv run python scripts/train_model.py --hours 3 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027 --resume
uv run python scripts/train_model.py --hours 5 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027 --resume
```

检查每个目标时长保存的固定提示词输出是否分别展示了中英文的新续写即可；不要把这项教学标准误读成对实际语言能力或生产可用性的保证。

## 必须保留的证据

每个运行目录记录模型配置、数据源与 tokenizer 的 SHA-256、目标与实际训练秒数、step 数、呈现给模型的 token 位置数、各语言覆盖数、验证 loss/perplexity 和固定提示词样例。语言覆盖数统计实际 batch 窗口中的输入 token 位置，窗口会重复，因此不等同于独立 token 数或完整语料遍历数。数据来源修订号、SHA-256、许可证与切分方法见[数据与许可](../datasets.md)及 `data/processed/experiment_manifest.json`。

原始语料与 checkpoint 不提交到 Git。权重发布前需单独复核 TinyStories 的 CDLA-Sharing-1.0 来源链、中文翻译数据的上游许可和适用的再发布条件；许可问题未解决时不发布。
