# 预训练循环与断点续训

本章把已有的 decoder-only Transformer 接到完整故事语料上。目标是让训练步骤可观察、可复现，也能在 CPU 限时运行；不会调用 PyTorch autograd。

## 从 token 流构造样本

预处理脚本把每篇故事独立编码为 `BOS + 内容 + EOS`，写入 little-endian int32 文件。`TokenStream` 通过 `torch.from_file` 映射文件，不把数 GB 的 token 数据复制到 Python 列表。

对序列长度 `T` 和随机起点 `s`，训练输入与目标是：

```text
x = tokens[s : s + T]
y = tokens[s + 1 : s + T + 1]
```

也就是在每个位置预测下一个 token。随机窗口可以跨过故事边界，但边界上保留 `EOS` 和下一篇的 `BOS`。双语训练流由两种语言的完整训练切分拼接；抽样在 token 位置上均匀，因此语言比例由两部分实际 token 数决定，不额外插入语言控制 token。

预处理命令：

```bash
uv run python scripts/download_datasets.py
uv run python scripts/prepare_training_data.py
```

Tokenizer 合并规则使用每种语言的 512 篇 reservoir 样本训练；初始字符表扫描整个训练切分。这个样本只限制 BPE 合并规则学习，不会下采样语言模型训练语料。

## 一次参数更新

`llm101.training.train_step` 展开完整步骤：

1. 采样 `batch_size × sequence_length` 输入和右移目标。
2. 调用 `TinyGPT.forward` 得到词表 logits。
3. 计算稳定的 softmax cross entropy 和 logits 显式梯度。
4. 调用手写 `TinyGPT.backward` 累积各参数梯度。
5. 按全模型 global norm 做梯度裁剪，再运行 AdamW。

默认参数：4 层、`d_model=256`、4 个注意力头、`d_ff=1024`、上下文 128；batch size 4、AdamW 学习率 `3e-4`、`β=(0.9, 0.999)`、`epsilon=1e-8`、weight decay `0.01`、global gradient norm 上限 1.0。参数可在训练命令行覆盖；当前主实验使用固定学习率，不使用 dropout 或 scheduler。

```bash
uv run python scripts/train_model.py --corpus english --hours 5
```

每个语言运行都会从独立随机初始化开始，彼此不共享权重。训练脚本按时长停止，并周期性写入 `artifacts/runs/.../metrics.jsonl` 与 `last.pt`。日志包含验证 loss/perplexity、训练步数、呈现给模型的 token 位置数，以及双语各语言的覆盖计数。覆盖计数会包含随机窗口重复采样，不代表去重后的语料遍历量。

## 验证和恢复

每 5 分钟默认用验证流抽样 20 个 batch；双语版分别记录混合、中、英文验证指标。结束时增加到至少 50 个 batch。运行配置还会记录验证 batch 数、验证/保存间隔、AdamW beta 和 epsilon，便于复现实验。loss/perplexity 是可比较的量化指标，固定提示词生成则用于人工检查可读性与语言表现。

每 5 分钟保存模型、AdamW 一阶/二阶矩、累计训练时间、语言覆盖计数和 batch sampler 的随机数状态。系统信号会在当前训练步骤完成后保存并退出。比如把英语实验从 5 小时延长到 12 小时：

```bash
uv run python scripts/train_model.py \
  --corpus english --hours 12 \
  --run-dir artifacts/runs/english-5h-seed20261027 --resume
```

恢复时数据摘要、模型配置、batch size、学习率、seed 等必须与原实验一致；目标总时长必须大于已累计训练时间。暂停的日历时间不计入训练时长。

训练结束会生成 `model-5h.pt`、`best-model.pt` 和固定提示词样例 JSON。原始数据、token 流、检查点都在 Git 忽略目录；是否对外发布权重还需完成数据许可复核。
