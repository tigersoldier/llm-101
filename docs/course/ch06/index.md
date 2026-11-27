# 第 6 章 训练循环

> 状态：大纲（待写）。本章小节页面随写作上线。

## 本章目标

- 把前向、损失、反向、裁剪和优化器组装成一次参数更新。
- 用教学语料构造 next-token 批次，理解 shift、所有位置平均和 perplexity。
- 在 CPU 上训练第 1/2 章的小模型，看到 loss 下降，而不是等到最终尺寸。
- 加入验证、loss 曲线、梯度范数、学习率和吞吐量等诊断。
- 用小批次过拟合做 sanity check，确认整条链路没有接错。
- 明确本章不做内存映射、checkpoint 和限时训练。

## 前置与交接

- 前置：第 5 章；需要优化器、完整反向传播和损失。
- 数据：第 2 章的英中教学语料；不接 TinyStories。
- 交接：`ch06/end` 是第 7 章起点；训练循环已经能跑，但数据仍是教学规模。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|6.1|从教学语料到批次|token 化、随机窗口、`BOS`/`EOS`、内存采样器|
|6.2|next-token 目标与损失|输入/目标 shift；所有位置平均；无 padding 时的边界；perplexity|
|6.3|一步更新：前向、反向、裁剪、优化|`train_step` 的顺序；梯度清零；非有限 loss 检查|
|6.4|验证、perplexity 与训练诊断|固定验证窗口；loss 曲线；梯度范数；学习率；吞吐量|
|6.5|小模型训练实验|小配置 + 教学语料；loss 下降；过拟合一个小批次|
|6.6|与限时训练的分工|本章不做的 checkpoint/续训/限时；这些留给第 10 章|

小节页面随写作上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch06/
├── start/
│   └── src/llm101/training.py     # 采样器 / train_step 空壳
└── end/
    ├── src/llm101/training.py
    └── tests/
```

- 新增：`training.py` 的内存采样器和 `train_step`。
- 复用：`losses.py`、`optim.py`、`model.py`。
- 数据：`ch02/start/samples/` 的教学语料。
- 不新增：`TokenStream` 的 mmap 实现、checkpoint、限时训练脚本。

## 本章验收

- `cd ch06/end && uv run pytest -q` 全绿。
- 固定 seed 下，两次运行得到同样的 loss 序列。
- 小模型在教学语料上训练若干步后 loss 明显下降，验证 PPL 可计算。
- 一个小批次可以被过拟合，证明前向/反向/优化器链路正确。
- 诊断输出包含 loss、学习率、梯度范数和吞吐量。

## 本章不做什么

- 不引入 int32/mmap `TokenStream`（第 7 章）。
- 不做 checkpoint、续训、限时停止（第 10 章）。
- 不用 5M 参数模型；最终配置留到第 8/10 章。
