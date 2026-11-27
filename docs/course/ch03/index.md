# 第 3 章 参数、层与手写梯度

> 状态：大纲（待写）。本章小节页面随写作进度上线。

## 本章目标

- 用 `Parameter` 保存 `value`、`grad` 和 `zero_grad`，把第 1 章的裸张量迁进来。
- 推导并实现 `Linear`、`Embedding`、`LayerNorm`、`GELU` 的反向传播。
- 用稳定的 softmax 交叉熵和 next-token 目标，把 logits 变成可优化的损失。
- 用中心差分、float64 和相对误差检查每个梯度。
- 把 GELU 从第 1 章的 tanh 近似换成精确 `erf` 形式，并证明差异在千分之一量级。
- 保持第 1 章的参数量测试：`parameters()` 改返回 `Parameter`，但 `numel()` 仍可数。

## 前置与交接

- 前置：第 2 章；需要张量形状、矩阵乘法、链式法则和 token id。
- 交接：`ch03/end` 是第 4 章起点；本章只到层和损失，不实现注意力、block 或整模型反向。
- 数值：GELU 精确化；其余前向数值保持不变。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|3.1|参数与梯度：`Parameter` 与手写反向|为什么不用 autograd；`value`/`grad`/`zero_grad`；梯度累加|
|3.2|Linear 与 Embedding 的反向|`dx`/`dW`/`db`；批次展平；重复 token 的 `index_add_`|
|3.3|LayerNorm 的反向|均值/方差、缩放平移、标准 LayerNorm 梯度|
|3.4|GELU：从 tanh 近似到精确式|精确式与导数；与第 1 章 tanh 近似的差异测试|
|3.5|Softmax 交叉熵与 next-token 目标|数值稳定的 softmax；shift；所有位置平均；perplexity|
|3.6|有限差分与梯度检查|中心差分、epsilon、float64、容差和常见错误|

小节页面随写作进度上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch03/
├── start/
│   ├── src/llm101/parameters.py   # Parameter 空壳
│   ├── src/llm101/layers.py       # backward 空壳
│   └── src/llm101/losses.py       # softmax 交叉熵空壳
└── end/
    ├── src/llm101/parameters.py
    ├── src/llm101/layers.py
    ├── src/llm101/losses.py
    └── tests/
```

- 修改：`parameters.py`、`layers.py`、`model.py` 的 `parameters()`；新增 `losses.py`。
- 复用：`attention.py` 的前向；第 2 章的 tokenizer 与词表。
- 不新增：优化器、训练循环、真实语料。

## 本章验收

- `cd ch03/end && uv run pytest -q` 全绿。
- 各层参数和输入的解析梯度与中心差分一致。
- GELU 精确式与第 1 章 tanh 近似的输出差异小于 2e-3。
- 交叉熵对 logits 的梯度与有限差分一致；单句 next-token loss 与手算一致。
- 第 1 章配置的参数量仍为 5,824。

## 本章不做什么

- 不实现注意力、block 或 `TinyGPT.backward`（第 4 章）。
- 不实现优化器（第 5 章）、不跑训练循环（第 6 章）。
- 不接真实语料。
