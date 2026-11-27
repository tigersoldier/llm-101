# 第 4 章 注意力、block 与整模型反向

> 状态：大纲（待写）。本章小节页面随写作进度上线。

## 本章目标

- 完成因果多头自注意力的反向传播：softmax 雅可比、mask、缩放、多头和输出投影。
- 完成 `TransformerBlock.backward`：残差、Pre-LN 和 MLP 两条分支。
- 完成 `TinyGPT.backward`：final norm、输出头，以及绑定词嵌入的梯度累加。
- 保留第 1 章返回的 cache 接口，同时用内部缓存支撑反向传播。
- 用 tiny config 做 block/整模型全参数有限差分；用专门测试检查绑定权重。
- 前向只做一页回顾，不重讲第 1 章的 Q/K/V、缩放和 mask。

## 前置与交接

- 前置：第 3 章；需要各层反向、softmax 交叉熵和 `Parameter`。
- 交接：`ch04/end` 是第 5 章起点；模型已经可以完整反向，但还没有优化器和训练循环。
- 不缩放：最终维度留到第 8 章。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|4.1|注意力的反向：softmax 与 mask|`dA`/`dV`/`dS`；softmax 向量雅可比；mask 位置零梯度；缩放因子|
|4.2|多头、缩放与输出投影|拆头/拼头梯度；Q/K/V 投影；`out_proj` 的输入梯度|
|4.3|残差与 Pre-LN 的 block 反向|两条残差路径；LayerNorm 反向；MLP 的 up/GELU/down|
|4.4|整模型反向与绑定权重|final norm；输出头；输入查表与输出投影共享同一张表时的梯度累加|
|4.5|缓存、形状与接口|第 1 章 `(output, cache)` 契约；内部缓存生命周期；`[T]`/`[B,T]` 兼容|
|4.6|有限差分与因果性验收|注意力全参数；block/整模型 tiny config；未来 token 不影响过去|

小节页面随写作进度上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch04/
├── start/
│   ├── src/llm101/attention.py    # backward 空壳
│   └── src/llm101/model.py        # block/model backward 空壳
└── end/
    ├── src/llm101/attention.py
    ├── src/llm101/model.py
    └── tests/
```

- 修改：`attention.py`、`model.py`；必要时给 cache 增加内部字段。
- 复用：第 3 章的 `Linear`、`LayerNorm`、`GELU` 反向和损失。
- 不新增：优化器、训练循环、真实语料。

## 本章验收

- `cd ch04/end && uv run pytest -q` 全绿。
- 注意力所有参数和输入的解析梯度与有限差分一致。
- block/整模型 tiny config 的全参数有限差分通过。
- 未出现在输入里的 token，其嵌入行只从输出头得到梯度。
- 改未来 token 不改变过去位置的输出；第 1 章的因果性测试仍通过。

## 本章不做什么

- 不缩放模型（第 8 章）。
- 不实现优化器（第 5 章）、不跑训练循环（第 6 章）。
- 不接真实语料，不做 KV cache。
