# 第 9 章 推理与生成

> 状态：大纲（待写）。本章小节页面随写作上线。

## 本章目标

- 实现自回归生成循环：编码 prompt、取最后一个位置的 logits、追加 token、遇到 EOS 停止。
- 实现贪心、temperature 和 top-k 采样，并固定 seed 保证可复现。
- 处理上下文截断、`BOS`/`EOS`、`max_new_tokens` 和解码时跳过特殊 token。
- 用第 6 章的小模型对比随机权重与训练后权重的输出。
- 讲清每步重算上下文的成本，以及 KV cache 为什么不在主线。
- 明确生成样例只用于观察，不作为统计评测。

## 前置与交接

- 前置：第 8 章；需要最终配置；也可以先用第 6 章的小模型跑通。
- 交接：`ch09/end` 是第 10 章起点；限时训练脚本会调用这里的 `generate`。
- 不做网页、不做 KV cache、不做 top-p/beam search。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|9.1|自回归循环与上下文截断|prompt → ids；取最后位置；追加；窗口截断；停止条件|
|9.2|贪心解码|`temperature=0` 的 argmax；确定性与重复问题|
|9.3|temperature 与 top-k 采样|logits/temperature；softmax；top-k 截断；seed 与可复现|
|9.4|特殊 token、停止条件与解码|`BOS`/`EOS`；`max_new_tokens`；`decode` 跳过特殊 token|
|9.5|随机权重与训练后模型对比|同一 prompt 的两组输出；小模型也能看出机制|
|9.6|重算成本与 KV cache 的取舍|每步重算注意力；KV cache 的动机；为什么不在主线|

小节页面随写作上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch09/
├── start/
│   ├── src/llm101/inference.py    # generate 空壳
│   └── scripts/generate.py        # CLI 空壳
└── end/
    ├── src/llm101/inference.py
    ├── scripts/generate.py
    └── tests/
```

- 新增：`inference.py`、`scripts/generate.py`。
- 复用：`TinyGPT.forward`、tokenizer、第 6/10 章的权重。
- 不新增：KV cache、top-p、beam search、网页。

## 本章验收

- `cd ch09/end && uv run pytest -q` 全绿。
- `temperature=0` 的输出确定；相同 seed + 设置可复现。
- top-k 只从截断后的候选采样；遇到 EOS 或 `max_new_tokens` 停止。
- prompt 编码/解码往返正确；特殊 token 不出现在最终文本里。
- 生成过程不使用 autograd；随机权重和训练后权重的输出可对比。

## 本章不做什么

- 不做 KV cache、top-p、beam search。
- 不做网页或交互界面。
- 不跑长训练、不做评测方法论（第 11 章）。
