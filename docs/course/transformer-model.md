# Decoder-only Transformer

本章把因果注意力、LayerNorm、GELU、线性层和残差连接组成一个可训练的 GPT 风格模型。完整实现位于 `src/llm101/model.py`，训练仍由调用方显式管理前向、反向、清梯度和优化器更新。

## Pre-LN Transformer block

每个 block 按以下顺序计算：

```text
x = x + CausalSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
MLP(x) = Linear_down(GELU(Linear_up(x)))
```

这是 Pre-LN 结构：LayerNorm 在注意力和 MLP 之前。每个残差分支的反向传播都把恒等路径的梯度与分支梯度相加。

## 词嵌入、位置与权重共享

输入 token ID 查找 token embedding，再加上可学习的绝对位置 embedding。最后的 LayerNorm 输出乘以 token embedding 矩阵的转置，得到词表 logits：

```text
logits = hidden_states @ token_embedding.T
```

输入查表与输出词表投影共享同一个参数矩阵，因此反向传播会把两条路径的梯度加到同一个参数上。不同语言版本各自训练独立模型，不共享权重。

## 默认尺寸与参数量

`GPTConfig` 默认使用 4 层、`d_model=256`、4 个注意力头、`d_ff=1024`、最大上下文 128 token。词表大小由 tokenizer 决定；共享词嵌入时总参数量约为 `3,192,320 + 256 × vocab_size`。例如词表 8,192 时约 529 万参数。

模型不使用 autograd。`TinyGPT.forward()` 返回每个位置的 logits，`TinyGPT.backward(grad_logits)` 显式反传并累加梯度；优化器则通过 `parameters()` 取得去重后的参数列表。
