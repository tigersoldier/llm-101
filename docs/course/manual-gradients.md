# 手写梯度：从公式到代码

本章使用 PyTorch 张量和矩阵乘法，但不调用 autograd。每个参数有一个单独的 `value` 和 `grad`；反向传播按链式法则显式计算梯度，并累加到参数上。

## 线性层

对 `y = xW + b`，上游给出 `dy`：

```text
dx = dy Wᵀ
dW = xᵀ dy
db = sum_rows(dy)
```

代码位于 `src/llm101/layers.py` 的 `Linear`。输入有 batch 和其他前导维度时，先把这些维度展平，再计算 `dW` 和 `db`。

## Embedding

Embedding 是按 token ID 从矩阵中取行。反向传播把每个位置的梯度加回对应 token 的那一行；相同 token 出现多次时，梯度要累加。实现使用张量的 `index_add_`，而不是 Python 循环。

## Softmax 与交叉熵

对单个目标 token，logits 的梯度为 `softmax(logits) - one_hot(target)`；批次平均时再除以样本数。实现用减去最大 logit 的方式保证数值稳定，并直接返回 loss 和 logits 梯度。

## 梯度校验

有限差分用小扰动近似某个参数的导数：

```text
grad[i] ≈ (loss(x + ε·eᵢ) - loss(x - ε·eᵢ)) / (2ε)
```

测试只对很小的张量运行，把有限差分结果与解析梯度对比。它计算昂贵，不用于训练。

LayerNorm 与 GELU 的反向传播也有可读实现和有限差分测试；主线把它们作为已提供的实现，重点推导线性层、Embedding、Softmax/交叉熵及后续的因果自注意力。
