# 因果自注意力

本章实现 decoder-only Transformer 的核心：因果多头自注意力。代码使用 PyTorch 张量完成矩阵运算，并显式计算反向传播，不调用 autograd。

## 从序列到 Q、K、V

给定隐藏状态 `X`，三个线性层分别得到 Query、Key、Value：

```text
Q = X Wq
K = X Wk
V = X Wv
```

把最后一维拆成 `n_heads` 个头。每个头的宽度为 `d_head = d_model / n_heads`，分数是：

```text
scores = Q Kᵀ / sqrt(d_head)
```

除以平方根可避免点积随头宽变大而过度放大。

## 因果遮罩与 Softmax

位置 `t` 只能查看 `0..t`，不能读取未来 token。实现把分数矩阵的严格上三角替换为负无穷，然后逐行做数值稳定的 Softmax。遮罩后每一行至少保留自身位置，因此归一化分母非零。

```text
weights = softmax(masked_scores)
context = weights V
```

合并各头后，再经过输出投影。这里暂不加入 dropout、缓存或其他性能优化，以便先看清计算过程。

## 反向传播

若 `C = A V`，上游梯度先得到 `dA = dC Vᵀ` 和 `dV = Aᵀ dC`。Softmax 的向量雅可比乘积可写成：

```text
dS = A * (dA - sum(dA * A, axis=-1, keepdim=True))
```

最后按点积关系把梯度传给 Q、K，并通过三个线性投影累加到输入。输出投影和 Q/K/V 投影都使用上一章的手写 `Linear.backward()`。

## 验证

`tests/test_attention.py` 用中心有限差分检查输入与所有投影参数的解析梯度；另有一个因果性测试，改变未来位置后确认之前位置的输出不变。有限差分只适用于很小的测试张量，不用于实际训练。
