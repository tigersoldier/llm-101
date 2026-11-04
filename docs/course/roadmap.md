# 课程路线

课程逐步完善同一个代码项目；假设读者会基本 Python，先构造可运行的 CPU 小模型，再理解工程取舍。

1. **文本与 next-token prediction**：从故事构造 token 序列、训练/验证切分和因果目标；数据加载见[数据与许可](../datasets.md)。
2. **教学版 BPE**：从 Unicode 字符学习频繁合并，自己实现训练、编码与解码；见[BPE 课程页](bpe.md)。这不是 GPT 的 byte-level tokenizer。
3. **损失与显式梯度**：线性层、Embedding、Softmax＋交叉熵的前向和手写反向传播；[课程页](manual-gradients.md)介绍公式，有限差分测试核对梯度。
4. **优化器**：`src/llm101/optim.py` 实现 SGD、AdamW 和断点状态保存；训练循环另做全局梯度裁剪。
5. **位置与注意力**：可学习绝对位置向量、因果掩码、多头自注意力及手写梯度，见[因果注意力](causal-attention.md)。
6. **Transformer block**：Pre-LN、残差连接、GELU 前馈网络和共享输入/输出嵌入组合为 decoder-only 模型，见[模型课程页](transformer-model.md)。
7. **预训练循环**：memory-mapped token 流、随机上下文窗口、next-token loss、验证 loss/perplexity、限时停止和断点续训，见[训练循环](training-loop.md)。
8. **生成**：贪心解码、temperature 和 top-k 采样，见[推理课程页](inference.md)。当前每步重新计算上下文，不使用 KV cache 或 top-p。
9. **KV cache**：在简单推理基线稳定后加入 cache，并比较输出一致性与速度。
10. **模型比较与发布**：对英文、中文和双语模型使用固定提示词与验证集；选满足标准的最短训练时长。实验过程见[限时实验方案](experiment-protocol.md)。

## 后续课程

指令微调等后训练、RoPE、GQA、MoE、生产级 tokenizer、beam search 和更完整的推理优化不属于基础模型主线，可作为后续扩展。
