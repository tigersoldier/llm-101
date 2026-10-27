# 课程路线

课程逐步完善同一个代码项目；不先做独立 bigram 模型。数学按需讲解，假设读者会基本 Python。

1. **文本与 next-token prediction**：训练样本如何从故事文本构造；训练集、验证集和 token 序列。数据加载和样本构造仍待实现。
2. **教学版 BPE**：从 Unicode 文本学习频繁合并，自己实现训练、编码与解码；当前实现见 [BPE 课程页](bpe.md)，GPT tokenizer 的工程细节留到进阶课。
3. **损失与梯度**：线性层、Embedding、Softmax＋交叉熵的前向和显式反向传播；[课程页](manual-gradients.md)介绍公式，有限差分测试核对梯度。
4. **优化器**：`src/llm101/optim.py` 已实现 SGD 和精简 AdamW；训练策略与梯度裁剪仍待展开。
5. **位置与注意力**：可学习绝对位置向量、因果掩码、多头自注意力及其手写梯度已实现，见[因果注意力课程页](causal-attention.md)。
6. **Transformer block**：Pre-LN、残差连接、GELU 前馈网络与共享输入/输出嵌入已组合为小型 decoder-only 模型，见[模型课程页](transformer-model.md)。
7. **预训练循环**：批次、next-token loss、验证集 loss/perplexity、检查点与可复现配置。
8. **生成**：贪心解码、temperature、top-k/top-p；介绍 min-p，主线不实现 beam search。
9. **KV cache**：先用每步重算的简单推理版本，再加入 KV cache 并对比输出与运行时间。
10. **模型比较与发布**：固定提示词评估中、英、双语版本；选最短达标检查点并发布最终权重。

## 后续课程

指令微调等后训练、RoPE、GQA、MoE、GPT 系列 tokenizer 和更完整的推理优化不属于基础模型主线，可作为后续扩展。
