# 课程路线

课程沿快照链推进：第 1 章先搭出完整但很小的 transformer，之后每章加一层细节，最后用限时实验训练一个中英混合模型。各章页面随写作上线；已上线的第 1 章见[第 1 章总览](ch01/index.md)。

1. **看得见的 transformer**：完整前向、字符级 tokenizer、SVG 图与命令行输出；权重随机，不训练。
2. **真正的 tokenizer**：教学版 BPE、英中 Unicode 例子、联合词表、压缩率与 OOV；建立教学语料。
3. **参数、层与手写梯度**：`Parameter`、各层反向、softmax 交叉熵、next-token 目标与有限差分。
4. **注意力、block 与整模型反向**：softmax 雅可比、因果 mask、多头、残差/Pre-LN、绑定权重梯度。
5. **优化器**：SGD、AdamW、偏差修正、weight decay、梯度裁剪、状态保存。
6. **训练循环**：小模型 + 教学语料，loss/perplexity、诊断指标、过拟合 sanity check。
7. **真实语料与最终 tokenizer**：TinyStories 中英、哈希切分、去重/配比、BPE 8192、token 流、许可。
8. **缩放与预算**：最终配置 256/4/4/1024/128、5,289,472 参数、FLOPs/激活内存、初始化。
9. **推理与生成**：贪心、temperature、top-k、EOS、上下文截断；不做 KV cache。
10. **限时训练、checkpoint 与续训**：按时长停止、信号安全保存、恢复、metrics/样例 artifact。
11. **实验、评估与发布**：1/3/5 小时对照、固定验证窗口、分语言 PPL、样例与局限、许可、后续方向。

旧版单页（BPE、手写梯度、因果注意力、训练循环、推理、实验等）会在对应章节上线后被替换；在当前阶段仍可当作参考。

## 不在基础主线

KV cache、top-p、beam search、指令微调/RLHF、RoPE、GQA、MoE、GPU/分布式训练和生产级 tokenizer 作为后续扩展；第 9 章会解释这些取舍。基础课程只做预训练，最终产物是一个中英混合的故事续写模型，不是指令微调的助手。
