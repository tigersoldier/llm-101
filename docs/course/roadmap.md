# 课程目录

课程沿快照链推进：第 1 章先搭出完整但很小的 transformer，之后每章加一层细节，最后用限时实验训练一个中英混合模型。每章有独立大纲页；小节页面随写作进度上线。

全书分四段：

- 第 1–2 章：前向与 tokenizer。
- 第 3–6 章：梯度、优化器与训练循环。
- 第 7–8 章：真实语料与缩放。
- 第 9–11 章：生成、限时训练与评估。

|章|标题|状态|这一章做什么|
|---|---|---|---|
|1|[看得见的 transformer](ch01/index.md)|已完成|完整前向、字符级 tokenizer、SVG 图与命令行输出；权重随机，不训练|
|2|[真正的 tokenizer](ch02/index.md)|写作中|2.1–2.3 已上线；BPE、英中码点、联合词表、压缩率与 OOV；建立教学语料|
|3|[参数、层与手写梯度](ch03/index.md)|大纲|`Parameter`、各层反向、softmax 交叉熵、next-token 目标与有限差分|
|4|[注意力、block 与整模型反向](ch04/index.md)|大纲|softmax 雅可比、因果 mask、多头、残差/Pre-LN、绑定权重梯度|
|5|[优化器](ch05/index.md)|大纲|SGD、AdamW、偏差修正、weight decay、梯度裁剪、状态保存|
|6|[训练循环](ch06/index.md)|大纲|小模型 + 教学语料，loss/perplexity、诊断指标、过拟合 sanity check|
|7|[真实语料与最终 tokenizer](ch07/index.md)|大纲|TinyStories 中英、哈希切分、去重/配比、BPE 8192、token 流、许可|
|8|[缩放与预算](ch08/index.md)|大纲|最终配置 256/4/4/1024/128、5,289,472 参数、FLOPs/激活内存、初始化|
|9|[推理与生成](ch09/index.md)|大纲|贪心、temperature、top-k、EOS、上下文截断；不做 KV cache|
|10|[限时训练、checkpoint 与续训](ch10/index.md)|大纲|按时长停止、信号安全保存、恢复、metrics/样例 artifact|
|11|[实验、评估与发布](ch11/index.md)|大纲|1/3/5 小时对照、固定验证窗口、分语言 PPL、样例与局限、许可、后续方向|

每章大纲页列出本章目标、前置与交接、小节清单、涉及代码、验收和“不做什么”。旧版单页（BPE、手写梯度、因果注意力、训练循环、推理、实验等）会在对应章节上线后被替换；在当前阶段仍可当作参考。

## 不在基础主线

KV cache、top-p、beam search、指令微调/RLHF、RoPE、GQA、MoE、GPU/分布式训练和生产级 tokenizer 作为后续扩展；第 9 章会解释这些取舍。基础课程只做预训练，最终产物是一个中英混合的故事续写模型，不是指令微调的助手。
