# LLM 101：从零训练一个小语言模型

本课程通过一个持续演进的代码项目，解释语言模型如何把文本变成 token、如何用 next-token prediction 训练 Transformer，以及如何逐 token 生成文本。

最终目标是训练一个约 500 万参数、上下文长度 128 token 的 dense decoder-only Transformer。它不追求知识问答能力；目标是对短故事提示生成一句可读、连贯的续写。

## 课程原则

- 使用 PyTorch 做张量和矩阵运算，不使用 autograd。
- 手写线性层、Embedding、Softmax 与交叉熵、因果自注意力等代表性梯度；其他实现会提供代码并解释其作用。
- 用有限差分在小张量上检查手写梯度。
- 所有代码在本地 CPU 运行，先重算注意力上下文，再实现 KV cache 对比。
- 基础课程只做预训练；指令微调、MoE、RoPE、GQA、GPT 系列 tokenizer 等放在后续课程。

## 一个中英混合模型

课程只训练一个中英混合模型：联合词表、两份完整训练语料按 token 量混合、不加语言控制 token。因此提示文本是语言线索，输出也可能混用两种语言。完整语料纳入数据管道；实验按设定时限停止，并记录实际读过的 token 数。

模型是预训练的故事续写器，不是指令微调的助手：给定故事开头会继续写，但不会执行“讲个笑话”这类请求，输出也常有语法和连贯性错误。验证指标和固定提示词样例见[实验结果](course/results.md)。

课程顺序和数据来源见[课程路线](course/roadmap.md)与[数据与许可](datasets.md)。完整代码快照位于[`final/`](https://github.com/tigersoldier/llm-101/tree/master/final)，可独立安装和运行；以后每章也会有一个同样形式的章节目录。
