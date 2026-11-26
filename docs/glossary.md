# 术语表

本书术语按首次出现顺序列出。中文一栏给出常见译法；“不译”表示没有公认中文译名，保留英文。最后一列给出首次介绍或实现该术语的小节；该小节已有页面时给链接，尚未写的先标节号。本表随章节写作补充。

|术语|中文|说明|详细介绍|
|---|---|---|---|
|model|模型|把输入加工成输出的程序；本书指这个很小的语言模型|[1.1](course/ch01/1.1.md)|
|transformer|不译|第 1 章要搭的模型结构；没有公认中文译名，保留英文|[第 1 章总览](course/ch01/index.md)（1.7 组装）|
|tokenizer|分词器、标记器、词元切分器|把文本切成 token，并把 token 映射成编号|[1.2](course/ch01/1.2.md)|
|token|词元、标记|切分后的最小单位；本章按单个字符切|[1.2](course/ch01/1.2.md)|
|vocabulary|词表、词汇表|所有可选 token 的清单|[1.2](course/ch01/1.2.md)|
|token id|词元编号|token 在词表里的位置，简称编号|[1.2](course/ch01/1.2.md)|
|special token|特殊词元、特殊符号|有固定用途的 token，如 `<unk>`、`<bos>`、`<eos>`|[1.2](course/ch01/1.2.md)|
|`<unk>`|未知词元|unknown 的缩写；词表外字符的替身|[1.2](course/ch01/1.2.md)|
|`<bos>`|起始词元|beginning of sequence 的缩写；标出输入的开头|[1.2](course/ch01/1.2.md)|
|`<eos>`|结束词元|end of sequence 的缩写；标出输入的结尾|[1.2](course/ch01/1.2.md)|
|tensor|张量|多维数组；每一维长度固定，二维就是矩阵|[1.3](course/ch01/1.3.md)|
|shape|形状|张量每一维的长度，如 `[T, C]`|[1.3](course/ch01/1.3.md)|
|feature|特征|特征向量的一维；本章 `C = 16`|[1.3](course/ch01/1.3.md)|
|feature vector|特征向量|一个位置的 `C` 个数字组成的一行；`[T, C]` 的每一行|[1.3](course/ch01/1.3.md)|
|one-hot|独热编码|词表大小维的向量，只有一个位置是 1，其余是 0|[1.3](course/ch01/1.3.md)|
|embedding|嵌入表、嵌入|embed 是“嵌入”；把离散 token 放进连续向量空间；表是 embedding table，行是 embedding vector|[1.3](course/ch01/1.3.md)|
|forward|前向、前向计算|从输入算到输出；PyTorch 模块的计算方法名；反向的 backward 第 3 章|[1.3](course/ch01/1.3.md)|
|backward|反向、反向传播|从输出或损失算回参数；第 3 章展开|[1.3](course/ch01/1.3.md)|
|weight|权重|模型里可调整的矩阵；Embedding 的表就是一张权重|[1.3](course/ch01/1.3.md)|
|output matrix|输出矩阵|把输出向量变成词表分数的矩阵，形状 `[16, 68]`；本节固定，1.7 可训练|[1.3](course/ch01/1.3.md)|
|score|分数|输出矩阵给每个词表编号的得分；最高分对应答案|[1.3](course/ch01/1.3.md)|
|position embedding|位置嵌入|为每个位置学一行可加向量，让模型知道顺序|[1.4](course/ch01/1.4.md)|
|position id|位置编号|序列中的位置 0..T-1，不是 token id|[1.4](course/ch01/1.4.md)|
|padding|填充|用填充符把不同长度的序列补齐到同一长度|[1.4](course/ch01/1.4.md)|
|context length|上下文长度|模型一次最多处理的位置数；本章 16|[1.4](course/ch01/1.4.md)|
|absolute position|绝对位置|位置表按 0、1、… 的绝对编号取行|[1.4](course/ch01/1.4.md)|
|relative position|相对位置|用两个位置的差表示距离；相对方案更容易外推|[1.4](course/ch01/1.4.md)|
|sinusoidal position encoding|正弦位置编码|原论文用固定正弦函数生成位置向量，不训练|[1.4](course/ch01/1.4.md)|
|LayerNorm|层归一化|Layer Normalization；对每个位置最后一维求均值/方差，再缩放平移|[1.5](course/ch01/1.5.md)|
|GELU|高斯误差线性单元|Gaussian Error Linear Unit；用 erf 或 tanh 近似的平滑激活|[1.5](course/ch01/1.5.md)|
|activation|激活函数|逐元素的非线性函数，如 ReLU、GELU|[1.5](course/ch01/1.5.md)|
|gain|增益|LayerNorm 的缩放参数，初始为 1|[1.5](course/ch01/1.5.md)|
|bias|偏置|归一化/线性层里的可学习偏移|[1.5](course/ch01/1.5.md)|
|Pre-LN|前置层归一化|先 LayerNorm 再进子层；与 Post-LN 相对|1.7|
|Post-LN|后置层归一化|先子层和残差再 LayerNorm|1.7|
|subLayer|子层|block 里的注意力或 MLP 分支；1.7 组装|[1.5](course/ch01/1.5.md)|
|ReLU|线性整流单元|Rectified Linear Unit；max(0,x)|[1.5](course/ch01/1.5.md)|
|BatchNorm|批归一化|Batch Normalization；沿批次维求统计量|[1.5](course/ch01/1.5.md)|
|sigmoid|S 形函数|把实数压到 0–1 的平滑函数|[1.5](course/ch01/1.5.md)|
|tanh|双曲正切|把实数压到 -1–1 的平滑函数|[1.5](course/ch01/1.5.md)|
|ELU|指数线性单元|Exponential Linear Unit；负数区平滑的 ReLU 变体|[1.5](course/ch01/1.5.md)|
|warmup|预热|训练初期把学习率从小逐渐调大|[1.5](course/ch01/1.5.md)|
|erf|误差函数|高斯积分的误差函数；GELU 的精确式用它|[1.5](course/ch01/1.5.md)|
|internal covariate shift|内部协变量偏移|深层网络各层输入分布随训练变化|[1.5](course/ch01/1.5.md)|
|vanishing gradient|梯度消失|深层网络反传时梯度逐层变小|[1.5](course/ch01/1.5.md)|
|running statistics|滑动统计量|BatchNorm 在训练中累积、推理时使用的均值/方差|[1.5](course/ch01/1.5.md)|
|epsilon|小常数|加在方差上防止除零；本章 1e-5|[1.5](course/ch01/1.5.md)|
|linear|线性层|`y = xW + b`|1.6|
|attention|注意力|让每个位置按相关程度加权读取其他位置|[1.6](course/ch01/1.6.md)|
|self-attention|自注意力|query/key/value 都来自同一串输入的注意力|[1.6](course/ch01/1.6.md)|
|query|查询|当前位置发出的“我想找什么”向量|[1.6](course/ch01/1.6.md)|
|key|键|每个位置提供的“我有什么标签”向量|[1.6](course/ch01/1.6.md)|
|value|值|每个位置实际被读取的向量|[1.6](course/ch01/1.6.md)|
|softmax|归一化指数|把一组分数变成和为 1 的权重|[1.6](course/ch01/1.6.md)|
|causal mask|因果掩码|把未来位置的分数填成 -inf，保证只能看前面|[1.6](course/ch01/1.6.md)|
|multi-head|多头|把 C 维拆成 H 个头并行做注意力|[1.6](course/ch01/1.6.md)|
|head|头|多头里的一个 D 维注意力子空间|[1.6](course/ch01/1.6.md)|
|scaled dot-product|缩放点积|QK^T/sqrt(D) 的打分方式|[1.6](course/ch01/1.6.md)|
|batch|批次|同时处理的一组输入；batch size 是批次大小，第一维 `B` 叫批次维；也说一批、批量处理|1.7|
|hidden state|隐藏状态|中间层输出的 `[B, T, C]` 数字|1.7|
|parameter|参数|模型里可调整的数字；`Parameter` 另留一块同形状缓冲|第 3 章|
|autograd|自动求导|PyTorch 的自动微分机制；本书不用|第 3 章|
