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
|one-hot|独热编码|词表大小维的向量，只有一个位置是 1，其余是 0|[1.3](course/ch01/1.3.md)|
|embedding|嵌入表、嵌入|embed 是“嵌入”；把离散 token 放进连续向量空间；表是 embedding table，行是 embedding vector|[1.3](course/ch01/1.3.md)|
|forward|前向、前向计算|从输入算到输出；PyTorch 模块的计算方法名；反向的 backward 第 3 章|[1.3](course/ch01/1.3.md)|
|backward|反向、反向传播|从输出或损失算回参数；第 3 章展开|[1.3](course/ch01/1.3.md)|
|weight|权重|模型里可调整的矩阵；Embedding 的表就是一张权重|[1.3](course/ch01/1.3.md)|
|output matrix|输出矩阵|把输出向量变成词表分数的矩阵，形状 `[16, 68]`；本节固定，1.7 可训练|[1.3](course/ch01/1.3.md)|
|score|分数|输出矩阵给每个词表编号的得分；最高分对应答案|[1.3](course/ch01/1.3.md)|
|position embedding|位置嵌入|为每个位置学一行可加向量，让模型知道顺序|[1.4](course/ch01/1.4.md)|
|position id|位置编号|序列中的位置 0..T-1，不是 token id|[1.4](course/ch01/1.4.md)|
|context length|上下文长度|模型一次最多处理的位置数；本章 16|[1.4](course/ch01/1.4.md)|
|absolute position|绝对位置|位置表按 0、1、… 的绝对编号取行|[1.4](course/ch01/1.4.md)|
|relative position|相对位置|用两个位置的差表示距离；相对方案更容易外推|[1.4](course/ch01/1.4.md)|
|sinusoidal position encoding|正弦位置编码|原论文用固定正弦函数生成位置向量，不训练|[1.4](course/ch01/1.4.md)|
|linear|线性层|`y = xW + b`|1.6|
|bias|偏置|线性层里的 `b`|1.6|
|batch|批|同时处理的多条输入，对应第一维 `B`|1.7|
|hidden state|隐藏状态|中间层输出的 `[B, T, C]` 数字|1.7|
|parameter|参数|模型里可调整的数字；`Parameter` 另留一块同形状缓冲|第 3 章|
|autograd|自动求导|PyTorch 的自动微分机制；本书不用|第 3 章|
