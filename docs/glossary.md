# 术语表

本书术语按首次出现顺序列出。中文一栏给出常见译法；“不译”表示没有公认中文译名，保留英文。最后一列链接到首次介绍或实现该术语的小节；本表随章节写作补充。

|术语|中文|说明|详细介绍|
|---|---|---|---|
|model|模型|把输入加工成输出的程序；本书指这个很小的语言模型|[1.1](course/ch01/1.1.md)|
|transformer|不译|第 1 章要搭的模型结构；没有公认中文译名，保留英文|[第 1 章总览](course/ch01/index.md)（1.7 组装）|
|tensor|张量|多维数组；每一维长度固定，二维就是矩阵|[1.1](course/ch01/1.1.md)|
|shape|形状|张量每一维的长度，如 `[B, T, C]`|[1.1](course/ch01/1.1.md)|
|argmax|取最大值位置|给一排数，返回最大值的下标；并列取第一个|[1.1](course/ch01/1.1.md)|
|tokenizer|分词器、标记器、词元切分器|把文本切成 token，并把 token 映射成编号|[1.2](course/ch01/1.2.md)|
|token|词元、标记|切分后的最小单位；本章按单个字符切|[1.2](course/ch01/1.2.md)|
|vocabulary|词表、词汇表|所有可选 token 的清单|[1.2](course/ch01/1.2.md)|
|token id|词元编号|token 在词表里的位置，简称编号|[1.2](course/ch01/1.2.md)|
|special token|特殊词元、特殊符号|有固定用途的 token，如 `<unk>`、`<bos>`、`<eos>`、`<pad>`|[1.2](course/ch01/1.2.md)|
|`<unk>`|未知词元|unknown 的缩写；词表外字符的替身|[1.2](course/ch01/1.2.md)|
|`<bos>`|起始词元|beginning of sequence 的缩写；标出输入的开头|[1.2](course/ch01/1.2.md)|
|`<eos>`|结束词元|end of sequence 的缩写；标出输入的结尾|[1.2](course/ch01/1.2.md)|
|`<pad>`|填充词元|padding 的缩写；把短的输入补齐；本章不用|[1.2](course/ch01/1.2.md)|
|parameter|参数|模型里可调整的数字；`Parameter` 另留一块同形状缓冲|[1.3](course/ch01/1.3.md)|
|autograd|自动求导|PyTorch 的自动微分机制；本书不用|[1.3](course/ch01/1.3.md)|
|batch|批|同时处理的多条输入，对应第一维 `B`|[1.3](course/ch01/1.3.md)|
|hidden state|隐藏状态|中间层输出的 `[B, T, C]` 数字|[1.3](course/ch01/1.3.md)|
|linear|线性层|`y = xW + b`|[1.3](course/ch01/1.3.md)|
|weight|权重|线性层里的 `W`|[1.3](course/ch01/1.3.md)|
|bias|偏置|线性层里的 `b`|[1.3](course/ch01/1.3.md)|
|embedding|嵌入表|按编号取行的查表|[1.3](course/ch01/1.3.md)|
