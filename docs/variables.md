# 变量表

本书的短变量分两类：**形状/数学符号**（$T$、$C$、$B$ 等）和**代码参数名**（`in_features`、`out_features` 等）。同一个量在不同层可能有多个名字；这张表给出对照，正文每节第一次用到短变量时也会再提醒一次。

## 形状与短变量

|符号|含义|本章取值|代码名|首次出现|
|---|---|---|---|---|
|$T$|序列长度，也就是位置数|13|`token_ids.shape[-1]`|[1.3](course/ch01/1.3.md)|
|$C$|每个位置的宽度，也就是特征维|16|`d_model`、`WIDTH`|[1.3](course/ch01/1.3.md)|
|$B$|批次大小，同时处理的序列条数|1、2、4|张量第 0 维|[1.4](course/ch01/1.4.md)|
|$H$|注意力头数|2|`n_heads`|[1.6](course/ch01/1.6.md)|
|$D$|每个头的宽度，`D = C / H`|8|`head_dim`|[1.6](course/ch01/1.6.md)|
|`vocab`|词表大小|68|`vocab_size`、`VOCAB_SIZE`|[1.2](course/ch01/1.2.md)|
|`max_seq_len`|上下文长度，位置表行数|16|`GPTConfig.max_seq_len`|[1.4](course/ch01/1.4.md)|
|`n_layers`|TransformerBlock 的层数|2|`GPTConfig.n_layers`|[1.8](course/ch01/1.8.md)|
|$d_{\text{ff}}$|MLP 隐藏层宽度|32|`d_ff`|[1.7](course/ch01/1.7.md)|

## 层与张量

|符号|含义|形状|首次出现|
|---|---|---|---|
|$X$|`Linear` 的输入|`[..., in_features]`|[1.6](course/ch01/1.6.md)|
|$W$|`Linear` 的权重|`[in_features, out_features]`|[1.6](course/ch01/1.6.md)|
|$b$|`Linear` 的偏置|`[out_features]`|[1.6](course/ch01/1.6.md)|
|$Y$|`Linear` 的输出|`[..., out_features]`|[1.6](course/ch01/1.6.md)|
|$o$|1.3 的输出向量|`[C]`|[1.3](course/ch01/1.3.md)|
|$W_{\text{out}}$|1.3 的固定输出矩阵|`[C, vocab]`|[1.3](course/ch01/1.3.md)|
|$E$|词嵌入表 / token embedding table（代码里叫 `token_embedding.weight`）|`[vocab, C]`|[1.3](course/ch01/1.3.md)|
|$scores$|1.3 的 68 个分数|`[vocab]`|[1.3](course/ch01/1.3.md)|
|$Q$、$K$、$V$|注意力的查询/键/值|`[T, C]`|[1.6](course/ch01/1.6.md)|
|$h_t$|位置 $t$ 的 token + 位置向量|`[C]`|[1.4](course/ch01/1.4.md)|
|$Z$|最终隐藏状态，也就是最终 LayerNorm 的输出（代码里叫 `hidden`）|`[B, T, C]`|[1.8](course/ch01/1.8.md)|
|`logits`|模型最后输出的词表分数|`[B, T, vocab]`|[1.8](course/ch01/1.8.md)|

## $C$ 与 `in_features` 不是同一件事

$C$ 是**特征维**：一个位置的向量有多少个数字，本章是 16，代码里叫 `d_model` 或 `WIDTH`。

`in_features` / `out_features` 是 **`Linear` 自己的输入/输出宽度**，不是新的特征维：

- Q/K/V 投影用 `Linear(C, C)`，所以 `in_features = out_features = C = 16`；
- 输出层用 `Linear(C, vocab)`，所以 `in_features = C = 16`，`out_features = vocab = 68`。

因此正文写“$C$（特征维）”时，不要用 `in_features` 代替；只有描述某个 `Linear` 的入/出宽度时，才用 `in_features` / `out_features`。

## 同一个量的别名

- $C$ = `d_model` = `WIDTH`（16）
- `vocab` = `vocab_size` = `VOCAB_SIZE`（68）
- $B$ = 张量第 0 维
- $H$ = `n_heads`（2）
- $D$ = `head_dim` = `C / H`（8）
