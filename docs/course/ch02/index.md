# 第 2 章 真正的 tokenizer

> 状态：大纲（待写）。本章小节页面随写作进度上线。

## 本章目标

- 把第 1 章的字符级 tokenizer 换成 BPE，并保持 `encode`、`decode`、`vocab_size`、`bos_id`、`eos_id` 五个接口不变。
- 从 Unicode 码点出发，理解相邻对合并、合并顺序和编码时重放。
- 用英中教学语料训练一个小词表，比较压缩率、OOV 和序列长度。
- 证明 tokenizer 替换不改变模型代码：固定 `GPTConfig`、seed 和手写 id 的 logits 回归。
- 明确 `<unk>`、`<bos>`、`<eos>` 的边界；本章不实现 padding，也不加入 `<pad>`。

## 前置与交接

- 前置：第 1 章；需要知道 token id、`CharacterTokenizer` 和 `TinyGPT` 的前向形状。
- 数据：`ch02/start/samples/` 的英中教学语料，供第 2–6 章使用。
- 交接：`ch02/end` 是第 3 章起点；模型结构不变，只换 tokenizer 和 `vocab_size`。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|2.1|从字符到子词：为什么要 BPE|字符级序列长、词表小；BPE 用常见片段换取更短序列|
|2.2|合并算法：从相邻对到词表|统计相邻对、按频率合并、合并顺序就是编码规则|
|2.3|训练词表：语料、特殊 token 与目标大小|从码点开始，用教学语料训练约 512 的词表；说明目标值与实际值|
|2.4|编码与解码：接口与未知字符|按 rank 重放合并、未知字符、特殊 token、往返与接口契约|
|2.5|中英混合与词表评估|中文无空格、联合词表、压缩率、OOV、序列长度与语言配比|
|2.6|接入模型与回归测试|更新 `GPTConfig.vocab_size`；固定 id logits 回归；tokenizer/模型词表一致性|

小节页面随写作进度上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch02/
├── start/
│   ├── samples/          # 英中教学语料
│   ├── src/llm101/       # 第 1 章代码 + BPE 空壳
│   └── tests/            # tokenizer 契约与红测试
└── end/
    ├── src/llm101/tokenizer.py
    └── tests/test_tokenizer.py
```

- 修改：`tokenizer.py`、`GPTConfig.vocab_size` 的调用点、tokenizer 测试。
- 复用：`model.py`、`layers.py`、`attention.py` 的前向不改。
- 不新增：训练、优化器、真实语料管道。

## 本章验收

- `cd ch02/end && uv run pytest -q` 全绿。
- 英文和中文样例都能往返；未知字符、特殊 token 的行为与第 1 章约定一致。
- 固定 `GPTConfig`、seed、手写 id 的 logits 与第 1 章回归一致。
- tokenizer 的 `vocab_size` 与模型配置一致；`TinyGPT` 在新词表下前向形状正确。
- 教学语料可以由固定文件或脚本重复得到。

## 本章不做什么

- 不接 TinyStories，不做 8192 词表（第 7 章）。
- 不引入 `<pad>`，不实现 padding 或 padding mask。
- 不训练、不反向传播、不优化。
