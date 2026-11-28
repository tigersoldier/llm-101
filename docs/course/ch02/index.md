# 第 2 章 真正的 tokenizer

> 状态：已规划。小节 2.1–2.6 的计划在 `ch02/sections/`；正文与代码待写。

第 1 章用字符级 tokenizer 把文字变成编号：一个字符一个编号，`the cat sat` 加上 `<bos>`/`<eos>` 是 13 个编号。第 2 章把英文样例延长一句：`the cat sat on the mat` 是 22 个字符，加 `<bos>`/`<eos>` 后 24 个编号，模型根本读不完。第 1 章的字符表还是固定的 68 个英文符号，中文 `小猫坐在垫子上` 的 7 个字全部变成 `<unk>`。

第 2 章把切分方式换成 BPE（Byte Pair Encoding，字节对编码）：从 Unicode 码点出发，反复合并语料中最常见的相邻符号对，用一份约 512 个 token 的联合词表同时覆盖英文和中文。`encode`、`decode`、`vocab_size`、`bos_id`、`eos_id` 五个接口保持不变，模型结构完全不动，只把 `GPTConfig.vocab_size` 换成新词表大小。模型仍然只做前向，权重是固定 seed 的随机初始化，输出没有语义。

每节都保留同一条可运行管线：`show_tokenizer.py --stage 1` 到 `--stage 6` 随 2.1–2.6 逐步替换占位组件。2.1 先跑第 1 章基线并用手写合并演示方向；2.2 实现相邻对合并；2.3 建词表；2.4 完成文本 ↔ 编号；2.5 做中英评估；2.6 接回模型并做固定 id 回归。

## 本章目标

- 用 BPE 替换字符级 tokenizer，保持五个接口和模型代码不变。
- 从 Unicode 码点出发，解释相邻对、合并频率、合并顺序和编码时的重放。
- 用英中教学语料训练一份约 512 个 token 的联合词表，比较压缩率、OOV 和序列长度。
- 明确 `<unk>`、`<bos>`、`<eos>` 的边界；本章不加入 `<pad>`。
- 用固定 `GPTConfig`、seed 和手写 id 的 logits 回归，证明 tokenizer 替换没有改模型代码。

## 前置与交接

- 前置：第 1 章；需要知道 token id、`CharacterTokenizer` 的五个接口和 `TinyGPT` 的前向形状。
- 数据：`ch02/start/samples/` 的英中教学语料，自写、许可明确，供第 2–6 章使用。
- 交接：`ch02/end` 是第 3 章起点；tokenizer 与 `samples/` 冻结，模型结构不变，只改 `GPTConfig.vocab_size`。

## 本章流程

固定示例有两条：英文 `the cat sat on the mat`（22 个字符，加 `<bos>`/`<eos>` 后 24 个编号，超过上下文 16）和中文 `小猫坐在垫子上`（7 个字符；第 1 章编码结果是 7 个 `<unk>`）。英文是第 1 章 `the cat sat` 的延长，两句意思对应；英文样例本身承担长度反例，中文样例承担覆盖反例，不再单独设长句。正文里的数字一律来自实际运行。

|小节|本节新增/换上的步骤|节末示例输出|
|---|---|---|
|2.1|第 1 章基线 + 手写片段合并占位|英文 24>16；中文 7 个 `<unk>`；手写合并后 13 个符号（加特殊 token 15）|
|2.2|`merge_pair`、`count_pairs`、`learn_merges`|实际得到的合并（最多 16 条）；英文/中文符号序列|
|2.3|`BPETokenizer.train`、词表、特殊 token|实际词表大小；英文/中文 ids|
|2.4|`encode`、`decode`、`save`、`load`|往返文本；未知字符的 `<unk>` 行为|
|2.5|`tokenizer_report.py`|字符数、token 数、压缩率、OOV|
|2.6|`GPTConfig.vocab_size` + 固定 id 回归|`[T, V]` logits、argmax；回归一致|

2.1 只建立问题和基线，不引入 BPE 名称；2.2 才介绍并实现合并；2.3 把符号串变成编号；2.4 完成文本与编号互转；2.5 做评估；2.6 接回模型并收束全章。模型架构不变，本章不重画 `architecture-N.svg`。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|2.1|起点与本章交付物|跑通第 1 章基线，看到 24>16、中文全 `<unk>`；明确五个接口和三个红测试|
|2.2|子词与相邻对合并（BPE）|统计相邻对、按频率合并；合并顺序就是编码规则|
|2.3|训练词表：语料、特殊 token 与目标大小|特殊 token、初始码点、合并结果怎样组成约 512 词表|
|2.4|编码与解码：接口与未知字符|按顺序重放合并、处理未知字符、保存/加载与接口契约|
|2.5|中英混合与词表评估|联合词表、压缩率、OOV、序列长度与语言差异|
|2.6|接入模型、回归测试与交接|只改 `vocab_size`，用固定 id logits 回归证明模型没变|

小节页面随写作进度上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch02/
├── README.md          # 章节总览、小节清单、写作流程
├── sections/          # 6 个小节的计划
├── start/             # （待建）BPE 空壳 + 3 个红测试 + 教学语料
└── end/               # （待建）完成态：BPE、报告脚本、回归测试
```

- 修改：`tokenizer.py`；新增 `scripts/show_tokenizer.py`、`scripts/tokenizer_report.py`、`samples/`。
- 复用：`model.py`、`layers.py`、`attention.py` 的前向不改。
- 不新增：训练、优化器、真实语料管道。

## 本章验收

- `cd ch02/end && uv run pytest -q` 全部通过（实际通过数在完成时记录）。
- 英文和中文固定样例都能往返；未知字符、特殊 token 的行为与第 1 章约定一致。
- 固定 `GPTConfig`（68）、seed 和手写 id 的 logits 与 `ch01/end` 回归一致。
- tokenizer 的 `vocab_size` 与模型配置一致；`TinyGPT` 在新词表下前向形状为 `[1, T, V]`。
- 教学语料由固定文件重复得到；`samples/README.md` 记录行数、字符数和许可。
- `show_tokenizer.py --stage 1` 到 `--stage 6` 都能跑；`uv run mkdocs build --strict` 通过。

## 本章不做什么

- 不接 TinyStories，不做 8192 词表（第 7 章）。
- 不引入 `<pad>`，不实现 padding 或 padding mask。
- 不训练模型、不反向传播、不引入优化器。
- 不做字节级 BPE、正则预切分或 GPT 系列 tokenizer。
- 不重画模型架构图；模型代码不改，只改 `GPTConfig.vocab_size`。
