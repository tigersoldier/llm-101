# 第 7 章 真实语料与最终 tokenizer

> 状态：大纲（待写）。本章小节页面随写作上线。

## 本章目标

- 接入英文 TinyStories 和中文 TinyStoriesZh，固定修订号与 SHA-256。
- 完成下载、校验、切分、去重/过滤、故事边界和语言配比。
- 用完整训练切分训练联合 BPE 词表（8192），并评估压缩率与 OOV。
- 把故事编码成 int32 token 流，提供可随机采样的 `TokenStream`。
- 记录 manifest、token 数、词表尺寸和语言覆盖，保证实验可复现。
- 讨论数据许可：代码、语料和权重的再发布边界不同。

## 前置与交接

- 前置：第 6 章；需要 tokenizer、训练循环和 perplexity。
- 数据：`data/raw/` 的 TinyStories 英中语料；原始数据不进入 Git。
- 交接：`ch07/end` 是第 8 章起点；此时最终词表和 token 流已经确定。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|7.1|数据源、修订号与许可|TinyStories 英中来源；固定 revision；许可与再发布边界|
|7.2|下载、校验与目录布局|下载脚本；SHA-256；`data/raw/` 与 `data/processed/` 的分工|
|7.3|切分、清洗与故事边界|英文 train/valid；中文归档哈希切分；去重/过滤；`BOS`/`EOS`|
|7.4|联合 tokenizer：BPE 8192|初始字符表、reservoir 样本、合并训练；压缩率与 OOV|
|7.5|Token 流与随机窗口|int32 编码；mmap `TokenStream`；随机窗口跨故事；语言覆盖计数|
|7.6|Manifest、复现与小样本验收|token/故事/词表统计；实验 manifest；CI 小样本|

小节页面随写作上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch07/
├── start/
│   ├── src/llm101/corpus.py       # 数据管道空壳
│   └── scripts/                   # download/prepare 空壳
└── end/
    ├── src/llm101/corpus.py
    ├── src/llm101/training.py     # TokenStream
    ├── scripts/download_datasets.py
    ├── scripts/prepare_training_data.py
    └── tests/
```

- 新增：`corpus.py`、下载/预处理脚本、`TokenStream`。
- 复用：第 2 章的 BPE 训练/编码/解码。
- 不新增：最终模型尺寸、训练脚本、checkpoint。

## 本章验收

- `cd ch07/end && uv run pytest -q` 全绿。
- manifest 可复现；token 数、词表尺寸、故事数和语言覆盖被记录。
- 抽样解码回原始故事；中英文都能编码/解码。
- `TokenStream` 的批次形状、随机窗口和语言覆盖计数正确。
- 数据许可与“原始数据不进 Git”的边界写清楚。

## 本章不做什么

- 不跑训练（第 10 章）。
- 不放大模型（第 8 章）。
- 不发布原始语料或权重；许可结论明确前只保留本地数据。
