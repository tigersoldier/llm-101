# 数据与许可

## 固定来源

下载脚本 `scripts/download_datasets.py` 将 Hugging Face 数据集修订号、文件大小和 SHA-256 固定在代码中；文件校验通过后才会安装到本地 `data/raw/`。

|语料|仓库与修订号|文件|大小（bytes）|SHA-256|
|---|---|---|---:|---|
|英文|[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64)，`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`|`TinyStories-train.txt`|1,924,281,556|`c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`|
|英文|同上|`TinyStories-valid.txt`|19,447,282|`94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`|
|中文|[`52AI/TinyStoriesZh`](https://huggingface.co/datasets/52AI/TinyStoriesZh/tree/96e998aab7ce3311f8a8497b2aa6ed5088630672)，`96e998aab7ce3311f8a8497b2aa6ed5088630672`|`TinyStories_all_data_zh_2M.tar.gz`|509,820,547|`d7ae2d3e99a67c2c26f735c6e2d53d30b141febbabfe1b689b2e89a6cbce593c`|

## 预处理与切分

```bash
uv run python scripts/download_datasets.py
uv run python scripts/prepare_training_data.py
```

英文使用数据源提供的 train/validation 文件。中文归档以流式方式读取，不按归档成员名解压到磁盘；对去除首尾空白后的故事文本计算 BLAKE2b-64，按固定 1% 阈值确定验证集。相同文本的重复故事会落在同一侧。已验证归档包含 1,994,291 篇非空故事；该切分得到 1,974,282 篇训练故事和 20,009 篇验证故事。

两种语言的初始字符表分别扫描完整训练切分；BPE 合并规则用固定 seed 的 512 篇 reservoir 样本学习，只训练一份联合词表。默认目标词表为 8,192；若完整字符表超过目标值，词表会自动扩至能够覆盖字符表。**Reservoir 样本仅用于 tokenizer 合并规则训练，语言模型本身仍使用完整训练语料，不预先下采样。**

预处理将每篇故事独立编码为 `BOS + 内容 + EOS`，保存为 little-endian int32 流。用联合词表分别编码英文与中文，再拼接为双语训练流；没有额外语言控制 token。具体 token 数、词表尺寸和故事数记录在本地 `data/processed/experiment_manifest.json`。

## 许可证与再发布

TinyStories 数据卡将英文故事标为由 GPT-3.5/GPT-4 生成，并标注 CDLA-Sharing-1.0。中文数据卡标注 MIT，但也说明它翻译自英文 TinyStories。中文翻译仓库的标签不能单独解决上游材料和翻译文本的再发布条件。

项目会保留仓库修订号、SHA-256、切分与转换方式及上游来源链。**目前模型权重再发布许可仍待人工复核；在复核完成前，不发布模型权重或原始语料。** 本页记录的是数据卡和已执行的工程步骤，不构成法律意见。

## 小型对照语料

[Global ASP 中文故事目录](https://github.com/global-asp/global-asp/tree/master/zh)包含少量中文故事，且不同故事许可不同；它可用于格式检查或补充对照，不能在未逐项筛选许可时整体使用。数据卡标注为公开，也不代表可以忽略具体许可和署名。
