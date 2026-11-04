# 本地数据目录

原始数据放在 `raw/`，切分、tokenizer 和二进制 token 流放在 `processed/`；这些目录均被 Git 忽略。

```bash
uv run python scripts/download_datasets.py
uv run python scripts/prepare_training_data.py
```

下载脚本按 Hugging Face 修订号固定文件，并在安装到 `raw/` 前验证文件大小和 SHA-256。预处理再次检查源文件摘要，创建 `processed/experiment_manifest.json`，其中记录数据修订、切分、tokenizer 配置、故事数和 token 数。

中文归档按故事内容的 BLAKE2b 哈希确定性切分验证集；英文使用数据源自带的训练/验证文件。BPE 合并规则只从每种语言的固定 reservoir 样本学习，但初始 Unicode 字符表扫描完整训练集；语言模型训练不会预先下采样。

不要把原始语料、预处理数据或训练检查点提交到仓库。数据来源、哈希和许可说明见 [`docs/datasets.md`](../docs/datasets.md)。
