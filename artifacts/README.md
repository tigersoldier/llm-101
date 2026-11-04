# 本地模型产物

训练输出保存在 `artifacts/runs/<corpus>-<hours>h-seed<seed>/`，并被 Git 忽略：

- `last.pt`：可续训 checkpoint，包含模型、AdamW 状态、计数器和 batch sampler 随机数状态。
- `best-model.pt`：当前运行中混合/验证 loss 最低的权重。
- `model-<hours>h.pt`：指定时长结束时的模型权重。
- `metrics.jsonl`、`run_manifest.json`、`samples-<hours>h.json`：训练指标、配置和固定提示词生成结果。

时长实验中的 checkpoint 用于评估；每种语言只把最短且满足质量标准的最终权重作为正式模型。发布前必须复核 TinyStories 上游许可、中文翻译来源链及模型输出。若许可条件不清楚，不公开权重。
