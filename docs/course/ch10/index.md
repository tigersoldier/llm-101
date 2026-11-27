# 第 10 章 限时训练、checkpoint 与续训

> 状态：大纲（待写）。本章小节页面随写作上线。

## 本章目标

- 把训练循环包成按时长停止的实验脚本，记录 step、tokens seen 和墙钟时间。
- 在训练中做验证、写 `metrics.jsonl`、生成 `run_manifest.json`。
- 保存模型、优化器、随机数状态和训练状态；区分 `last.pt`、`best-model.pt`、`model-<h>h.pt`。
- 处理 SIGINT/SIGTERM，在当前 step 后安全保存并退出；支持 `--resume`。
- 调用第 9 章的 `generate`，用固定提示词产出 `samples-<h>h.json`。
- 定义实验协议和短跑验收，不要求在本章跑完 1/3/5 小时。

## 前置与交接

- 前置：第 9 章；需要 `generate`、最终配置、真实 token 流和优化器状态。
- 数据/配置：第 7 章语料与 tokenizer，第 8 章最终配置。
- 交接：`ch10/end` 是第 11 章起点；此时已经有可复现的训练 artifact。

## 小节清单

|小节|标题|这一节做什么|
|---|---|---|
|10.1|限时训练循环|按墙钟时间停止；step 与 tokens seen；训练/验证交替|
|10.2|验证、指标与日志|验证流；loss/perplexity；`metrics.jsonl`；语言覆盖；`run_manifest.json`|
|10.3|Checkpoint 的内容与格式|模型/优化器/RNG/训练状态；last/best/定时权重；原子写入|
|10.4|信号安全与续训|SIGINT/SIGTERM；当前 step 后保存；`--resume`；配置一致性|
|10.5|固定提示词样例生成|调用 `generate`；四条中英 prompt；`samples-<h>h.json`|
|10.6|实验协议与短跑验收|1/3/5 小时方案；短跑 smoke；artifact 清单；不要求全跑|

小节页面随写作上线；标题是路线图，正文在各自小节展开。

## 涉及代码

```text
ch10/
├── start/
│   ├── src/llm101/checkpoint.py   # 保存/加载空壳
│   └── scripts/train_model.py     # 限时训练 CLI 空壳
└── end/
    ├── src/llm101/checkpoint.py
    ├── scripts/train_model.py
    └── tests/
```

- 新增：`checkpoint.py`、`scripts/train_model.py`。
- 复用：第 6 章训练循环、第 7 章 `TokenStream`、第 8 章配置、第 9 章 `generate`。
- 不新增：新的模型结构、新的优化器、新的评估方法。

## 本章验收

- `cd ch10/end && uv run pytest -q` 全绿。
- 短跑能产出 `last.pt`、`best-model.pt`、`model-<h>h.pt`、`metrics.jsonl`、`run_manifest.json`、`samples-<h>h.json`。
- 续训后的下一步行为与不中断运行一致；配置或数据不匹配时拒绝恢复。
- 信号触发后保存并可恢复；样例 JSON 可复现。
- 实验协议写清楚 1/3/5 小时的关系和记录字段。

## 本章不做什么

- 不重讲优化器和训练循环。
- 不做评估方法论、许可发布（第 11 章）。
- 不要求本章跑完 9 小时；完整对照在第 11 章。
