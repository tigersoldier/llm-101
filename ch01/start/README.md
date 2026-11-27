# 第 1 章 起点：脚手架

本章目标见 `../README.md`，小节计划见 `../sections/`。

## 先跑一次

```bash
cd ch01/start
uv sync --extra dev
uv run pytest -q
```

此时应当看到 **3 个失败**，分别对应：

|测试|在哪一节转绿|
|---|---|
|`test_forward_shapes`|1.8 组装整模型|
|`test_causal_mask_hides_future`|1.6 因果注意力|
|`test_context_limit`|1.4 位置与上下文上限|

失败信息里的 `TODO 1.x` 指出该去哪个小节找答案。不要先看 `../end/`，那是完成态。

## 阅读顺序

1. `src/llm101/tokenizer.py` —— 文本怎么变成 id
2. `src/llm101/parameters.py` —— 参数是什么
3. `src/llm101/layers.py` —— 四个零件
4. `src/llm101/attention.py` —— 注意力
5. `src/llm101/model.py` —— 拼成模型

## 目录

```text
ch01/start/
├── pyproject.toml
├── README.md
├── samples/stories.txt
├── src/llm101/{tokenizer,parameters,layers,attention,model}.py
└── tests/test_structure.py
```

接口（函数签名、返回形状）由脚手架固定，实现留空；后续章节的 `start/` 就是本章 `end/`，所以签名一旦定下就不要随意改。
