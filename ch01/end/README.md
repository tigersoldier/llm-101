# 第 1 章完成态：看得见的 transformer

## 本章新增了什么

第 1 章从字符级 tokenizer 开始，搭出一个完整但很小的 decoder-only transformer：

- `tokenizer.py`：字符级 tokenizer，负责文本与编号互转；
- `layers.py`：`Linear`、`Embedding`、`LayerNorm`、`GELU`；
- `attention.py`：因果多头自注意力；
- `model.py`：`GPTConfig`、`TransformerBlock`、`TinyGPT`；
- `placeholder.py` 与 `scripts/show_placeholder.py`：按小节阶段展示同一条管线；
- `scripts/render_architecture.py`：生成各节的架构图。

模型只有 2 层、`d_model=16`、2 个头、`d_ff=32`，上下文长度 16，词表 68，共享输入/输出嵌入，共 5,824 个参数。它只做前向，不训练，权重是固定 seed 的随机初始化。

## 怎么验证

```bash
cd ch01/end
uv sync --extra dev
uv run pytest -q
uv run python scripts/show_placeholder.py --stage 7 "the cat sat"
```

预期：

- `pytest` 39 passed；
- 最后一行输出 `max 0.205 at 3`、`answer 3`、`character 'd'`。

`--stage N` 只使用第 N 节已有的步骤，`--stage 7` 是完整模型。本章没有 stage 8，也没有 web server、前端或逐层 trace。

## 下一章入口

`ch01/end/` 是第 2 章 `ch02/start/` 的来源。第 2 章只替换 `CharacterTokenizer` 为 BPE tokenizer，保持 `encode`、`decode`、`vocab_size` 等接口不变；第 3 章再给这些模块加 `backward`。`ch01/end/` 冻结后，后续章节在这份完成态上增量修改。
