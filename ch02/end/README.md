# 第 2 章进度：2.1–2.2 完成

## 本节新增了什么

- `src/llm101/baseline.py`：冻结的第 1 章 68 符号字符表、`character_encode`、`character_decode`、`hand_merge`、`build_report` 和 `context_error`。
- `scripts/show_tokenizer.py`：stage 1 的命令行输出，打印英文/中文基线、未知编号数和手写合并结果。
- `tests/test_show_tokenizer.py`：4 项测试，检查 stage 1 的英文/中文基线和 stage 2 的合并结果。
- `docs/assets/ch02/01-hand-merge.svg`：图 2.1-1，手写合并的工作过程。
- `docs/assets/ch02/01-token-budget.svg`：图 2.1-2，字符级编号与上下文 16 的对比。
- 2.2 新增 `merge_pair`、`count_pairs`、`learn_merges`，以及 `--stage 2` 和 4 张 2.2 图。

`src/llm101/tokenizer.py` 的合并函数已经实现；`tests/test_tokenizer.py` 现在 7 passed、2 failed，剩余失败属于 2.4 的 tokenizer 往返和 2.6 的词表接入。

## 怎么验证

```bash
cd ch02/end
uv sync --extra dev
uv run python scripts/show_tokenizer.py --stage 1
uv run python scripts/show_tokenizer.py --stage 2
uv run pytest tests/test_show_tokenizer.py -q
```

预期：

- 英文 `the cat sat on the mat`：22 个字符、24 个编号，`model` 行是 `ValueError: sequence length 24 exceeds the model context of 16`。
- 中文 `小猫坐在垫子上`：7 个字符、9 个编号，其中 7 个 `<unk>`，`decoded` 是空字符串。
- 手写合并：英文 13 个符号，加 `<bos>`/`<eos>` 是 15。
- `tests/test_show_tokenizer.py`：4 passed。
- `uv run pytest tests/test_tokenizer.py -q`：7 passed、2 failed（2.4/2.6 的剩余契约）。

2.2 完成时，`tests/test_tokenizer.py` 仍有 2 个失败；这不影响 2.2 的验收。

## 下一节入口

2.3 在 2.2 的合并列表之上建立词表：收集教学语料的码点、加入三个特殊 token、按目标大小训练。`samples/` 在 2.3 建立；2.2 的 10 条演示短句只用于看算法，不是最终语料。
