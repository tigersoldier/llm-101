# 第 2 章进度：2.1–2.3 完成

## 本节新增了什么

- `src/llm101/baseline.py`：冻结的第 1 章 68 符号字符表、`character_encode`、`character_decode`、`hand_merge`、`build_report` 和 `context_error`。
- `scripts/show_tokenizer.py`：stage 1 的命令行输出，打印英文/中文基线、未知编号数和手写合并结果。
- `tests/test_show_tokenizer.py`：5 项测试，检查 stage 1 的英文/中文基线、stage 2 的合并结果和 stage 3 的 512 词表。
- `docs/assets/ch02/01-hand-merge.svg`：图 2.1-1，手写合并的工作过程。
- `docs/assets/ch02/01-token-budget.svg`：图 2.1-2，字符级编号与上下文 16 的对比。
- 2.2 新增 `merge_pair`、`count_pairs`、`learn_merges`，以及 `--stage 2` 和 4 张 2.2 图。
- 2.3 新增 `samples/` 教学语料、`BPETokenizer.__init__`/`train`/只读属性、`--stage 3`、`tests/test_samples.py` 与图 2.3-1/2.3-2。

`src/llm101/tokenizer.py` 的合并函数和词表训练已经实现；`tests/test_tokenizer.py` 现在 13 passed、2 failed，剩余失败属于 2.4 的 tokenizer 往返和 2.6 的词表接入。

## 怎么验证

```bash
cd ch02/end
uv sync --extra dev
uv run python scripts/show_tokenizer.py --stage 1
uv run python scripts/show_tokenizer.py --stage 2
uv run python scripts/show_tokenizer.py --stage 3
uv run pytest tests/test_show_tokenizer.py -q
uv run pytest tests/test_samples.py -q
```

预期：

- 英文 `the cat sat on the mat`：22 个字符、24 个编号，`model` 行是 `ValueError: sequence length 24 exceeds the model context of 16`。
- 中文 `小猫坐在垫子上`：7 个字符、9 个编号，其中 7 个 `<unk>`，`decoded` 是空字符串。
- 手写合并：英文 13 个符号，加 `<bos>`/`<eos>` 是 15。
- `tests/test_show_tokenizer.py`：5 passed。
- `uv run pytest tests/test_tokenizer.py -q`：13 passed、2 failed（2.4/2.6 的剩余契约）。
- `uv run pytest tests/test_samples.py -q`：3 passed。

2.3 完成时，`tests/test_tokenizer.py` 仍有 2 个失败；这不影响 2.3 的验收。

## 下一节入口

2.4 在已训练的词表上实现 `encode`/`decode`/`save`/`load`：按 rank 重放合并、处理未知字符、默认跳过特殊 token。
