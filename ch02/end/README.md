# 第 2 章进度：2.1 完成

## 本节新增了什么

- `src/llm101/baseline.py`：冻结的第 1 章 68 符号字符表、`character_encode`、`character_decode`、`hand_merge`、`build_report` 和 `context_error`。
- `scripts/show_tokenizer.py`：stage 1 的命令行输出，打印英文/中文基线、未知编号数和手写合并结果。
- `tests/test_show_tokenizer.py`：3 项测试，检查英文 24>16、中文 7 个 `<unk>`、手写合并 13 个符号。
- `docs/assets/ch02/01-hand-merge.svg`：图 2.1-1，手写合并的工作过程。
- `docs/assets/ch02/01-token-budget.svg`：图 2.1-2，字符级编号与上下文 16 的对比。

`src/llm101/tokenizer.py` 仍是 BPE 空壳；`tests/test_tokenizer.py` 的 3 个红测试要到 2.2–2.4 才逐个转绿，2.6 再把它们纳入整章验收。

## 怎么验证

```bash
cd ch02/end
uv sync --extra dev
uv run python scripts/show_tokenizer.py --stage 1
uv run pytest tests/test_show_tokenizer.py -q
```

预期：

- 英文 `the cat sat on the mat`：22 个字符、24 个编号，`model` 行是 `ValueError: sequence length 24 exceeds the model context of 16`。
- 中文 `小猫坐在垫子上`：7 个字符、9 个编号，其中 7 个 `<unk>`，`decoded` 是空字符串。
- 手写合并：英文 13 个符号，加 `<bos>`/`<eos>` 是 15。
- `tests/test_show_tokenizer.py`：3 passed。

2.1 完成时，`uv run pytest -q` 仍会看到 `tests/test_tokenizer.py` 的 3 个 BPE 测试失败；这不影响 2.1 的验收。

## 下一节入口

2.2 在 `src/llm101/tokenizer.py` 里实现 `merge_pair`、`count_pairs` 和 `learn_merges`，并把 `--stage 2` 加到 `show_tokenizer.py`。教学语料 `samples/` 在 2.3 建立；本章所有数字都以实际运行为准。
