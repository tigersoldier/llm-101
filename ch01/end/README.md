# 第 1 章 完成态

本目录是第 1 章全部小节写完后应达到的状态。它从 `../start/` 整体复制而来，内容由 `../sections/1.1` 到 `1.9` 依次填充；全部写完后用三段式小结替换本说明。

## 完成时应包含

```text
ch01/end/
├── pyproject.toml          # 与 start 相同，无 demo 依赖
├── README.md               # 本章小结（由 1.9 写）
├── samples/stories.txt
├── src/llm101/{tokenizer,parameters,layers,attention,model,pipeline}.py
├── scripts/{show_placeholder.py,render_architecture.py}
└── tests/{test_structure.py,test_tokenizer.py,test_layers.py,test_placeholder.py}
```

## 与 start 的关系

- 同一批文件、同一批函数签名；差别只是实现与新增的测试、管线、`scripts/`。
- 所有 `TODO 1.x` 都被替换，`start` 里 3 个红测试在 `end` 中全绿。
- 下一章（第 2 章）的 `start/` 必须与这里逐字节一致，改动请回写到本章。

## 验收

```bash
cd ch01/end
uv sync --extra dev
uv run pytest -q
uv run python scripts/show_placeholder.py --stage 7 "the cat sat"
```
