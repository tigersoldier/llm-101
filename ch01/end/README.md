# 第 1 章 完成态

本目录是第 1 章全部小节写完后应达到的状态，目前**只有这份说明**：内容由 `../sections/1.1` 到 `1.11` 依次填充。

## 完成时应包含

```text
ch01/end/
├── pyproject.toml          # 与 start 相同，可加 demo 依赖
├── README.md               # 本章小结（由 1.11 写）
├── samples/stories.txt
├── src/llm101/{tokenizer,parameters,layers,attention,model}.py
├── web/{app.py,schemas.py,static/{index.html,app.js,style.css}}
├── scripts/serve_demo.py
└── tests/{test_structure.py,test_tokenizer.py,test_layers.py,test_api.py}
```

## 与 start 的关系

- 同一批文件、同一批函数签名；差别只是实现与新增的测试、`web/`、`scripts/`。
- 所有 `TODO 1.x` 都被替换，`start` 里 3 个红测试在 `end` 中全绿。
- 下一章（第 2 章）的 `start/` 必须与这里逐字节一致，改动请回写到本章。

## 验收

```bash
cd ch01/end
uv sync --extra dev --extra demo
uv run pytest -q
uv run python scripts/serve_demo.py   # 打开 http://127.0.0.1:8000
```
