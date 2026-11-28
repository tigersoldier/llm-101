# 第 2 章 起点：新 tokenizer 的空壳

本章要把第 1 章的字符级 tokenizer 换成从语料学习词表的 BPE tokenizer，同时保留五个接口：`encode`、`decode`、`vocab_size`、`bos_id`、`eos_id`。模型结构不变，只把 `GPTConfig.vocab_size` 换成新词表大小。

## 本章要做什么

- 2.2：实现相邻对合并、频率统计和合并顺序。
- 2.3：建立英中教学语料，训练约 512 个 token 的联合词表。
- 2.4：实现编码、解码、未知字符和保存/加载。
- 2.5：统计压缩率、OOV 与序列长度。
- 2.6：接回 `TinyGPT`，用固定 id logits 回归证明模型代码没变。

教学语料会在 2.3 加入 `samples/`；本节先用测试里的 `low`、`lower`、`lowest` 三条短文本。

## 先跑一次

```bash
cd ch02/start
uv sync --extra dev
uv run pytest tests/test_tokenizer.py -q
```

此时应当看到 **3 个失败**：

|测试|在哪一节转绿|
|---|---|
|`test_learn_merges_shrinks_symbols`|2.2|
|`test_tokenizer_contract_round_trip`|2.4|
|`test_vocab_size_matches_model`|2.4 可跑，2.6 纳入整章验收|

失败信息里的 `TODO 2.2`、`TODO 2.3` 指出第一个缺口；`2.4` 完成编解码后，前两个测试才具备完整行为。不要先看 `../end/`，那是逐节完成态。

## 阅读顺序

1. `src/llm101/tokenizer.py` —— 五成员接口与三个空壳函数
2. `tests/test_tokenizer.py` —— 三个红测试分别检查什么
3. `src/llm101/model.py` —— 只看 `GPTConfig.vocab_size` 怎么进 `TinyGPT`
4. 2.3 加入的 `samples/README.md` —— 教学语料的来源、行数和许可

## 目录

```text
ch02/start/
├── pyproject.toml
├── README.md
├── src/llm101/{tokenizer,parameters,layers,attention,model}.py
└── tests/{test_tokenizer,test_layers,test_structure}.py
```

`parameters.py` 仍是第 3 章的空壳；`layers.py`、`attention.py`、`model.py` 与第 1 章完成态一致。`test_structure.py` 和 `test_layers.py` 应当保持通过；`test_tokenizer.py` 的 3 个失败是本章的起点。
