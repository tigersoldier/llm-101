# 教学版 BPE tokenizer

BPE（Byte Pair Encoding）把文本切成 token。这里实现一个刻意简化的版本：从 Unicode 码点开始，反复合并语料中最常见的相邻符号对。它不做 GPT tokenizer 的字节映射、正则预切分或特殊的空白符处理。

## 合并步骤

以 `low low lower` 为例，初始序列按字符表示。算法统计相邻符号对的频率，选最高频的一对合并成一个新符号；随后重新统计并继续，直到达到词表大小或没有足够频繁的符号对。

合并顺序本身就是 tokenizer 的规则。编码时从字符序列开始，按学习到的顺序重放合并；解码时把 token 对应的字符串拼接起来。

## 运行示例

```python
from llm101.tokenizer import BPETokenizer

examples = ["hello world", "hello there", "你好，世界。", "你好，朋友。"]
tokenizer = BPETokenizer.train(examples, vocab_size=40)

ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)
print(ids)
print(tokenizer.decode(ids))
print(tokenizer.vocab_size)
```

`vocab_size` 至少要容纳所有训练文本中出现的不同字符和四个特殊 token（`<pad>`、`<unk>`、`<bos>`、`<eos>`）。训练中未见过的字符编码为 `<unk>`；因此对含未知字符的文本，decode 不保证无损还原。

## 实现边界

- 代码位于仓库文件 `src/llm101/tokenizer.py`。
- 训练、英文/中文往返、未知字符和 JSON 保存/加载都有单元测试。
- 为突出合并原理，这版允许合并跨空格的字符；它不是生产 tokenizer，也不是 GPT byte-level BPE。
- BPE trainer 会重复扫描输入并把序列保存在内存中。大语料可通过 `max_texts` 限制 tokenizer 训练样本；这只影响词表学习，不会自动裁掉语言模型预训练语料。之后会根据真实语料的规模选择可复现的词表训练样本。

运行测试：

```bash
uv run pytest tests/test_tokenizer.py -q
```
