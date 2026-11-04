# 自回归推理与生成

训练阶段让模型从左到右预测下一个 token；推理阶段把新 token 接到上下文末尾，再重复预测。当前实现位于 `src/llm101/inference.py`，只用前向计算，不使用 autograd。

## 解码步骤

1. 把提示文本编码为 token，并在开头加入 `BOS`。
2. 将最近最多 `max_seq_len` 个 token 输入 `TinyGPT`。
3. 取最后一个位置的 logits，应用采样策略，得到下一个 token。
4. 遇到 `EOS` 或达到 `max_new_tokens` 时停止。
5. 解码完整 token 序列，跳过 `BOS`、`EOS` 等特殊标记。

`temperature=0` 使用贪心 argmax。正温度会先把 logits 除以 temperature，再 softmax；`top_k` 可限制候选 token。相同模型、prompt、seed 与设置会得到相同采样结果。

```bash
uv run python scripts/generate.py \
  --model artifacts/runs/bilingual-5h-seed20261027/model-5h.pt \
  --tokenizer data/processed/tokenizers/bilingual.json \
  --prompt "小兔子走进了" \
  --max-new-tokens 64 --temperature 0.8 --top-k 40 --seed 20261027
```

训练脚本会用四条固定提示词（两条英文、两条中文）生成样例，并保存到对应运行目录的 `samples-<hours>h.json`，方便比较不同训练时长下的中英文表现。

## 当前实现的限制

每生成一个 token，程序都会重新计算完整窗口的注意力；它没有 KV cache，长输出会越来越慢。这个简单版本便于直接看懂因果掩码和 next-token 循环。后续章节再加入 KV cache，并用一致性测试确认逐 token 输出不变。

当前解码器支持贪心、temperature 和 top-k，不支持 top-p、beam search 或指令对话模板。这里训练的是短故事续写的基础预训练模型，不应把它描述成指令微调或通用问答模型。
