# 实验结果

本页记录已完成的限时训练指标和固定提示词样例。实验方法和时长规则见[限时实验方案](experiment-protocol.md)。所有指标来自本地 CPU 运行，原始 checkpoint 与 token 流不进入 Git。

## 做了什么

- 只训练一个中英混合模型：联合词表、两份完整训练语料、不加语言控制 token。
- 模型为 4 层、`d_model=256`、4 头、`d_ff=1024`、上下文 128，共享输入/输出嵌入。
- 训练使用完整语料，按墙钟时间停止；每次运行记录实际呈现给模型的 token 位置数。
- 中英文训练 token 分别约占 59% 和 41%，与两种语料的 token 量比例一致，没有额外重采样。

## 双语模型：1、3、5 小时的同一训练轨迹
先训练 1 小时，再从同一 checkpoint 续训到 3 小时、5 小时。下表是三个时长分别保存的权重，在同一批固定验证窗口上重新评估的结果（batch 4、上下文 128、每个子集 50 个 batch，固定随机种子），因此可以直接横向比较。

|训练时长|step|呈现 token 数|混合 PPL|英文 PPL|中文 PPL|
|---|---:|---:|---:|---:|---:|
|1 小时|16,031|8,207,872|38.66|37.62|44.50|
|3 小时|46,880|24,002,560|21.24|20.17|24.39|
|5 小时|77,416|39,636,992|18.21|17.21|20.89|

1 小时到 3 小时的改善最大；3 小时到 5 小时仍在下降，但幅度明显变小。perplexity 依赖词表和训练分布，这里的数字只在同一词表和同一验证窗口内可比。

另一次独立的 5 小时从头运行（`bilingual-5h-seed20261027`）结束时记录：step 74,787、呈现 38,290,944 token，混合 loss 2.924 / PPL 18.61，英文子集 17.46，中文子集 20.25。它们与上表用的随机种子不同，因此只作为重复运行的参考。

## 生成样例

以下为双语 5 小时权重，`temperature=0.8`、`top-k=40`、最多续写 64 token。输出保留原样，包括拼写和语法错误。

英文提示词 `Once upon a time, there was a little girl named`：

```text
Once upon a time, there was a little girl named. Her name was Sam. She loved to wear her favourite dress. Every day she put on her warm dress, wearing a very special dress.
One day, she was playing a game with her friend the sink. She got very excited and showed her doll a new dress. Her friend was so thankful. She played with her doll all day.
```

中文提示词 `小兔子走进了`：

```text
小兔子走进了树林。小兔子想知道如何爬树，只发现树林中央有小小的老树。小兔子非常害怕，不知道该怎么办。
突然，小兔子开始大声地爬树。小兔子听到树林的声音，环顾四周。他发现地上有一个黑色的大洞。
小兔子找到了一根大胡萝卜，把它埋在地里。
```

样例说明模型确实在按上下文续写新的故事文本，同时也能看到它没有掌握的事实、搭配和因果一致性。

## 这不是指令微调模型

直接用「讲一个小兔子的笑话：」测试 1、3、5 小时三个检查点时，输出都是普通故事续写，没有形成笑话。这类提示之所以失败，是因为训练目标只有 next-token prediction：模型学到的是短故事的分布，不是遵循请求。要获得问答能力需要后训练数据，不在本课程的基础主线内。

## 复现

```bash
uv run python scripts/train_model.py --hours 1 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027
uv run python scripts/train_model.py --hours 3 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027 --resume
uv run python scripts/train_model.py --hours 5 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027 --resume
```

权重再发布仍需先完成上游数据许可复核，见[数据与许可](../datasets.md)。
