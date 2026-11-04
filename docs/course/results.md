# 实验结果

本页记录已完成的限时训练指标和固定提示词样例。实验方法和时长规则见[限时实验方案](experiment-protocol.md)。所有指标来自本地 CPU 运行，原始 checkpoint 与 token 流不进入 Git。

## 做了什么

- 三种模型分别随机初始化，权重不共享：英文、中文、中英双语。
- 模型均为 4 层、`d_model=256`、4 头、`d_ff=1024`、上下文 128，共享输入/输出嵌入。
- 训练使用完整语料，按墙钟时间停止；每次运行记录实际呈现给模型的 token 位置数。
- 双语模型的中英文训练 token 分别约占 59% 和 41%，与两种语料的 token 量比例一致，没有额外重采样。

## 双语模型：1、3、5 小时的同一训练轨迹

先训练 1 小时，再从同一 checkpoint 续训到 3 小时、5 小时。下表是三个时长分别保存的权重，在同一批固定验证窗口上重新评估的结果（batch 4、上下文 128、每个子集 50 个 batch，固定随机种子），因此可以直接横向比较。

|训练时长|step|呈现 token 数|混合 PPL|英文 PPL|中文 PPL|
|---|---:|---:|---:|---:|---:|
|1 小时|16,031|8,207,872|38.66|37.62|44.50|
|3 小时|46,880|24,002,560|21.24|20.17|24.39|
|5 小时|77,416|39,636,992|18.21|17.21|20.89|

1 小时到 3 小时的改善最大；3 小时到 5 小时仍在下降，但幅度明显变小。perplexity 依赖词表和训练分布，这里的数字只在同一 tokenizer、同一验证窗口内可比。

## 三个 5 小时基线

各自运行结束时记录的最终验证结果（验证 batch 数为 50，随机种子随运行变化，因此不宜与其他表逐位对照）。

|模型|step|呈现 token 数|验证 loss|验证 PPL|
|---|---:|---:|---:|---:|
|英文|93,924|48,089,088|2.895|18.08|
|中文|78,107|39,990,784|3.079|21.74|
|双语（混合）|74,787|38,290,944|2.924|18.61|
|双语（仅英文子集）|—|—|2.860|17.46|
|双语（仅中文子集）|—|—|3.008|20.25|

中文和英文使用不同词表，两个 PPL 不应互相比较；表中的中文数值更高，只说明该模型在自己的验证分布上的拟合程度。

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

直接用「讲一个小兔子的笑话：」测试三个检查点时，输出都是普通故事续写，没有形成笑话。这类提示之所以失败，是因为训练目标只有 next-token prediction：模型学到的是短故事的分布，不是遵循请求。要获得问答能力需要后训练数据，不在本课程的基础主线内。

## 复现

```bash
uv run python scripts/train_model.py --corpus bilingual --hours 1 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027
uv run python scripts/train_model.py --corpus bilingual --hours 3 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027 --resume
uv run python scripts/train_model.py --corpus bilingual --hours 5 \
  --run-dir artifacts/runs/bilingual-progressive-seed20261027 --resume
```

权重再发布仍需先完成上游数据许可复核，见[数据与许可](../datasets.md)。
