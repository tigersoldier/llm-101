# 课程重构计划：快照链

记录日期：2026-11-23。状态：**规划已定，尚未实现任何章节目录。**

本文件记录课程结构的决定，供以后回顾。当前仓库里的 `src/`、`scripts/`、`tests/` 仍是旧的“先实现、再补文档”形态；按本计划改造后，它们会被拆进 `ch01/` … `ch11/` 的快照链，`final/` 由 `ch11/end` 产生。

## 1. 教学目标的变化

旧路线是“先讲 tokenizer、手写梯度、注意力，最后才拼出模型”，读者要读到第 5 章才见到一个能跑的 transformer，前置太长，也看不到全貌。

新路线：**第 1 章先搭出完整但很小的 decoder-only transformer，并用交互网页把每层的输入输出显示出来；之后每章在这个骨架上加一层细节，最终收敛到 `final/`。**

这也是为什么第 1 章不是一个演示用的 `demo.py`：第 1 章的代码就是最终代码的雏形，后续章节只做增量。

## 2. 目录结构

```text
ch01/start/   ch01/end/
ch02/start/   ch02/end/
...
ch11/start/   ch11/end/
final/                    # 与 ch11/end 逐字节一致
```

- `start/`：上一章 `end/` 的完整副本，加上本章脚手架——新模块的空壳（函数体 `raise NotImplementedError`）、本章要跑通但此刻是红的新测试、以及写明“本章目标 / 先跑哪个测试 / 阅读顺序”的 README。
- `end/`：本章完成态，测试全绿，可作为下一章 `start` 的来源。
- `ch01/start/` 没有上一章，只包含工程脚手架：`pyproject.toml`、空包、第 1 章会失败的测试、README。
- `final/` 是最后一章 `end/` 的副本，作为可分发成果。

## 3. 已定决策

1. 目录名用 `start/` 与 `end/`，避免与根目录的 `final/` 混淆。
2. `uv.lock` 只保留在仓库根和 `final/`；各章 README 用 `uv sync`（不带 `--locked`），避免 22 份锁文件副本淹没历史。
3. 交互页面的前向用 JavaScript 实现，可输入任意文本；用 Node 跑同一组输入与 Python 对齐（容差 1e-5），Node 不存在时跳过该测试。放弃“预计算固定提示词”方案，因为它只能选句子，失去交互感。
4. CI 只运行每个 `end/` 的测试（共用一套环境，`PYTHONPATH` 指向各快照）；`start/` 不跑测试，只校验它与上一章 `end/` 逐字节一致。
5. 第 1 章直接采用与 `final/` 相同的模块划分，只是暂时只有 `forward`。

## 4. 连续性规则

- **同名模块只增语义，不改已有行为。** 第 3 章给 `layers.py` 加 `backward`，不重写 `forward`。
- **必须重构时附带等价性测试**：例如第 2 章把字符级词表换成 BPE，要有一个测试证明同一权重下手写输入得到的 logits 与重构前一致。
- 每章 `start/README.md` 固定写三段：本章要做什么、先跑哪个失败测试、按什么顺序读代码。
- 每章 `end/README.md` 固定写：本章新增了什么、怎么验证、下一章的入口在哪。

## 5. 章节清单

### 第 1 章 看得见的 transformer

- `start/`：工程脚手架 + 三个失败测试（输出形状、因果掩码、上下文上限）。
- `end/`：`parameters.py`（`Parameter`，含全零 `grad` 但本章不用）、`layers.py`（`Linear`、`Embedding`、`LayerNorm`、`GELU`，只有 `forward`）、`attention.py`（`MultiHeadSelfAttention`，只有 `forward`）、`model.py`（`GPTConfig`、`TransformerBlock`、`TinyGPT`）、`scripts/export_demo.py`、`docs/assets/demo/` 交互页、Node 对齐测试。
- 验收：`pytest` 全绿；页面能输入文本并逐层查看张量；改后段 token 不影响前段注意力输出。

### 第 2 章 真正的 tokenizer

- 增量：`tokenizer.py`（BPE）替换字符级词表；`GPTConfig.vocab_size` 来自训练好的词表；页面显示切词结果与 token id。
- 等价性测试：固定权重下，手写 token id 序列的 logits 与重构前一致。

### 第 3 章 参数、层与手写梯度

- 增量：给 `parameters.py`、`layers.py` 加 `backward`，新增 `losses.py`（softmax 交叉熵）；有限差分测试。
- 页面不变，本章解释页面里那些数字是**怎么算出来的**。

### 第 4 章 因果多头自注意力（拆解）

- 增量：注意力 `backward`；Q/K/V 推导；缩放因子；掩码实现细节。
- 验收：注意力全参数有限差分；改未来 token 不影响过去。

### 第 5 章 整模型与维度选择

- 增量：把配置提到 `d_model=256`、4 层、4 头、`d_ff=1024`、上下文 128；共享输入/输出嵌入；参数量 5,289,472（`vocab=8192`）。
- 验收：整模型全参数有限差分；参数量与理论值一致。

### 第 6 章 优化器

- 增量：`optim.py`（SGD、AdamW、`state_dict`）；全模型梯度裁剪。
- 验收：AdamW 单步与手算一致；状态往返后行为一致。

### 第 7 章 训练循环

- 增量：`training.py`（`TokenStream`、`clip_grad_norm_`、`train_step`）、验证 loss/perplexity。
- 数据：仓库内自带的自写短句，不依赖外部语料。

### 第 8 章 真实语料管道

- 增量：`corpus.py`、`scripts/download_datasets.py`、`scripts/prepare_training_data.py`；固定修订号与 SHA-256；中文归档的哈希切分；并行编码；manifest。
- 本章开始讨论数据许可。

### 第 9 章 限时训练、checkpoint 与续训

- 增量：`checkpoint.py`、`scripts/train_model.py`；按时长停止、信号安全保存、`--resume`。
- 验收：短跑产出 `last.pt`、`best-model.pt`、`model-<h>h.pt`、`metrics.jsonl`、`run_manifest.json`、`samples-<h>h.json`。

### 第 10 章 推理与生成

- 增量：`inference.py`、`scripts/generate.py`；贪心、temperature、top-k、`EOS` 停止。
- 页面升级：加载训练过的权重，同一个界面切换“随机权重 / 训练后权重”，对比输出。

### 第 11 章 实验、评估与发布

- 增量：`experiment-protocol.md`、`results.md`、许可说明；`ch11/end` 即 `final/`。
- 验收：1/3/5 小时数字可用同一批固定验证窗口重算；`final/` 独立 `uv sync` 与 `pytest` 通过。

## 6. 第 1 章详细设计

- 词表：字符级 68 个 token（大小写字母、空格、常用标点、`BOS`/`EOS`），任意英文可输入。
- 结构：2 层 Pre-LN decoder-only，`d_model=32`、2 头、`d_ff=64`、上下文 16，可学习绝对位置，共享输入/输出嵌入。
- 规模：19,840 个参数；float32 权重 78 KiB，网页端 JSON 约 0.1–0.2 MB。
- 权重：固定 seed 随机初始化，只有 forward。页面必须明确写“输出是噪声，看的是结构与数据流”，避免读者误以为模型已经会说话。
- 页面交互：
  - 输入文本 → 实时前向；
  - 逐层步进：token 与 id → 词嵌入 + 位置 → 每层 LayerNorm → Q/K/V → 每头注意力矩阵（因果掩码区域灰显）→ 注意力输出 → 残差加 → MLP 隐藏层 → 残差加；
  - 每步显示张量形状与这一层参数量，热力图悬停显示数值；
  - 末端显示 logits 与 top-5 概率条，`temperature` 滑块影响采样。
- 测试：`tests/test_structure.py`（形状、掩码、上下文上限）、`tests/test_demo_parity.py`（Node 执行页面 JS，与 Python 激活对齐 1e-5）。

## 7. 数据与许可边界

- 第 1–7 章使用仓库内自带的自写短句（约 100 条），不嵌入任何数据集的原文，避免把 CDLA-Sharing 内容带进代码仓库。
- 第 8 章起接入真实语料：英文 `roneneldan/TinyStories`、中文 `52AI/TinyStoriesZh`，沿用现有 SHA-256 固定与哈希切分。
- 权重再发布仍需先完成上游许可复核；在结论明确前只发布代码与文档。

## 8. 工程约束

- 每个 `end/` 可独立 `uv sync`（不带 `--locked`）并 `pytest`。
- 快照只含代码、测试、脚本、`pyproject.toml` 与 README；不含 `data/`、`artifacts/`、`site/`、`.venv/`。
- 新增 `scripts/check_snapshots.py`：校验 `chNN/start` 与 `ch(N-1)/end` 逐字节一致、`final` 与 `ch11/end` 一致，并对每个 `end/` 运行测试。
- 体积估算：每份快照约 450 KB 文本、约 5 千行；23 份合计约 10 MB，Git 增量压缩后更小。
- 页面资源放在 `docs/assets/demo/`，由 MkDocs 原样复制；`mkdocs build --strict` 必须继续通过。

## 9. 与现状的关系

- 现有 11 个模块与 46 项测试是各章 `end/` 的内容来源；第 1 章需要从中删出“只有 forward”的版本，其余章节按上述增量重新分配。
- 现有 `docs/course/*.md` 将拆入对应章节，`roadmap.md` 改为章节索引。
- 现有根目录 `final/` 暂时保留；`ch11/end` 完成后由其取代并保持同步。
- 代码已收敛为单一中英混合模型，章节计划沿用这一结论，不再保留纯英文或纯中文训练路径。

## 10. 尚未决定

- KV cache 是否作为第 12 章（当前代码未实现，`roadmap.md` 已列为后续）。
- 第 10 章页面加载训练后权重的具体形式：训练一个小号 2 层模型专门给页面用，还是对大模型做量化后导出。
- 章节的中文标题与固定 URL 命名。
