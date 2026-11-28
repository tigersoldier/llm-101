# 课程重构计划：快照链

记录日期：2026-11-23；最近调整：2026-11-27。状态：**第 1 章已完成；第 2 章已规划、代码未开始；第 3–11 章按 §5 执行，本文件仍是可调整的规划草案。**

> 本文件记录的是目前讨论的结论，不是最终决定。章节划分、目录命名、模型规模、页面形式都可能在实现过程中调整；不要把它当成不可变的规范。每次调整请同步修改相关段落，并在末尾“修订记录”追加一条，避免文档与实际代码不一致。

本文件汇总目前的课程结构设想，供以后回顾。当前仓库里的 `src/`、`scripts/`、`tests/` 仍是旧的“先实现、再补文档”形态；`ch01/` 已落地，`ch02/` 已有规划（`README.md` 与 `sections/`，`start/`/`end/` 待建），`ch03/` … `ch11/` 待建，`final/` 由 `ch11/end` 产生。

## 1. 教学目标的变化

旧路线是“先讲 tokenizer、手写梯度、注意力，最后才拼出模型”，读者要读到第 5 章才见到一个能跑的 transformer，前置太长，也看不到全貌。

新路线：**第 1 章先搭出完整但很小的 decoder-only transformer，用 SVG 图、命令行输出和测试把每一步的输入、输出与形状展示出来；之后按“会学习 → 真实数据 → 放大 → 生成 → 限时训练 → 评估”加细节，最终收敛到 `final/`。**

全书分四段：第 1–2 章建立前向与 tokenizer；第 3–6 章讲梯度、优化器和训练循环；第 7–8 章接真实语料并放大；第 9–11 章做生成、限时训练和评估。第 1 章的代码是最终代码的雏形，后续章节只做增量。

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

以下十七项是目前采纳的方案；若后续实验或读者反馈暴露问题，可以再改，改动时同步更新本章节与修订记录。

1. 目录名用 `start/` 与 `end/`，避免与根目录的 `final/` 混淆。
2. `uv.lock` 只保留在仓库根和 `final/`；各章 README 用 `uv sync`（不带 `--locked`），避免 22 份锁文件副本淹没历史。
3. **不做 web server，也不做浏览器前端。** 第 1 章用 SVG 图、`show_placeholder.py` 的命令行输出和 `pytest` 展示数据流；原理在 1.1–1.8 已经足够清楚，不需要再实现一套逐层 API/页面。后续章节如果确实需要交互，再单独评估，不作为第 1 章的交付物。
4. CI 运行根目录测试和每个 `end/` 的测试（共用一套环境，`PYTHONPATH` 指向各快照）；`start/` 不跑测试，只由 `scripts/check_snapshots.py` 校验核心库文件与上一章 `end/` 一致；`final/` 与 `ch11/end` 逐字节一致。
5. 第 1 章采用与 `final/` 相同的核心模块划分（`tokenizer`、`layers`、`attention`、`model`），只是暂时只有 `forward`；`placeholder` 等章节专用文件不进入 `final/`。
6. 第 1 章不引入 HTML/JS/CSS 或前端构建步骤；图放在 `docs/assets/ch01/`，运行示例由命令行脚本输出。
7. **术语不提前出现。** 专业术语第一次出现不得早于具体介绍并实现它的那一章。在此之前，要么不使用该术语、也不提及相关内容，要么用通俗语言简单描述，并明确告诉读者后续会在哪一章介绍。第 1 章内部按小节执行同一规则：正文只使用本节或更早小节已经介绍过的术语；代码标识符可以照实展示，但要说明它对应哪件事。小节清单与导航是路线图，可以提前列出标题，正文不能假定读者已经认识这些名字。
8. **术语译名与术语表。** 有通用中文译名的术语第一次出现时附上中文译名；有多种通用译法的全部列出（例如 tokenizer：分词器、标记器、词元切分器）。transformer 这类没有公认译名的术语不强译，保留原文。缩写与符号第一次出现时写出完整单词（如 `<bos>` = beginning of sequence）。书末维护术语表 `docs/glossary.md`：按首次出现顺序列出术语、中文译名、一句话说明，并链接到详细介绍或实现它的小节；每写完一节，把新术语补进表中。
9. **图解、例子与数学补充。** 结构、数据流、矩阵或多阶段变化能说明问题的概念与计算，各配 SVG 图；只承载一条公式、删掉也不损失信息的图不画。完整图规则见 `docs/figures.md`。给出至少一组输入/输出例子（可运行的代码或可手算的数值）。数学补充只写大学及以上的内容；大学之前的数学（取余、基础向量/矩阵运算等）不解释，除非本节特别注明。没有需要补充的数学就不写这一节。图放在 `docs/assets/chNN/`，文件名用“序号-语义名”（如 `16-attention-pipeline.svg`）；架构图用 `architecture-N.svg`。
10. **可运行检查点。** 第 1 章每节结束都必须有一条能跑的管线：用同一个示例输入，展示本节改动后的完整输入→输出与过程；未实现的组件用简单占位，架构图标出真组件与占位组件。第 2 章起每章至少有一个章末端到端命令/测试，节内是否画管线按需。详见第 12 节。
11. **Python 代码必须带严格类型注解。** 正文、示例和章节代码里的每个函数/方法都要写参数与返回注解（包括 `-> None`）；示例与真实代码保持一致。`tests/test_docs_examples.py` 会扫描 `docs/**/*.md` 的 Python 代码块。
12. **概念本质与扩展阅读。** 概念在最早需要它的章节引入，先讲本质（如 tensor 是多维数组：标量、向量、矩阵都是它的特例）；有重要新概念的章节在最后加“扩展阅读”，讲为什么需要它、没有它会怎样。
13. **本节需要实现的函数、参考实现与逐函数示例。** 只要求读者实现最终算法相关部分（进入累计库、被后续章节或最终模型使用）；示例、冻结基线、手写占位和一次性脚本直接给完整代码，不设空壳和独立参考实现块。对最终算法相关的函数，按教科书式逐函数介绍：函数名、如何被使用、输入输出、输入输出样例、带注释的签名；不使用表格。函数有独立概念小节时，介绍放进该小节，不要集中堆在节首。再给参考实现；最后逐个函数给实际结果与解释，必要时配矩阵/多阶段图；能用公式表达的，先给矩阵/张量公式，再给代数式或逐元素展开。
14. **教科书式写作风格。** 全书按 `STYLE.md` 执行：新概念与专有函数逐个介绍；复杂实现与算法给编号步骤、最小例子、中间结果和图例；每节按“概念 → 函数 → 分步分析 → 参考实现 → 结果”组织。
15. **概念的引入时机。** 概念只在读者能理解其意图时引入；判断标准与反例见 `STYLE.md` 的“引入时机”；第 1 章的逐项审计见第 13 节。
16. **第 1 章 API 是基线。** 后续章节保留 `ch01/end` 的公开接口和数值语义；`final/` 只作算法参考，第 11 章再由 `ch11/end` 重新生成。允许的数值“毕业”只有第 3 章的精确 GELU 和第 8 章的初始化，且必须附差异测试与文档说明。
17. **章节专用文件不进入累计库。** `placeholder.py`、演示脚本、该章测试和教学语料属于章节专用；`start/` 可以删除上一章的章节专用文件，但要在 README 和快照清单中列出。

## 4. 连续性规则

- **第 1 章 API 是基线。** 后续章节保留 `ch01/end` 的公开接口和数值语义；同名模块只增语义，不重写 `forward` 的返回结构。`final/` 只作算法参考，第 11 章再由 `ch11/end` 重新生成。
- **允许两处数值“毕业”。** 第 3 章把 `GELU` 从 1.5 的 tanh 近似换成精确 `erf` 形式，并附输出差异测试；第 8 章放大维度时把 `Linear` 初始化从 `0.02` 换成 `sqrt(2/(in_features+out_features))`，并说明理由。其他前向数值变化必须单独提案，不得静默发生。
- **必须重构时附带等价性测试。** 第 2 章换 tokenizer 的等价性定义为：同一 `GPTConfig`、同一 seed、同一手写 id 序列的 logits 回归（证明模型代码没变），加上 tokenizer 契约测试（`encode`/`decode`/`vocab_size`/`bos_id`/`eos_id`、未知字符、特殊 token、往返）；不以“同一文本的新旧 token id”作等价。
- **章节专用文件与累计库分开。** `placeholder.py`、`show_placeholder.py`、`render_architecture.py`、该章的演示测试和教学语料属于章节专用；`start/` 可以删除上一章的章节专用文件，但必须在 README 和快照清单中列出。累计库核心是 `src/llm101/` 中跨章使用的模块（`tokenizer`、`parameters`、`layers`、`attention`、`model`、`losses`、`optim`、`training`、`corpus`、`checkpoint`、`inference`）。
- 每章 `start/README.md` 固定写三段：本章要做什么、先跑哪个失败测试、按什么顺序读代码。
- 每章 `end/README.md` 固定写：本章新增了什么、怎么验证、下一章的入口在哪。
- 章节目录里的 `pyproject.toml` 保持同一份模板；`uv.lock` 只保留根目录和 `final/`。

## 5. 章节清单

全书按“看得见的前向 → 会学习 → 真实数据 → 放大 → 生成 → 限时训练 → 评估”推进。第 1 章已完成；第 2 章已有小节计划；第 3–11 章按下表执行。每章的大纲页在 `docs/course/chNN/index.md`，列出目标、前置、小节、代码和验收；小节页面随写作上线。

### 第 1 章 看得见的 transformer

- `start/`：工程脚手架 + 三个失败测试（输出形状、因果掩码、上下文上限）。
- `end/`：`tokenizer.py`（字符级）、`parameters.py`（第 3 章的占位空壳，本章不导入）、`layers.py`（`Linear`、`Embedding`、`LayerNorm`、`GELU`，只有 `forward`）、`attention.py`（`MultiHeadSelfAttention`，只有 `forward`）、`model.py`（`GPTConfig`、`TransformerBlock`、`TinyGPT`，只保留 `forward`）、`placeholder.py`、`scripts/show_placeholder.py`、`scripts/render_architecture.py`。
- 小节：`ch01/sections/1.1.md`–`1.9.md` 已全部完成，正文在 `docs/course/ch01/`。
- 验收：`ch01/end` 的 39 个测试全绿；`show_placeholder.py --stage 7` 输出与 `TinyGPT.forward` 一致；改后段 token 不影响前段注意力输出；1.9 只做验收、小结与交接。
- 本章不做什么：不换 BPE、不做反向传播、不训练、不读真实语料、不做网页。

### 第 2 章 真正的 tokenizer

- 小节：2.1–2.6，计划与验收在 `ch02/sections/`；2.1 是起点（固定示例、第 1 章基线、三个红测试），2.2–2.5 各讲一个机制，2.6 接回模型并收尾。详见 §14–15。
- `start/`：`ch01/end/` 的副本，删除第 1 章专用文件（`placeholder.py`、`show_placeholder.py`、`render_architecture.py`、`test_placeholder.py`），把 `tokenizer.py` 换成 BPE 空壳，保留 `test_structure.py`，新增 `samples/` 与三个红测试。
- `end/`：`tokenizer.py`（BPE）、`samples/`、`scripts/show_tokenizer.py`、`scripts/tokenizer_report.py`、`tests/test_tokenizer.py`、`tests/test_report.py`、`tests/test_regression.py`、`tests/data/ch01-logits.json`。
- 增量：`tokenizer.py` 换成 BPE；提供 `BPETokenizer.train`、`encode`、`decode`、保存/加载；`GPTConfig.vocab_size` 来自教学词表（目标约 512，实际由语料决定，不要求精确值）。
- 内容：从 Unicode 码点出发的合并；英文空格与中文无空格的处理；联合词表；`<unk>`/`<bos>`/`<eos>`；词表大小、压缩率、OOV 与序列长度；建立英中教学语料（各 200 行，自写，CC BY 4.0），供第 2–6 章使用。
- 契约：`encode`/`decode`/`vocab_size`/`bos_id`/`eos_id` 行为不变；构造/训练/加载接口可以不同，但要提供 `train`/`save`/`load`；词表顺序固定为特殊 token → 初始码点（字典序）→ 合并结果；`decode` 默认跳过特殊 token，未知码点编码成一个 `<unk>`，往返对未知字符有损（与第 1 章一致）；`<pad>` 暂不加入，直到真正实现 padding。
- 等价性：固定 `GPTConfig`（68）、seed 和手写 id 的 logits 回归（证明模型代码没变）；tokenizer 契约测试；tokenizer 与模型 `vocab_size` 一致性测试。回归 golden 从 `ch01/end` 生成，不用新文本 id。
- 验收：英文/中文往返；未知字符；特殊 token；合并示例；新词表下 `TinyGPT` 前向形状正确；教学语料可由固定文件重复得到；`show_tokenizer.py --stage 1` 到 `--stage 6` 可跑。
- 不做什么：不接 TinyStories；不做 8192 词表（第 7 章）；不加 `<pad>`；不做字节级 BPE；不训练。

### 第 3 章 参数、层与手写梯度

- 增量：`Parameter`（`value`、`grad`、`zero_grad`、`numel`）；`Linear`、`Embedding`、`LayerNorm`、`GELU` 的 `backward`；`losses.py` 的稳定 softmax 交叉熵；next-token 目标（shift、所有位置平均、perplexity）；中心差分与 float64 校验。
- 数值毕业：GELU 从 1.5 的 tanh 近似换成精确 `erf` 形式，附输出差异小于 2e-3 的测试；其余前向数值不变。
- 迁移：`parameters()` 从裸张量改为 `Parameter`，保留 `numel()`；第 1 章的参数量测试仍应通过。
- 验收：各层有限差分；GELU 新旧差异；交叉熵梯度；单句 next-token loss 与手算一致。
- 不做什么：不实现注意力/block/整模型反向（第 4 章）；不优化、不训练。

### 第 4 章 注意力、block 与整模型反向

- 增量：注意力 `backward`（softmax 雅可比、因果 mask、多头、输出投影）；`TransformerBlock.backward`（残差、Pre-LN、MLP）；`TinyGPT.backward`（final norm、绑定输出头的梯度累加）；内部缓存与第 1 章返回的 cache 共存。
- 内容：前向只做一页回顾，不重讲 Q/K/V、缩放和 mask；重点讲梯度形状、mask 的零梯度、多头合并、绑定权重。
- 验收：注意力全参数有限差分；block/整模型 tiny config 全参数有限差分；未出现在输入里的 token 只从输出头得到梯度；改未来 token 不影响过去。
- 不做什么：不缩放（第 8 章）；不优化、不训练。

### 第 5 章 优化器

- 增量：`optim.py` 的 SGD、AdamW（偏差修正、decoupled weight decay、epsilon、`state_dict`）；全模型梯度裁剪；可选 warmup/学习率调度的动机与取舍。
- 验收：AdamW 单步与手算一致；状态往返后行为一致；裁剪后的全局范数正确。
- 不做什么：不接数据、不跑训练循环；不引入最终模型尺寸。

### 第 6 章 训练循环

- 增量：`training.py` 的内存采样器与 `train_step`；next-token shift、所有位置平均损失、验证 loss/perplexity；loss 曲线、梯度范数、学习率、吞吐量等诊断；过拟合一个小批次的 sanity check。
- 数据：第 2 章的教学语料；第 1/2 章的小配置；不依赖外部语料。
- 验收：固定 seed 可复现；小模型 loss 下降；验证 PPL 可计算；小批次可过拟合。
- 不做什么：不引入 int32/mmap `TokenStream`（第 7 章）；不做 checkpoint/续训/限时（第 10 章）；不用 5M 模型。

### 第 7 章 真实语料与最终 tokenizer

- 增量：`corpus.py`、`scripts/download_datasets.py`、`scripts/prepare_training_data.py`；固定修订号与 SHA-256；中文归档的哈希切分；去重/过滤；中英配比；BPE 8192；并行编码；int32 `TokenStream`；manifest；许可讨论。
- 内容：故事边界与 `BOS`/`EOS`；随机窗口跨故事；无语言控制 token；token 预算与语言覆盖计数；tokenizer 压缩率与 OOV。
- 验收：manifest 可复现；token 数与词表尺寸记录；抽样解码正确；`TokenStream` 批次形状正确；许可说明齐全。
- 不做什么：不训练（第 10 章）；不缩放（第 8 章）。

### 第 8 章 缩放与预算

- 增量：最终 `GPTConfig`（`d_model=256`、4 层、4 头、`d_ff=1024`、上下文 128、`vocab=8192`）；参数量 5,289,472；每 token FLOPs、激活内存、批次大小与上下文成本；`Linear` 初始化从 0.02 换成 `sqrt(2/(in_features+out_features))`；`GPTConfig` 校验；最终配置的抽样梯度检查。
- 内容：为什么 4 层/256 宽/4 头/4C/128 上下文；放大后 Pre-LN、初始化和梯度裁剪的关系；不再重复 1.8 的权重绑定与参数量基础。
- 验收：参数量与理论值一致；一次前向/反向的 CPU 时间与内存估算；最终配置抽样梯度检查；tiny config 全参数有限差分仍通过。
- 不做什么：不跑完整 1/3/5 小时训练（第 10 章）；不换 tokenizer。

### 第 9 章 推理与生成

- 增量：`inference.py`、`scripts/generate.py`；贪心、temperature、top-k、`EOS` 停止、上下文截断；固定 seed 可复现；用第 6 章的小模型对比随机权重与训练后权重。
- 内容：为什么每步重算上下文；KV cache、top-p、beam search 为什么不在主线；生成样例只用于观察，不作为评测。
- 验收：temperature=0 确定性；top-k 截断；EOS 停止；prompt 往返；生成不依赖 autograd。
- 不做什么：不做 KV cache/top-p/beam；不做网页；不跑长训练。

### 第 10 章 限时训练、checkpoint 与续训

- 增量：`checkpoint.py`、`scripts/train_model.py`；按时长停止、信号安全保存、`--resume`；`metrics.jsonl`、`run_manifest.json`、`last.pt`、`best-model.pt`、`model-<h>h.pt`；用第 9 章 `generate` 产出 `samples-<h>h.json`；固定提示词；实验协议。
- 数据/配置：第 7 章语料与 tokenizer，第 8 章最终配置。
- 验收：短跑产出全部 artifact；续训后行为与不中断运行一致；样例 JSON 可复现；不要求 1/3/5 小时全跑。
- 不做什么：不重讲优化器/训练循环；不做评估方法论（第 11 章）。

### 第 11 章 实验、评估与发布

- 增量：`results.md`、评估方法、许可说明、`ch11/end` 即 `final/`；1/3/5 小时同一轨迹比较；固定验证窗口；分语言 PPL；多 seed/采样方差；固定提示词人读样例；记忆/污染/局限；后续方向。
- 验收：结果可用 `ch11/end` 重算；`final/` 独立 `uv sync` 与 `pytest` 通过；许可与“不是什么模型”写清楚；给出后续课程方向。
- 不做什么：不做指令微调/RLHF/分布式训练；不发布未复核许可的权重。

## 6. 第 1 章详细设计

- 词表：字符级 68 个 token（大小写字母、空格、常用标点、`BOS`/`EOS`），任意英文可输入。
- 结构：2 层 Pre-LN decoder-only，`d_model=16`、2 头（每头 8 维）、`d_ff=32`、上下文 16，可学习绝对位置，共享输入/输出嵌入。
- 规模：5,824 个参数；float32 权重约 23 KiB。刻意保持这么小，是为了让每层的张量能一屏放下、数值可以手算核对；第 8 章再把维度提到 `d_model=256`、4 层，并说明每一步放大了什么。
- 权重：固定 seed 随机初始化，只有 forward。图注和命令行输出必须写明“输出是噪声，看的是结构与数据流”，避免读者误以为模型已经会说话。
- 展示方式：1.3–1.8 的 SVG 图 + `show_placeholder.py --stage N`；不实现逐层 API、服务器或网页。
- 测试：`tests/test_structure.py`（形状、掩码、上下文上限）和 `tests/test_placeholder.py`；不新增 `test_api.py`，不依赖 Node。

## 7. 数据与许可边界

- 第 1 章使用内置的 6 行示例；第 2 章建立英中教学语料（数百条，自写或明确许可），供第 2–6 章使用，不嵌入任何数据集原文，避免把 CDLA-Sharing 内容带进代码仓库。
- 第 7 章起接入真实语料：英文 `roneneldan/TinyStories`、中文 `52AI/TinyStoriesZh`，沿用现有 SHA-256 固定与哈希切分，并补去重/过滤、语言配比和 token 预算。
- 第 7 章开始讨论数据许可；权重再发布仍需先完成上游许可复核，在结论明确前只发布代码与文档。

## 8. 工程约束

- 每个 `end/` 可独立 `uv sync`（不带 `--locked`）并 `pytest`。
- 所有 import 放在文件顶部；正文示例代码块把 import 放在块顶部，函数、条件分支或循环内不写局部 import。新增代码避免在函数内部定义函数；可复用逻辑提到模块级私有函数。
- 快照只含代码、测试、脚本、`pyproject.toml` 与 README；不含 `data/`、`artifacts/`、`site/`、`.venv/`。
- 新增 `scripts/check_snapshots.py`：校验 `chNN/start` 与 `ch(N-1)/end` 的累计库核心文件一致；章节专用文件、README、新空壳/红测试按清单允许差异；`final/` 与 `ch11/end` 逐字节一致；对每个 `end/` 运行测试。
- CI 增加章节矩阵：根目录测试、每个 `end/` 测试、`scripts/check_snapshots.py`、文档示例注解检查；`start/` 只做快照校验，不跑测试。
- 体积估算：每份快照约 450 KB 文本、约 5 千行；23 份合计约 10 MB，Git 增量压缩后更小。
- 第 1 章不放前端静态资源，也没有服务器启动步骤；`docs/` 只放说明页和 SVG 图，所以站点不需要额外构建步骤。

## 9. 与现状的关系

- 现有 `src/`、`scripts/`、`tests/` 是旧实现，作为各章 `end/` 的算法来源，不再继续演进；章节快照是正文的唯一代码权威。
- 现有 `docs/course/*.md` 按新章节拆分：`bpe.md`→第 2 章，`manual-gradients.md`→第 3 章，`causal-attention.md`→第 4 章，`transformer-model.md`→第 1 章前向 + 第 8 章缩放，`training-loop.md`→第 6/10 章，`inference.md`→第 9 章，`experiment-protocol.md`→第 10 章，`results.md`→第 11 章。对应章节上线后旧页删除或改成跳转。
- `roadmap.md` 改为章节索引，与新顺序一致。
- 现有根目录 `final/` 只作算法参考；第 11 章由 `ch11/end` 重新生成，已发布的 1/3/5 小时结果标注为旧实现结果，或由第 11 章重跑。
- 代码已收敛为单一中英混合模型，章节计划沿用这一结论，不再保留纯英文或纯中文训练路径。

## 10. 交接清单

截至 2026-11-27 的状态：

- 第 1 章已完成：`ch01/start/` 是脚手架（3 个红测试），`ch01/end/` 是完成态（39 passed），正文在 `docs/course/ch01/1.1.md`–`1.9.md`。
- 第 2 章已规划：`ch02/README.md` 与 `ch02/sections/2.1.md`–`2.6.md` 已写；2.1–2.3 已写完（`ch02/start/`、`ch02/end/`、图 2.1-1/2.1-2、2.2-1–2.2-4 与 2.3-1/2.3-2、正文），2.4–2.6 待写。
- `scripts/check_snapshots.py` 尚未实现；CI 只跑根目录和 `ch01/end`，还没有章节矩阵。
- 第 1 章正文里指向旧章号的句子暂不回填；等第 2 章正文开工或章节号稳定后统一做一次指针修订，以本文件 §5 为准。

第 2 章实施顺序：

1. 读 `ch02/README.md`、`ch02/sections/2.1.md`、本文件 §14–15、`ch01/README.md`、`STYLE.md`。
2. 从 `ch01/end/` 复制出 `ch02/start/`；按 2.1 的清单删除 `placeholder.py`、`show_placeholder.py`、`render_architecture.py`、`test_placeholder.py`，把 `tokenizer.py` 换成 BPE 空壳，写 `ch02/start/README.md`（本章要做什么、先跑哪个失败测试、按什么顺序读代码）。
3. 建立 `ch02/start/samples/` 的英中教学语料与 `samples/README.md`；写三个红测试。
4. 按 2.1–2.6 顺序写正文与 `ch02/end/` 代码；测试放 `ch02/end/tests/`：tokenizer 契约、报告指标、固定 id logits 回归、tokenizer/模型词表一致性。
5. 同时补 `scripts/check_snapshots.py` 与 CI 章节矩阵；不要等第 11 章再补。
6. 不接 TinyStories、不做 8192 词表、不训练、不加 `<pad>`；这些分别留给第 7 章和真正需要 padding 的章节。

写作约束：第 2 章只换 tokenizer，不引入反向传播、优化器或真实语料；`ch01/end/` 的公开接口是基线，`final/` 不是。

## 11. 尚未决定

- KV cache 不进入主线；第 9 章讲清每步重算的成本，KV cache 作为附录或第 12 章扩展，暂不排入 11 章。
- top-p、beam search、指令微调/RLHF、GPU/分布式训练不进入基础主线；第 11 章给后续方向。
- 章节中文标题与固定 URL 命名按 §5 的标题执行；页面统一放在 `docs/course/chNN/`。
- 教学语料的形式已定：自写静态文件 `samples/english.txt`、`samples/chinese.txt`，许可随仓库文字采用 CC BY 4.0；最终条数与字符数在 2.3 写作时按实际统计记录；不把 TinyStories 原文带进第 2–6 章。

## 12. 第 1 章的螺旋编排

目标：每节结束时，读者都能用同一个示例输入跑一遍完整流程，看到本节改动带来的输入→输出变化；还没实现的部分用简单占位组件。第 1 章从第一节起就有可运行的“看得见的程序”，而不是等到 1.7 才见到完整模型。

### 固定示例

- 输入：`the cat sat`（11 个字符；加 `BOS`/`EOS` 后 13 个编号，未超过上下文上限 16）。
- 每节末尾记录：1.1/1.2 是编号、答案编号和字符；1.3–1.6 是编号和最后一个位置的 16 个数字；1.7 是 block 输出的数字；1.8 起是分数、答案和字符。

### 管线随小节增长

每节只出现当时已有的步骤：

- 1.1：文字 → 编号 → 答案（占位）
- 1.2：编号换成真 tokenizer
- 1.3：答案换成每个位置的数字（Embedding）
- 1.4：加入位置
- 1.5：加入调匀/筛选
- 1.6：加入混合
- 1.7：换成 `TransformerBlock`，输出 block 的数字；1.8：整模型输出 68 个分数，再用 argmax 得到答案

后面的步骤在对应小节才出现，不提前画成占位。

### 占位组件（1.1）

|阶段|占位做法|
|---|---|
|文字 → 编号|`ord(ch) % 68`|
|答案|`(sum(ids) * 31) % 68`|
|答案编号 → 字符|`ALPHABET[answer % 26]`，`ALPHABET = "abcdefghijklmnopqrstuvwxyz"`|

这些占位故意简单、确定，读者可以手算；它们只负责让 1.1 跑通，不是教学重点。

### 每节的状态

|节|本节新增/换上的步骤|节末示例输出|
|---|---|---|
|1.1|占位编号、占位答案|11 个编号 → 答案 19 → 't'|
|1.2|`CharacterTokenizer`|13 个编号 → 答案 11 → 'l'|
|1.3|`Embedding`|查表数字（16 个）|
|1.4|位置嵌入|同一编号在不同位置的数字不同|
|1.5|`LayerNorm`、`GELU`|调匀后的数字|
|1.6|因果多头注意力|混合后的数字|
|1.7|`TransformerBlock`（残差、Pre-LN、MLP）|block 输出 `[13,16]` 的数字|
|1.8|完整 `TinyGPT`（两层、残差、共享输出）|68 个分数 → argmax → 字符|
|1.9|验收、小结与交接|同一答案 + 最终检查结果|

1.9 不改模型，也不增加 stage；它只做回顾、测试和交接。

小节顺序沿用现有的 1.2 tokenizer、1.3 embedding；如果要把 embedding 提前到 1.2，只需对调这两节。

### 运行方式

```bash
cd ch01/end
uv run python scripts/show_placeholder.py --stage 1 "the cat sat"
```

- `--stage N` 只使用第 N 节已有的步骤；后面的步骤不出现。
- 每节正文末尾贴 `--stage N` 的实际输出：1.1/1.2 是编号、答案和字符，1.3 起是编号和最后一个位置的数字；输出随实现变化，不手写假数字。

### 架构图

- `uv run python scripts/render_architecture.py --stage N` 生成 `docs/assets/ch01/architecture-<N>.svg`。
- 只画第 N 节已有的步骤；真组件用实线彩色，占位用虚线灰色。后面的组件不提前画。
- 每节正文嵌入本节机制样例管线图和架构图：机制样例图在前，架构图在后；图注写明本节新增的步骤和位置全书统一。

### 新增文件

- `ch01/end/src/llm101/placeholder.py`：占位函数与 `run(text, stage)`；后面的组件在对应小节才调用；所有 import 放在文件顶部，1.1 只走占位分支。
- `ch01/end/scripts/show_placeholder.py`、`ch01/end/scripts/render_architecture.py`。
- `ch01/end/tests/test_placeholder.py`：1.1 的确定性输出、各阶段形状、最终输出与 `TinyGPT.forward` 一致。
- `ch01/end/README.md` 的目录树与阅读顺序补上管线。
- `docs/assets/ch01/architecture-{1..8}.svg`。

### 与现有页面的关系

- 1.1 从“只看脚手架”改为“最小可运行管线 + 脚手架巡览”；第一节就能看到输入→输出。
- 1.2/1.3 已有内容保留，补“本节结束后的管线输出与架构图”；1.2 的示例输入从 17 个字符的 `The little rabbit` 改为 11 个字符的 `the cat sat`（避免超过上下文上限）。
- 1.4 起按“图解 + 例子 + 数学补充 + 本节管线输出 + 架构图”的同一模板写。
- 章节目录、导航、术语表不变；小节标题不变。

## 13. 第 1 章编排审计（概念引入时机）

按 `STYLE.md` 的“引入时机”检查第 1 章。标准：如果本节/本章的管线用不到某个概念，读者就无法判断“没有它会怎样”，应推迟到真正需要它的地方。

### 审计

|概念/组件|现在的位置|第一次真正用到|判断|改动|
|---|---|---|---|---|
|tensor（张量）|1.1|1.3 的 Embedding|1.1 的管线只输出字符，不产生张量|把张量概念和矩阵图移到 1.3；1.1 删掉这两节|
|`<pad>`|1.2|后续真正需要补齐时|本章从不补齐序列|从 1.2 删除；到需要 padding 的章节再讲|
|`Parameter` / `grad`|1.3（已只留 value）|第 3 章反向传播|`Parameter` 的动机是记录调整量，1.3 用不到|第 3 章再引入 `Parameter` 和 `grad`；1.3 的权重直接用张量|
|`Linear`|1.3|1.6 的 Q/K/V、1.7 的 MLP/输出|1.3 的管线只到 Embedding，`Linear` 不改变本节输出|移到 1.6 首次使用；1.7 再用 MLP/输出|
|`parameters()` / 参数量|1.3、1.7|1.7 汇总模型|1.3 不需要|推迟到 1.7|
|批次 / 隐藏状态|已从 1.3 移除|1.7 或训练|—|保持推迟|
|`d_model`|1.3 提及|1.7 的配置|1.3 只用 `C`|1.3 只写 `C`，`d_model` 到 1.7|
|`Embedding`|1.3|1.3 的管线|本节输出就是它|保留|
|`LayerNorm` / `GELU`|1.5|1.7 的 block/MLP|零件本身的意图（调匀、筛选）可独立说明|保留；1.5 要明确它们服务于 1.7 的 block|
|因果注意力|1.6|1.7 的 block|“每个位置参考前面”可独立说明|保留|

### 调整后的编排

- 1.1：文字 → 编号 → 字符；不引入张量。
- 1.2：字符级 tokenizer、`<unk>`/`<bos>`/`<eos>`；不引入 `<pad>`。
- 1.3：张量概念、`Embedding`、`[T]`/`[T, C]`；权重用张量，不引入 `Parameter`/`Linear`。
- 1.4：位置嵌入。
- 1.5：`LayerNorm`、`GELU`（说明它们将用于 1.7 的 block）。
- 1.6：因果注意力，用 `Linear` 做 Q/K/V；把 1.3 的 Linear 手算例子移到这里。
- 1.7：Transformer block：残差、Pre-LN、MLP；1.8：整模型、共享输出、参数量、分数与 argmax。
- 1.8 完成；1.9 只做验收、小结与交接，不再增加 trace/服务器/页面。

### 落地顺序

1. 1.1：删掉张量/矩阵图，保留占位管线和逐函数示例。
2. 1.2：删掉 `<pad>` 段。
3. 1.3：删掉 `Parameter`、`Linear`、`parameters()`；保留张量、`Embedding`、形状和手算查表；小节标题改为“嵌入表与张量形状”。
4. 1.6：引入 `Linear`（Q/K/V）和线性层手算例子。
5. 1.7：引入 MLP 和 block；1.8：引入整模型、共享输出、`parameters()`、参数量和 argmax。
6. 第 3 章：引入 `Parameter` + `grad`，把 1.3/1.6/1.7 的裸张量包起来，并加等价性测试；GELU 换成精确 `erf`，附差异测试。
7. 第 4 章：注意力、block、整模型反向；第 5–11 章按 §5 的新顺序执行。
8. 上述每步同步更新小节计划、README、章节总览、导航、图和测试。

## 14. 第 2 章详细设计与螺旋编排

目标：每节结束时，读者都能用同一对固定样例跑一遍 tokenizer 管线，看到本节改动带来的输入→输出变化；2.6 接回模型并收束全章。第 2 章不重画模型架构图，管线重点是 tokenizer；模型只在 2.6 出现一次。

### 固定示例与基线

- 英文：`the cat sat on the mat`（22 个字符；加 `<bos>`/`<eos>` 后 24 个编号，超过上下文 16）。它是第 1 章 `the cat sat` 的延长，前三个词不变。
- 中文：`小猫坐在垫子上`（7 个字符；第 1 章字符表没有中文，编码结果是 7 个 `<unk>`，加特殊 token 后 9 个编号）。
- 失败点：英文样例本身就是长度失败（24>16）；中文样例是覆盖失败（7 个 `<unk>`）。不再单独设长句。

2.1 在 `ch01/end` 上跑第 1 章字符基线；`ch02/end` 从 2.2 起用 BPE。中文在 2.1 全部是 `<unk>`，2.3 建好码点表之后才有真实编号；正文要把这两个阶段分开写。

### 管线随小节增长

|节|新增/换上的步骤|节末示例输出|
|---|---|---|
|2.1|第 1 章字符基线 + 手写片段合并占位|英文 24>16；中文 7 个 `<unk>`；手写合并后 13 个符号（加特殊 token 15）|
|2.2|`merge_pair`、`count_pairs`、`learn_merges`|实际得到的合并（最多 16 条）；英文/中文符号序列|
|2.3|`BPETokenizer.train`、词表顺序、特殊 token|实际词表大小；英文/中文 ids|
|2.4|`encode`、`decode`、`save`、`load`|往返文本；未知字符的 `<unk>` 行为|
|2.5|`token_count`、`compression_ratio`、`unknown_count`、报告脚本|字符数、token 数、压缩率、OOV|
|2.6|`GPTConfig.vocab_size` + 固定 id 回归|`[T, V]` logits、argmax；回归一致|

### 2.1 的占位与删除

- 手写合并只用于 2.1 演示：`a`+`t`→`at`、`s`+`at`→`sat`、`m`+`at`→`mat`、`t`+`h`→`th`、`th`+`e`→`the`，英文样例从 22 个符号降到 13 个，加特殊 token 后是 15，进入上下文 16。它故意简单、确定、可以手算，不是算法；`cat` 和 `on` 故意保留为字符，说明手写合并是部分的。
- 2.1 是基线节，不要求读者实现新函数；`baseline.py` 的代码按“字符表与编码 → 手写合并 → 报告与上下文检查”三段直接嵌入正文，不设独立的“参考实现”块，也不做签名空壳。
- `ch02/start/` 删除第 1 章专用文件 `placeholder.py`、`scripts/show_placeholder.py`、`scripts/render_architecture.py`、`tests/test_placeholder.py`；不做兼容层。
- `CharacterTokenizer` 在 `ch02/start/` 和 `ch02/end/` 中删除；2.1 的基线在 `ch01/end` 里跑，`show_tokenizer.py --stage 1` 用 `ch02/end/src/llm101/baseline.py` 里冻结的字符表副本，只用于对比。
- `ch02/end/` 不再生成 `architecture-N.svg`；模型没变，逐节重画架构图是噪声。2.6 用简化模型栈画 `15-tokenizer-model.svg`，并注明完整架构见 1.8。

### API 与契约

- 新增 `BPETokenizer`：`train`、`vocabulary`、`vocab_size`、`unk_id`、`bos_id`、`eos_id`、`encode`、`decode`、`save`、`load`；`__init__(initial_tokens, merges, *, special_tokens=SPECIAL_TOKENS)` 负责校验并建词表。
- 三个模块级辅助函数：`merge_pair(symbols, pair)`、`count_pairs(sequences)`、`learn_merges(texts, num_merges, *, min_frequency=2)`；`Pair = tuple[str, str]`。
- 三个指标函数：`token_count(tokenizer, text)`、`compression_ratio(tokenizer, text)`、`unknown_count(tokenizer, text)`；放在 `src/llm101/tokenizer.py`，因为只依赖 tokenizer。
- 词表顺序固定：特殊 token（按 `special_tokens` 给定顺序）→ 初始码点（字典序）→ 合并结果（合并顺序）。`<unk>`/`<bos>`/`<eos>` 的 id 是 0/1/2；`bos_id`/`eos_id` 的数值从第 1 章的 66/67 变成 1/2，含义不变。
- `encode`/`decode` 的签名与关键字参数与第 1 章一致；未知码点编码成一个 `<unk>`，`decode` 默认跳过特殊 token，所以往返对未知字符有损。特殊 token 只由 `add_bos`/`add_eos` 控制。
- `save` 只写 `format_version`、`special_tokens`、`initial_tokens`、`merges` 四个字段；`load` 重建词表。不保存派生词表，避免文件与代码不一致。
- 本章不加入 `<pad>`；到真正需要补齐序列的章节再引入。

### 教学语料

- `samples/english.txt`、`samples/chinese.txt`：各 200 行，一行一句，UTF-8/LF，无空行；主题限定在动物、家庭、天气、日常动作；全部自写，随仓库文字采用 CC BY 4.0；不下载、不嵌入外部数据集原文。
- `samples/README.md` 记录行数、字符数、许可、固定示例出现次数；`samples/heldout_english.txt`、`samples/heldout_chinese.txt` 只给 2.5 评估，不参与训练。
- 固定示例 `the cat sat on the mat`、`小猫坐在垫子上` 各整行出现至少 3 次；留出句里至少有一个训练语料没有的生僻字，用于 OOV 演示。
- 目标词表 512：$V = S + A + M$，其中 $S$ 是特殊 token 数、$A$ 是初始码点数、$M$ 是合并新增的 token 数（按字符串去重后）。如果初始码点数 $A$ 太大、可用合并少于 100，补充语料或把目标调低到 $3+A+100$，并在正文记录实际值；不硬凑 512。

### 测试与验收

- `ch02/start/tests/` 三个红测试：`test_learn_merges_shrinks_symbols`（2.2）、`test_tokenizer_contract_round_trip`（2.4）、`test_vocab_size_matches_model`（2.6 验收）。失败信息含 `TODO 2.x`。
- `ch02/end/tests/`：`test_tokenizer.py`（合并、词表、编解码、保存/加载、约定）、`test_report.py`（指标与报告）、`test_samples.py`（语料格式与固定示例）、`test_regression.py`（固定 id logits 回归、词表一致性）、`test_structure.py`（第 1 章模型测试，保持通过）。
- 回归 golden：旧配置 `vocab_size=68`、`d_model=16`、2 层、2 头、`d_ff=32`、`max_seq_len=16`、`generator=manual_seed(7)`、手写 ids `[5, 12, 3, 9]`；从 `ch01/end` 生成 logits 存到 `tests/data/ch01-logits.json`；比较用 `atol=1e-6`、`rtol=0`。
- 验收：`ch02/end` 全绿；`ch01/end` 仍 39 passed；`show_tokenizer.py --stage 1` 到 `--stage 6` 可跑；根目录测试与 `mkdocs build --strict` 通过；`samples/README.md` 统计齐全。

### 图与资产

- 图放在 `docs/assets/ch02/`，序号 01–16；同一小节可以共用序号，语义名：`01-hand-merge`、`01-token-budget`、`02-pair-loop`、`03-merge-tree`、`04-symbols`、`05-subword-methods`、`06-vocabulary-layers`、`07-vocab-growth`、`08-encode-decode`、`09-unknown`、`10-save-load`、`11-unicode`、`12-compression`、`13-oov`、`14-tokenizer-eval`、`15-tokenizer-model`、`16-regression`。
- 2.2、2.4、2.5 的扩展阅读各配一张；2.1、2.3、2.6 的正文图不承担扩展阅读。
- 改动 SVG 后把引用处 `?v=N` 加一；按 `docs/figures.md` 和 `llm101-figure-review` 在 1400/700/390 审核。

### 与第 1 章风格的差异（有意）

- 不逐节重画架构图：模型不变，只在 2.6 画一次简化模型栈；每节仍有机制样例图。
- 不新增 2.7 验收节：2.1 兼作起点，2.6 兼作接入模型、验收与交接；六个小节各讲一个机制。
- 仍保留“每节一个可运行 stage、同一对固定样例、节末贴实际输出”的螺旋做法。

### 落地顺序

1. 建 `ch02/start/` 与 `ch02/end/`：复制 `ch01/end/`，删除第 1 章专用文件，换 BPE 空壳与三个红测试。
2. 写教学语料与 `samples/README.md`；跑 `test_samples.py`。
3. 2.1 正文与 `show_tokenizer.py --stage 1`；2.2 合并算法；2.3 词表；2.4 编解码与保存/加载。
4. 2.5 指标与报告；2.6 模型接入、golden 回归与全章验收。
5. 同步 `docs/course/ch02/index.md`、`mkdocs.yml`、`docs/glossary.md`、`docs/variables.md`，并补 `scripts/check_snapshots.py` 与 CI 章节矩阵。

## 15. 第 2 章编排审计（概念引入时机）

按 `STYLE.md` 的“引入时机”检查第 2 章。

|概念/组件|计划位置|第一次真正用到|判断|处理|
|---|---|---|---|---|
|常见片段、符号|2.1|2.1 的占位管线|只需要说明“把片段当一个单位”|2.1 不出现 BPE、子词、码点、相邻对|
|BPE、子词、码点、相邻对、合并顺序|2.2|2.2 的合并算法|需要先有符号序列和频率|2.2 写全称、译名、来历和手算|
|词表训练、特殊 token 顺序|2.3|2.3 的 `train`|需要先有合并列表|2.2 只讲合并，不讲词表训练和 id|
|`encode`/`decode`、未知字符哨兵|2.4|2.4 的编码/解码|需要先有词表|2.3 只检查 `vocabulary`，不实现编解码|
|保存/加载|2.4|2.4 的 `save`/`load`|需要完整 tokenizer|不放到 2.3|
|压缩率、OOV、序列长度|2.5|2.5 的报告|需要编码和留出句|2.4 只做往返与未知字符|
|联合词表|2.5|2.5 的中英对比|需要两种语言都能编码|2.3 只建一份词表，2.5 才比较|
|固定 id logits 回归|2.6|2.6 的测试|需要旧模型和 golden|不在 2.5 提前；不用新文本 id|
|`vocab_size` 接入模型|2.6|2.6 的前向|tokenizer 冻结后才接|2.3 只算词表大小|
|`<pad>`|推迟|真正需要 padding 的章节|本章不补齐序列|不加|
|字节级 BPE|2.2 扩展阅读|不实现|只解释与码点级的差别|不写字节级代码|
|`initial_alphabet`、`max_texts`|第 7 章|真实语料的采样训练|教学语料不需要|本章不引入|
|`CharacterTokenizer`|2.1 基线|2.1|2.2 起被替换|`ch02/end` 删除；2.1 用 `ch01/end` 与脚本冻结副本|
|模型架构图|2.6 简化模型栈|2.6|模型没变，逐节重画是噪声|不生成 `architecture-N.svg`；2.6 画 `15-tokenizer-model.svg`|

## 修订记录

- 2026-11-23：建立本文件，记录快照链结构、四项工程决策与十一章清单。
- 2026-11-23：可视化改为 Python web server + 前端分离（**已废止**，见 §3.3）：服务器导入本章代码，每层一个 API，浏览器不重复计算；可视化资源只放章节目录内，`docs/` 不再复制；相应去掉 Node 对齐测试。
- 2026-11-23：第 1 章配置从 `d_model=32`/`d_ff=64`（19,840 参数）缩小到 `d_model=16`/`d_ff=32`（5,824 参数），理由是页面展示与手算核对优先；同时明确本文件只是草案，后续可调整。
- 2026-11-23：生成 `ch01/` 骨架（小节骨架 + `start` 脚手架 + `end` 占位），并新增第 10 节交接清单。
- 2026-11-26：确定术语出现顺序：专业术语不早于介绍并实现它的章（章内按小节）；第 1 章的章节总览与 1.1 按此重写。
- 2026-11-26：术语首次出现附中文译名（多种通用译法全列），无公认译名不强译；新增书末术语表 `docs/glossary.md`。
- 2026-11-26：要求每节配 SVG 结构/数据流图与输入输出例子，节末加“数学补充”；1.3 按此重写。
- 2026-11-26：第 1 章改为螺旋编排：每节用同一示例输入跑通管线，未实现组件用占位，节末给输出与架构图；详见第 12 节。
- 2026-11-26：要求文章与示例里的 Python 代码带完整类型注解；新增 `tests/test_docs_examples.py` 检查 `docs/**/*.md`。
- 2026-11-26：tensor 概念提前到 1.1，先讲“多维数组”的本质；新增“扩展阅读”讲概念动机。
- 2026-11-26：每节按“接口 → 参考实现 → 逐函数实例结果”组织，解释结果并配矩阵/多阶段图。
- 2026-11-26：全书采用教科书式风格（`STYLE.md`）：概念与函数逐个介绍，复杂实现给分步分析与图例。
- 2026-11-26：新增“概念的引入时机”规范（`STYLE.md`）与第 1 章编排审计（第 13 节）：tensor 移到 1.3、`<pad>` 删除、`Parameter` 移到第 3 章、`Linear` 移到 1.6。
- 2026-11-27：公式改用 MkDocs `pymdownx.arithmatex` + MathJax 渲染；正文一律用 `$...$`/`$$...$$`，不再用代码块或反引号写公式。
- 2026-11-27：新增全书变量表 `docs/variables.md`；短变量首次出现必须说明含义；修正“X 是 [..]”为“X 的形状是 [..]”，并区分特征维 `C` 与 `Linear` 的 `in_features`/`out_features`。
- 2026-11-27：写作规范补充“每个公式下面给最小计算范例”；1.6 展开“头/拆头/多头”和“softmax 的性质、动机、与注意力的关系”。
- 2026-11-27：规范补充“矩阵/向量/张量值和运算用数学公式，不用代码块”；1.6 的矩阵例子改为 `bmatrix`/`aligned`，1.5/1.4 的小例子同步。
- 2026-11-27：取消第 1 章的 web server、浏览器前端和逐层 trace（含 `forward_with_trace`、stage 8、`test_api.py`、`web/`）；第 1 章收为 1.1–1.9，1.9 只做验收、小结与交接，原理展示继续用 SVG 图、命令行输出和测试。
- 2026-11-27：补写作规范：新增 `docs/figures.md` 图规则入口；`STYLE.md` 增补“断言配例、先修顺序、术语来历、库函数介绍、每节两张图、公式复核、轴标签/边距”；`ch01/README.md` 增补提交与发布、指针自检、图审核；修正过期 1.10/服务器引用、39/40 测试数字、1.7 上游图角色和根 README 章节数；CI 增加 `ch01/end` 测试。
- 2026-11-27：按全书流程重排第 5–11 章为“优化器 → 小模型训练 → 真实语料/最终 tokenizer → 缩放 → 推理 → 限时训练 → 评估”；明确第 1 章 API 为基线、`final/` 第 11 章再生成、第 2 章 tokenizer 契约测试、第 3 章 GELU 精确化、第 8 章初始化/缩放、教学语料和快照校验规则；同步第 10/11 节交接与未决项。
- 2026-11-27：完成第 2 章规划：新增 `ch02/README.md`、`ch02/sections/2.1.md`–`2.6.md`，确定固定英中样例、2.1 手写合并占位、六个 stage、BPE 契约、教学语料规格、固定 id logits 回归、17 张图的语义名与概念引入审计；新增 §14–15，更新第 2 章清单、交接清单与路线图状态。
- 2026-11-27：第 2 章 2.1 已写完：`ch02/start/` 的 BPE 空壳与 3 个红测试、`ch02/end/` 的 stage-1 基线演示与图 2.1-2；固定样例改为同义平行对：英文 `the cat sat on the mat`、中文 `小猫坐在垫子上`；英文样例本身承担 24>16 的长度反例，不再单独设长句；2.1 的手写合并改为 `a+t`、`s+at`、`m+at`、`t+h`、`th+e` 五步，22 个符号降到 13 个，加特殊 token 后 15。

- 2026-11-27：为 2.1 增加手写合并工作示意图（图 2.1-1，22→19→18→17→13）；原 token 预算图改为图 2.1-2。

- 2026-11-27：2.2 已写完：`merge_pair`、`count_pairs`、`learn_merges`，stage 2、7 项合并测试与 4 张图；2.3 起建立教学语料和词表。
- 2026-11-27：2.2 评审修正：字节级 BPE 的首次介绍改为 2.2 扩展阅读，并同步术语表、章节 README 与正文的固定样例说明。

- 2026-11-27：2.3 已写完：教学语料 `samples/`（400 行、12,709 字符、143 码点）、`BPETokenizer.train` 与 512 词表、stage 3、词表层图与增长图；2.4 起实现编解码。
