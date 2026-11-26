# 课程重构计划：快照链

记录日期：2026-11-23。状态：**规划草案，尚未实现任何章目录。**

> 本文件记录的是目前讨论的结论，不是最终决定。章节划分、目录命名、模型规模、页面形式都可能在实现过程中调整；不要把它当成不可变的规范。每次调整请同步修改相关段落，并在末尾“修订记录”追加一条，避免文档与实际代码不一致。

本文件汇总目前的课程结构设想，供以后回顾。当前仓库里的 `src/`、`scripts/`、`tests/` 仍是旧的“先实现、再补文档”形态；若本计划落地，它们会被拆进 `ch01/` … `ch11/` 的快照链，`final/` 由 `ch11/end` 产生。

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

以下八项是目前采纳的方案；若后续实验或读者反馈暴露问题，可以再改，改动时同步更新本章节与修订记录。

1. 目录名用 `start/` 与 `end/`，避免与根目录的 `final/` 混淆。
2. `uv.lock` 只保留在仓库根和 `final/`；各章 README 用 `uv sync`（不带 `--locked`），避免 22 份锁文件副本淹没历史。
3. 可视化用 **Python web server + 浏览器前端**的分离结构：服务器导入本章代码，为每一层提供一个 API，请求体是上一层的输出，响应是本层的中间量与输出；浏览器只发请求和渲染，不重复实现任何数学。服务器用 FastAPI + uvicorn（作为 `demo` 可选依赖），测试用 `TestClient` 校验每个端点的形状，并有一个“接口链结果 == `TinyGPT.forward`”的等价性测试。放弃浏览器端 JS 前向方案：它要么重写一遍数学、容易与 Python 漂移，要么只能选预置句子。
4. CI 只运行每个 `end/` 的测试（共用一套环境，`PYTHONPATH` 指向各快照）；`start/` 不跑测试，只校验它与上一章 `end/` 逐字节一致。
5. 第 1 章直接采用与 `final/` 相同的模块划分，只是暂时只有 `forward`。
6. 可视化资源（HTML/JS/CSS）只存在于章节目录内，由本地服务器提供；`docs/` 只放说明页，不复制页面文件。
7. **术语不提前出现。** 专业术语第一次出现不得早于具体介绍并实现它的那一章。在此之前，要么不使用该术语、也不提及相关内容，要么用通俗语言简单描述，并明确告诉读者后续会在哪一章介绍。第 1 章内部按小节执行同一规则：正文只使用本节或更早小节已经介绍过的术语；代码标识符可以照实展示，但要说明它对应哪件事。小节清单与导航是路线图，可以提前列出标题，正文不能假定读者已经认识这些名字。
8. **术语译名与术语表。** 有通用中文译名的术语第一次出现时附上中文译名；有多种通用译法的全部列出（例如 tokenizer：分词器、标记器、词元切分器）。transformer 这类没有公认译名的术语不强译，保留原文。书末维护术语表 `docs/glossary.md`：按首次出现顺序列出术语、中文译名、一句话说明，并链接到详细介绍或实现它的小节；每写完一节，把新术语补进表中。

## 4. 连续性规则

- **同名模块只增语义，不改已有行为。** 第 3 章给 `layers.py` 加 `backward`，不重写 `forward`。
- **必须重构时附带等价性测试**：例如第 2 章把字符级词表换成 BPE，要有一个测试证明同一权重下手写输入得到的 logits 与重构前一致。
- 每章 `start/README.md` 固定写三段：本章要做什么、先跑哪个失败测试、按什么顺序读代码。
- 每章 `end/README.md` 固定写：本章新增了什么、怎么验证、下一章的入口在哪。

## 5. 章节清单

### 第 1 章 看得见的 transformer

- `start/`：工程脚手架 + 三个失败测试（输出形状、因果掩码、上下文上限）。
- `end/`：`tokenizer.py`（字符级）、`parameters.py`（`Parameter`，含全零 `grad` 但本章不用）、`layers.py`（`Linear`、`Embedding`、`LayerNorm`、`GELU`，只有 `forward`）、`attention.py`（`MultiHeadSelfAttention`，只有 `forward`）、`model.py`（`GPTConfig`、`TransformerBlock`、`TinyGPT`，并把逐层中间量暴露出来）、`web/`（FastAPI 应用 + 静态前端）、`scripts/serve_demo.py`、`tests/test_api.py`。
- 小节骨架：`ch01/sections/1.1.md` 到 `1.11.md`，每份只写目标、内容安排、涉及代码、验收与写作注意，正文留给后续。
- 验收：`pytest` 全绿；`uv run python scripts/serve_demo.py` 后浏览器能输入文本并逐层查看张量；接口链拼接出的 logits 与 `TinyGPT.forward` 一致；改后段 token 不影响前段注意力输出。

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
- 结构：2 层 Pre-LN decoder-only，`d_model=16`、2 头（每头 8 维）、`d_ff=32`、上下文 16，可学习绝对位置，共享输入/输出嵌入。
- 规模：5,824 个参数；float32 权重 23 KiB，网页端 JSON 约 40 KB。刻意保持这么小，是为了让每层的张量能一屏放下、数值可以手算核对；第 5 章再把维度提到 `d_model=256`、4 层，并说明每一步放大了什么。
- 权重：固定 seed 随机初始化，只有 forward。页面必须明确写“输出是噪声，看的是结构与数据流”，避免读者误以为模型已经会说话。
- 页面交互（前端只负责显示）：
  - 输入文本 → 调 `/api/tokenize` 与 `/api/embed`；
  - 逐层步进：每一层一次请求，拿回 LayerNorm、Q/K/V、每头注意力矩阵（因果掩码区域灰显）、注意力输出、残差、MLP 隐藏层、残差；
  - 每步显示张量形状与这一层参数量，热力图悬停显示数值；
  - 末端调 `/api/logits` 得到 logits 与 top-5，`temperature` 滑块调 `/api/sample`。
- 服务器接口：`GET /api/config`、`POST /api/tokenize`、`POST /api/embed`、`POST /api/layers/{0,1}`、`POST /api/logits`、`POST /api/sample`。每层请求体是上一层的输出，响应体是该层全部中间量与输出，形状与参数名固定。
- 测试：`tests/test_structure.py`（形状、掩码、上下文上限）、`tests/test_api.py`（端点形状与接口链等价性，容差 1e-6）。不再依赖 Node。

## 7. 数据与许可边界

- 第 1–7 章使用仓库内自带的自写短句（约 100 条），不嵌入任何数据集的原文，避免把 CDLA-Sharing 内容带进代码仓库。
- 第 8 章起接入真实语料：英文 `roneneldan/TinyStories`、中文 `52AI/TinyStoriesZh`，沿用现有 SHA-256 固定与哈希切分。
- 权重再发布仍需先完成上游许可复核；在结论明确前只发布代码与文档。

## 8. 工程约束

- 每个 `end/` 可独立 `uv sync`（不带 `--locked`）并 `pytest`。
- 快照只含代码、测试、脚本、`pyproject.toml` 与 README；不含 `data/`、`artifacts/`、`site/`、`.venv/`。
- 新增 `scripts/check_snapshots.py`：校验 `chNN/start` 与 `ch(N-1)/end` 逐字节一致、`final` 与 `ch11/end` 一致，并对每个 `end/` 运行测试。
- 体积估算：每份快照约 450 KB 文本、约 5 千行；23 份合计约 10 MB，Git 增量压缩后更小。
- 前端静态资源放在章节目录内（如 `ch01/end/web/static/`），由本地服务器提供；`docs/` 只放说明页，不复制页面文件，所以站点不需要额外构建步骤。

## 9. 与现状的关系

- 现有 11 个模块与 46 项测试是各章 `end/` 的内容来源；第 1 章需要从中删出“只有 forward”的版本，其余章节按上述增量重新分配。
- 现有 `docs/course/*.md` 将拆入对应章节，`roadmap.md` 改为章节索引。
- 现有根目录 `final/` 暂时保留；`ch11/end` 完成后由其取代并保持同步。
- 代码已收敛为单一中英混合模型，章节计划沿用这一结论，不再保留纯英文或纯中文训练路径。

## 10. 交接清单

截至 2026-11-23 的状态：

- `ch01/` 骨架已建立：`README.md`（总览、小节清单、写作流程、进度勾选）、`sections/1.1.md`–`1.11.md`（目标 / 内容安排 / 涉及代码 / 验收 / 写作注意）、`start/`（模块空壳 + 3 个红测试 + 示例文本）、`end/`（只有说明，待填写）。
- `start/` 已验证可运行：`cd ch01/start && uv sync --extra dev && uv run pytest -q` → 3 failed，全部是 `NotImplementedError`，分别对应 1.4（上下文上限）、1.6（掩码）、1.7（形状）。
- 章节目录还没有接入 CI；`scripts/check_snapshots.py` 尚未实现。提交时不要带上 `uv.lock`、`.venv/`、缓存或 `egg-info`。

新 session 从 1.1 开始时的步骤：

1. 读 `ch01/sections/1.1.md` 与 `ch01/README.md` 的写作流程。
2. 把 `ch01/start/` 整体复制为 `ch01/end/`（只做一次），之后只改 `end/`。
3. 正文写到 `docs/course/ch01/<编号>.md`；1.1 时顺带建 `index.md` 与 `mkdocs.yml` 子导航。
4. 每节按“验收”跑测试；新测试放 `ch01/end/tests/`。
5. 完成后把 `sections/<编号>.md` 的状态改为“完成”，并在 `ch01/README.md` 的进度清单里打勾。

写作约束：`start/` 里的函数签名已固定，后续章节依赖它们；第 1 章只做前向；前端只发请求和渲染。

## 11. 尚未决定

- KV cache 是否作为第 12 章（当前代码未实现，`roadmap.md` 已列为后续）。
- 第 10 章页面加载训练后权重的具体形式：训练一个小号 2 层模型专门给页面用，还是对大模型做量化后导出。
- 章节的中文标题与固定 URL 命名。

## 修订记录

- 2026-11-23：建立本文件，记录快照链结构、四项工程决策与十一章清单。
- 2026-11-23：可视化改为 Python web server + 前端分离：服务器导入本章代码，每层一个 API，浏览器不重复计算；可视化资源只放章节目录内，`docs/` 不再复制；相应去掉 Node 对齐测试。
- 2026-11-23：第 1 章配置从 `d_model=32`/`d_ff=64`（19,840 参数）缩小到 `d_model=16`/`d_ff=32`（5,824 参数），理由是页面展示与手算核对优先；同时明确本文件只是草案，后续可调整。
- 2026-11-23：生成 `ch01/` 骨架（小节骨架 + `start` 脚手架 + `end` 占位），并新增第 10 节交接清单。
- 2026-11-26：确定术语出现顺序：专业术语不早于介绍并实现它的章（章内按小节）；第 1 章的章节总览与 1.1 按此重写。
- 2026-11-26：术语首次出现附中文译名（多种通用译法全列），无公认译名不强译；新增书末术语表 `docs/glossary.md`。
