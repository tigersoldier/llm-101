# 第 2 章 真正的 tokenizer

第 1 章用字符级 tokenizer 把文字变成编号：一个字符一个编号，`the cat sat` 加上 `<bos>`/`<eos>` 是 13 个编号，已经占了上下文 16 个位置中的 13 个。第 2 章把英文样例延长一句：`the cat sat on the mat` 是 22 个字符，加 `<bos>`/`<eos>` 后 24 个编号，模型根本读不完；第 1 章的字符表还是固定的 68 个英文符号，中文 `小猫坐在垫子上` 的 7 个字全部变成 `<unk>`。

第 2 章把切分方式换成 BPE（Byte Pair Encoding，字节对编码）：从 Unicode 码点出发，反复合并语料中最常见的相邻符号对，用一份约 512 个 token 的联合词表同时覆盖英文和中文。`encode`、`decode`、`vocab_size`、`bos_id`、`eos_id` 五个接口保持不变，模型结构完全不动，只把 `GPTConfig.vocab_size` 换成新词表大小。模型仍然只做前向，权重是固定 seed 的随机初始化，输出没有语义。

整体规划见仓库根目录 `PLAN.md` 的 §5 与 §14–15，全书写作风格见 `STYLE.md`；本章小节计划在 `sections/`。

## 目录

```text
ch02/
├── README.md          # 本文件：章节总览、小节清单、写作流程
├── sections/          # 6 个小节的计划：目标、内容安排、涉及代码、验收
├── start/             # BPE 空壳 + 3 个红测试；samples/ 是 2.3 教学语料
└── end/               # 2.1 完成态；BPE、报告与回归随 2.2–2.6 加入
```

## 小节清单

|小节|标题|一句话目标|
|---|---|---|
|2.1|起点与本章交付物|跑通第 1 章基线，看到英文 24>16、中文全是 `<unk>`；明确五个接口和三个红测试|
|2.2|子词与相邻对合并（BPE）|统计相邻对、按频率合并、合并顺序就是编码规则|
|2.3|训练词表：语料、特殊 token 与目标大小|特殊 token、初始码点、合并结果怎样组成 512 词表|
|2.4|编码与解码：接口与未知字符|按顺序重放合并、处理未知字符、保存/加载与接口契约|
|2.5|中英混合与词表评估|联合词表、压缩率、OOV、序列长度与语言差异|
|2.6|接入模型、回归测试与交接|只改 `vocab_size`，用固定 id logits 回归证明模型没变|

## 本章流程（与第 1 章同一节奏）

第 1 章的螺旋编排是“每节用同一个示例输入跑一遍完整管线，节末贴实际输出”；本章沿用同一节奏，但管线的重点是 tokenizer，模型只在 2.6 出现一次。

### 固定示例

- 英文：`the cat sat on the mat`（22 个字符；加 `<bos>`/`<eos>` 后 24 个编号，超过上下文 16）。它是第 1 章 `the cat sat` 的延长，前三个词不变。
- 中文：`小猫坐在垫子上`（7 个字符；第 1 章字符表没有中文，编码结果是 7 个 `<unk>`，加特殊 token 后 9 个编号）。
- 失败点：英文样例本身就是长度失败（24>16）；中文样例是覆盖失败（7 个 `<unk>`）。不再单独设长句样例。

两条固定样例一直用到 2.6：英文用来看空格和常见字母组合，中文用来看“没有空格”和“码点级切分”。正文里的数字一律来自实际运行，不写预期值。

### 管线随小节增长

|节|本节新增/换上的步骤|节末示例输出|
|---|---|---|
|2.1|第 1 章字符基线 + 手写片段合并占位|英文 24>16；中文 7 个 `<unk>`；手写合并后 13 个符号（加特殊 token 15）|
|2.2|`merge_pair`、`count_pairs`、`learn_merges`|16 条合并；英文 5 个、中文 2 个片段|
|2.3|`BPETokenizer.train`、词表顺序、特殊 token|512 个 token；英文 4、中文 3 个片段|
|2.4|`encode`、`decode`、`save`、`load`|往返文本；未知字符的 `<unk>` 行为|
|2.5|`tokenizer_report.py`|英文/中文的字符数、token 数、压缩率、OOV|
|2.6|`GPTConfig.vocab_size` + 固定 id 回归|`[T, V]` logits、argmax 片段；回归一致|

2.1 只建立问题和基线，不引入 BPE 名称；2.2 才介绍 BPE 并实现合并；2.3 建好词表与编号顺序；2.4 完成文本 ↔ 编号；2.5 做评估；2.6 接回模型并收束全章。

### 2.1 的占位组件

|阶段|占位做法|
|---|---|
|文本 → 编号|第 1 章 `CharacterTokenizer`，固定 68 个英文符号|
|常见片段 → 一个符号|手写 `a`+`t`→`at`、`s`+`at`→`sat`、`m`+`at`→`mat`、`t`+`h`→`th`、`th`+`e`→`the`|

手写合并只用于 2.1 演示，让英文样例从 22 个符号降到 13 个（`the`、空格、`c`、`at`、空格、`sat`、空格、`o`、`n`、空格、`the`、空格、`mat`），加 `<bos>`/`<eos>` 正好 15，进入上下文 16。它故意简单、确定、可以手算，不是教学重点。2.2 起改由语料统计决定合并哪些对。

### start/ 与 end/ 的清单

`ch02/start/` 从 `ch01/end/` 复制后：

- 删除：`src/llm101/placeholder.py`、`scripts/show_placeholder.py`、`scripts/render_architecture.py`、`tests/test_placeholder.py`、`CharacterTokenizer` 及其测试。
- 保留：`model.py`、`layers.py`、`attention.py`、`parameters.py`、`tests/test_structure.py`、`tests/test_layers.py`。
- 新增：`tokenizer.py` 的 BPE 空壳与三个红测试；`samples/` 与 `samples/README.md`（2.3）。

`ch02/end/` 不保留第 1 章专用文件，也不做兼容层；2.1 的字符基线在 `ch01/end` 里跑，`show_tokenizer.py --stage 1` 使用 `src/llm101/baseline.py` 里冻结的字符表副本。

### 运行方式

```bash
cd ch02/end
uv run python scripts/show_tokenizer.py --stage 1
uv run python scripts/show_tokenizer.py --stage 2
uv run python scripts/show_tokenizer.py --stage 3
uv run python scripts/show_tokenizer.py --stage 4
uv run python scripts/show_tokenizer.py --stage 5
uv run python scripts/show_tokenizer.py --stage 6
```

- `--stage N` 只使用第 N 节已有的步骤；后面的步骤不出现。
- stage 1 的字符基线是 `src/llm101/baseline.py` 里冻结的副本，只用于对比；库接口从 2.2 起是 BPE。
- 每节正文末尾贴 `--stage N` 的实际输出：2.1 是编号、符号数和 `<unk>` 数；2.2 起是符号序列或 ids；2.6 是 logits 形状、argmax 和回归结果。

### 图与架构

- 图放在 `docs/assets/ch02/`，文件名用“序号-语义名”（如 `02-pair-loop.svg`）。
- 每节至少一张机制样例图；2.2、2.4、2.5 的扩展阅读各加一张。
- 模型架构不变，本章不重画 `architecture-N.svg`。2.6 的 `15-tokenizer-model.svg` 用简化模型栈把 tokenizer 和模型接起来，并写明完整架构见 1.8。
- 新增或修改 SVG 后，按 `docs/figures.md` 和 `llm101-figure-review` 在 1400/700/390 审核；没通过不算完成。

## 写作流程

每节按同一节奏进行：

1. 读 `sections/<编号>.md` 的“目标 / 内容安排 / 验收”。
2. 正文写到 `docs/course/ch02/<编号>.md`；页面上线时同步更新 `mkdocs.yml` 的子导航和 `docs/course/ch02/index.md` 的小节链接。
3. 代码在 `ch02/end/` 里实现。**2.1 时先把 `ch01/end/` 复制成 `ch02/start/`**，按 `sections/2.1.md` 的清单删除第 1 章专用文件、换上 BPE 空壳和三个红测试；之后只改 `end/`，`start/` 保持脚手架原样。
4. 跑该节“验收”里列出的测试或命令；新增测试放 `ch02/end/tests/`。
5. 全节写完先做一次指针自检：`grep -nE "第 [0-9]+ (章|节)|[0-9]+\.[0-9]+ " docs/course/ch02/*.md ch02/README.md`，确认没有指向已取消或已改号的章节或小节。
6. 把 `sections/<编号>.md` 的“状态：待写”改成“状态：完成”，并在本文件的进度清单里打勾。
7. 全章写完后，`ch02/end/` 即为第 3 章 `start/` 的来源。

**术语顺序**：正文只使用本节或更早小节已经介绍过的术语。2.1 不出现 BPE、子词、码点、相邻对；用“常见片段”“符号”，并明确“2.2 会讲算法”。2.2 第一次出现 BPE 时写全称与中文译名；码点、相邻对、合并顺序，以及字节级 BPE、WordPiece、Unigram 都在 2.2 介绍。

**词表训练不是模型训练**：2.3 第一次出现“训练 tokenizer”时，明确说明这是统计合并，没有参数、没有损失、没有反向传播；模型训练在第 6 章。

**类型注解**：正文和示例里的 Python 代码必须带完整参数与返回注解（含 `-> None`），示例与真实代码一致；`tests/test_docs_examples.py` 检查 `docs/**/*.md` 的 Python 代码块。

**函数结构**：每节按概念小节组织，函数在自己的概念小节里给用法、输入输出、输入输出样例和带注释的签名（不用表格）；再给参考实现；最后给每个函数的实际结果。复杂实现按 `STYLE.md` 先写编号步骤、最小例子和中间结果，再配矩阵或多阶段图。只对最终算法相关部分要求读者实现；示例、冻结基线、手写占位等临时算法（例如 2.1 的 `baseline.py`）直接按概念分段嵌入代码，不设空壳和独立的“参考实现”块。

### start/README 与 end/README 的固定段落

`ch02/start/README.md`：

1. 本章要做什么：把字符级 tokenizer 换成 BPE，保持五个接口，模型代码不改；先看教学语料与三个红测试。
2. 先跑哪个失败测试：`cd ch02/start && uv sync --extra dev && uv run pytest -q`，应看到 3 个失败，失败信息指向 `TODO 2.2`、`TODO 2.3`；第一个测试在 2.2 转绿，后两个在 2.4 转绿，2.6 纳入整章验收。
3. 按什么顺序读代码：`tokenizer.py` 的 BPE 空壳 → `samples/README.md` → `tests/test_tokenizer.py` 的三个红测试 → `model.py` 的 `GPTConfig`（只看 `vocab_size`）。

`ch02/end/README.md`：

1. 本章新增了什么：BPE tokenizer、英中教学语料、指标与报告脚本、固定 id logits 回归。
2. 怎么验证：`cd ch02/end && uv sync --extra dev && uv run pytest -q`；`uv run python scripts/show_tokenizer.py --stage 6`；记录实际测试数与输出。
3. 下一章入口：`ch02/end/` 是第 3 章 `start/` 的来源；tokenizer 与 `samples/` 冻结，模型结构不变，第 3 章加 `Parameter`、`backward` 和交叉熵。

## 术语与引入时机

每节写完把新术语补进 `docs/glossary.md`（BPE、子词、码点、相邻对、合并、合并顺序、字节级 BPE、WordPiece、Unigram、联合词表、压缩率、OOV），把短变量补进 `docs/variables.md`（$V$、$A$、$M$、$R$、$N$；$T$ 在 tokenizer 章节表示 token 数）。第 2 章的值单独成列或成节，不覆盖第 1 章的取值。

|术语/概念|首次出现|处理|
|---|---|---|
|常见片段 / 符号|2.1|用通俗语言说明，不出现 BPE|
|BPE、子词、码点、相邻对、合并顺序|2.2|BPE 写全称 Byte Pair Encoding 与中文译名；码点写 Unicode code point|
|词表训练、特殊 token 顺序、`<unk>`/`<bos>`/`<eos>`|2.3|特殊 token 已在第 1 章出现；这里只定顺序与 id|
|重放合并、未知字符、保存/加载|2.4|未知字符行为与第 1 章一致：一个码点一个 `<unk>`，解码默认跳过|
|联合词表、压缩率、OOV、序列长度|2.5|OOV 在第 1 章术语表已出现，这里第一次量化|
|固定 id logits 回归|2.6|只用旧配置和手写 id，不用新文本 id|
|`<pad>`|推迟|本章不加；到真正需要补齐序列的章节再引入|
|字节级 BPE|2.2 扩展阅读|只解释与码点级的差别，不实现|
|`initial_alphabet`、`max_texts`|第 7 章|本章不引入|

## 本章不做什么

- 不训练模型、不做反向传播、不引入优化器；模型仍然只做前向，输出是随机权重的噪声。
- 不接 TinyStories，不做 8192 词表（第 7 章）。
- 不加入 `<pad>`，不实现 padding 或 padding mask。
- 不做字节级 BPE、正则预切分、空白符特殊处理、GPT 系列 tokenizer。
- 不重画模型架构图；模型代码不改，只改 `GPTConfig.vocab_size`。

## 进度

- [x] 2.1 起点与本章交付物
- [x] 2.2 子词与相邻对合并（BPE）
- [x] 2.3 训练词表：语料、特殊 token 与目标大小
- [ ] 2.4 编码与解码：接口与未知字符
- [ ] 2.5 中英混合与词表评估
- [ ] 2.6 接入模型、回归测试与交接

## 提交与发布

- 直接提交并推送到主分支，不再开 PR；一次提交只做一件事（一节正文、一组图、或一次规范更新）。
- 推送后确认 Tests 与 Pages 工作流跑完，再回报“可在线查看”并给出页面地址。
- 只改被点名的范围；用户点名改正文时，不改小节标题、不改小节顺序。
- 图和正文都改了时，先确认 SVG 的 `?v=N` 已更新，再检查页面是否拿到新图。

## 交接

`ch02/end/` 是第 3 章 `start/` 的来源。第 3 章沿用本章的 tokenizer、`vocab_size` 和 `samples/`，添加 `Parameter`、各层 `backward` 和交叉熵；第 4 章做注意力/block/整模型反向；第 5–6 章做优化器与训练循环；第 7 章才换成真实语料与 8192 词表。`ch02/end/` 冻结后，本章不再改 tokenizer 的五个接口。
