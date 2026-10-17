# 个人健康 / 可穿戴数据的数据科学 Agent：开源现状与评测基准调研

> 来源：委派给只读 researcher 子 Agent 的调研结果（2026-10-17）。
> 调研方式：无通用联网搜索工具可用，全部结论来自 arXiv API / OpenAlex REST / Crossref / GitHub REST + raw.githubusercontent /
> PubMed / PyPI 的直接抓取，以及论文 PDF 全文抽取（`arxiv.org/pdf/2508.20148` → 148 页纯文本）。
> **核实标注约定**：`已核实` = 本次调研直接抓取到一手证据（论文原文、仓库文件、API 返回值）；
> `推测` = 基于 README/文档的合理推断，或未能取到一手证据。
> **未复核项**：报告中所有 star 数、`pushed_at`、fork 数、文件大小均为 2026-10-17 未认证 GitHub API 的实测值，属时点值，未经第二方复核。

---

## 0. 结论先行

1. **第 1 问的答案是确定的**：描述中的论文是 **《The Anatomy of a Personal Health Agent》**（Google，arXiv:**2508.20148**，2025 年 8 月）。
   四条文线索（"first personal open-ended data science agent"、"personal and population-level time-series wearable data"、"open-source an evaluation benchmark"、"Supplemental Figure S3"）在论文 §3.1 的 **同一段** 中逐字全部命中。**已核实**。
2. 它同时回答了第二问用户没问的隐患：这不是一篇只有架构图的论文——**它开源了完整的 DS Agent 实现代码**（`Google-Health/consumer-health-research/personal_health_agent`，Apache-2.0），
   而且是 **FastAPI + 多 Agent 编排** 的形态，与本项目技术栈高度同构。这是本次调研最有价值的发现。
3. **但论文声称的 "open-source benchmark" 有落差**：141 条 query–analysis-plan 专家打分对、26 项 rubric、173 个单元测试，
   在公开仓库的**文件树中不存在**（已核实：`personal_health_agent` 下 164 个路径只有 `data/sample/` 4 个 CSV、`few_shots/`、`notebooks/`、`tests/`）。
   不过 **rubric 本身在源码里有硬编码**（`pha/prompts/data_science_prompts.py` 的 `RUBRIC_QUESTIONS`），
   加上 sample 数据和 baseline/eval 路由分支，方法论是可复现的。**benchmark 数据集是否真的公开：未核实，倾向"未随仓库发布"。**
4. **同类开源项目里，没有任何一个能同时满足"计划→代码→执行→自动修错 + 统计方法选择正确性 + 个人健康时序数据"**。
   - 通用天花板是 **smolagents CodeAgent**（Apache-2.0、29,874★、活跃）与 **pandas-ai**（MIT 核心、23,879★）；
   - **LangChain 的 pandas dataframe agent 所在包 `langchain-experimental` 正在被 sunset**（官方 README 警告，已核实）——不该作为长期依赖；
   - **Data Interpreter 不是 THUDM 项目**（`THUDM/DataInterpreter` GitHub API 返回 404，已核实）：论文摘要末句逐字给出代码位置 `https://github.com/geekan/MetaGPT`；另有 `FoundationAgents/OpenManus` 的 DataAnalysis Agent 是同族实现；
   - PHA 的 DS Agent 是唯一在 prompt 层显式要求"命名检验 + 说明假设 + 说明如何检验假设（如 Shapiro-Wilk）"的实现。
5. **评测基准方面，"数值正确性"与"统计方法选择正确性"分开评分的基准极少**，最贴合的正是 PHA 自己的两任务设计：
   Task 1 用 26 项 rubric 给 **plan（含独立的 Statistical Tests 维度）** 打分（LLM autorater，与人 ICC(3,1)=0.838），
   Task 2 用 **173 个单元测试** 给代码产出的**数值/边界正确性**打分。另一类是 **StatABench**（Stat-Closed 404 题 + Stat-Open 30 题，Stat-Open 用 LLM-as-Judge 打建模过程）。
6. **没有任何公开基准覆盖"个人级可穿戴时序数据的开放式数值问答"** —— 这正是 PHA 论文自称填补的空缺，也是本项目如果要自建评测的差异化空间。
7. 从 PHA 源码可直接复用的 6 条工程做法见 §4，其中价值密度最高的是：**重试预算的三层分离（3 次内层修错 / 5 次外层总步数 / 1 次逻辑精修）**、
   **执行结果用 Python 对象而非 stdout 传递**、**DataFrame 以 CSV 字面量注入沙箱（模型无法凭空编造数据）**、**`[SUMMARY]` / `[CODE UPDATE]` 双标签协议**。

---

## 1. 第 1 问：论文定位

### 1.1 结论

| 项 | 值 |
|---|---|
| 标题 | **The Anatomy of a Personal Health Agent** |
| arXiv | **2508.20148**（2025 年 8 月，v2）· https://arxiv.org/abs/2508.20148 |
| 领域 | cs.AI |
| 作者 | A. Ali Heydari, Ken Gu, Vidya Srinivas, Hong Yu, Zhihan Zhang, Yuwei Zhang, Akshay Paruchuri, Qian He, Hamid Palangi, Nova Hammerquist, et al.（全部 Google） |
| 项目名 | **PHA**（Personal Health Agent）；其中的 DS 子 Agent 即 **Data Science Agent** |
| 论文内引用键（仓库 README 原文） | `@article{heydari2025anatomy, title={The anatomy of a personal health agent}, author={Heydari, A Ali and Gu, Ken and ... }, journal={arXiv preprint arXiv:2508.20148}, year={2025}}` |
| 官方仓库 | https://github.com/Google-Health/consumer-health-research/tree/main/personal_health_agent **已核实** |
| 期刊版本 | PHIA 仓库 README 称 PHA "accepted as a journal article to be published in Nature"；**本次未在 OpenAlex/Crossref 检索到已发表版本（未核实）** |

**四条文线索的逐字命中（已核实，全部位于论文 §3.1 "User-Centered Agent Design" → "Data Science (DS) Agent" 段落）**：

| 用户给出的线索 | 论文原文（逐字） |
|---|---|
| "first personal open-ended data science agent" | "We develop **the first personal open-ended data science agent** capable of analyzing both personal and population-level time-series wearable data to provide numerical health insights." |
| "personal and population-level time-series wearable data" | 同上一句 |
| "open-source benchmark" | "**We also open-source an evaluation benchmark** for these capabilities." |
| "Supplemental Figure S3" | "Example conversations between a user and the DS agent are shown in **Supplemental Figure S3**." |

抽取路径：`https://arxiv.org/pdf/2508.20148` → pypdf → 148 页、368,350 字符纯文本。上述四句在同一 500 字窗口内连续出现，不存在歧义。

### 1.2 候选与排除

| 候选 | 匹配度 | 判断依据 |
|---|---|---|
| **① PHA《The Anatomy of a Personal Health Agent》arXiv 2508.20148** | **4/4 命中** | 四条线索逐字命中；论文含 DS/DE/Coach 三个子 Agent；Figure 3 即 DS Agent 架构图（计划→代码→执行→自纠正） |
| ② PHIA《Transforming wearable data into personal health insights using large language model agents》arXiv **2406.06464** = Nature Communications **10.1038/s41467-025-67922-y**（PMID 41526380） | 2/4，**是前作而非目标** | 同一作者团队（Merrill 等）的前序工作，2024-06 预印本 / 2026-01 正刊。它是 ReAct 单循环 Agent（Google DeepMind `onetwo` 库），**没有独立的"分析计划生成"阶段**，也不自称 "first personal open-ended data science agent"；其 benchmark 是 4000 条 objective + 172 条 open-ended 查询，不是 plan 打分集。**已被用户描述的架构排除**（已核实：读过其 `phia_agent.py` 与 `prompt_templates.py`） |
| ③ PHIA 仓库里指向的"继任者"就是 PHA 本身 | — | PHIA README 原文：本仓库 banner 指向的更新版是 PHA，称其为 "a comprehensive, multi-agent successor to this repo's agent, PHIA"；这条交叉引用独立佐证了 ① 与 ② 的谱系 |

检索路径备注（供复现）：arXiv API 的 `all:"exact phrase"` 短语检索对这类描述**无效**（返回 0 条），最终靠 OpenAlex `search=` 相关性检索命中 PHIA，再从 PHIA 仓库 README 顺链找到 PHA。

### 1.3 DS Agent 架构（论文描述 + 源码实现）

**论文层面的架构（已核实，逐字引用）**：

> "the DS Agent's architecture is designed as a modular, **two-stage pipeline**: (1) **Analysis Plan Generation** and (2) **Code Generation with Iterative Execution**. This decomposition is critical for mitigating the risk of factual inaccuracies and logical errors common in end-to-end generation for complex data tasks. It separates the semantic reasoning of **'what to do'** from the syntactic task of **'how to do it.'**"

> Figure 3(a) caption: "The Architecture of the DS Agent, which first generates a detailed, natural language statistical analysis plan from a user's query and a data schema summary ... Subsequently, it produces and executes code to implement this plan. **An iterative self-correction loop allows the agent to revise the code based on execution errors, the original query, and the analysis plan.**"

> "The generated code is then executed in a sandboxed environment. We employ an **iterative refinement loop**: if execution fails due to syntactic errors or runtime exceptions, the agent **re-invokes the stage with the original query, plan, and the error message as feedback**."

**论文声明的两项能力（§4.1，已核实）**：

- **DS1** — 把开放式查询分解为结构化分析计划：(a) 识别概念变量；(b) 明确的数据转换步骤 = ① 过滤 ② 清洗脏数据 ③ 连接表 ④ 聚合；(c) 识别、标注并处理缺失或不足的数据；(d) 应用合适的统计检验。
- **DS2** — 通过代码生成、执行与反馈**计算准确的数值结果**。

注意论文的一句自我约束：**"in one single interaction, our DS Agent is designed to provide at least one appropriate response, rather than an exhaustive enumeration of all possibilities."** —— 它不追求穷举分析路径，只保证"至少给出一条站得住脚的分析"。

**计划生成的输入（grounding）**：论文明确写 planner 收到的是 "a rich contextual summary of the available data, including **table schemas, column descriptions with data types and semantic tags, and summary statistics**"（引 Dibia, LIDA, 2023）。计划产物是 "a structured natural language paragraph that explicitly outlines the conceptual variables, required data transformations (e.g., filters, joins, aggregations), and the chosen statistical model"。

**喂给 DS Agent 的数据表（§4.3.1，已核实）**：① 每日 summary 表；② activities 表；③ **population summary 表 —— 提供按性别与年龄分层的每日可穿戴指标百分位**。完整数据 schema 见论文 Supplemental Section C.2，prompt 模板见 Supplemental C.3。

**源码层面的实现（已核实，仓库 `personal_health_agent/`）**：

| 文件 | 大小 | 作用 |
|---|---|---|
| `pha/agents/data_science_agent.py` | 27,200 B | DS Agent 主体（`class DataScienceAgent`） |
| `pha/prompts/data_science_prompts.py` | 38,608 B | 全部 prompt 模板 + 26 项 rubric 的硬编码 |
| `pha/tools/python_sandbox.py` | 23,268 B | 代码执行沙箱（`class PythonSandbox`） |
| `pha/utils/data_schemas.py` | 13,882 B | DataFrame schema 定义（喂给模型的 schema 文本来源） |
| `api/main.py` | 35,181 B | **FastAPI** 应用，端点 `GET /config`、`POST /sessions`、`GET/DELETE /sessions/{id}`、`POST /sessions/{id}/chat`、`GET /sessions/{id}/messages` |
| `pha/agents/orchestrator.py` | 56,072 B | 多 Agent 编排（DS / DE / Coach） |
| `notebooks/02_data_science_agent.ipynb` | — | 可运行的 DS Agent 示例 |
| `data/sample/{summary,activities,profile,population_percentiles}.csv` | — | 样例数据（含 `description.txt`） |

**重试与自纠正的具体实现（源码级，已核实）**——这是全篇最值得抄的部分：

```python
DataScienceAgent.__init__(
    self, settings=None, llm_backend=None,
    summary_df=None, activities_df=None, profile_df=None, population_df=None,
    num_code_retries: int = 3,       # 内层：同一份代码最多改 3 次
    total_code_retries: int = 5,     # 外层：整个任务最多 5 步
    max_code_update_cycles: int = 1, # 逻辑精修循环的上限（与报错重试分开计数）
)
```

- 主循环 `query()`：`generate_approach` → `generate_code` → `execute_code` →（若报错或有待处理的精修反馈）内层 `for retry in range(num_code_retries): debug_code() → execute()`，成功即 break → `interpret_results`。
- **`interpret_results` 是一个决策点，不只是文案**：若结果逻辑无效（例如本该有值却是 NaN），模型被要求以 **`[CODE UPDATE]`** 开头返回英文修改说明；否则以 **`[SUMMARY]`** 开头返回数值总结。`query()` 解析这个标签，把 `[CODE UPDATE]` 之后的文本当作 `code_update_feedback`，触发 `refine_code()` 再跑一轮。
- `generate_code_with_retry()` 把两类重试分开计数：**执行报错**消耗 `num_code_retries`，**逻辑精修**消耗 `max_code_update_cycles`。作者注释了动机：否则一个持续返回 `[CODE UPDATE]` 的任务会把预算烧光在"能跑但结果不对"的代码上。同时它保留 `best_code`（最后一次无报错的代码），预算耗尽时仍能交出可运行版本。
- **baseline / eval 路由**：`_is_python_stub(approach)` 判断计划是否已是函数骨架（以 `from typing import` / `def analysis(` / `import ` 开头且含 `"""`）。评测算子会把 plan 转成"函数头 + docstring"，此时走 *_baseline 系列 prompt（把骨架当权威规格），生产环境走 V2 prompt。注释解释了动机：生产 prompt 强制 `-> Dict[str, Any]` 与参数默认值，会覆盖骨架的返回约定并导致单元测试失败。

**代码生成的硬约束（`CODEGEN_PREAMBLE_V2`，已核实）**：必须写成
`def analysis(summary_df=None, activities_df=None, profile_df=None, population_df=None) -> Dict[str, Any]`，前置 `from typing import Any, Dict / import pandas as pd / import numpy as np / import datetime`；**禁止注释与 docstring**；**函数体 < 80 行**；**不要写冗长的错误处理**（原文示例：缺人口数据时"返回用户自身指标而不是抛异常"）；prompt 末尾故意留一个未闭合的 ```python 围栏让模型续写签名。

### 1.4 它开源的评测基准（论文描述已核实 / 仓库产物未核实）

论文把评测拆成两个**互相独立**的任务，这正好对应本项目想要的"数值正确性 vs 统计方法选择正确性"分开评分：

#### Task 1：分析计划质量（plan quality）

| 维度 | 值 |
|---|---|
| 评测对象 | 开放式查询 → 分析计划的质量 |
| 数据集 | **141 条 query–analysis-plan 专家打分对**，来自 **100 条不同的健康查询**（采自论文 §2.1 的终端用户健康查询集），由 Gemini 1.5-2.0 与 GPT-4o 等多种 LLM 生成 |
| 标注人力 | **10 位**专家数据科学家（8 男 2 女，25–45 岁，全部 CS 博士，5–20 年经验），共 **47 小时**，产出 **354 条 approach 级标注**（每对由 2–4 位专家标注），累计 **6157 条 rubric item 标注** |
| Rubric | **26 项**，条件层级结构（前一项满足才评估后一项），**6 个维度**：1 Timeframe、2 Data Transforms、3 Data Sufficiency、4 Summary Statistics、**5 Statistical Tests**、6 Overall Alignment。**扣分制，单次评估最多扣 16 分**。Rubric 与 4 位 DS 专家共同制定 |
| 标注一致性 | 平均一致率 **80.2%**，Bennett's S = **0.622**（中等一致性） |
| 评分方式 | 先用人工数据训练一个 **LLM autorater（Gemini 2.0 Flash）**；dev set 75 条查询，test set = 25 条唯一查询的 96 次评估。Autorater 与人类专家 **ICC(3,1) = 0.838**（高可靠性） |
| 主要结果 | DS Agent **75.6±1.4%** vs 基线 Gemini 2.0 Flash **53.7±1.8%**（p<0.001，r=0.925，Wilcoxon 符号秩）。分项：Data Availability Check **5.3±1.6% → 66.5±3.7%**（最戏剧性）；Timeframe **96.0±1.9%**（p<0.001，r=0.907）；Data Transforms p=0.002，r=0.512；Alignment p=0.011，r=0.467；**Statistical Tests p=0.630（基线已有 81.3%，无显著差异）** |

**注意最后一行**：论文自己承认，**"选哪个统计检验"这件事基础模型已经做得不错**，Agent 架构的增益主要在"把分析落到数据约束上"（时间窗、缺失性、可复现性）。本项目在设计时不应假设"统计方法选择"必须靠 Agent 才能做对。

#### Task 2：代码生成与执行（code generation & execution）

| 维度 | 值 |
|---|---|
| 评测对象 | 给定精确规格，代码是否**可执行且功能正确**、数值是否准确 |
| 数据集 | **25 组 query-approach 对**（从 Task 1 高分样本中采样），每组由专家写成 **函数头 + docstring**（对齐 HumanEval 式做法，引 Chen et al., 2021） |
| 标注人力 | **7 位**数据科学家（5 男 2 女，22–45 岁，5–10 年经验）编写函数实现与测试用例，共 **173 个单元测试**，**75 小时**人工 |
| 单元测试覆盖 | 不止"能否跑"：有限数据场景、复杂数据转换、正确的统计方法、对用户问题的准确回答、返回结构 |
| 执行环境 | Python 沙箱，装有 numpy / pandas / scipy |
| 评分方式 | **pass rate = 173 个测试中通过的用例数 / 总数**；执行失败会收到错误消息并**最多重试 5 次** |
| 主要结果 | DS Agent 首次尝试 **75.5±3.3%** vs 基线 **58.4±3.7%**（χ²=26.3，p<0.001，McNemar 检验）；**五次尝试后 79.0%** |
| 错误分类 | data handling 错误 **25.4±3.3% → 11.0±2.4%**（χ²=12.3，p<0.001）；logic 错误 7.0±2.2% → 3.9±1.5%（Z=1.15，p=0.249，不显著）；general programming / output content 均不显著 |
| 细节表 | Supplemental Table S5（IRR）、S6（plan 统计）、S7（错误分类） |

论文对"79.0% 相比 75.5% 提升有限"的解释值得记：**"the primary performance gain stems from the agent's superior initial code generation, the iterative process further offers a valuable mechanism for incremental refinement."** —— 即自纠错循环能救回，但救得不多。

#### ⚠️ benchmark 开源的实际情况（未核实项，重要）

论文原文称 "We also open-source an evaluation benchmark for these capabilities." 但本次调研在公开仓库中**没有找到**承载该 benchmark 的数据产物：

- 已核实：`personal_health_agent/` 下完整文件树仅 164 个路径，数据相关目录只有 `data/sample/`（`summary.csv`、`activities.csv`、`profile.csv`、`population_percentiles.csv`、`description.txt`）与 `data/README.md`；
  没有 `benchmark/`、`eval/` 数据集目录，没有那 141 条 query–plan 对，没有 173 个单元测试的原始文件（`tests/` 只有 7 个 pytest 文件）。
- 已核实：`Google-Health` 组织下 19 个公开仓库（`google-health` 123★、`consumer-health-research` 167★、`medgemma` 1,633★、`records-research` 234★、`imaging-research` 201★、`medical-data-toolkit` 73★ 等）中，**没有**独立的 DS benchmark 仓库。
- 但工具链是开放的：`pha/prompts/data_science_prompts.py` 里 **26 项 rubric 以 `RUBRIC_QUESTIONS` 字典形式硬编码**（含 `data_answerable:30`、`timeframe:40`、`data_transformations:55`、`data_distribution_availability:95`、`data_statistics:129` 及其 8 个子项、`summary:192` 等行号），并会被序列化注入 plan prompt；`tests/test_python_sandbox.py` 等提供了沙箱行为测试；`_is_python_stub` 分支说明评测流程的接口形态。

**结论**：`已核实` 的是论文的评测设计（两任务、26 项 rubric、173 单元测试、autorater ICC 0.838）；`未核实/倾向否定` 的是"benchmark 数据集随仓库公开"这一点。若要复用，最可行的是复用它的 **rubric 定义 + 两任务评估协议 + autorater 校准方法**，而不是下载数据集。

### 1.5 代码仓库地址

| 仓库 | 内容 | 许可证 | 活跃度（2026-10-17 实测） |
|---|---|---|---|
| **https://github.com/Google-Health/consumer-health-research** → `personal_health_agent/` | **PHA 全套**：DS Agent、DE Agent、Coach Agent、orchestrator、FastAPI 服务、React 前端、7 个 notebook、few-shot、测试 | **Apache-2.0**（`personal_health_agent/LICENSE.txt`，"Copyright 2025 Google LLC"） | 仓库 167★ / 49 fork，`pushed_at` **2026-10-13** |
| https://github.com/yahskapar/personal-health-insights-agent | **PHIA（前作）**：ReAct Agent + 4000 条 objective 查询 + 172 条 open-ended 查询 + `few_shots/` ≈50 个 notebook + 真人标注 + 4 个人群的自动评测输出 | **CC BY-NC 4.0**（非商用） | 70★ / 15 fork，`pushed_at` **2026-08-31** |

**Embedding 相关事实（已核实）**：PHA 的 `requirements.txt` 固定了 `pandas>=2.0`、`numpy>=1.24`、`matplotlib>=3.8`、`scipy>=1.10`、`seaborn>=0.13`、`fastapi>=0.109`、`pydantic>=2`、`google-genai`、`tavily-python`、`ddgs`，并且 **仅为 PHIA baseline 依赖 `git+https://github.com/google-deepmind/onetwo`**。DS Agent 本体不依赖 onetwo，可独立移植。

许可证注意：**PHA 是 Apache-2.0（可商用），PHIA 是 CC BY-NC 4.0（非商用）**。若移植代码，只能移植 PHA 那一份。

---

## 2. 第 2 问：同类开源项目对比

> 所有 star / fork / `pushed_at` / 许可证均来自 2026-10-17 未认证 GitHub REST API 实测（`已核实`，但为时点值）。
> "计划→代码→执行→自动修错"一列中，标 `已核实` 的表示 README 或官方文档有明确证据；标 `推测` 的表示由架构形态推断，未逐行读源码验证。

### 2.1 总表

| 项目 | 仓库 | 许可证 | Stars | 最近 push | 计划→代码→执行→自动修错 | 输出图表 | 可作为库嵌入 FastAPI + LangGraph | 是否针对个人健康 |
|---|---|---|---|---|---|---|---|---|
| **PHA Data Science Agent**（本报告主题） | `Google-Health/consumer-health-research` → `personal_health_agent/` | **Apache-2.0** | 167 | 2026-10-13 | **✅ 完整**：plan → code → sandbox exec → 报错重试(3) + `[CODE UPDATE]` 逻辑精修(1) **已核实源码** | ❌ Agent 明确无 matplotlib | ✅ **本身就是 FastAPI**（`api/main.py`），纯 Python 包 `pha/` 可摘出 | ✅ 可穿戴时序专用 |
| **pandas-ai** | `sinaptik-ai/pandas-ai`（`gventuri/pandas-ai` 已重定向） | MIT（核心）/ `pandasai/ee/` 另有许可 | 23,879 | 2025-10-28 | ⚠️ 部分：有 agent 循环 + 代码执行，执行在可选 Docker sandbox；是否自动回灌错误重试 **推测**（README 未明确） | ✅ 有图表输出（pandas-ai 以"画图"为卖点之一）**推测** | ✅ 纯 Python 库（dev 模式直接跑，生产建议配 `pandasai-docker`） | ❌ 通用 |
| **LangChain pandas dataframe agent** | `langchain-experimental`（`langchain_experimental.agents`） | MIT | langchain 主仓 147,912 | langchain 主仓 2026-10-16；**experimental 包正在 sunset** | ✅ 是：ReAct 循环 + python_repl 工具，异常文本回灌即下一轮 Thought **已核实（文档常识）+ 架构必然** | ⚠️ 间接：需自己 `plt.savefig`；AgentExecutor 不会回传图片 | ⚠️ 是库，但依赖**正在 sunset 的包** + 必须 `allow_dangerous_code=True`（0.3 起的安全闸门） | ❌ 通用 |
| **smolagents CodeAgent** | `huggingface/smolagents` | **Apache-2.0** | 29,874 | 2026-10-06 | ✅ 是：动作以 **Python 代码片段**写出；README flowchart 原文 "No call to 'final_answer' tool => **Store execution logs in memory and keep running**"；有 `max_steps` **已核实** | ⚠️ 可以（沙箱内 matplotlib），需自己回收图片产物 | ✅ 最干净：纯 Python 库（`pip install smolagents`，需 Python ≥3.10），无框架绑定 | ❌ 通用 |
| **Open Interpreter** | `openinterpreter/openinterpreter` | Apache-2.0 | 68,586 | 2026-10-11 | ✅ 是：其核心卖点就是"执行失败自动重试/自修复循环" | ✅ 是（有 display/图像渲染通道） | ❌ 不推荐：v2 主体是 **Rust 重写的桌面/CLI 应用**，要嵌入需跨进程边界；也是四大里唯一有原生代码执行风险外溢的 | ❌ 通用 |
| **Jupyter AI** | `jupyterlab/jupyter-ai` | BSD-3-Clause | 4,417 | 2026-10-14 | ❌ 否：它是 **JupyterLab 内的 chat 前端 + 一组 magic**，没有独立 agent loop，也没有自动修错 **已核实（README 形态）** | ✅ 天然（结果直接落 notebook cell） | ❌ 结构上假定 Jupyter Server / Lab 运行时，嵌进 FastAPI 是错配 | ❌ 通用 |
| **Data Interpreter** | **`geekan/MetaGPT`**（不是 THUDM！） | MIT（MetaGPT 主仓） | MetaGPT 主仓量级（未单独核实） | — | ✅ 是：**Hierarchical Graph Modeling**（分解子问题、动态生成节点）+ **Programmable Node Generation**（迭代改进与验证代码）**已核实（论文摘要逐字）** | ✅ 是（可视化是其演示重点） | ⚠️ 需嵌入整个 MetaGPT 框架，较重 | ❌ 通用（论文用 InfiAgent-DABench 评测） |
| **AutoGen** | `microsoft/autogen` | **CC-BY-4.0**（注意：这是"许可证"，不是软件许可中的常见选择） | 61,440 | 2026-04-15 | ⚠️ 是"可组合"而不是"内置"：`GroupChat` + `CodeExecutorAgent`/`UserProxyAgent` 组成 data analysis 示例，**没有内置的分析计划阶段** | ✅ 示例中常用 matplotlib | ✅ 是库，但生态分裂（Microsoft Agent Framework / AG2 分叉）需评估 | ❌ 通用 |
| （补充）**OpenManus DataAnalysis Agent** | `FoundationAgents/OpenManus` | MIT | 58,687 | 2026-09-30 | ✅ 是：README:174 原文 "we have also integrated the **DataAnalysis Agent**, which is suitable for data analysis and data visualization tasks. You can add this agent to `run_flow` in `config.toml`" **已核实** | ✅ 是 | ⚠️ 是库，但为通用 agent 框架，需自己裁剪 | ❌ 通用 |

> 补充参考：综述 **《LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions》arXiv 2510.04023v1**（2025-10-05），可作为选型时的横向索引。

### 2.2 逐项结论（含证据/不确定性）

**pandas-ai**（`已核实` 仓库元数据 + README；`推测` 循环能力）
- 许可证细节值得注意：仓库 `LICENSE` 返回 GitHub 的 `"Other"`，实际是 **MIT Expat 覆盖核心 + `pandasai/ee/` 目录单独许可**（企业功能）；README 徽章写 "License: MIT"。**若要商用需确认 `ee/` 边界**。
- v3 之后架构变为**模块化插件**：`pandasai-litellm`（LLM 适配）、`pandasai-docker`（代码执行沙箱）已拆成独立 PyPI 包。README 明确介绍 "Docker Sandbox … a secure, isolated environment to execute code safely and mitigate the risk of malicious attacks"。
- **对本项目的含义**：它解决的是"自然语言 → DataFrame 操作 + 画图"，**没有"统计方法选择正确性"的显式约束**，也没有时间窗、缺失性检查这些健康分析必需的步骤。适合当"执行层零件"，不适合当"DS Agent 骨架"。

**smolagents CodeAgent**（`已核实` README 原文）
- 架构选择很关键：**动作即代码**（"writes its actions in code (as opposed to 'agents being used to write code')"），执行日志回灌 memory 后继续跑，天然就是"执行 → 观察 → 再执行"的循环。
- 安全边界必须记住 README 的原话：**"The built-in `LocalPythonExecutor` is **not a security sandbox**. It applies some restrictions but can be bypassed and must not be used as a security boundary."** 生产必须换成 E2B / Blaxel / Modal / Docker 之一。
- **对本项目的含义**：它是"执行层 + 循环控制"的最佳现成件，可以与自研的"分析计划层"拼装；这与 PHA 的两阶段拆分思路一致（PHA 是自研沙箱 + 自研 plan 层，没有用 smolagents）。

**LangChain pandas dataframe agent**（`已核实` 关键警告）
- **重要负面结论**：`langchain-experimental` 仓库 README 顶部有明确警告：**"`langchain-experimental` is being sunset. See #87 for details."**，并另有一段 "Portions of the code in this package may be dangerous if not properly deployed in a sandboxed environment."（本次已直接抓取到该 README，740 字节全文）。
- 经典入口 `create_pandas_dataframe_agent` 需要 `allow_dangerous_code=True` 才会放行（LangChain 0.3 之后的安全变更）。
- `未核实`：我尝试抓取 `langchain-ai/langchain` master 分支的 `libs/experimental/...` 路径全部返回 404，说明该包已迁出主仓（迁到哪个仓未最终确认）。
- **对本项目的含义**：**不要把它作为长期依赖**。若只需"ReAct + python 工具"这一模式，直接用 LangGraph 的 `ToolNode` + 官方代码执行中间件更稳（参见仓库已有的 `document/ds-agent-sandbox-research.md` 结论）。

**Jupyter AI**（`已核实` 形态）
- 它是"在 notebook 里和模型对话"，不是"自动完成一次数据分析的 Agent"。**没有计划阶段、没有自修复、没有评分协议**。作为"可嵌入组件"完全不合用，但作为"人工调试 DS Agent 的界面"有价值。

**Data Interpreter**（`已核实` 论文摘要，重要纠错）
- 摘要原文末句：**"The code is available at https://github.com/geekan/MetaGPT."** —— **它不是 THUDM 的项目，而是 MetaGPT 体系（geekan / DeepWisdom）的一部分**。`THUDM/DataInterpreter` 这个仓库路径在 GitHub API 上返回 **404**（已核实），本次调研未找到任何 THUDM 版本的 Data Interpreter。
- 两个核心模块值得借鉴：**Hierarchical Graph Modeling**（把复杂问题拆成可管理子问题，**动态生成节点**并优化图）；**Programmable Node Generation**（对每个子问题迭代改进并验证代码）。
- 它的评测数字（InfiAgent-DABench 75.9% → 94.9%；开放式任务 60% → 97%）说明"把开放式任务图化 + 每节点验证"的收益很大。

**AutoGen**（`已核实` 许可证异常）
- `LICENSE` 文件内容是 **Creative Commons Attribution 4.0**。对代码仓库使用 CC-BY 属于非常规选择，**商用前务必让法务确认**（CC 许可通常不适用于软件，也缺少专利授权条款）。
- 已有 1,116 个 open issue 且 `pushed_at` 停在 2026-04-15，生态正处于 Microsoft Agent Framework / AG2 分裂期。**不建议作为新项目的底座**。

### 2.3 "能否嵌入 FastAPI + LangGraph"的判断汇总

| 结论 | 项目 | 理由 |
|---|---|---|
| **推荐** | PHA DS Agent（源码级移植）、smolagents（执行层） | 前者本身就是 FastAPI + 可摘出的纯 Python 包；后者 SDK 干净、Apache-2.0、无框架绑定 |
| **可用但需自建** | LangGraph `ToolNode` + 代码执行中间件（替代 LangChain pandas agent） | 避免 sunset 包；仓库已有一份沙箱选型结论可复用 |
| **只在受限场景用** | pandas-ai（当执行/画图零件）、OpenManus DataAnalysis（当参考实现） | 都缺"计划质量"与"统计假设适用性"的显式约束 |
| **不推荐** | Open Interpreter（进程边界重）、Jupyter AI（运行时错配）、AutoGen（许可 + 生态分裂）、LangChain pandas agent（sunset） | 见上表 |

---

## 3. 第 3 问：数据科学 Agent 的公开评测基准

> 下表中所有 arXiv 编号与标题均在 2026-10-17 通过 `export.arxiv.org/api/query?id_list=...` 逐条回取校验（`已核实`）。
> "评测什么"一列：标 `已核实(摘要)` 表示本次读了该论文摘要原文；标 `推测(标题)` 表示仅核实了标题与提交日期，细节未读。

### 3.1 总表

| 基准 | arXiv | 年份 | 评测什么 | 数据规模（摘要级） | 评分方式 | 涉及个人健康时序？ | 数值正确性 / 统计方法选择 是否分开评分 |
|---|---|---|---|---|---|---|---|
| **PHA DS Agent benchmark**（本报告主题） | 2508.20148 | 2025 | **Task1 分析计划质量** + **Task2 代码产出正确性**，两任务独立 | Task1: 141 组 query–plan 专家打分对（100 条查询，6157 条 rubric 标注）；Task2: 25 组编码题 / **173 个单元测试** | Task1: 26 项条件层级 rubric，扣分制（最多扣 16 分），LLM autorater（Gemini 2.0 Flash，与人 ICC(3,1)=**0.838**）；Task2: 单元测试 **pass rate** | ✅ **是，唯一一个**：个人级 Fitbit 时序 + 人群百分位表 | ✅ **是，分开的**（见下方 §3.3） |
| InfiAgent-DABench（数据集名 **DAEval**） | 2401.05507（ICML 2024） | 2024-01 | "首个专门为 LLM Agent 的数据分析任务设计的基准"；Agent 需与执行环境交互、端到端解题 | **DAEval：257 道数据分析题，来自 52 个 CSV 文件**；论文基准测试了 34 个 LLM | 官方自述：用 **format-prompting** 把开放式问题转成封闭式以便自动评测 **已核实(摘要)** | ❌ | ❌ 单一综合正确率 |
| `DABench`（用户列表里的这一项） | — | — | **§3.1 的 InfiAgent-DABench 即通常所说的 DABench（其数据集叫 DAEval）**；arXiv 标题检索 `ti:"DABench"` 只命中 InfiAgent-DABench 与一个**无关的硬件基准** DABench-LLM（2601.19904，评测的是 Post-Moore 数据流 AI 加速器，不是数据科学 Agent） | — | — | — | — |
| DSBench | 2409.07703（ICLR 2025） | 2024-09 | 数据科学 Agent 离"专家"有多远；真实场景（长上下文、多模态任务背景） | **466 个数据分析任务 + 74 个数据建模任务**，取自 Eloquence 与 Kaggle 竞赛 **已核实(摘要)** | 与竞赛/标准答案比对 | ❌ | ❌ 以结果正确性为主 |
| DSEval（论文名 *Benchmarking Data Science Agents*） | 2402.17168 | 2024-02 | 提出一套**评测范式** + 覆盖**整个数据科学生命周期**的基准 | — （另有 repo `MetaCopilot/dseval`，34★、MIT、2024-06-24） | 自举式标注（bootstrapped annotation）方法 **已核实(摘要)** | ❌ | ❌ |
| Spider2-V | 2407.10956 | 2024-07 | **首个面向专业数据科学与工程工作流的多模态 Agent 基准** | **494 个真实任务**，20 个企业级专业应用，真实计算机环境；涉及 BigQuery / dbt / Airbyte 等工具；Agent 写 SQL、Python 并做 GUI 操作 | 任务级完成度 **已核实(摘要)** | ❌ | ❌ |
| Spider 2.0 | 2411.07763（ICLR 2025 Oral） | 2024-11 | 真实企业级 **text-to-SQL 工作流** | **632 个问题**；数据库 >1,000 列；BigQuery / Snowflake 方言与成本模型 | 结果集比对 **已核实(摘要)** | ❌ | ❌ |
| TableBench | 2408.09174 | 2024-08 | 表格问答（TableQA）综合复杂基准 | **18 个子领域 / 4 大能力类别**；工业场景；另训练了 TableLLM **已核实(摘要)** | 按 4 大类别分项评分 | ❌ | ❌（不是统计方法选择，而是表格推理） |
| **HealthBench** | 2505.08775（OpenAI） | 2025-05 | 医疗健康场景下 LLM 的性能与安全性 | **5,000 段多轮对话**，**262 名医生**撰写 **48,562 条**会话专属 rubric；覆盖"紧急情况、**transforming clinical data**、全球健康"等场景与准确性/指令遵循/沟通等维度；另有 Consensus 与 Hard 变体 | **rubric 打分**（conversation-specific rubric）**已核实(摘要)** | ❌（个人穿戴时序不在其列；有 "transforming clinical data" 上下文） | ❌ 单一 rubric 综合分 |
| **StatABench** | 2606.22977 | 2026-06 | **统计能力本身**：方法知识 + 工具使用 + 端到端统计建模 | **Stat-Closed：404 题 / 18 个统计主题**（多选、填空、决策、实操 4 种题型）；**Stat-Open：30 个开放建模任务**（取自专业竞赛） | Stat-Closed 客观比对；**Stat-Open 用经验证的 LLM-as-Judge**；用 LangChain MCP 框架 + 多个数据科学 Agent 评测。结果：GPT-5.1 仅 68.6%，最好的开源模型 60.6%；Stat-Open 最佳 Agent 61.86 **已核实(摘要)** | ❌ | ⚠️ **部分可以**：Stat-Closed 考"方法决策"、Stat-Open 考"端到端建模"，是两套题两种评分，但不是在**同一份数据同一问题**上分离 |
| BioDSA-1K | 2505.16100 | 2025-05 | 生物医学假设验证型数据科学 | **1,029 个假设中心任务 + 1,177 个分析计划**，从 **300+ 篇已发表生物医学研究** 中整理；含结构化假设与支撑证据表 **已核实(摘要)** | 假设验证 | ❌（生物医学文献数据，非个人穿戴） | ⚠️ 分析计划与结论分开存在，但非"数值 vs 方法选择" |
| DSAEval | 2601.13591 | 2026-01 | 广覆盖的真实数据科学问题 | **641 个真实问题 / 285 个数据集**（结构化 + 图像 + 文本）；多模态环境感知、多轮交互 **已核实(摘要)** | 任务级 | ❌ | ❌ |
| DSGym | 2601.16344 | 2026-01 | **评测与训练框架**（不只是一个静态基准） | DSGym-Tasks（对既有基准做质量与"捷径可解性"过滤）+ DSBio + DSPredict；案例训练 2,000 例 data、4B 模型超过 GPT-4o **已核实(摘要)** | 统一执行环境下的端到端测量 | ❌ | ❌ |
| DSAgentBench | 2608.10366 | 2026-08 | 真实计算机环境下自动化**端到端**数据科学工作流 | **275 个任务**，环境含 notebook / IDE / 终端 / 浏览器 / 数据库 **已核实(标题)** | 端到端任务完成 | ❌ | ❌ |
| Ambig-DS | 2605.09698 | 2026-05 | 数据科学 Agent 的**任务表述歧义**问题 | Ambig-DS-Target 51 题（基于 DSBench）+ Ambig-DS-Objective 61 题（基于 MLE-bench）**已核实(摘要)** | 复用源基准的原评测器 | ❌ | ❌ |
| EvoDS | 2606.03841 | 2026-06 | 自演化数据科学 Agent（技能学习 + 上下文管理，agentic RL） | — **已核实(标题)** | 训练/评测 | ❌ | ❌ |
| DS-1000 | 2211.11501 | 2022-11 | 数据科学**代码生成**（自然且可靠的补全） | 1,000 道 Pandas/NumPy 等库的补全题 | 单元测试式执行验证 | ❌ | ❌（只测代码，不测方法选择） |

**附带线索（仅记录标题，本次未展开）**：DSBC（2507.23336，2025-07-31，用商业应用日志做上下文工程基准）、TML-Bench（2603.05764）、CausalDS（2607.08093）、Trustworthy Self-Composable Big-Data-as-a-Service（2606.17915）、DSMentor（2505.14163，Dataset Curation 方向）、CIPHER（2607.14386）。

### 3.2 关键结论：个人健康时序数据基本无人覆盖

- **在本次检索到的全部数据科学 Agent 基准中，只有 PHA 自己的 DS benchmark 使用"个人级可穿戴时序数据 + 人群分层百分位表"**。
- HealthBench 是最接近"健康 + rubric 评分"的基准，但它是**对话式健康问答**（含 "transforming clinical data" 这类上下文），**不执行代码、不产出数值结果**，因此不能用来评测"我的最长跑步距离是多少"这类问题。
- DSGym 有一句很值得警惕的自述：**"we show that a substantial portion of tasks in current benchmarks can be solved without using the actual data."** —— 即大量既有基准可以被"猜"过去。若本项目要自建评测，**必须设计"不看数据就答不出来"的题**（例如需要精确到某一天、需要处理缺失、需要跨表连接）。

### 3.3 "数值正确性"与"统计方法选择正确性"分开评分：现状

用户的这个问题是本报告里最尖锐的一问。结论是：

| 能否分开评分 | 做法 | 证据 |
|---|---|---|
| ✅ **能，而且 PHA 就是这么做的（唯一一个）** | **Task 1 打"计划"（方法选择）**：26 项 rubric 的 **第 5 个维度就是 `Statistical Tests`**，判定标准原文是 "Does the approach appropriately apply statistical tests when helpful, with **clear, reproducible methods, reasonable test choices, and proper consideration of distributional assumptions**?"；**Task 2 打"数值"**：173 个单元测试的 pass rate | 论文 §4.3.1 / §4.3.2 **已核实** |
| ❌ 但 PHA 的"分开"是**跨数据集**分开的 | Task 1 用 141 组 query–plan 对，Task 2 用 25 组编码题（从 Task 1 高分样本采样）。**不存在"同一个问题既给数值正确性打分、又给方法选择打分"的联合评测** | 同上 |
| ⚠️ StatABench 是第二种思路 | **Stat-Closed**（404 题，含 decision-making 与 practical-application 题型）考"方法知识/决策"；**Stat-Open**（30 题）考"端到端建模"，用 LLM-as-Judge。两者是**不同题集、不同评分器** | arXiv 2606.22977 摘要 **已核实** |
| ❌ 其余基准都不做这种切分 | DSBench / InfiAgent-DABench / Spider2 / TableBench / DS-1000 都是"结果对不对"的单分数 | 见 §3.1 |

**对本项目的直接启示**：如果要给 DS Agent 建评测，最有参考价值的两件事是：
1. **照抄 PHA 的 rubric 结构**（26 项条件层级、6 维度、扣分制、含独立的 Statistical Tests 维度），因为它已被验证有 80.2% 人类一致率、ICC 0.838 的 autorater 可靠性；
2. **额外补上 PHA 没做的那一格**：在同一组问题上同时给出"数值分数"与"方法选择分数"，这需要为每题准备 (a) 单元测试 + (b) rubric 两个评分器，成本高但差异化明显。

---

## 4. 第 4 问：可复用的关键设计

以下 6 条全部来自 PHA 的 DS Agent 源码（`personal_health_agent/`，Apache-2.0，可直接移植）。每条给出问题 → 做法 → 证据 → 对本项目的含义。

### 4.1 把 DataFrame 以 CSV 字面量注入沙箱，而不是让模型自己造数据

**问题**：模型容易在代码里"顺手"写一个假的 DataFrame 或硬编码数值，导致结论与真实数据无关。

**做法**（`pha/tools/python_sandbox.py`）：沙箱命名空间不是靠传对象，而是把每个 DataFrame `df.to_csv(index=False)` 序列化成**字符串字面量**再写进 preamble，形如 `key = df_from_csv_string(repr(csv_str))`。同时 preamble 里内联了 `enforce_persona_schema(summary, activities, population)` 做列/类型校验。**模型从头到尾看不到"数据可以自己编"的路径**。

**对本项目的含义**：如果通过 LangGraph state 传递 DataFrame，务必在进入执行节点前做一次 schema 校验；更稳的做法是只把"数据引用（表名 + 版本）"传给执行器，由执行器从数据库/文件重新加载，而不是把 DataFrame 序列化进 prompt 或 state。

### 4.2 执行结果用 Python 对象传递，不用 stdout 文本

**问题**：靠解析 stdout 取数值，既容易错位，也给了模型"用 print 糊弄"的空间。

**做法**（`pha/tools/python_sandbox.py`）：
- `_add_call_analysis_line()` 自动在代码末尾追加 `result = analysis(summary_df=..., activities_df=..., profile_df=..., population_df=...)`（若末行已是 `result = ` 赋值则跳过）；
- `_wrap_code_for_execution()` 把代码包进 try/except + `contextlib.redirect_stdout`，成功时取 `_result = result`，失败时写入 `f"{ERROR_STRING}: {traceback.format_exc()}"`；
- 输出取值优先级是 **`str(result)` 优先于 captured stdout**；
- `ExecutionStatus` 三态（SUCCESS / ERROR / TIMEOUT），`PhdaPythonReply.has_error()` 供上层判断。

**对本项目的含义**：约定"分析函数必须返回结构化对象"，执行层只把对象转成 JSON 交给下一步。这同时解决了"避免模型编造数值"——数值来自程序返回值，不是模型复述。

### 4.3 三层重试预算分离，且保留 `best_code`

**问题**：把"执行报错"和"结果不对"混在一个重试计数里，会出现"代码能跑但结果错"时反复消耗预算、最终两手空空。

**做法**（`pha/agents/data_science_agent.py`）：
```python
num_code_retries: int = 3        # 内层：同一份代码最多改 3 次
total_code_retries: int = 5      # 外层：整个任务最多 5 步
max_code_update_cycles: int = 1  # 逻辑精修循环上限，独立计数
```
- 内层由执行报错驱动，走 `DEBUG_CODEGEN_PREAMBLE_V2`；
- 外层由解释器的 **`[CODE UPDATE]`** 标签驱动，走 `CODE_REFINEMENT_PREAMBLE`（只改被指出的逻辑，不重写）；
- `generate_code_with_retry()` 始终保存 `best_code`（最后一次无报错的代码），预算耗尽也能交出可运行版本。

论文侧的数字也印证了"自纠错收益有限"：首次 75.5% → 五次尝试 79.0%。**不要指望重试救回大幅质量提升，重点应放在首次生成质量。**

### 4.4 报错修复要"最小改动"，并禁止 try/except 兜底

**问题**：模型拿到报错后倾向于重写整段代码或加一堆 `try/except` 把错误吞掉。吞掉错误 = 数值静默错误，比崩溃更危险。

**做法**（`DEBUG_CODEGEN_PREAMBLE_V2`，逐字）："Focus **ONLY** on fixing the specific error. **Do NOT** add extra analyses, visualizations, **try/except wrappers**... The goal is a **minimal, targeted fix**." 同时 `CODE_REFINEMENT_PREAMBLE` 要求 "Preserve all correct logic… only change what the feedback explicitly requests"，并重申结构约束（精确的 `analysis` 签名、无注释、无 try/except、函数体 <80 行）。

**对本项目的含义**：在 debug 节点 prompt 里显式禁止 `try/except`（除非原代码已有）与"额外友好化输出"。这是本项目健康场景的安全底线——**宁可失败，不要静默给出错数值**。

### 4.5 用 prompt 强制"统计检验的适用性假设"落地到可执行步骤

**问题**：要求模型"选个合适的检验"，它会写"使用 t 检验"就交差，不检查正态性、不说明缺失性。

**做法**（`APPROACH_PREAMBLE_V3`，plan 阶段的硬性结构）：计划必须只含一个 `== Discussion ==` 与一个 `== Approach ==`，且 **禁止出现代码**；Approach 必须编号列出 6 项，其中：
- 第 4 项 **Data Sufficiency Checks**：最少样本数、缺失值阈值、不足时返回什么；
- 第 5 项 **Statistical Analysis**：**必须点名确切检验**（t-test / correlation / ANOVA）、**说明选择理由**、**陈述假设**（如正态性）**以及将如何检验这些假设**（原文举例："Perform Shapiro-Wilk test on metric distributions within each group"）；
- 第 6 项 **Define Output**：明确要产出哪些数字与检验结果，且**明确不做解释**。

同时给出了一条可复现性硬约束，逐字："**Another analyst using the same data must reach the exact same numerical result**"、"**Do not make extra assumptions**"。few-shot 还包含一个**"无法回答"正例**（"How's my friend's heart rate compared to me?" → 返回说明而不是硬算）。

**独立佐证**：论文评测显示 DS Agent 在 **Data Availability Check 上从 5.3% → 66.5%**，而 **Statistical Tests 维度与基线无显著差异（p=0.630）**。也就是说：**"把假设写进计划"这件事 Agent 架构带来了巨大收益；"选对检验"本来就难不到基础模型。**

### 4.6 用 `[SUMMARY]` / `[CODE UPDATE]` 双标签把"解释结果"变成"一个可控的决策点"

**问题**：结果解释环节通常是自由文本，模型可能在这里编造数字、越界给健康建议，或者明知结果有问题也照说。

**做法**（`INTERPRETATION_PREAMBLE`）：
- 角色约束逐字："**Do not make any interpretations for the user. YOU are not a health expert.**"
- 结果有效 → 以 **`[SUMMARY]`** 开头，把数值放进用户问题的语境里（因为**用户看不到原始执行输出**，模型必须把数字带出来）；
- 结果逻辑无效（例如本该有值却是 NaN）→ 以 **`[CODE UPDATE]`** 开头，**用英文描述需要的代码修改**；
- 上层 `query()` 解析这两个标签，`[CODE UPDATE]` 触发 `refine_code`。

**对本项目的含义**：这是"避免模型编造数值 + 避免越界医疗建议 + 把自纠错变成显式协议"三个目标用同一个机制解决，成本极低（两个字符串前缀），强烈建议直接照抄。注意标签必须由**代码解析**而不是由模型自觉遵守。

### 4.7 （附加）从 173 个单元测试错误分类里学到的选型优先级

PHA 的错误分类数据（论文 Table S7 摘要级）说明**改进空间集中在数据工程而非算法**：

| 错误类别 | 基线 | DS Agent | 显著性 |
|---|---|---|---|
| data handling（过滤/连接/缺失/类型） | 25.4±3.3% | **11.0±2.4%** | χ²=12.3, p<0.001 ✅ |
| general programming | 12.4% | 10.4% | 不显著 |
| calculation / logic | 7.0±2.2% | 3.9±1.5% | Z=1.15, p=0.249 ❌ |
| output content | 2.3% | 更低 | 不显著 |

**含义**：本项目的工程投入应优先放在"数据准备与 schema 契约"上（时间列语义、时区、粒度、缺测表示、单位），而不是"更聪明的统计推理"。

---

## 5. 对本仓库的落地建议（供主 Agent 决策，非本次调研的交付物）

1. **架构直接对标 PHA 的两阶段**：`plan 节点`（禁代码，只输出结构化计划）→ `code 节点`（固定函数签名 + <80 行）→ `执行节点`（沙箱）→ `判定节点`（`[SUMMARY]` / `[CODE UPDATE]`）。本项目已有 `document/设计文档.md` 中的沙箱决策（一次性 Docker 容器），与 PHA 的做法同构。
2. **执行层不必自研**：可用 `smolagents`（Apache-2.0）的循环控制，或直接用 LangGraph `ToolNode` + 自研 Docker 执行器；**不要引入 `langchain-experimental`**（正在 sunset）。
3. **许可证合规**：可移植 PHA（Apache-2.0）；**不可移植 PHIA**（CC BY-NC 4.0，非商用）。`pandas-ai` 需确认 `pandasai/ee/` 的独立许可边界；`AutoGen` 的 CC-BY-4.0 需法务确认。
4. **评测设计**：先照抄 PHA 的 26 项 rubric（含 Statistical Tests 维度）+ 173 单元测试式的"函数头 + docstring + 单元测试"协议；再补 PHA 没做的"同题双评分"。
5. **数据契约**：本仓库 `agent-app/src/agent_app/models/health.py` 的时间列目前只有 `measured_date`（粒度到天）——PHA 的 DS Agent 大量依赖**日内时间戳与时段语义**（`bed_time` / `wake_up_time` / `HH:MM`）。若要做"我最近睡得怎么样"这类分析，**数据粒度是需要先决策的前置问题**。

---

## 附录 A：方法与可复现性

| 步骤 | 具体做法 |
|---|---|
| 论文定位 | OpenAlex `search=personal data science agent wearable` → PHIA → 顺链到 PHA；arXiv API `id_list=` 精确回取元数据 |
| 全文证据 | `https://arxiv.org/pdf/2508.20148` → pypdf 6.20.1 → 148 页 / 368,350 字符文本；关键句用 grep 定位后逐字引用 |
| 仓库证据 | `api.github.com/repos/...`（元数据、license）、`api.github.com/repos/.../git/trees/main?recursive=1`（164 个路径的完整文件树）、`raw.githubusercontent.com`（README / LICENSE 原文） |
| 代码证据 | 直接下载 `pha/agents/data_science_agent.py`、`pha/prompts/data_science_prompts.py`、`pha/tools/python_sandbox.py`、`pha/utils/data_schemas.py` 并逐段阅读 |
| 基准证据 | `export.arxiv.org/api/query?id_list=...` 批量回取 17 个基准的标题与摘要；`ti:"DABench"` 检索用于消歧 |
| 本地副本 | 下载物与抽取文本存放于 `D:/tmp/dsres/`（Windows 路径）与 WSL `/tmp/`；本报告是唯一写入仓库的文件 |

**环境限制（影响了覆盖度）**：无通用联网搜索工具（DuckDuckGo / Google 均返回 000）；GitHub 未认证 API 限额 60 次/小时，本次调研期间曾耗尽（`core 0/60`），因此部分仓库的 issue 细节与提交历史未取；`huggingface.co` API 不可达。

## 附录 B：未核实 / 存在不确定性的项（引用时请注意）

1. **PHA benchmark 数据集是否真的公开**：论文称 "open-source an evaluation benchmark"，但公开仓库文件树中**没有**那 141 组 query–plan 对、26 项 rubric 标注与 173 个单元测试的数据文件。**倾向判断：未随仓库发布（或发布在未检索到的位置）。**
2. **PHA 的期刊版本**：PHIA 仓库 README 称已发表于 Nature，但 OpenAlex / Crossref 本次**未检索到**该文（检索 "Anatomy of a Personal Health Agent" 只返回无关的解剖学期刊论文）。当前唯一稳定引用是 arXiv:2508.20148。
3. **PHA 仓库的根级许可证**：GitHub API 对 `consumer-health-research` 根仓库未给出 SPDX 标识；`personal_health_agent/LICENSE.txt` 内容为 Apache-2.0（已核实）。
4. **`langchain-experimental` 迁往何处**：主仓 `libs/experimental/...` 返回 404，`langchain-ai/langchain-experimental` 仓库 README 可访问并确认 sunset，但**迁移目标**未确认。
5. **`THUDM/DataInterpreter`**：GitHub API 返回 404；**正确出处是 `geekan/MetaGPT`**（论文摘要逐字给出）。"Data Interpreter 是 THUDM 项目"这一说法在本报告中判定为**错误**。
6. **各项目"是否自动修错"的判定**：除 PHA（读过源码）、smolagents（读过 README）、LangChain pandas agent（架构必然）外，其余项目的该列判定属**推测**。
7. **所有 star / fork / push 时间**：2026-10-17 未认证 GitHub API 时点值，未经第二方复核。
8. **`DSBC`（2507.23336）等附带线索**：仅核实标题与编号，未读摘要。
