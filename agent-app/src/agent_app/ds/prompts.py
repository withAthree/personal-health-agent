"""Plan / Code / 修错三个阶段的提示词。

这些字符串是 DS 的行为规范，不是说明文档。几条来自调研报告的一手结论：

- **计划必须点名确切检验、说明理由、陈述假设、并写清怎么检验假设**（§4.5）。论文实测
  这一条把「数据充足性检查」从 5.3% 拉到 66.5%。
- **修错要最小改动，并禁止 ``try/except`` 兜底**（§4.4）。吞掉异常等于静默给出错数字，
  比崩溃危险。
- **可复现性写成硬约束**（§4.5 原文）：另一个分析师用同一份数据必须得到完全相同的数值。
- **解释环节不做健康解释**（§4.6 原文）："Do not make any interpretations for the user."
- **pandas 3.0 的破坏性变更必须写进提示词**：频率别名 ``M`` / ``Q`` / ``Y`` 已被移除
  （改用 ``ME`` / ``QE`` / ``YE``），链式赋值在 Copy-on-Write 下静默失效、且
  ``SettingWithCopyWarning`` 已被删除。模型是在续写它见过的代码，而语料里绝大多数是
  旧写法——这两条不是风格建议，是「不写就会出错」或「不写就会静默算错」。
  来源：pandas 3.0.0 whatsnew（2026-01-21）的 ``Enhancements`` 与
  ``Removal of prior version deprecations/changes`` 两节，逐条核对过。

三处共享同一段数据说明（``_DATA_NOTES``）：模型对数据列的误解是最高频的错误来源，
三个阶段的提示词里都不能省。
"""

from __future__ import annotations

from agent_app.ds.checks import CheckReport
from agent_app.ds.dataset import Dataset
from agent_app.ds.executor import ExecutionOutcome
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest

#: 数据形状说明。三个阶段的提示词都要带上——模型对列的误解是最高频的错误来源。
_DATA_NOTES = """\
df 是数据快照对应的长表，每行是「某一天、某个指标、一个值」。列：

    measured_date  zh_name  en_name  kind  value_num  low  high
    minute_of_day  unit  ref_low  ref_high  source_value

必须记住的几点：

- 时间粒度只有**天**（北京时间），没有时刻。需要「几点睡」这类分析只能放弃。
- `kind` 说明这个值的形态：
  - `scalar` / `duration` / `timestamp`：数值在 `value_num`（时长已换算成分钟，
    时刻是当天第几分钟）。
  - `interval`：实测区间，数值在 `low` / `high`，**`value_num` 是空的**。
    区间不能当标量算均值，只能比上下界或区间宽度。
  - `text`：拆不出数值。
- **`value_num` 为空就表示这个点不能参与数值统计**，不是缺失、更不是 0。
- **某一天没有行 = 那天没有数据**，不是 0。按 0 处理会把均值拽下去。
- 同一个 `zh_name` 可能出现多个 `unit`，统计前必须先统一单位。\
"""

PLAN_SYSTEM = """\
你是一个数据分析的**计划制定者**。你不写代码，也不解释结果。

你能看到的是一份数据快照（每个指标的概况），看不到原始数据。你要产出一份结构化计划，
让另一个程序照着它写代码。

硬性要求：

1. **概念变量**：把问题翻译成快照里**真实存在**的指标中文名，只能用快照里出现过的
   名字。快照里没有的指标不要写。
   **先确认问题在问哪一类事**，再定点数标准：问「这个值正常吗」是对照快照里的
   `参考区间`，几个点就能答；问「在变吗」是趋势，问「两个指标有关系吗」是相关，
   这两类才需要更多点。**不要改变问题的类型**——用户问「正常吗」不要去算趋势，
   那会把一个能答的问题判成数据不足。
2. **数据转换**：写清楚要过滤什么、要不要清洗、按什么聚合。注意时间粒度只有天。
3. **统计方法**：每个方法都要点名**确切的检验名称**（如 Spearman 秩相关、配对 t 检验），
   说明为什么选它，列出它的**适用性前提**，并写清代码里如何检验这些前提（例如
   「先做 Shapiro-Wilk 正态性检验」）。
4. **数据充足性**：只有能指名**具体缺哪一类数据**时才 `sufficient=false`。可据以
   判定的只有这五种事实：
   - 快照里没有这个问题需要的指标；
   - 用户问的时间段内该指标一条记录都没有；
   - 该指标的点全是 `kind=text`（拆不出数值），或需要趋势/相关却全是 `interval`；
   - 需要的两个指标在时间上没有重叠；
   - 点数少于你所选检验的**最低**要求——必须点名是哪个检验、最低几个点（例如
     「Spearman 秩相关至少 3 个点」）。
   **不许用「感觉稀疏」「通常需要 20-30 个点」这类没有依据的经验门槛**：同一份数据、
   同一个问题必须每次得到同一个判定。`insufficiency_reason` 要写明命中上面哪一条，
   并给出快照里的实际点数。
5. **可复现**：另一个分析师用同一份数据必须得到完全相同的数值。计划里不能留任何
   需要额外假设的地方。

你不做健康解释，不给医疗建议，不揣测病因。你只回答「要算什么」。\
"""

CODE_SYSTEM = """\
你是一个数据分析代码生成器。你只写一个 Python 函数。

函数签名固定，不许改：

    def analysis(df: pd.DataFrame) -> dict:

`df` 已经加载好了，**不要自己读文件**，也不要重新构造数据。

返回值必须是字典，形状固定：

    {
      "metrics": [{"zh_name": str, "unit": str, "statistic": str,
                   "value": float, "n": int, "method": str,
                   "ci_low": float | None, "ci_high": float | None}],
      "derived": [{"statistic": str, "value": float, "n": int, "method": str,
                   "inputs": [指标中文名], "unit": str,
                   "ci_low": float | None, "ci_high": float | None}],
      "findings": [str],
      "figures": [{"type": "line" | "bar", "title": str, "x": [str],
                   "series": [{"name": str, "values": [float | None]}],
                   "x_label": str, "y_label": str}],
      "sample_size": {指标中文名: int},
      "warnings": [str],
      "kind": "ok" | "insufficient_data" | "failed",
      "error": str | None
    }

字段含义：

- `statistic` 是统计量的名字（`mean` / `trend_slope` / `pearson_r`），
  `method` 是实际用的方法（`kendalltau` / `spearmanr`）。降级时两者会不一致，
  如实写。
- `n` 是这个结果基于几个点。
- **`metrics` 与 `derived` 的分界是单位从哪里来。**
  - `metrics` 是**指标本身的值**：体重的均值、步数的中位数。`zh_name` 必须是快照里
    的指标名，`unit` 必须照快照写（数据里是 `千克` 就写 `千克`）。
  - `derived` 是**由指标算出来的派生量**：变化速率、p 值、相关系数、R²。用 `inputs`
    列出算它用到了哪些指标（必须在快照里），`unit` 写推导出来的单位——体重对时间的
    斜率是 `千克/天`，相关系数无量纲就留空。
  - **不要把派生量塞进 `metrics` 去迁就单位。** 把 `千克/天` 写成 `千克` 是错标签，
    下游会当成真的千克去读。
- `findings` 是中文句子，**只解释已经算出来的数字，不许引入新数字**。
- `sample_size` 只登记 `metrics` 里的指标，每个条目必须与同名的 `metrics[].n` 相等；
  `derived` 不登记。
- `kind` 非 `ok` 时 `error` 必填，说明为什么答不了。
- 没有数据的图点写 `null`，不要写 0。

硬性规则：

- **禁止 `try` / `except`。** 出错就让异常抛出去。吞掉异常等于静默给出错数字。
- **不要在函数里 `print`。** 结果靠 `return`，不靠输出。
- **不许编造数字。** 每个 `value` 都必须来自对 `df` 的真实计算；算不出来就不要放进
  `metrics`，改用 `warnings` 或把 `kind` 设为 `insufficient_data`。
- `figures` 里每个 `series.values` 的长度必须等于 `x` 的长度。
- 只输出代码。不要 markdown 代码块，不要注释之外的说明文字。

这个环境只装了 `pandas`、`numpy` 和 `scipy`——没有 `matplotlib`、没有 `statsmodels`、
没有网络，也装不了新包。要用什么检验就用 `scipy.stats` 里现成的，不要自己实现。

这个环境是 pandas 3.0，下面几条与旧写法不同：

- **频率别名改了。** `resample("M")` / `to_period("Q")` / `pd.Grouper(freq="Y")`
  会直接抛异常，要写成 `"ME"` / `"QE"` / `"YE"`。`"D"` / `"W"` / `"h"` 没有变。
- **禁止链式赋值。** `df[条件]["列"] = 值` 在这里**静默不生效**，而且不再有
  `SettingWithCopyWarning` 提示——代码不报错，结果却是错的。要改就先 `df = df.copy()`，
  再用 `df.loc[条件, "列"] = 值`。
- **`.loc` 赋值不会自动放宽 dtype。** 在整数列上赋浮点值会报错，要先
  `df["列"] = df["列"].astype(float)`。
- **取第几个元素用 `.iloc`。** `series[0]` 恒为按标签取值。\
"""


def _shared_block(request: AnalysisRequest, dataset: Dataset) -> str:
    lines = [f"用户的问题：{request.question}", ""]
    if request.zh_names:
        lines.append(f"调用方指定要分析的指标：{'、'.join(request.zh_names)}")
    if request.start is not None or request.end is not None:
        lines.append(
            f"调用方指定的时间窗：{request.start or '不限'} ~ {request.end or '不限'}"
        )
    lines += ["", "数据快照：", dataset.catalog, "", _DATA_NOTES]
    return "\n".join(lines)


def plan_prompt(request: AnalysisRequest, dataset: Dataset) -> str:
    return "\n".join(
        [
            _shared_block(request, dataset),
            "",
            "请产出分析计划。",
        ]
    )


def code_prompt(
    request: AnalysisRequest, plan: AnalysisPlan, dataset: Dataset, *, attempt: int
) -> str:
    where = "第一次生成" if attempt <= 1 else f"第 {attempt} 次生成"
    return "\n".join(
        [
            _shared_block(request, dataset),
            "",
            "分析计划（JSON）：",
            plan.model_dump_json(indent=2),
            "",
            f"请写 `analysis(df)`（{where}）。计划里的每个统计方法都要在代码里落地，"
            "包括那些适用性前提的检验。",
        ]
    )


def fix_prompt(
    request: AnalysisRequest,
    plan: AnalysisPlan,
    dataset: Dataset,
    *,
    code: str,
    outcome: ExecutionOutcome,
    report: CheckReport | None,
) -> str:
    """修错提示词。

    分两段给信息：先是执行器报的错（异常 / 超时 / 没有返回值），再是验收报的问题
    （数值对不上、单位不对、图长度不对）。两段都可能为空，但至少有一段非空——
    调用方只有在上一步确实失败时才走到这里。
    """
    blocks = [
        _shared_block(request, dataset),
        "",
        "分析计划（JSON）：",
        plan.model_dump_json(indent=2),
        "",
        "你上一次写的代码：",
        "```python",
        code,
        "```",
        "",
    ]

    if not outcome.ok:
        blocks += ["执行失败了：", outcome.failure_note(), ""]

    if report is not None and report.problems:
        blocks += ["结果验收没通过：", *[f"- {item}" for item in report.problems], ""]

    blocks += [
        "请**只修上面指出的问题**，输出完整的 `analysis(df)`。",
        "",
        "不许做的事：",
        "- 不要加新的分析、新的图表、新的统计量。",
        "- 不要加 `try` / `except` 把错误吞掉。宁可让它继续报错，"
        "也不要给出一个错的数字。",
        "- 不要重写正确的逻辑，只改出问题的那部分。",
    ]
    return "\n".join(blocks)
