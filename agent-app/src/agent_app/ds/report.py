"""把 ``AnalysisResult`` 渲染成给主 agent 看的文本。

tool 的返回值只能是一个字符串（见 ``agent_app/ds/tool.py``），主 agent 读这段文本写
最终回答。所以渲染要满足三条：

1. **数字只来自执行结果。** 这里不计算、不换算、不补全，只排版。
2. **每个数字带着它的指标、量纲和样本量。** 主 agent 看不到 ``AnalysisResult`` 本身，
   丢了这三样它就分不清 ``-0.158`` 是相关系数还是斜率，也不知道它基于几个点。
3. **原值输出，不做显示四舍五入。** 写 ``77.3`` 会让主 agent 抄走一个 ``metrics``
   里没有的数字，将来要验「最终回答的每个数字都能在结果里找到」就会误报。要不要把
   ``77.31666666666666`` 写成人话是主 agent 的表述自由，原料给原值。
"""

from __future__ import annotations

from agent_app.ds.schemas import (
    AnalysisPlan,
    AnalysisResult,
    DerivedStatistic,
    MetricOutcome,
)

_HEADINGS = {
    "ok": "数据分析结果",
    "insufficient_data": "数据不足，这个问题答不了",
    "failed": "数据分析没能完成",
}


def _number(value: float) -> str:
    """原值输出，只把 ``70.0`` 写成 ``70``。

    ``repr`` 是 Python 里唯一能保证 ``float(repr(x)) == x`` 的写法。
    """
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _measured(value: float, unit: str) -> str:
    return f"{_number(value)} {unit}".rstrip()


def _interval(low: float | None, high: float | None) -> str:
    if low is None and high is None:
        return ""
    left = _number(low) if low is not None else "?"
    right = _number(high) if high is not None else "?"
    return f"，区间 {left} ~ {right}"


def _metric_line(outcome: MetricOutcome) -> str:
    method = f"，{outcome.method}" if outcome.method != outcome.statistic else ""
    return (
        f"- {outcome.zh_name} · {outcome.statistic} = "
        f"{_measured(outcome.value, outcome.unit)}"
        f"{_interval(outcome.ci_low, outcome.ci_high)}（n={outcome.n}{method}）"
    )


def _derived_line(item: DerivedStatistic) -> str:
    source = "、".join(item.inputs) if item.inputs else "来源未标明"
    return (
        f"- {source} · {item.statistic} = "
        f"{_measured(item.value, item.unit)}"
        f"{_interval(item.ci_low, item.ci_high)}（n={item.n}，{item.method}）"
    )


def _plan_lines(plan: AnalysisPlan | None) -> list[str]:
    if plan is None:
        return []
    if plan.start is None and plan.end is None:
        window = "全部历史"
    else:
        window = f"{plan.start or '起始不限'} ~ {plan.end or '至今'}"
    names = "、".join(plan.zh_names) if plan.zh_names else "未指定"
    return [f"- 分析窗口：{window}", f"- 涉及指标：{names}"]


def render_result(result: AnalysisResult) -> str:
    """把一次分析排成一段文本。任何 ``kind`` 都能渲染。"""
    lines = [_HEADINGS[result.kind]]
    if result.error:
        lines.append(f"- 原因：{result.error}")
    lines += _plan_lines(result.plan)

    if result.metrics or result.derived:
        lines += ["", "数值（全部由代码在真实数据上算出）"]
        lines += [_metric_line(item) for item in result.metrics]
        lines += [_derived_line(item) for item in result.derived]

    if result.findings:
        lines += ["", "结论"]
        lines += [f"- {item}" for item in result.findings]

    if result.warnings:
        lines += ["", "需要注意"]
        lines += [f"- {item}" for item in result.warnings]

    if result.attempts:
        lines += ["", f"（计划 + 代码共跑了 {result.attempts} 轮）"]

    return "\n".join(lines)


def plan_note(plan: AnalysisPlan) -> str:
    """一句话说明这次要算什么，给前端做过程提示用。

    用模板拼，不经模型：这是对事实的转述，模型只会把它写成另一种口吻的结论。整句
    带「DS Agent 工作中」前缀，前端拿到什么就显示什么，不替服务端拼句子。
    """
    names = "、".join(plan.zh_names[:3])
    if len(plan.zh_names) > 3:
        names = f"{names} 等 {len(plan.zh_names)} 项"
    if not names:
        names = "健康数据"

    window = ""
    if plan.start is not None and plan.end is not None:
        window = f" {plan.start.isoformat()} 至 {plan.end.isoformat()} 的"
    elif plan.start is not None:
        window = f" {plan.start.isoformat()} 起的"
    return f"DS Agent 工作中：正在分析{window}{names}"
