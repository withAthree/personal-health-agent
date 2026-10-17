"""DS 契约（``agent_app.ds.schemas``）的单元测试。

不需要数据库、Redis 或网络。重点验证「靠类型系统兜住的硬约束」是否真的兜住了：
NaN / inf 进不来，非 ``ok`` 的结果不能静默返回，倒置的时间窗在入口就被拒。
"""

from datetime import date

import pytest
from pydantic import ValidationError

from agent_app.ds.schemas import (
    AnalysisPlan,
    AnalysisRequest,
    AnalysisResult,
    Figure,
    MetricOutcome,
    Series,
    StatisticSpec,
)


def _outcome(value: float = 7.2) -> MetricOutcome:
    return MetricOutcome(
        zh_name="睡眠时长",
        unit="小时",
        statistic="mean",
        value=value,
        n=42,
        method="mean",
    )


# --- AnalysisRequest -------------------------------------------------------


def test_request_defaults() -> None:
    """只有 ``question`` 是必填的。"""
    request = AnalysisRequest(question="我最近睡得好吗")

    assert request.zh_names == []
    assert request.start is None
    assert request.end is None


def test_request_rejects_inverted_window() -> None:
    """起点晚于终点直接失败，不要让下游用空结果假装「没数据」。"""
    with pytest.raises(ValidationError, match="时间窗起点晚于终点"):
        AnalysisRequest(question="x", start=date(2026, 10, 12), end=date(2026, 1, 1))


def test_request_allows_same_day_window() -> None:
    request = AnalysisRequest(
        question="x", start=date(2026, 10, 12), end=date(2026, 10, 12)
    )

    assert request.start == request.end


def test_request_rejects_unknown_field() -> None:
    """``extra="forbid"``：模型多塞字段要立刻失败，而不是被静默丢掉。"""
    with pytest.raises(ValidationError):
        AnalysisRequest(question="x", user_id="someone-else")


# --- MetricOutcome --------------------------------------------------------


def test_outcome_roundtrip() -> None:
    outcome = _outcome()

    assert outcome.value == 7.2
    assert outcome.n == 42
    assert outcome.ci_low is None
    assert outcome.ci_high is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_outcome_rejects_non_finite(value: float) -> None:
    """算崩的结果必须构造不出来，而不是留给 Check 步去发现。"""
    with pytest.raises(ValidationError):
        _outcome(value)


def test_outcome_rejects_empty_sample() -> None:
    """``n`` 至少为 1——没有样本量的统计量不该存在。"""
    with pytest.raises(ValidationError):
        MetricOutcome(
            zh_name="睡眠时长",
            unit="小时",
            statistic="mean",
            value=1.0,
            n=0,
            method="mean",
        )


# --- Figure / Series ------------------------------------------------------


def test_figure_holds_gaps_as_none() -> None:
    """缺失点用 null，不用 0 填——0 会把折线拽到底。"""
    figure = Figure(
        type="line",
        title="睡眠时长趋势",
        x=["2026-10-10", "2026-10-11", "2026-10-12"],
        series=[Series(name="睡眠时长", values=[7.5, None, 6.1])],
        x_label="日期",
        y_label="小时",
    )

    assert figure.series[0].values[1] is None


def test_figure_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        Series(name="睡眠时长", values=[float("nan")])


def test_figure_accepts_numeric_ticks() -> None:
    """刻度是数字还是字符串，对下游没有区别，不该为它烧一轮重试。

    真实回归：Q3 的第 2 轮，diff 全文只改了 `x = list(range(n))` →
    `[str(i) for i in ...]`。
    """
    figure = Figure(
        type="line",
        title="步数",
        x=[1, 2, 3],
        series=[Series(name="步数", values=[1.0, 2.0, 3.0])],
    )

    assert figure.x == ["1", "2", "3"]


def test_figure_still_rejects_a_tick_that_is_neither_number_nor_string() -> None:
    with pytest.raises(ValidationError):
        Figure(type="line", title="步数", x=[None], series=[])


def test_figure_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Figure(type="scatter", title="x", x=[], series=[])


# --- AnalysisResult -------------------------------------------------------


def test_result_ok() -> None:
    result = AnalysisResult(
        kind="ok",
        plan=AnalysisPlan(
            zh_names=["睡眠时长"],
            statistics=[
                StatisticSpec(
                    name="trend_slope",
                    method="Mann-Kendall",
                    rationale="睡眠时长不满足正态假设",
                    assumptions=["数据点独立", "无未处理的强季节项"],
                )
            ],
            sufficient=True,
        ),
        metrics=[_outcome()],
        findings=["近一年睡眠时长均值 7.2 小时。"],
        sample_size={"睡眠时长": 42},
        attempts=1,
    )

    assert result.kind == "ok"
    assert result.error is None
    assert result.metrics[0].n == result.sample_size["睡眠时长"]


@pytest.mark.parametrize("kind", ["insufficient_data", "failed"])
def test_result_non_ok_requires_error(kind: str) -> None:
    with pytest.raises(ValidationError, match="必须填 error"):
        AnalysisResult(kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["insufficient_data", "failed"])
def test_result_non_ok_can_still_carry_partial_output(kind: str) -> None:
    """答不了也得把已经算出来的部分带回去，那部分对用户有价值。"""
    result = AnalysisResult(
        kind=kind,  # type: ignore[arg-type]
        metrics=[_outcome()],
        sample_size={"睡眠时长": 42},
        error="过去 30 天只有 3 个睡眠时长数据点，不够做趋势检验。",
        attempts=3,
    )

    assert result.metrics
    assert result.error is not None


def test_result_defaults_are_empty() -> None:
    result = AnalysisResult(kind="ok")

    assert result.plan is None
    assert result.metrics == []
    assert result.figures == []
    assert result.findings == []
    assert result.sample_size == {}
    assert result.warnings == []
    assert result.attempts == 0
    assert result.error is None


def test_result_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult(kind="partial")


def test_result_json_roundtrip() -> None:
    """结果要能直接写进 ``messages.tool_calls`` / ``messages.charts`` 两列。"""
    result = AnalysisResult(
        kind="ok",
        metrics=[_outcome()],
        figures=[
            Figure(
                type="bar",
                title="各指标数据点",
                x=["睡眠时长"],
                series=[Series(name="点数", values=[42.0])],
            )
        ],
        sample_size={"睡眠时长": 42},
        attempts=1,
    )

    restored = AnalysisResult.model_validate_json(result.model_dump_json())

    assert restored == result
