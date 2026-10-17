"""``ds/report.py``：把结果排成主 agent 能读的文本。

这个渲染器是「所有数字必须由代码算出」这条约束的落点——主 agent 看不到
``AnalysisResult`` 本身，只能从这段文本里抄数字。所以这里钉三件事：数字按原值出现、
每个数字带着指标与样本量、失败时说清是哪种失败。
"""

from __future__ import annotations

from typing import Any

from tests.ds_factories import derived, outcome

from agent_app.ds.report import render_result
from agent_app.ds.schemas import AnalysisResult


def _result(**overrides: Any) -> AnalysisResult:  # noqa: ANN401 - 透传构造参数
    payload: dict[str, Any] = {"kind": "ok"}
    payload.update(overrides)
    return AnalysisResult.model_validate(payload)


def test_numbers_are_printed_at_full_precision() -> None:
    """不做显示四舍五入。

    写成 ``77.3`` 会让主 agent 抄走一个 ``metrics`` 里没有的数字，将来要验「最终回答
    的每个数字都能在结果里找到」就会误报。
    """
    result = _result(
        metrics=[outcome("体重", 18, 77.31666666666666, unit="千克")],
        sample_size={"体重": 18},
    )

    assert "77.31666666666666 千克" in render_result(result)


def test_number_carries_its_metric_method_and_sample_size() -> None:
    result = _result(
        metrics=[outcome("体重", 18, 70.0, unit="千克")],
        sample_size={"体重": 18},
    )

    assert "- 体重 · 平均值 = 70 千克（n=18，mean）" in render_result(result)


def test_method_is_hidden_when_it_repeats_the_statistic_name() -> None:
    result = _result(
        metrics=[outcome("体重", 18, 70.0, unit="千克", method="平均值")],
        sample_size={"体重": 18},
    )

    text = render_result(result)

    assert "- 体重 · 平均值 = 70 千克（n=18）" in text


def test_derived_statistic_names_the_metrics_it_used() -> None:
    result = _result(
        derived=[derived("trend_slope", 18, -0.071, inputs=["体重"], unit="千克/天")],
        sample_size={"体重": 18},
    )

    assert "- 体重 · trend_slope = -0.071 千克/天（n=18，linregress）" in render_result(
        result
    )


def test_derived_without_inputs_says_so() -> None:
    """``inputs`` 为空是 Check 要拦的形态，但渲染器也得能把它排出来。"""
    result = _result(derived=[derived("p", 18, 0.03)])

    assert "来源未标明 · p = 0.03（n=18，linregress）" in render_result(result)


def test_dimensionless_value_has_no_trailing_space() -> None:
    result = _result(derived=[derived("spearman_rho", 35, -0.158, inputs=["步数"])])

    assert "= -0.158（n=35，linregress）" in render_result(result)


def test_confidence_interval_is_shown_when_present() -> None:
    entry = outcome("体重", 18, 70.0, unit="千克")
    entry["ci_low"] = 69.5
    entry["ci_high"] = 70.5
    result = _result(metrics=[entry], sample_size={"体重": 18})

    assert "- 体重 · 平均值 = 70 千克，区间 69.5 ~ 70.5（n=18，mean）" in render_result(
        result
    )


def test_findings_and_warnings_have_their_own_sections() -> None:
    result = _result(
        metrics=[outcome("步数", 31, 7937.971428571429, unit="步")],
        sample_size={"步数": 35},
        findings=["平均每天约 7938 步。"],
        warnings=["步数：窗口内可数值化的点有 35 个，这个结果用了 31 个"],
        attempts=2,
    )

    text = render_result(result)

    assert "结论\n- 平均每天约 7938 步。" in text
    assert "需要注意\n- 步数：窗口内可数值化的点有 35 个，这个结果用了 31 个" in text
    assert "（计划 + 代码共跑了 2 轮）" in text


def test_plan_window_and_metrics_appear_when_the_plan_is_known() -> None:
    result = _result(
        plan={
            "zh_names": ["体重", "步数"],
            "start": "2026-09-10",
            "end": "2026-10-14",
            "sufficient": True,
        }
    )

    text = render_result(result)

    assert "- 分析窗口：2026-09-10 ~ 2026-10-14" in text
    assert "- 涉及指标：体重、步数" in text


def test_no_plan_means_no_window_lines() -> None:
    assert "分析窗口" not in render_result(_result())


def test_failure_keeps_the_numbers_that_were_already_computed() -> None:
    """重试耗尽时也可能带着算出来的部分结果，那部分对用户有价值。"""
    result = _result(
        kind="failed",
        error="4 轮都没通过验收",
        metrics=[outcome("体重", 18, 70.0, unit="千克")],
        sample_size={"体重": 18},
    )

    text = render_result(result)

    assert text.startswith("数据分析没能完成\n- 原因：4 轮都没通过验收")
    assert "70 千克" in text


def test_insufficient_data_says_what_is_missing() -> None:
    result = _result(kind="insufficient_data", error="窗口内只有 2 个血压点")

    text = render_result(result)

    assert text.startswith("数据不足，这个问题答不了\n- 原因：窗口内只有 2 个血压点")
