"""验收层：结果能不能信。全部是纯代码，这里的用例不碰模型也不碰子进程。"""

from __future__ import annotations

from datetime import date

from tests.ds_factories import derived, outcome, point

from agent_app.ds.checks import CheckReport, check_result
from agent_app.ds.dataset import Dataset, build_dataset
from agent_app.ds.normalize import MetricPoint
from agent_app.ds.schemas import AnalysisResult


def _dataset(*points: MetricPoint) -> Dataset:
    return build_dataset(list(points))


def _result(
    *,
    metrics: list[dict[str, object]] | None = None,
    derived: list[dict[str, object]] | None = None,
    sample_size: dict[str, int] | None = None,
    findings: list[str] | None = None,
    figures: list[dict[str, object]] | None = None,
    warnings: list[str] | None = None,
    plan: dict[str, object] | None = None,
    kind: str = "ok",
) -> AnalysisResult:
    data: dict[str, object] = {
        "kind": kind,
        "metrics": metrics or [],
        "derived": derived or [],
        "sample_size": sample_size or {},
        "findings": findings if findings is not None else ["结论。"],
        "figures": figures or [],
        "warnings": warnings or [],
    }
    if plan is not None:
        data["plan"] = plan
    if kind != "ok":
        data["error"] = "答不了"
    return AnalysisResult.model_validate(data)


def test_empty_dataset_fails_everything() -> None:
    report = check_result(_result(), build_dataset([]))

    assert not report.ok
    assert "数据快照是空的" in report.problems[0]


def test_sound_result_passes() -> None:
    dataset = _dataset(
        point("体重", "70", unit="公斤"), point("体重", "71", unit="公斤")
    )
    result = _result(
        metrics=[outcome("体重", 2, 70.5, unit="公斤")],
        sample_size={"体重": 2},
    )

    report = check_result(result, dataset)

    assert report.ok
    assert report.warnings == ()


def test_n_larger_than_available_points_is_rejected() -> None:
    """把区间值当成数值算进去时，n 会大于可数值化的点数。"""
    dataset = _dataset(point("血压", "120"), point("血压", "52-171", kind="interval"))
    result = _result(metrics=[outcome("血压", 2)], sample_size={"血压": 2})

    report = check_result(result, dataset)

    assert not report.ok
    assert any("超过窗口内可数值化的" in item for item in report.problems)


def test_metric_without_any_numeric_point_cannot_produce_a_statistic() -> None:
    dataset = _dataset(point("血压", "52-171", kind="interval"))
    result = _result(metrics=[outcome("血压", 1)], sample_size={"血压": 1})

    report = check_result(result, dataset)

    assert not report.ok
    assert any("一个可数值化的点都没有" in item for item in report.problems)


def test_sample_size_must_agree_with_metric_n() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(metrics=[outcome("体重", 1, unit="公斤")], sample_size={"体重": 9})

    report = check_result(result, dataset)

    assert not report.ok
    assert any("同一个事实不能有两个说法" in item for item in report.problems)


def test_missing_sample_size_is_rejected() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(metrics=[outcome("体重", 1, unit="公斤")], sample_size={})

    report = check_result(result, dataset)

    assert any("没有登记样本量" in item for item in report.problems)


def test_metric_outside_the_snapshot_is_rejected() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(metrics=[outcome("血糖", 1)], sample_size={"血糖": 1})

    report = check_result(result, dataset)

    assert any("不在本次数据快照里" in item for item in report.problems)


def test_unit_must_exist_in_the_data() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(metrics=[outcome("体重", 1, unit="斤")], sample_size={"体重": 1})

    report = check_result(result, dataset)

    assert any("不在数据里的" in item for item in report.problems)


def test_unit_synonym_is_not_a_mistake() -> None:
    """目录给模型看的是归一化之后的单位，模型写同义词不算错。

    ``normalize_unit("公斤") == "千克"``——两边同一个单位。判错不会让答案变对，
    只会白烧一轮重试预算（总共只有 4 轮）。
    """
    dataset = _dataset(point("体重", "70", unit="千克"))
    result = _result(metrics=[outcome("体重", 1, unit="公斤")], sample_size={"体重": 1})

    report = check_result(result, dataset)

    assert report.ok, report.problems


def test_filtering_rows_is_a_warning_not_a_failure() -> None:
    """只用了一部分点是分析的正常部分，但用户必须看得到。"""
    dataset = _dataset(
        point("体重", "70", unit="公斤"),
        point("体重", "71", day=date(2026, 5, 2), unit="公斤"),
        point("体重", "72", day=date(2026, 5, 3), unit="公斤"),
    )
    result = _result(
        metrics=[outcome("体重", 2, 70.5, unit="公斤")], sample_size={"体重": 2}
    )

    report = check_result(result, dataset)

    assert report.ok
    assert any("这个结果用了 2 个" in item for item in report.warnings)


def test_ok_with_nothing_in_it_is_rejected() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(findings=[])

    report = check_result(result, dataset)

    assert any("没有内容的成功不算成功" in item for item in report.problems)


def test_non_ok_result_may_have_nothing_in_it() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(kind="insufficient_data", findings=[])

    report = check_result(result, dataset)

    assert report.ok


def test_series_length_must_match_axis() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(
        metrics=[outcome("体重", 1, unit="公斤")],
        sample_size={"体重": 1},
        figures=[
            {
                "type": "line",
                "title": "体重趋势",
                "x": ["2026-05-01", "2026-05-02"],
                "series": [{"name": "体重", "values": [70.0]}],
            }
        ],
    )

    report = check_result(result, dataset)

    assert any("必须等长" in item for item in report.problems)


def test_figure_without_series_is_rejected() -> None:
    dataset = _dataset(point("体重", "70", unit="公斤"))
    result = _result(
        metrics=[outcome("体重", 1, unit="公斤")],
        sample_size={"体重": 1},
        figures=[{"type": "line", "title": "体重趋势", "x": [], "series": []}],
    )

    report = check_result(result, dataset)

    assert any("没有任何数据序列" in item for item in report.problems)


def test_unit_conflict_must_be_addressed_in_warnings() -> None:
    dataset = _dataset(
        point("体重", "70", unit="公斤"),
        point("体重", "154", day=date(2026, 5, 2), unit="斤"),
    )
    result = _result(
        metrics=[outcome("体重", 1, 70.0, unit="公斤")], sample_size={"体重": 1}
    )

    report = check_result(result, dataset)

    assert report.ok  # 不阻断
    assert any("出现了多个单位" in item for item in report.warnings)


def test_unit_conflict_already_explained_is_not_repeated() -> None:
    dataset = _dataset(
        point("体重", "70", unit="公斤"),
        point("体重", "154", day=date(2026, 5, 2), unit="斤"),
    )
    result = _result(
        metrics=[outcome("体重", 1, 70.0, unit="公斤")],
        sample_size={"体重": 1},
        warnings=["体重：已把斤折算成公斤"],
    )

    report = check_result(result, dataset)

    assert not any("出现了多个单位" in item for item in report.warnings)


def test_merged_warnings_keeps_order_and_drops_duplicates() -> None:
    report = CheckReport(problems=(), warnings=("甲", "乙"))

    merged = report.merged_warnings(["乙", "", "丙"])

    assert merged == ["甲", "乙", "丙"]


def test_derived_statistic_keeps_its_own_unit() -> None:
    """1b 的真实回归：斜率是「千克/天」，不该被逼着写成「千克」。

    改之前「单位必须来自数据」这条对派生量也生效，模型连拒三轮之后把 ``千克/天``
    改成了 ``千克``——验收把对的标签换成了错的。
    """
    dataset = _dataset(
        point("体重", "70", unit="千克"),
        point("体重", "71", day=date(2026, 5, 2), unit="千克"),
    )
    result = _result(
        derived=[derived("trend_slope", 2, -0.07, inputs=["体重"], unit="千克/天")],
        findings=["体重在这段时间里平均每天下降 0.07 千克。"],
    )

    report = check_result(result, dataset)

    assert report.ok, report.problems


def test_derived_statistic_needs_a_source_in_the_snapshot() -> None:
    dataset = _dataset(point("体重", "70", unit="千克"))
    result = _result(
        derived=[derived("trend_slope", 1, 0.1, inputs=["步数"], unit="千克/天")]
    )

    report = check_result(result, dataset)

    assert not report.ok
    assert any("不在本次数据快照里" in item for item in report.problems)


def test_derived_statistic_cannot_use_more_points_than_its_weakest_input() -> None:
    dataset = _dataset(
        point("步数", "8000"),
        point("步数", "9000", day=date(2026, 5, 2)),
        point("睡眠时长", "420"),
    )
    result = _result(
        derived=[derived("spearman_rho", 3, -0.16, inputs=["步数", "睡眠时长"])]
    )

    report = check_result(result, dataset)

    assert not report.ok
    assert any("派生量不可能用到比输入还多的点" in item for item in report.problems)


def test_derived_statistic_without_inputs_is_rejected() -> None:
    dataset = _dataset(point("体重", "70"))
    result = _result(derived=[derived("trend_slope", 1, 0.1)])

    report = check_result(result, dataset)

    assert not report.ok
    assert any("没有声明 inputs" in item for item in report.problems)


def test_metric_outcome_still_refuses_a_derived_unit() -> None:
    """分界要守住：``metrics`` 里的单位仍然必须来自数据。"""
    dataset = _dataset(point("体重", "70", unit="千克"))
    result = _result(
        metrics=[outcome("体重", 1, 70.0, unit="千克/天")],
        sample_size={"体重": 1},
    )

    report = check_result(result, dataset)

    assert not report.ok
    assert any("不在数据里的" in item for item in report.problems)


def test_code_cannot_overrule_a_sufficient_plan() -> None:
    """计划说数据够，代码没有资格改判。

    真实回归：代码给血压加了个「必须是区间型」的前提，把 18 条标量行全部过滤掉，
    然后交回 ``insufficient_data``。用户看到的是「数据不足」，而计划和数据都认为
    数据是够的。
    """
    dataset = _dataset(point("体重", "70", unit="千克"))
    result = _result(
        kind="insufficient_data",
        plan={"zh_names": ["体重"], "sufficient": True},
    )

    report = check_result(result, dataset)

    assert not report.ok
    assert any("计划判定数据充足" in item for item in report.problems)


def test_insufficient_plan_may_still_refuse() -> None:
    """计划自己说不充分时，拒答是计划层的正常结论，不该被这条规则拦住。"""
    dataset = _dataset(point("体重", "70", unit="千克"))
    result = _result(
        kind="insufficient_data",
        plan={
            "zh_names": ["体重"],
            "sufficient": False,
            "insufficiency_reason": "窗口内只有 1 个点。",
        },
    )

    report = check_result(result, dataset)

    assert report.ok, report.problems
