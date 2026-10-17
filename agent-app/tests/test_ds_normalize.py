"""指标归一化层的单元测试。

不需要数据库、Redis 或网络：``agent_app.ds.normalize`` 是纯函数。样本值取自
``parsing/prompt.py`` 里抽取规则给出的真实写法。
"""

import uuid
from datetime import date

import pytest

from agent_app.ds.normalize import (
    MetricPoint,
    normalize_metric,
    normalize_metrics,
    normalize_unit,
    parse_reference_range,
    parse_value,
)
from agent_app.models.health import HealthMetric


def _metric(value: str, unit: str = "", reference_range: str = "") -> HealthMetric:
    """造一条 ``health_metrics`` 记录。不入库，只用于归一化。"""
    return HealthMetric(
        user_id=uuid.uuid4(),
        measured_date=date(2026, 10, 12),
        en_name="",
        zh_name="睡眠时长",
        value=value,
        unit=unit,
        reference_range=reference_range,
        position=0,
    )


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("2.69", "公里", 2.69),
        ("136", "次/分钟", 136.0),
        ("97", "%", 97.0),
        ("0", "米", 0.0),
        ("-0.5", "", -0.5),
        ("2.69", "", 2.69),
    ],
)
def test_scalar_value(value: str, unit: str, expected: float) -> None:
    parsed = parse_value(value, unit)

    assert parsed.kind == "scalar"
    assert parsed.value_num == expected


def test_scalar_value_normalizes_unit() -> None:
    # 次/分钟 与 次/分 是同一个单位，不归一就会把一条序列劈成两条。
    assert parse_value("136", "次/分钟").unit == "次/分"
    assert parse_value("136", "次/分").unit == "次/分"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1小时 24 分钟", 84.0),
        ("6 小时 47 分钟", 407.0),
        ("10小时33分钟", 633.0),
        ("2小时", 120.0),
        ("45分钟", 45.0),
        ("1分30秒", 1.5),
    ],
)
def test_composite_duration_becomes_minutes(value: str, expected: float) -> None:
    parsed = parse_value(value, "")

    assert parsed.kind == "duration"
    assert parsed.value_num == expected
    assert parsed.unit == "分钟"


def test_interval_keeps_bounds_and_no_scalar() -> None:
    # 整夜实测范围不能当标量算均值。
    parsed = parse_value("52-171", "次/分")

    assert parsed.kind == "interval"
    assert parsed.value_num is None
    assert parsed.low == 52.0
    assert parsed.high == 171.0


@pytest.mark.parametrize("value", ["12.5-20.0", "12.5~20.0", "52至171"])
def test_interval_separator_variants(value: str) -> None:
    assert parse_value(value, "次/分").kind == "interval"


def test_time_of_day_becomes_minute_of_day() -> None:
    # 户外健走时间：语义是「发生在几点」，不是「值等于 21.72」。
    parsed = parse_value("2026-10-12 21:43", "")

    assert parsed.kind == "timestamp"
    assert parsed.minute_of_day == 21 * 60 + 43
    assert parsed.value_num is None


def test_bare_clock() -> None:
    assert parse_value("07:05", "").minute_of_day == 7 * 60 + 5


@pytest.mark.parametrize("value", ["阴性", "未见异常", "", "肝内钙化灶"])
def test_text_value_is_kept_not_dropped(value: str) -> None:
    parsed = parse_value(value, "")

    assert parsed.kind == "text"
    assert parsed.value_num is None
    assert parsed.low is None


def test_value_with_trailing_unit_uses_it() -> None:
    parsed = parse_value("136次/分", "")

    assert parsed.kind == "scalar"
    assert parsed.value_num == 136.0
    assert parsed.unit == "次/分"


def test_given_unit_wins_over_trailing_unit() -> None:
    # 抽取阶段已经拆出单位时，以它为准，不从值里再猜一次。
    assert parse_value("97%", "%").unit == "%"


def test_fullwidth_digits_are_flattened() -> None:
    assert parse_value("２.６９", "").value_num == 2.69
    assert normalize_unit("次／分钟") == "次/分"


def test_inf_and_nan_are_not_numbers() -> None:
    assert parse_value("nan", "").kind == "text"
    assert parse_value("inf", "").kind == "text"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("130-175", (130.0, 175.0)),
        ("3.5~5.5", (3.5, 5.5)),
        ("<5", (None, 5.0)),
        ("≤5", (None, 5.0)),
        (">10", (10.0, None)),
        ("≥10", (10.0, None)),
        ("阴性", (None, None)),
        ("", (None, None)),
    ],
)
def test_parse_reference_range(
    raw: str, expected: tuple[float | None, float | None]
) -> None:
    assert parse_reference_range(raw) == expected


def test_normalize_unit_keeps_unknown_unit() -> None:
    assert normalize_unit("次/分钟") == "次/分"
    assert normalize_unit("千米") == "公里"
    assert normalize_unit("") == ""
    # 认不出来就原样返回，不猜。
    assert normalize_unit("自定义XYZ") == "自定义XYZ"


def test_normalize_metric_keeps_source_text() -> None:
    point = normalize_metric(_metric("1小时 24 分钟", "", "130-175"))

    assert isinstance(point, MetricPoint)
    assert point.kind == "duration"
    assert point.value_num == 84.0
    assert point.measured_date == date(2026, 10, 12)
    assert point.zh_name == "睡眠时长"
    # 转换有损，原文必须留下。
    assert point.source_value == "1小时 24 分钟"
    assert point.ref_low == 130.0
    assert point.ref_high == 175.0


def test_normalize_metrics_keeps_order() -> None:
    points = normalize_metrics(
        [_metric("2.69", "公里"), _metric("阴性"), _metric("52-171", "次/分")]
    )

    assert [point.kind for point in points] == ["scalar", "text", "interval"]
    assert [point.source_value for point in points] == ["2.69", "阴性", "52-171"]


def test_none_columns_are_treated_as_empty() -> None:
    """``health_metrics`` 的字符串列有 ``server_default``，但未入库的对象上是 ``None``。

    以前这里会抛 ``AttributeError: 'NoneType' object has no attribute 'strip'``，
    把整轮分析崩在数据准备阶段。
    """
    assert normalize_unit(None) == ""
    assert parse_reference_range(None) == (None, None)

    parsed = parse_value(None)

    assert parsed.kind == "text"
    assert parsed.value_num is None
    assert parse_value("70", None).unit == ""
