"""DS 测试用的构造器。

``MetricPoint`` 有 13 个字段，多数用例只关心其中两三个。这里给一组默认值，让每个
用例只写它真正在意的那部分。
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Literal

from agent_app.ds.normalize import MetricPoint, normalize_unit
from agent_app.models.health import HealthMetric

DEFAULT_DATE = date(2026, 5, 1)

#: 造点的方式。``interval`` 的 ``value`` 写成 ``"52-171"``。
PointKind = Literal["scalar", "interval", "text"]


def point(
    zh_name: str,
    value: str,
    *,
    day: date = DEFAULT_DATE,
    unit: str = "",
    en_name: str = "",
    kind: PointKind = "scalar",
) -> MetricPoint:
    """造一个 ``MetricPoint``。

    ``kind="interval"`` 造的是 ``52-171`` 这类实测区间：没有 ``value_num``，只有
    ``low`` / ``high``，正是 Check 层要拦的那类输入。``kind="text"`` 造拆不出数值的
    点，它们保留在快照里但不参与统计。

    单位与生产路径一致地走一遍 ``normalize_unit``：``point(..., unit="公斤")`` 造出来的
    点的 ``unit`` 是 ``"千克"``——而目录给模型看的正是 ``"千克"``。
    """
    value_num: float | None = None
    low: float | None = None
    high: float | None = None

    if kind == "scalar":
        value_num = float(value)
    elif kind == "interval":
        low_text, _, high_text = value.partition("-")
        low, high = float(low_text), float(high_text)

    return MetricPoint(
        measured_date=day,
        zh_name=zh_name,
        en_name=en_name,
        kind=kind,
        value_num=value_num,
        low=low,
        high=high,
        minute_of_day=None,
        unit=normalize_unit(unit),
        ref_low=None,
        ref_high=None,
        source_value=value,
    )


def outcome(
    zh_name: str,
    n: int,
    value: float = 1.0,
    *,
    unit: str = "",
    method: str = "mean",
) -> dict[str, object]:
    """造一个沙箱返回值里的 ``metrics`` 条目。返回 dict 而不是模型，方便改坏它。"""
    return {
        "zh_name": zh_name,
        "unit": unit,
        "statistic": "平均值",
        "value": value,
        "n": n,
        "method": method,
    }


def derived(
    statistic: str,
    n: int,
    value: float = 1.0,
    *,
    inputs: list[str] | None = None,
    unit: str = "",
    method: str = "linregress",
) -> dict[str, object]:
    """造一个沙箱返回值里的 ``derived`` 条目。

    ``inputs`` 缺省时是空列表——那正是 Check 要拦的「派生量没有来源」。
    """
    return {
        "statistic": statistic,
        "value": value,
        "n": n,
        "method": method,
        "inputs": list(inputs) if inputs is not None else [],
        "unit": unit,
    }


def metric_row(
    user_id: uuid.UUID,
    zh_name: str,
    value: str,
    *,
    day: date = DEFAULT_DATE,
    unit: str = "",
    en_name: str = "",
    reference_range: str = "",
    position: int = 0,
) -> HealthMetric:
    """造一条尚未入库的 ``health_metrics`` 行。

    这里刻意不做任何归一化：``value`` 一律按数据库里的原样传字符串，测试要看的正是
    ``normalize_metric`` 怎么处理它。
    """
    return HealthMetric(
        user_id=user_id,
        measured_date=day,
        zh_name=zh_name,
        value=value,
        unit=unit,
        en_name=en_name,
        reference_range=reference_range,
        position=position,
    )
