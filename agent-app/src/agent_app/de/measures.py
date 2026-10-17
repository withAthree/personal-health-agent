"""把 ``health_metrics`` 的一行读成「这个值相对参考区间落在哪」。

这里只做算术：报告给了区间，就判断值在区间内、下方还是上方。**不做医学分层**
（「正常高值」「空腹血糖受损」这类结论要有指南依据，由 DE 拿检索结果去说），
也**不内置任何参考区间**——报告没给区间就返回 ``unknown``，把不确定性交回 DE。

``reference_range`` 是随报告进来的文本，形态不统一。实测样本里出现过：
``3.9-6.1`` / ``90-139`` / ``0.108-0.282`` / ``≤26`` / ``9--50`` / ``40-50``。
解析不出来就明确说解析不出来，不猜。

**不比较单位。** 区间和值来自同一行、同一份报告，单位天然一致；工具也不接受
外部传入的数值，所以不存在「mg/dL 的区间比 mmol/L 的值」这种错配。真要放开传入
数值，单位校验必须同时补上。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Literal

from sqlalchemy import or_, select

from agent_app.core.database import async_session_maker
from agent_app.models.health import HealthMetric

Position = Literal["below", "within", "above", "unknown"]
"""一个值相对参考区间的落点。``unknown`` 表示没法判断，原因写在 ``reason`` 里。"""

MAX_SERIES = 3
"""一次查询最多返回几个不同的指标。

用户说「血糖」时可能同时命中「空腹血糖」「餐后血糖」「平均睡眠血糖」。全返回会让
工具输出膨胀，只返回 3 个又要求 DE 自己选，折中：按最近测量时间排序取前 3，并在
渲染时说明还命中了几个。
"""

READINGS_PER_SERIES = 6
"""每个指标最多带回几条历史。1 条最新 + 5 条更早，够 DE 看出方向，不至于淹没它。"""


@dataclass(frozen=True)
class Reference:
    """解析出来的参考区间。

    ``low`` 与 ``high`` 至少有一个非 ``None``；两个都是 ``None`` 的情况由调用方在
    构造前排除。
    """

    low: float | None
    high: float | None


@dataclass(frozen=True)
class Reading:
    """一条测量记录，以及它相对参考区间的落点。"""

    value_text: str
    """原文数值，如 ``"6.3"`` 或 ``"10小时33分钟"``。渲染时原样回显。"""
    value: float | None
    """能转成数字时才有值。"""
    unit: str
    reference_text: str
    measured_on: date
    position: Position
    delta: float | None
    """超出边界多少；``above`` 为正、``below`` 为负，``within`` / ``unknown`` 为空。"""
    reason: str = ""
    """``position`` 为 ``unknown`` 时的原因，直接给 DE 看。"""


@dataclass(frozen=True)
class Series:
    """同一指标的若干条记录，按测量日期倒序，第一条是最新。"""

    name: str
    unit: str
    readings: tuple[Reading, ...]
    total: int
    """库里这个指标一共多少条。``readings`` 只带回其中最近几条。"""


@dataclass(frozen=True)
class Lookup:
    """一次指标查询的完整结果。"""

    query: str
    series: tuple[Series, ...]
    omitted: int
    """命中了但因为 ``MAX_SERIES`` 没返回的指标个数。"""
    available: tuple[str, ...] = ()
    """一条都没命中时，库里现有的指标名，用来帮 DE 换个词重查。"""


# 数值区间：``3.9-6.1`` / ``9--50`` / ``40~50`` / ``0.4～1.7``。
# 分隔符允许重复出现，因为报告里出现过 ``9--50`` 这种排版错误（第二个横线是手误，
# 语义仍是 9 到 50）。数字本身不接受负号——医学参考区间没有负数下界，接受负号
# 反而会让 ``9--50`` 被解析成 ``9`` 到 ``-50``。
_INTERVAL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-~～—－]{1,2}\s*(\d+(?:\.\d+)?)\s*$")

# 单边上限：``<5.2`` / ``≤26`` / ``＜6.8``。
_UPPER = re.compile(r"^\s*[<＜≤]\s*=?\s*(\d+(?:\.\d+)?)\s*$")

# 单边下限：``>60`` / ``≥1.0``。
_LOWER = re.compile(r"^\s*[>＞≥]\s*=?\s*(\d+(?:\.\d+)?)\s*$")


def parse_reference(text: str) -> Reference | None:
    """把参考范围的原文解析成上下界；解析不出来返回 ``None``。

    能处理三种写法：双边区间、单边上限、单边下限。单边符号按闭区间处理——``<5.2``
    与 ``≤5.2`` 在判断「5.2 算不算偏高」上会差一个点，但报告本身也没区分，向下取
    闭区间不会把明确正常的说成异常。
    """
    raw = text.strip()
    if not raw:
        return None

    if match := _INTERVAL.match(raw):
        low = float(match.group(1))
        high = float(match.group(2))
        # 排版颠倒（``6.1-3.9``）时按数值大小摆正，不因为写法异常就放弃判断。
        return Reference(low=min(low, high), high=max(low, high))

    if match := _UPPER.match(raw):
        return Reference(low=None, high=float(match.group(1)))

    if match := _LOWER.match(raw):
        return Reference(low=float(match.group(1)), high=None)

    return None


def parse_value(text: str) -> float | None:
    """把指标值转成数字；转不了返回 ``None``。

    ``value`` 字段允许是 ``10小时33分钟`` 这类文本，转不成数字不是错误，只是没法
    比较。
    """
    try:
        return float(text.strip())
    except ValueError:
        return None


def compare(value: float, reference: Reference) -> tuple[Position, float | None]:
    """判断数值落在区间的哪一侧，并算出超出量。

    Returns:
        ``(position, delta)``。``delta`` 只在越界时有值：``above`` 为正、``below``
        为负，单位与指标本身一致。
    """
    if reference.low is not None and value < reference.low:
        return "below", round(value - reference.low, 4)
    if reference.high is not None and value > reference.high:
        return "above", round(value - reference.high, 4)
    return "within", None


def _to_reading(row: HealthMetric) -> Reading:
    """把一行记录转成 ``Reading``，包括判定。"""
    value = parse_value(row.value)
    reference = parse_reference(row.reference_range)

    if value is None:
        return Reading(
            value_text=row.value,
            value=None,
            unit=row.unit,
            reference_text=row.reference_range,
            measured_on=row.measured_date,
            position="unknown",
            delta=None,
            reason="这个指标的数值不是纯数字，没法直接和区间比较",
        )

    if reference is None:
        if not row.reference_range.strip():
            reason = "报告没有给出这个指标的参考区间"
        else:
            reason = f"报告给的参考区间 {row.reference_range!r} 不是可解析的数值区间"
        return Reading(
            value_text=row.value,
            value=value,
            unit=row.unit,
            reference_text=row.reference_range,
            measured_on=row.measured_date,
            position="unknown",
            delta=None,
            reason=reason,
        )

    position, delta = compare(value, reference)
    return Reading(
        value_text=row.value,
        value=value,
        unit=row.unit,
        reference_text=row.reference_range,
        measured_on=row.measured_date,
        position=position,
        delta=delta,
    )


async def load_series(user_id: uuid.UUID, metric: str) -> Lookup:
    """按指标名查这位用户的记录，每条记录都带上与参考区间的比较结果。

    匹配方式是**包含**，不是相等：用户说「血糖」要能命中「空腹血糖」。中文指标名
    里带修饰词是常态（空腹、餐后、平均、静息），精确匹配会让大部分自然提问查不到。

    Args:
        user_id: 当前用户。
        metric: 指标名关键词，中文名和英文缩写都会匹配。

    Returns:
        查询结果。一个都没命中时 ``series`` 为空，由渲染层给出可读提示。
    """
    keyword = metric.strip()
    stmt = (
        select(HealthMetric)
        .where(
            HealthMetric.user_id == user_id,
            or_(
                HealthMetric.zh_name.contains(keyword, autoescape=True),
                HealthMetric.en_name.contains(keyword, autoescape=True),
            ),
        )
        .order_by(HealthMetric.measured_date.desc(), HealthMetric.position)
    )

    async with async_session_maker() as db:
        rows = list((await db.execute(stmt)).scalars().all())

    if not rows:
        return Lookup(
            query=keyword,
            series=(),
            omitted=0,
            available=await _available_names(user_id),
        )

    grouped: dict[str, list[HealthMetric]] = {}
    for row in rows:
        grouped.setdefault(row.zh_name, []).append(row)

    # 指标之间也按最近测量时间排序：用户更可能问的是最近在看的那一项。
    ordered = sorted(
        grouped.items(),
        key=lambda item: max(row.measured_date for row in item[1]),
        reverse=True,
    )

    series: list[Series] = []
    for name, items in ordered[:MAX_SERIES]:
        readings = tuple(_to_reading(row) for row in items[:READINGS_PER_SERIES])
        series.append(
            Series(
                name=name,
                unit=items[0].unit,
                readings=readings,
                total=len(items),
            )
        )

    return Lookup(
        query=keyword,
        series=tuple(series),
        omitted=max(0, len(ordered) - MAX_SERIES),
    )


async def _available_names(user_id: uuid.UUID) -> tuple[str, ...]:
    """库里这位用户有哪些指标名。只在没命中的时候查，避免正常路径多一次往返。"""
    stmt = (
        select(HealthMetric.zh_name)
        .where(HealthMetric.user_id == user_id)
        .distinct()
        .order_by(HealthMetric.zh_name)
    )
    async with async_session_maker() as db:
        return tuple((await db.execute(stmt)).scalars().all())
