"""健康数据工具：把 ``agent_app/health/store.py`` 的确定性查询包成模型能调的接口。

三个工具对应三类问题：**有哪些数据**（清单）、**某个指标的逐次数值**（序列）、
**报告写了什么结论**（流水账）。每个工具的参数上限都夹在 Python 里——参数是模型
填的，不夹住它就能要求「把这张表全给我」，一下把上下文塞满。

身份不是参数（见 ``agent_app/tools/__init__.py``）；``used`` 是本模块唯一的副作用：
工具每读到一批数据就把来源文件名追加进去，节点据此填 ``Contribution.sources``。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import date

from langchain_core.tools import BaseTool, tool

from agent_app.health.store import (
    MetricPoint,
    MetricSeries,
    RecordEntry,
    list_metric_series,
    read_metric_series,
    read_records,
)

#: 工具参数 ``limit`` 的默认值与上限。
_DEFAULT_POINT_LIMIT = 60
_MAX_POINT_LIMIT = 400
_DEFAULT_RECORD_LIMIT = 20
_MAX_RECORD_LIMIT = 50


def build_health_tools(user_id: uuid.UUID, used: list[str]) -> list[BaseTool]:
    """构造属于 ``user_id`` 的健康数据工具。

    Args:
        user_id: 从图运行的 context 取出的用户身份（见 ``nodes._current_user_id``）。
        used: 来源记录器。工具读到数据时把来源文件名追加进来（去重由调用方负责），
            供 ``Contribution.sources`` 使用——「每个数字都能追到一份文件」靠它。
    """

    @tool
    async def list_health_metrics() -> str:
        """列出这位用户已入库的健康指标：指标名、单位、参考范围、测量次数、时间跨度。

        取数的第一步：``read_health_metric`` 的 ``zh_name`` 必须是这份清单里的名字
        （记不全时也可以用其中一段，会按包含关系匹配）。返回「还没有入库任何健康指标」
        说明这位用户确实没有可用数据，此时不要凭常识或典型值编数据。
        """
        return _render_series(await list_metric_series(user_id))

    @tool
    async def read_health_metric(
        zh_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = _DEFAULT_POINT_LIMIT,
    ) -> str:
        """读取某个指标（如「空腹血糖」「体重」「睡眠时长」）的逐次测量值，按时间从早到晚。

        Args:
            zh_name: 指标名，取自 ``list_health_metrics``；也可只写其中一段做包含匹配。
            start_date: 只看这之后的测量值，``YYYY-MM-DD``；不填不限。
            end_date: 只看这之前的测量值，``YYYY-MM-DD``；不填不限。
            limit: 最多返回多少条，默认 60，取最近的那些。
        """
        points = await read_metric_series(
            user_id,
            zh_name,
            start=start_date,
            end=end_date,
            limit=_clamp(limit, _MAX_POINT_LIMIT),
        )
        if not points:
            return await _render_missing(user_id, zh_name)
        _remember(used, (point.source_file for point in points))
        return _render_points(points)

    @tool
    async def read_health_records(
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = _DEFAULT_RECORD_LIMIT,
    ) -> str:
        """读取体检 / 检查报告的结论条目（如「空腹血糖偏高」「轻度脂肪肝」），从新到旧。

        指标序列（``read_health_metric``）只有数值，报告的**文字结论**在这里。
        两个都取往往才够用：数值看趋势，结论看医生写了什么。

        Args:
            start_date: 只看这之后的报告，``YYYY-MM-DD``；不填不限。
            end_date: 只看这之前的报告，``YYYY-MM-DD``；不填不限。
            limit: 最多返回多少份报告，默认 20，取最近的。
        """
        entries = await read_records(
            user_id,
            start=start_date,
            end=end_date,
            limit=_clamp(limit, _MAX_RECORD_LIMIT),
        )
        _remember(used, (entry.source_file for entry in entries))
        return _render_records(entries)

    return [list_health_metrics, read_health_metric, read_health_records]


def _clamp(limit: int, maximum: int) -> int:
    """把模型给的 ``limit`` 夹进 ``[1, maximum]``。"""
    return max(1, min(limit, maximum))


def _remember(used: list[str], names: Iterable[str | None]) -> None:
    """记录这批数据的来源文件名（按出现顺序去重）。"""
    for name in names:
        if name and name not in used:
            used.append(name)


def _label(zh_name: str, en_name: str, unit: str, reference_range: str) -> str:
    """拼「指标名（英文名，单位，参考范围）」，缺项自动省略。"""
    bits = [bit for bit in (en_name, unit) if bit]
    if reference_range:
        bits.append(f"参考 {reference_range}")
    return f"{zh_name}（{'，'.join(bits)}）" if bits else zh_name


def _render_series(series: Sequence[MetricSeries]) -> str:
    """指标清单。"""
    if not series:
        return "这位用户还没有入库任何健康指标。"
    lines = [f"已入库 {len(series)} 个指标："]
    lines.extend(
        f"- {_label(item.zh_name, item.en_name, item.unit, item.reference_range)}："
        f"{item.points} 次，{item.first_date.isoformat()}"
        f" ~ {item.last_date.isoformat()}"
        for item in series
    )
    return "\n".join(lines)


def _render_points(points: Sequence[MetricPoint]) -> str:
    """逐次测量值。

    按指标名分块：``zh_name`` 用了包含匹配时一次可能命中好几条序列
    （「血糖」→ 空腹血糖 + 餐后 2 小时血糖），混在一起按日期排没人读得懂。
    """
    lines: list[str] = []
    for zh_name, group in _group_by_name(points).items():
        head = group[0]
        lines.append(
            f"{_label(zh_name, head.en_name, head.unit, head.reference_range)}："
            f"共 {len(group)} 次，{group[0].measured_date.isoformat()} ~ "
            f"{group[-1].measured_date.isoformat()}"
        )
        lines.extend(
            f"{point.measured_date.isoformat()}  {point.value}" for point in group
        )
    sources = list(dict.fromkeys(p.source_file for p in points if p.source_file))
    if sources:
        lines.append(f"（数据来源：{'、'.join(sources)}）")
    return "\n".join(lines)


def _group_by_name(points: Sequence[MetricPoint]) -> dict[str, list[MetricPoint]]:
    """按指标名分组；``store`` 返回的顺序已经保证同名相邻。"""
    grouped: dict[str, list[MetricPoint]] = {}
    for point in points:
        grouped.setdefault(point.zh_name, []).append(point)
    return grouped


async def _render_missing(user_id: uuid.UUID, zh_name: str) -> str:
    """没查到数据时的返回：顺手报上可用指标名，省掉模型再问一轮。

    空结果有两种原因——名字不对，或这段时间确实没测。把可用的指标列出来，模型自己
    判断该换名字还是该如实告诉用户「这段时间没有记录」。
    """
    series = await list_metric_series(user_id)
    if not series:
        return f"没有查到「{zh_name}」的数据：这位用户还没有入库任何健康指标。"
    names = "、".join(item.zh_name for item in series)
    return (
        f"没有查到「{zh_name}」的数据。可用的指标有：{names}。"
        "换成这些名字，或去掉时间范围再试。"
    )


def _render_records(entries: Sequence[RecordEntry]) -> str:
    """报告结论条目。"""
    if not entries:
        return "没有查到检查记录。"
    lines: list[str] = []
    for entry in entries:
        label = entry.date_raw or entry.measured_date.isoformat()
        source = f"（来源：{entry.source_file}）" if entry.source_file else ""
        lines.append(f"{label}{source}")
        lines.extend(f"- {finding}" for finding in entry.findings)
    return "\n".join(lines)
