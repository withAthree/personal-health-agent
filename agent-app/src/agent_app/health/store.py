"""用户已确认的健康数据读取。

与 ``agent_app.memory.store`` 同一条纪律：**每个查询都带 ``user_id``**。这些函数
是 DS / DE 取数的唯一入口，``user_id`` 由调用方从图上下文（``MemoryContext``）
注入，绝不允许来自模型输出或请求参数——漏掉这个条件就是跨用户泄漏，而不是
「多取了几行」。

读的是两张正式表：

``health_metrics``
    同一天同一指标只有一个值（后来者覆盖先前的），适合看趋势。
``health_records``
    只增不改的流水账，记录每次确认入库的日期与结论（``findings``）。

``value`` 一律原样返回字符串：``2.69`` / ``10小时33分钟`` / ``52-171`` 都是库里
合法值，该不该当数字算、怎么算，是模型与分析代码的事，这一层不做数值解析。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select

from agent_app.core.database import async_session_maker
from agent_app.models.file import FileMetadata
from agent_app.models.health import HealthMetric, HealthRecord

#: 单个指标一次最多返回多少个点（够看一年多的每日数据）。
DEFAULT_METRIC_LIMIT = 400

#: 流水账一次最多返回多少条。
DEFAULT_RECORD_LIMIT = 50


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """一个指标（``zh_name`` 唯一）的概览。"""

    zh_name: str
    en_name: str
    unit: str
    reference_range: str
    points: int
    first_date: date
    last_date: date


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """某一天某个指标的值。"""

    measured_date: date
    zh_name: str
    en_name: str
    value: str
    unit: str
    reference_range: str
    #: 这个值是哪次上传带来的（原始文件名）。用于回答「数字从哪来」。
    source_file: str | None


@dataclass(frozen=True, slots=True)
class RecordEntry:
    """一次确认入库留下的记录结论。"""

    measured_date: date
    date_raw: str
    findings: list[str]
    source_file: str | None


async def list_metric_series(user_id: uuid.UUID) -> list[MetricSeries]:
    """该用户有哪些指标、各有多少天数据、覆盖哪段时间。

    指标名的写法难以预料（问「血糖」时库里可能是「空腹血糖」或「餐后2小时血糖」），
    取数之前先看这张目录，比猜名字可靠。按「最新数据日期」倒序，最近的在前。
    """
    async with async_session_maker() as db:
        result = await db.execute(
            select(
                HealthMetric.zh_name,
                func.max(HealthMetric.en_name),
                func.max(HealthMetric.unit),
                func.max(HealthMetric.reference_range),
                func.count(),
                func.min(HealthMetric.measured_date),
                func.max(HealthMetric.measured_date),
            )
            .where(HealthMetric.user_id == user_id)
            .group_by(HealthMetric.zh_name)
            .order_by(func.max(HealthMetric.measured_date).desc(), HealthMetric.zh_name)
        )
    return [
        MetricSeries(
            zh_name=zh_name,
            en_name=en_name or "",
            unit=unit or "",
            reference_range=reference_range or "",
            points=points,
            first_date=first_date,
            last_date=last_date,
        )
        for (
            zh_name,
            en_name,
            unit,
            reference_range,
            points,
            first_date,
            last_date,
        ) in result
    ]


async def read_metric_series(
    user_id: uuid.UUID,
    zh_name: str,
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int = DEFAULT_METRIC_LIMIT,
) -> list[MetricPoint]:
    """某个指标在 ``[start, end]`` 内的值，按日期升序（同一天只有一个值）。

    ``zh_name`` 先精确匹配；一条都没有时退化为包含匹配（「血糖」能捞到「空腹血糖」），
    每个点都带自己的 ``zh_name``，调用方据此区分命中的是哪几条序列。
    """
    points = await _fetch_metric_points(user_id, zh_name, start, end, limit, exact=True)
    if points:
        return points
    return await _fetch_metric_points(user_id, zh_name, start, end, limit, exact=False)


async def read_records(
    user_id: uuid.UUID,
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int = DEFAULT_RECORD_LIMIT,
) -> list[RecordEntry]:
    """流水账：每次确认入库的日期与结论，按测量日期倒序（最近的在前）。

    ``findings`` 是自由文本结论（「肝内钙化灶」），保留原文，不做归类。
    """
    conditions = [HealthRecord.user_id == user_id]
    if start is not None:
        conditions.append(HealthRecord.measured_date >= start)
    if end is not None:
        conditions.append(HealthRecord.measured_date <= end)

    async with async_session_maker() as db:
        result = await db.execute(
            select(
                HealthRecord.measured_date,
                HealthRecord.date_raw,
                HealthRecord.findings,
                FileMetadata.original_name,
            )
            .outerjoin(FileMetadata, HealthRecord.file_id == FileMetadata.id)
            .where(*conditions)
            .order_by(HealthRecord.measured_date.desc())
            .limit(limit)
        )
    return [
        RecordEntry(
            measured_date=measured_date,
            date_raw=date_raw,
            findings=list(findings or []),
            source_file=source_file,
        )
        for measured_date, date_raw, findings, source_file in result
    ]


async def _fetch_metric_points(
    user_id: uuid.UUID,
    zh_name: str,
    start: date | None,
    end: date | None,
    limit: int,
    *,
    exact: bool,
) -> list[MetricPoint]:
    """按名字取点，返回按日期升序的序列（可能命中多条序列）。"""
    conditions = [HealthMetric.user_id == user_id]
    if exact:
        conditions.append(HealthMetric.zh_name == zh_name)
    else:
        conditions.append(
            HealthMetric.zh_name.ilike(_like_pattern(zh_name), escape="\\")
        )
    if start is not None:
        conditions.append(HealthMetric.measured_date >= start)
    if end is not None:
        conditions.append(HealthMetric.measured_date <= end)

    async with async_session_maker() as db:
        result = await db.execute(
            select(
                HealthMetric.zh_name,
                HealthMetric.en_name,
                HealthMetric.measured_date,
                HealthMetric.value,
                HealthMetric.unit,
                HealthMetric.reference_range,
                FileMetadata.original_name,
            )
            .outerjoin(FileMetadata, HealthMetric.source_file_id == FileMetadata.id)
            .where(*conditions)
            # 先按「每个序列最近的点」取，再把每个序列翻回升序：
            # 直接升序加 limit 会拿到最老的几条，趋势就断了。
            .order_by(HealthMetric.zh_name, HealthMetric.measured_date.desc())
            .limit(limit)
        )

    grouped: dict[str, list[MetricPoint]] = {}
    for row in result:
        grouped.setdefault(row.zh_name, []).append(
            MetricPoint(
                measured_date=row.measured_date,
                zh_name=row.zh_name,
                en_name=row.en_name or "",
                value=row.value,
                unit=row.unit or "",
                reference_range=row.reference_range or "",
                source_file=row.original_name,
            )
        )
    return [point for points in grouped.values() for point in reversed(points)]


def _like_pattern(name: str) -> str:
    """把指标名转成安全的包含匹配模式（``%`` / ``_`` / ``\\`` 按字面量处理）。"""
    escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
