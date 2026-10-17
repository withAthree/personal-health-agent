"""把数据库里的 ``health_metrics`` 读成可分析的点。

DS 的其余部分都不碰数据库：``run_analysis()`` 只接收 ``list[MetricPoint]``，沙箱里跑的
代码更是连不上库。这个模块是唯一把数据库接进来的地方，也是唯一需要异步会话的地方。

窗口是**闭区间**：``start`` 与 ``end`` 当天都包含在内。``health_metrics.measured_date``
粒度到天，所以不存在「当天几点」这种边界问题。
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import select

from agent_app.core.database import async_session_maker
from agent_app.ds.normalize import MetricPoint, normalize_metrics
from agent_app.models.health import HealthMetric


async def load_metric_points(
    user_id: uuid.UUID,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[MetricPoint]:
    """读某用户在 ``[start, end]`` 内的全部指标点。两个端点省略时读全部历史。

    **不按指标名过滤。** 选哪些指标参与分析是 Plan 阶段的职责，在数据进模型之前就
    过滤掉，会让模型看不到它本该考虑的东西（用户问「我最近状态怎么样」，真正相关的
    可能是一个它没想到的名字）。``AnalysisRequest.zh_names`` 是给 Plan 的提示，不是
    查询条件。

    **不排定顺序。** 顺序由 ``build_dataset`` 决定（按 ``(measured_date, zh_name)``
    排）。这里不再排一次：Postgres 排中文用的是数据库的排序规则，和 Python 的码点顺序
    不保证一致（本机 ``en_US.utf8`` 下实测一致，换成按拼音排的排序规则就会分歧），
    两处都排的话，生效的是哪一个取决于调用路径。
    """
    stmt = select(HealthMetric).where(HealthMetric.user_id == user_id)
    if start is not None:
        stmt = stmt.where(HealthMetric.measured_date >= start)
    if end is not None:
        stmt = stmt.where(HealthMetric.measured_date <= end)

    async with async_session_maker() as db:
        result = await db.execute(stmt)
        rows = list(result.scalars().all())

    # ``HealthMetric`` 只在这个函数里出现，出了这里就是纯 ``MetricPoint``。
    return normalize_metrics(rows)
