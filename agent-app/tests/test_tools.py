"""健康数据工具（``agent_app/tools/health.py``）的契约测试。

不需要数据库、Redis 或网络：把 ``health.store`` 的三个读函数换成桩，只校验工具对
模型暴露的参数、渲染结果与来源收集。真库行为由 ``tests/test_health_store.py`` 覆盖。
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pytest
from langchain_core.tools import BaseTool

from agent_app.health.store import MetricPoint, MetricSeries, RecordEntry
from agent_app.tools import health as health_tools

DAY = dt.date(2026, 10, 1)


def _series(zh_name: str, *, points: int = 3) -> MetricSeries:
    return MetricSeries(
        zh_name=zh_name,
        en_name="FPG",
        unit="mmol/L",
        reference_range="3.9-6.1",
        points=points,
        first_date=DAY,
        last_date=DAY,
    )


def _point(
    value: str,
    *,
    day: int = 0,
    zh_name: str = "空腹血糖",
    source_file: str | None = None,
) -> MetricPoint:
    return MetricPoint(
        measured_date=DAY + dt.timedelta(days=day),
        zh_name=zh_name,
        en_name="FPG",
        value=value,
        unit="mmol/L",
        reference_range="3.9-6.1",
        source_file=source_file,
    )


def _record(findings: list[str], *, source_file: str | None = None) -> RecordEntry:
    return RecordEntry(
        measured_date=DAY,
        date_raw="2026年10月1日",
        findings=findings,
        source_file=source_file,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    series: list[MetricSeries] | None = None,
    points: list[MetricPoint] | None = None,
    records: list[RecordEntry] | None = None,
) -> dict[str, list[Any]]:
    """把三个读函数换成桩，返回它们各自收到的调用参数。"""
    seen: dict[str, list[Any]] = {"series": [], "points": [], "records": []}

    async def _list_series(user_id: uuid.UUID) -> list[MetricSeries]:
        seen["series"].append(user_id)
        return list(series or [])

    async def _read_points(
        user_id: uuid.UUID,
        zh_name: str,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        limit: int = 0,
    ) -> list[MetricPoint]:
        seen["points"].append(
            {
                "user_id": user_id,
                "zh_name": zh_name,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        return list(points or [])

    async def _read_records(
        user_id: uuid.UUID,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        limit: int = 0,
    ) -> list[RecordEntry]:
        seen["records"].append(
            {"user_id": user_id, "start": start, "end": end, "limit": limit}
        )
        return list(records or [])

    monkeypatch.setattr(health_tools, "list_metric_series", _list_series)
    monkeypatch.setattr(health_tools, "read_metric_series", _read_points)
    monkeypatch.setattr(health_tools, "read_records", _read_records)
    return seen


def _name(tools: list[BaseTool], name: str) -> BaseTool:
    return next(item for item in tools if item.name == name)


def test_the_three_tools_expose_no_identity_argument() -> None:
    """身份只从闭包进来：工具参数里**不允许**出现 user_id。"""
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    assert [item.name for item in tools] == [
        "list_health_metrics",
        "read_health_metric",
        "read_health_records",
    ]
    for item in tools:
        schema = item.args_schema.model_json_schema()
        assert "user_id" not in schema["properties"]
    metric = _name(tools, "read_health_metric").args_schema.model_json_schema()
    assert metric["required"] == ["zh_name"]
    assert set(metric["properties"]) == {"zh_name", "start_date", "end_date", "limit"}


async def test_list_health_metrics_renders_the_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install(
        monkeypatch, series=[_series("空腹血糖", points=12), _series("体重")]
    )
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    text = await _name(tools, "list_health_metrics").ainvoke({})
    assert "已入库 2 个指标：" in text
    assert (
        "- 空腹血糖（FPG，mmol/L，参考 3.9-6.1）：12 次，2026-10-01 ~ 2026-10-01"
        in text
    )
    assert len(seen["series"]) == 1


async def test_list_health_metrics_says_so_when_there_is_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    assert "还没有入库任何健康指标" in await _name(
        tools, "list_health_metrics"
    ).ainvoke({})


async def test_read_health_metric_passes_the_identity_window_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install(monkeypatch, points=[_point("6.4")])
    user_id = uuid.uuid4()
    tools = health_tools.build_health_tools(user_id, [])
    await _name(tools, "read_health_metric").ainvoke(
        {"zh_name": "血糖", "start_date": "2026-10-01", "end_date": "2026-10-13"}
    )
    call = seen["points"][0]
    assert call["user_id"] == user_id
    assert call["zh_name"] == "血糖"
    assert call["start"] == dt.date(2026, 10, 1)
    assert call["end"] == dt.date(2026, 10, 13)
    assert call["limit"] == health_tools._DEFAULT_POINT_LIMIT


async def test_read_health_metric_clamps_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install(monkeypatch, points=[_point("6.4")])
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    metric = _name(tools, "read_health_metric")
    await metric.ainvoke({"zh_name": "空腹血糖", "limit": 0})
    await metric.ainvoke({"zh_name": "空腹血糖", "limit": 10**6})
    assert [call["limit"] for call in seen["points"]] == [
        1,
        health_tools._MAX_POINT_LIMIT,
    ]


async def test_read_health_metric_groups_two_series_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """包含匹配一次命中好几条序列时，各成一块（``store`` 保证同名相邻、时间升序）。"""
    _install(
        monkeypatch,
        points=[
            _point("6.2", day=0),
            _point("6.4", day=1),
            _point("8.8", day=0, zh_name="餐后2小时血糖"),
            _point("9.1", day=1, zh_name="餐后2小时血糖"),
        ],
    )
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    text = await _name(tools, "read_health_metric").ainvoke({"zh_name": "血糖"})
    assert text.splitlines() == [
        "空腹血糖（FPG，mmol/L，参考 3.9-6.1）：共 2 次，2026-10-01 ~ 2026-10-02",
        "2026-10-01  6.2",
        "2026-10-02  6.4",
        "餐后2小时血糖（FPG，mmol/L，参考 3.9-6.1）：共 2 次，2026-10-01 ~ 2026-10-02",
        "2026-10-01  8.8",
        "2026-10-02  9.1",
    ]


async def test_read_health_metric_collects_sources_in_order_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        points=[
            _point("6.4", source_file="体检报告.pdf"),
            _point("6.2", source_file="体检报告.pdf"),
            _point("6.1"),
            _point("6.0", source_file="门诊记录.pdf"),
        ],
    )
    used: list[str] = []
    tools = health_tools.build_health_tools(uuid.uuid4(), used)
    text = await _name(tools, "read_health_metric").ainvoke({"zh_name": "空腹血糖"})
    assert used == ["体检报告.pdf", "门诊记录.pdf"]
    assert text.endswith("（数据来源：体检报告.pdf、门诊记录.pdf）")


async def test_read_health_metric_offers_the_available_names_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install(monkeypatch, series=[_series("体重"), _series("睡眠时长")])
    used: list[str] = []
    tools = health_tools.build_health_tools(uuid.uuid4(), used)
    text = await _name(tools, "read_health_metric").ainvoke({"zh_name": "尿酸"})
    assert "没有查到「尿酸」的数据" in text
    assert "可用的指标有：体重、睡眠时长" in text
    assert used == []
    assert len(seen["series"]) == 1


async def test_read_health_metric_says_there_is_no_data_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    text = await _name(tools, "read_health_metric").ainvoke({"zh_name": "尿酸"})
    assert "还没有入库任何健康指标" in text


async def test_read_health_records_renders_findings_and_collects_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install(
        monkeypatch,
        records=[
            _record(["空腹血糖偏高", "建议内分泌科随诊"], source_file="体检报告.pdf"),
            _record(["轻度脂肪肝"]),
        ],
    )
    used: list[str] = []
    tools = health_tools.build_health_tools(uuid.uuid4(), used)
    text = await _name(tools, "read_health_records").ainvoke({"limit": 5})
    assert text.splitlines() == [
        "2026年10月1日（来源：体检报告.pdf）",
        "- 空腹血糖偏高",
        "- 建议内分泌科随诊",
        "2026年10月1日",
        "- 轻度脂肪肝",
    ]
    assert used == ["体检报告.pdf"]
    assert seen["records"][0]["limit"] == 5


async def test_read_health_records_without_any_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    tools = health_tools.build_health_tools(uuid.uuid4(), [])
    assert await _name(tools, "read_health_records").ainvoke({}) == "没有查到检查记录。"
