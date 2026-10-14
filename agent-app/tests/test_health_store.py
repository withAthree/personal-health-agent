"""功能单元 3：健康数据读取的按用户隔离、排序与匹配。

需要 ``docker compose up -d`` 中的 Postgres 可用：用例会真实读写
``health_metrics`` / ``health_records``。不访问网络，也不调模型。
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import date, timedelta

import pytest

from agent_app.core.database import async_session_maker
from agent_app.health.store import (
    list_metric_series,
    read_metric_series,
    read_records,
)
from agent_app.models.file import FileMetadata
from agent_app.models.health import HealthMetric, HealthRecord

#: 所有用例共用的「第 0 天」，日期断言都相对它算，避免依赖运行时间。
_BASE_DAY = date(2026, 3, 2)


@pytest.fixture
async def two_users(
    register_account: Callable[[], Awaitable[str]],
    created_users: list[uuid.UUID],
) -> tuple[uuid.UUID, uuid.UUID]:
    """两个互不相干的账号：健康数据必须按 ``user_id`` 严格隔离。"""
    await register_account()
    await register_account()
    return created_users[0], created_users[1]


def _metric(
    user_id: uuid.UUID,
    zh_name: str,
    value: str,
    *,
    day: int = 0,
    en_name: str = "",
    unit: str = "",
    reference_range: str = "",
    source_file_id: uuid.UUID | None = None,
) -> HealthMetric:
    """构造一条未落库的指标（只供这些用例使用）。"""
    return HealthMetric(
        id=uuid.uuid4(),
        user_id=user_id,
        zh_name=zh_name,
        en_name=en_name,
        value=value,
        unit=unit,
        reference_range=reference_range,
        measured_date=_BASE_DAY + timedelta(days=day),
        source_file_id=source_file_id,
    )


def _record(
    user_id: uuid.UUID,
    findings: list[str],
    *,
    day: int = 0,
    date_raw: str = "",
) -> HealthRecord:
    """构造一条未落库的流水账。"""
    return HealthRecord(
        id=uuid.uuid4(),
        user_id=user_id,
        measured_date=_BASE_DAY + timedelta(days=day),
        date_raw=date_raw,
        findings=findings,
    )


def _file(user_id: uuid.UUID, original_name: str) -> FileMetadata:
    """构造一条未落库的文件元信息（只为给指标一个来源）。"""
    file_id = uuid.uuid4()
    return FileMetadata(
        id=file_id,
        user_id=user_id,
        filename=f"{file_id}.pdf",
        original_name=original_name,
        file_path=f"{user_id}/{file_id}.pdf",
        file_size=1024,
        mime_type="application/pdf",
        extension="pdf",
        status="confirmed",
    )


async def _insert(*rows: HealthMetric | HealthRecord | FileMetadata) -> None:
    """直接往表里插数据（绕过上传与确认流程）。账号删除时随外键级联清理。"""
    async with async_session_maker() as db:
        db.add_all(list(rows))
        await db.commit()


async def test_list_metric_series_groups_by_name_and_puts_the_latest_first(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, _ = two_users
    await _insert(
        _metric(mine, "体重", "78.5", day=0, unit="kg"),
        _metric(mine, "体重", "78.2", day=2, unit="kg"),
        _metric(
            mine,
            "空腹血糖",
            "7.1",
            day=5,
            en_name="FPG",
            unit="mmol/L",
            reference_range="3.9-6.1",
        ),
    )

    series = await list_metric_series(mine)

    assert [item.zh_name for item in series] == ["空腹血糖", "体重"]
    blood = series[0]
    assert (blood.points, blood.first_date, blood.last_date) == (
        1,
        _BASE_DAY + timedelta(days=5),
        _BASE_DAY + timedelta(days=5),
    )
    assert (blood.en_name, blood.unit, blood.reference_range) == (
        "FPG",
        "mmol/L",
        "3.9-6.1",
    )
    weight = series[1]
    assert (weight.points, weight.first_date, weight.last_date) == (
        2,
        _BASE_DAY,
        _BASE_DAY + timedelta(days=2),
    )


async def test_list_metric_series_is_empty_for_a_fresh_user(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    _, theirs = two_users

    assert await list_metric_series(theirs) == []


async def test_read_metric_series_only_sees_the_owner_and_returns_ascending(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, theirs = two_users
    await _insert(
        _metric(mine, "空腹血糖", "6.9", day=2),
        _metric(mine, "空腹血糖", "7.2", day=1),
        _metric(theirs, "空腹血糖", "5.0", day=1),
    )

    points = await read_metric_series(mine, "空腹血糖")

    assert [point.value for point in points] == ["7.2", "6.9"]
    assert [point.measured_date for point in points] == [
        _BASE_DAY + timedelta(days=1),
        _BASE_DAY + timedelta(days=2),
    ]
    their_points = await read_metric_series(theirs, "空腹血糖")
    assert [point.value for point in their_points] == ["5.0"]


async def test_read_metric_series_falls_back_to_partial_match(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, _ = two_users
    await _insert(
        _metric(mine, "空腹血糖", "7.0", day=1),
        _metric(mine, "餐后2小时血糖", "11.2", day=1),
        _metric(mine, "体重", "78.0", day=1),
    )

    points = await read_metric_series(mine, "血糖")

    assert sorted((point.zh_name, point.value) for point in points) == [
        ("空腹血糖", "7.0"),
        ("餐后2小时血糖", "11.2"),
    ]


async def test_read_metric_series_treats_wildcards_as_text(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, _ = two_users
    await _insert(
        _metric(mine, "空腹血糖", "7.0", day=1),
        _metric(mine, "体重", "78.0", day=1),
    )

    assert await read_metric_series(mine, "%") == []
    assert await read_metric_series(mine, "_") == []


async def test_read_metric_series_filters_by_date_range(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, _ = two_users
    await _insert(*[_metric(mine, "体重", f"7{day}.0", day=day) for day in range(4)])

    points = await read_metric_series(
        mine,
        "体重",
        start=_BASE_DAY + timedelta(days=1),
        end=_BASE_DAY + timedelta(days=2),
    )

    assert [point.value for point in points] == ["71.0", "72.0"]


async def test_read_metric_series_limit_keeps_the_newest_points(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, _ = two_users
    await _insert(*[_metric(mine, "体重", f"7{day}.0", day=day) for day in range(5)])

    points = await read_metric_series(mine, "体重", limit=2)

    assert [point.value for point in points] == ["73.0", "74.0"]
    assert [point.measured_date for point in points] == [
        _BASE_DAY + timedelta(days=3),
        _BASE_DAY + timedelta(days=4),
    ]


async def test_read_records_returns_newest_first_and_only_the_owner(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, theirs = two_users
    await _insert(
        _record(mine, ["肝内钙化灶", "轻度脂肪肝"], day=0, date_raw="2026年3月2日"),
        _record(mine, ["甲状腺结节（TI-RADS 3 类）"], day=21, date_raw="2026年3月23日"),
        _record(theirs, ["未见异常"], day=1, date_raw="2026年3月3日"),
    )

    records = await read_records(mine)

    assert [record.findings for record in records] == [
        ["甲状腺结节（TI-RADS 3 类）"],
        ["肝内钙化灶", "轻度脂肪肝"],
    ]
    assert records[0].measured_date == _BASE_DAY + timedelta(days=21)
    assert records[0].date_raw == "2026年3月23日"
    assert [record.findings for record in await read_records(theirs)] == [["未见异常"]]


async def test_source_file_name_is_returned_for_traceability(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    mine, _ = two_users
    report = _file(mine, "2026年体检报告.pdf")
    await _insert(report)
    await _insert(_metric(mine, "空腹血糖", "7.0", day=1, source_file_id=report.id))

    points = await read_metric_series(mine, "空腹血糖")

    assert [point.source_file for point in points] == ["2026年体检报告.pdf"]
    assert points[0].en_name == ""
