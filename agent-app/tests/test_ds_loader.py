"""``load_metric_points`` 的集成测试。

这个文件需要 ``docker compose up -d`` 里的 Postgres。它验的是「真的从表里读出来了
什么」，所以不用 mock 会话——把 SQLAlchemy 换掉之后剩下的东西就不值得测了。
"""

from __future__ import annotations

import csv
import io
import uuid
from collections.abc import Awaitable, Callable
from datetime import date

import pytest
from tests.ds_factories import metric_row

from agent_app.core.database import async_session_maker
from agent_app.ds.dataset import build_dataset
from agent_app.ds.loader import load_metric_points
from agent_app.models.health import HealthMetric

Register = Callable[[], Awaitable[str]]

APRIL = date(2026, 4, 1)
MAY = date(2026, 5, 1)
JUNE = date(2026, 6, 1)


@pytest.fixture
async def user_id(account_token: str, created_users: list[uuid.UUID]) -> uuid.UUID:
    """注册一个真实用户。``created_users`` 会在用例结束后连着数据一起删掉。"""
    assert account_token
    return created_users[-1]


async def _insert(rows: list[HealthMetric]) -> None:
    async with async_session_maker() as db:
        db.add_all(rows)
        await db.commit()


async def test_empty_user_yields_nothing(user_id: uuid.UUID) -> None:
    assert await load_metric_points(user_id) == []


async def test_all_rows_come_back_when_no_window_is_given(user_id: uuid.UUID) -> None:
    await _insert(
        [
            metric_row(user_id, "体重", "70", day=APRIL, unit="公斤"),
            metric_row(user_id, "体重", "71", day=JUNE, unit="公斤"),
        ]
    )

    points = await load_metric_points(user_id)

    assert [item.measured_date for item in points] == [APRIL, JUNE]


async def test_window_includes_both_endpoints(user_id: uuid.UUID) -> None:
    """闭区间：窗口两端的当天都算在内。"""
    await _insert(
        [
            metric_row(user_id, "体重", "69", day=date(2026, 3, 31)),
            metric_row(user_id, "体重", "70", day=APRIL),
            metric_row(user_id, "体重", "71", day=MAY),
            metric_row(user_id, "体重", "72", day=date(2026, 5, 2)),
        ]
    )

    points = await load_metric_points(user_id, start=APRIL, end=MAY)

    assert [item.measured_date for item in points] == [APRIL, MAY]


async def test_open_ended_windows(user_id: uuid.UUID) -> None:
    await _insert(
        [
            metric_row(user_id, "体重", "69", day=date(2026, 3, 31)),
            metric_row(user_id, "体重", "70", day=APRIL),
            metric_row(user_id, "体重", "71", day=MAY),
        ]
    )

    after = await load_metric_points(user_id, start=APRIL)
    before = await load_metric_points(user_id, end=APRIL)

    assert [item.measured_date for item in after] == [APRIL, MAY]
    assert [item.measured_date for item in before] == [date(2026, 3, 31), APRIL]


async def test_csv_order_comes_from_the_dataset_not_the_database(
    user_id: uuid.UUID,
) -> None:
    """排序只有一处：``build_dataset``。

    数据库排中文的顺序和 Python 的码点顺序不保证一致，loader 里再排一次就会有两处
    权威，谁也说不清哪一处最终生效。
    """
    await _insert(
        [
            metric_row(user_id, "体重", "70", day=MAY),
            metric_row(user_id, "步数", "8000", day=APRIL),
            metric_row(user_id, "血压", "120/80", day=MAY),
            metric_row(user_id, "体重", "71", day=APRIL),
        ]
    )

    dataset = build_dataset(await load_metric_points(user_id))
    rows = list(csv.reader(io.StringIO(dataset.csv_text)))[1:]

    assert [(row[0], row[1]) for row in rows] == [
        (APRIL.isoformat(), "体重"),
        (APRIL.isoformat(), "步数"),
        (MAY.isoformat(), "体重"),
        (MAY.isoformat(), "血压"),
    ]


async def test_the_same_rows_always_produce_the_same_csv(user_id: uuid.UUID) -> None:
    """重跑一次不能得到不同的字节，否则失败无法复现。"""
    await _insert(
        [
            metric_row(user_id, "体重", "70", day=MAY),
            metric_row(user_id, "步数", "8000", day=APRIL),
            metric_row(user_id, "血压", "120/80", day=MAY),
        ]
    )

    first = build_dataset(await load_metric_points(user_id)).csv_text
    second = build_dataset(await load_metric_points(user_id)).csv_text

    assert first == second
    assert first.splitlines()[0].startswith("measured_date,zh_name,")


async def test_values_are_normalized_on_the_way_out(user_id: uuid.UUID) -> None:
    """出来的是 ``MetricPoint``，不是数据库行：字符串值在这里就拆干净了。"""
    await _insert(
        [
            metric_row(user_id, "睡眠时长", "10小时33分钟", day=MAY),
            metric_row(user_id, "血压", "52-171", day=MAY),
            metric_row(user_id, "体重", "2.69", day=MAY, unit="千克"),
            metric_row(user_id, "起床时间", "2026-05-01 07:20", day=MAY),
        ]
    )

    points = {item.zh_name: item for item in await load_metric_points(user_id)}

    assert points["睡眠时长"].value_num == 633.0
    assert points["睡眠时长"].unit == "分钟"
    assert points["血压"].kind == "interval"
    assert points["血压"].value_num is None
    assert points["血压"].low == 52.0
    assert points["血压"].high == 171.0
    assert points["体重"].value_num == 2.69
    assert points["体重"].unit == "千克"
    assert points["起床时间"].kind == "timestamp"
    assert points["起床时间"].minute_of_day == 440


async def test_source_text_is_kept_for_audit(user_id: uuid.UUID) -> None:
    """转换是有损的（``52-171`` 丢掉了开闭方向），原文必须留着。"""
    await _insert([metric_row(user_id, "血压", "52-171", day=MAY, unit="mmHg")])

    (point,) = await load_metric_points(user_id)

    assert point.source_value == "52-171"


async def test_reference_range_is_parsed(user_id: uuid.UUID) -> None:
    await _insert(
        [
            metric_row(
                user_id,
                "空腹血糖",
                "6.1",
                day=MAY,
                unit="mmol/L",
                reference_range="3.9-6.1",
            )
        ]
    )

    (point,) = await load_metric_points(user_id)

    assert (point.ref_low, point.ref_high) == (3.9, 6.1)


async def test_other_users_rows_are_invisible(
    user_id: uuid.UUID,
    register_account: Register,
    created_users: list[uuid.UUID],
) -> None:
    """查询必须带 ``user_id`` 条件：数据越权比算错更严重。"""
    await register_account()
    intruder = created_users[-1]

    await _insert(
        [
            metric_row(user_id, "体重", "70", day=MAY),
            metric_row(intruder, "体重", "99", day=MAY),
        ]
    )

    mine = await load_metric_points(user_id)
    theirs = await load_metric_points(intruder)

    assert [item.value_num for item in mine] == [70.0]
    assert [item.value_num for item in theirs] == [99.0]
