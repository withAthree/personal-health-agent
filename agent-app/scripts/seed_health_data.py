"""给开发库造一份演示用的健康数据（编出来的，只用于验证取数链路）。

在 ``agent-app`` 目录下运行::

    ./.venv/Scripts/python.exe scripts/seed_health_data.py

做三件事：

1. 找到或创建演示账号 ``13800000001`` / ``test-passw0rd``；
2. 删掉该账号已有的 ``health_metrics`` / ``health_records``；
3. 写入约 5 周的指标序列（步数、睡眠时长、体重、血糖、血压、血脂）与 2 次体检结论。

随机数用固定种子，重跑得到同样的数字。数据刻意编得有话说：血糖整体在降但仍高于
参考范围，体重缓降，步数不规律，睡眠普遍不足，体检结论里带「肝内钙化灶」这类
自由文本。写完用 ``agent_app.health.store`` 读一遍打印出来，顺带当一次自检。
"""

from __future__ import annotations

import asyncio
import random
import sys
import uuid
from datetime import UTC, date, datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_app.core.database import async_session_maker
from agent_app.core.security import hash_password
from agent_app.health.store import list_metric_series, read_metric_series, read_records
from agent_app.models.health import HealthMetric, HealthRecord
from agent_app.models.user import User

#: 演示账号：手机号即账号名，密码固定，便于反复重跑与手工登录。
PHONE = "13800000001"
PASSWORD = "test-passw0rd"
NICKNAME = "演示用户"

#: 北京时间的「今天」：库里的 ``measured_date`` 一律按北京时间取日期。
_CST = timezone(timedelta(hours=8))

#: 造多少天的数据。
_DAYS = 35

#: 固定种子，保证每次生成的数字一致。
_SEED = 20261014


def _today() -> date:
    """北京时间的今天。"""
    return datetime.now(UTC).astimezone(_CST).date()


async def main() -> None:
    """建账号、清旧数据、写新数据、再读一遍打印。"""
    async with async_session_maker() as db:
        user_id = await _ensure_user(db)
        await db.execute(delete(HealthMetric).where(HealthMetric.user_id == user_id))
        await db.execute(delete(HealthRecord).where(HealthRecord.user_id == user_id))
        metrics = _metrics(user_id)
        records = _records(user_id)
        db.add_all([*metrics, *records])
        await db.commit()

    print(f"账号 {PHONE} / {PASSWORD}  user_id={user_id}")
    print(f"写入 health_metrics {len(metrics)} 条、health_records {len(records)} 条")
    await _report(user_id)


async def _ensure_user(db: AsyncSession) -> uuid.UUID:
    """找到演示账号，没有就创建；密码与昵称每次重置，保证文档里的登录信息有效。"""
    user = (
        await db.execute(select(User).where(User.phone == PHONE))
    ).scalar_one_or_none()
    if user is None:
        user = User(
            id=uuid.uuid4(),
            phone=PHONE,
            password_hash=hash_password(PASSWORD),
            nickname=NICKNAME,
        )
        db.add(user)
    else:
        user.password_hash = hash_password(PASSWORD)
        user.nickname = NICKNAME
    await db.flush()
    return user.id


def _metrics(user_id: uuid.UUID) -> list[HealthMetric]:
    """按天生成一份指标序列。"""
    rng = random.Random(_SEED)
    today = _today()
    rows: list[HealthMetric] = []
    position = 0

    def add(
        day_ago: int,
        zh_name: str,
        value: str,
        *,
        en_name: str = "",
        unit: str = "",
        reference_range: str = "",
    ) -> None:
        nonlocal position
        position += 1
        rows.append(
            HealthMetric(
                id=uuid.uuid4(),
                user_id=user_id,
                measured_date=today - timedelta(days=day_ago),
                zh_name=zh_name,
                en_name=en_name,
                value=value,
                unit=unit,
                reference_range=reference_range,
                position=position,
            )
        )

    for index in range(_DAYS):
        day_ago = _DAYS - 1 - index
        steps = max(900, min(19000, int(rng.gauss(7600, 2500))))
        add(day_ago, "步数", str(steps), unit="步")

        minutes = max(240, min(600, int(rng.gauss(385, 55))))
        add(day_ago, "睡眠时长", f"{minutes // 60}小时{minutes % 60:02d}分钟")

        if index % 2 == 0:
            weight = 78.6 - 0.085 * index + rng.gauss(0, 0.25)
            add(day_ago, "体重", f"{weight:.1f}", unit="kg")

        if index % 3 == 0:
            glucose = 7.35 - 0.033 * index + rng.gauss(0, 0.18)
            add(
                day_ago,
                "空腹血糖",
                f"{glucose:.1f}",
                en_name="FPG",
                unit="mmol/L",
                reference_range="3.9-6.1",
            )

        if index % 4 == 0:
            systolic = round(138 - 0.22 * index + rng.gauss(0, 4))
            diastolic = round(88 - 0.16 * index + rng.gauss(0, 3))
            add(
                day_ago,
                "收缩压",
                str(systolic),
                en_name="SBP",
                unit="mmHg",
                reference_range="90-139",
            )
            add(
                day_ago,
                "舒张压",
                str(diastolic),
                en_name="DBP",
                unit="mmHg",
                reference_range="60-89",
            )

    # 血脂只在两次体检里测过，点数少，正好试试「最近的点 + 升序」有没有接对。
    for day_ago, ldl in ((26, "3.62"), (5, "3.21")):
        add(
            day_ago,
            "低密度脂蛋白胆固醇",
            ldl,
            en_name="LDL-C",
            unit="mmol/L",
            reference_range="0-3.4",
        )
    return rows


def _records(user_id: uuid.UUID) -> list[HealthRecord]:
    """两次体检结论：一次年度体检、一次复查。"""
    today = _today()
    first = today - timedelta(days=26)
    second = today - timedelta(days=5)
    return [
        HealthRecord(
            id=uuid.uuid4(),
            user_id=user_id,
            measured_date=first,
            date_raw=f"{first.year}年{first.month}月{first.day}日",
            findings=["肝内钙化灶", "轻度脂肪肝", "甲状腺结节（TI-RADS 3 类）"],
        ),
        HealthRecord(
            id=uuid.uuid4(),
            user_id=user_id,
            measured_date=second,
            date_raw=f"{second.year}年{second.month}月{second.day}日",
            findings=["空腹血糖偏高", "建议内分泌科随诊"],
        ),
    ]


async def _report(user_id: uuid.UUID) -> None:
    """用 ``agent_app.health.store`` 读一遍，确认数据真的能按预期取出来。"""
    print("\n指标目录（最近有数据的在前）：")
    for item in await list_metric_series(user_id):
        print(
            f"  {item.zh_name:<12} {item.points:>3} 天  "
            f"{item.first_date} ~ {item.last_date}  "
            f"{item.unit}  参考 {item.reference_range or '—'}"
        )

    print("\n空腹血糖（全部，升序）：")
    for point in await read_metric_series(user_id, "血糖"):
        print(f"  {point.measured_date}  {point.value} {point.unit}  / {point.zh_name}")

    print("\n流水账（最近的在前）：")
    for record in await read_records(user_id):
        print(f"  {record.measured_date}  {', '.join(record.findings)}")


if __name__ == "__main__":
    if sys.platform == "win32":
        # psycopg 用不了 Windows 默认的 ProactorEventLoop。
        asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(main())
