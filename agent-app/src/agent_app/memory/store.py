"""长期记忆的读写。

**每个查询都带 ``user_id``**：记忆是直接注入 system prompt 的，漏掉这个条件
就是跨用户泄漏，而不是「多取了几行」。写操作同样按 ``user_id`` 过滤——
抽取器给出的 ``target_id`` 来自模型输出，不能当可信输入。
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_app.config import settings
from agent_app.core.database import async_session_maker
from agent_app.models.memory import MemoryItem

#: 「必须在场」的类别：天然条数少，且漏掉一次就是一次错答，
#: 因此不受 ``memory_max_per_loose_category`` 限制。
ALWAYS_INCLUDED = ("health_fact", "goal", "obstacle")

#: 读取时先按时间多扫一些行，再在 Python 里做按类别裁剪。
#: 「每个宽松类别各留最近 N 条」没法用一条简单 SQL 表达，而单用户记忆量
#: 本来就在几百行以内，多扫几十行的代价可以忽略。
_SCAN_MULTIPLIER = 4

#: 扫描行数的下限，避免 ``memory_max_injected`` 调小时连最近几条都取不全。
_MIN_SCAN = 200

#: 喂给抽取器的现有记忆清单上限：它只是用来判断「这条是新记忆还是要改正
#: 已有记忆」，条数太多只会把 prompt 撑大。
DIGEST_LIMIT = 200


async def fetch_memories(user_id: uuid.UUID) -> list[MemoryItem]:
    """取该用户要注入模型的记忆（已按上限裁剪，越新越优先）。"""
    if not settings.memory_enabled:
        return []
    async with async_session_maker() as db:
        result = await db.execute(
            select(MemoryItem)
            .where(MemoryItem.user_id == user_id)
            .order_by(MemoryItem.created_at.desc())
            .limit(max(settings.memory_max_injected * _SCAN_MULTIPLIER, _MIN_SCAN))
        )
        rows = list(result.scalars().all())
    return _trim(rows)


def _trim(rows: list[MemoryItem]) -> list[MemoryItem]:
    """按类别与总数上限裁剪 ``rows``（入参已按时间倒序）。"""
    kept: list[MemoryItem] = []
    loose_seen: dict[str, int] = {}
    for row in rows:
        if row.category not in ALWAYS_INCLUDED:
            seen = loose_seen.get(row.category, 0)
            if seen >= settings.memory_max_per_loose_category:
                continue
            loose_seen[row.category] = seen + 1
        kept.append(row)
        if len(kept) >= settings.memory_max_injected:
            break
    return kept


async def fetch_digest(user_id: uuid.UUID) -> list[MemoryItem]:
    """取喂给抽取器的现有记忆清单（越新越优先）。"""
    async with async_session_maker() as db:
        result = await db.execute(
            select(MemoryItem)
            .where(MemoryItem.user_id == user_id)
            .order_by(MemoryItem.created_at.desc())
            .limit(DIGEST_LIMIT)
        )
        return list(result.scalars().all())


async def add_memory(
    *,
    user_id: uuid.UUID,
    category: str,
    content: str,
    confidence: float | None,
    source_session_id: uuid.UUID | None,
) -> MemoryItem | None:
    """新增一条记忆。

    ``(user_id, category, content)`` 完全一样时什么都不做并返回 ``None``：
    抽取器每轮都能看到现有清单，本该输出 ``update``，这里是兜底。
    """
    async with async_session_maker() as db:
        duplicate = (
            await db.execute(
                select(MemoryItem.id).where(
                    MemoryItem.user_id == user_id,
                    MemoryItem.category == category,
                    MemoryItem.content == content,
                )
            )
        ).first()
        if duplicate is not None:
            return None
        item = MemoryItem(
            user_id=user_id,
            category=category,
            content=content,
            confidence=confidence,
            source_session_id=source_session_id,
        )
        db.add(item)
        await db.commit()
        return item


async def update_memory(
    user_id: uuid.UUID,
    memory_id: uuid.UUID,
    *,
    content: str,
    confidence: float | None,
) -> bool:
    """改写一条属于该用户的记忆；不属于他的一律当不存在。"""
    async with async_session_maker() as db:
        item = await _load_owned(db, user_id, memory_id)
        if item is None:
            return False
        item.content = content
        if confidence is not None:
            item.confidence = confidence
        await db.commit()
        return True


async def delete_memory(user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
    """删除一条属于该用户的记忆。"""
    async with async_session_maker() as db:
        item = await _load_owned(db, user_id, memory_id)
        if item is None:
            return False
        await db.delete(item)
        await db.commit()
        return True


async def _load_owned(
    db: AsyncSession,
    user_id: uuid.UUID,
    memory_id: uuid.UUID,
) -> MemoryItem | None:
    """按 (记忆 id, 用户 id) 取一行——所有权校验就在这里。"""
    result = await db.execute(
        select(MemoryItem).where(
            MemoryItem.id == memory_id,
            MemoryItem.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()
