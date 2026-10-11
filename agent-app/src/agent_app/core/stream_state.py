"""本轮对话过程（block）的聚合态，存在 Redis 里。

它只服务展示：把思考 / 正文这些过程块在服务端留一份，用户刷新页面后可以回放。
最终正文仍然只进 ``messages`` 账本（见 ``chat.py`` 顶部说明与
``document/中间过程可见-设计方案.md``）。

**故障策略与 ``core/token_store.py`` 相反，不要「统一风格」。**
``token_store`` 在判定权限，Redis 不可用必须 fail-closed（抛
``TokenStoreUnavailableError`` → 503）；这里只是「刷新后还能看到刚才的过程」这一
增强能力，所以必须 **fail-soft**：所有 Redis 异常都只记 warning 后返回或跳过，
绝不抛给调用方，对话与流式必须照常工作。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_app.core.redis import get_redis

logger = structlog.get_logger(__name__)

#: 一个会话同一时刻只有一轮生成，新的一轮直接覆盖旧值。
_KEY_PREFIX = "chat:stream:"

#: 过程数据的存活时间：超过这个时间再刷新，就只剩正文可看了。
_TTL_SECONDS = 30 * 60

#: delta 在内存里累积，最多每这么久落一次 Redis。块边界与结束时强制落盘。
_FLUSH_INTERVAL_SECONDS = 0.2


@dataclass
class StreamBlock:
    """一个过程块。"""

    block_id: str
    index: int
    step: int
    kind: str
    text: str = ""
    status: str = "running"
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转成可直接 JSON 序列化的结构。"""
        return {
            "block_id": self.block_id,
            "index": self.index,
            "step": self.step,
            "kind": self.kind,
            "text": self.text,
            "status": self.status,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StreamBlock:
        """从 Redis 里读回的结构还原。"""
        return cls(
            block_id=str(raw.get("block_id", "")),
            index=int(raw.get("index", 0)),
            step=int(raw.get("step", 0)),
            kind=str(raw.get("kind", "text")),
            text=str(raw.get("text", "")),
            status=str(raw.get("status", "complete")),
            meta=raw.get("meta"),
        )


@dataclass
class StreamState:
    """一轮生成的完整过程状态。"""

    run_id: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    blocks: list[StreamBlock] = field(default_factory=list)
    compaction: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


def _key(session_id: uuid.UUID) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _to_dict(state: StreamState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "status": state.status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "blocks": [block.to_dict() for block in state.blocks],
        "compaction": state.compaction,
        "usage": state.usage,
    }


def _from_dict(raw: dict[str, Any]) -> StreamState:
    blocks_raw = raw.get("blocks") or []
    finished_at = raw.get("finished_at")
    return StreamState(
        run_id=str(raw.get("run_id", "")),
        status=str(raw.get("status", "done")),
        started_at=float(raw.get("started_at", 0.0)),
        finished_at=float(finished_at) if finished_at is not None else None,
        blocks=[StreamBlock.from_dict(item) for item in blocks_raw],
        compaction=raw.get("compaction"),
        usage=raw.get("usage"),
    )


async def _save(key: str, state: StreamState) -> None:
    """写状态；任何 Redis 异常都吞掉（fail-soft）。"""
    try:
        payload = json.dumps(_to_dict(state), ensure_ascii=False)
        await get_redis().set(key, payload, ex=_TTL_SECONDS)
    except Exception as exc:
        logger.warning("stream_state_save_failed", key=key, error=str(exc))


async def load(session_id: uuid.UUID) -> StreamState | None:
    """读回一轮的过程状态。

    没有记录、记录损坏、Redis 不可用三种情况都返回 ``None``：前端拿不到过程就
    降级为只显示正文，不影响任何正确性。
    """
    key = _key(session_id)
    try:
        raw = await get_redis().get(key)
    except Exception as exc:
        logger.warning("stream_state_load_failed", key=key, error=str(exc))
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("stream_state_corrupt", key=key)
        return None
    return _from_dict(data)


async def clear(session_id: uuid.UUID) -> None:
    """丢弃某会话的过程块（删除会话时调用）。

    与 ``load`` / ``_save`` 一样 fail-soft：Redis 异常只记 warning。删会话本身
    在账本里已经完成，不能被这个纯展示数据的清理拖失败。
    """
    key = _key(session_id)
    try:
        await get_redis().delete(key)
    except Exception as exc:
        logger.warning("stream_state_clear_failed", key=key, error=str(exc))


class StreamStateWriter:
    """按节流把 ``StreamState`` 写进 Redis。

    ``mark_dirty()`` 只在距上次落盘超过 ``_FLUSH_INTERVAL_SECONDS`` 时才写：delta
    是连续到达的，顺手检查即可，不需要后台定时任务。块边界、``reset``、结束必须
    用 ``flush()`` 强制写，否则最后不足 200ms 的文本会丢。
    """

    def __init__(self, session_id: uuid.UUID, state: StreamState) -> None:
        self._key = _key(session_id)
        self.state = state
        self._last_flush = 0.0

    async def flush(self) -> None:
        """立刻写一次（失败只记日志）。"""
        self._last_flush = time.monotonic()
        await _save(self._key, self.state)

    async def mark_dirty(self) -> None:
        """距上次落盘够久才写。"""
        if time.monotonic() - self._last_flush < _FLUSH_INTERVAL_SECONDS:
            return
        await self.flush()
