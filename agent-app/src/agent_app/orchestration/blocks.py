"""过程块事件：节点往前端流里发内容的唯一出口。

过程块只服务展示（``core/stream_state.py`` 的 Redis 聚合态，TTL 30 分钟），
**不进账本、不进 checkpoint、不回喂模型**——三条边界见 ``orchestration/state.py``
的模块 docstring。

事件形状与 ``api/v1/chat.py`` 的 SSE 事件同名（``block-start`` / ``block-delta`` /
``block-stop``），节点只负责「发什么内容、属于哪个 agent、第几轮」；块的**身份**
（``block_id`` / ``index``）由 ``chat.py`` 在转发时补上——它才是聚合态的所有者。

**为什么增量事件也带 ``agent``**：并行跑的节点（两个支持 Agent）没法靠「最近打开的
那个块」对号入座。同一个 agent 在同一时刻最多只会有一个打开的块——本图里每个节点
一次只产一个块，同一 agent 的两个块是先后关系（``thinking`` 完了才 ``text``）。
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

logger = structlog.get_logger(__name__)

#: 事件类型，与 ``api/v1/chat.py`` 的 SSE ``type`` 一致。
BLOCK_START = "block-start"
BLOCK_DELTA = "block-delta"
BLOCK_STOP = "block-stop"

#: 块类型，取值见 ``schemas/chat.py`` 的 ``StreamKind``；本图用得到的是这两个。
BLOCK_TEXT = "text"
BLOCK_THINKING = "thinking"

#: 结束状态，取值见 ``schemas/chat.py`` 的 ``StreamBlockStatus``。
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

#: 出错信息进日志前的截断长度（与 ``memory/extract.py`` 同一口径）。
_MAX_ERROR_CHARS = 200


def _writer() -> StreamWriter | None:
    """取当前图执行的流写入器；不在图里时返回 ``None``。

    ``get_stream_writer()`` 在非图上下文（单元测试直接调节点函数）会抛错，而节点
    必须能被单独调用，所以这里吞掉异常：丢的只是过程展示，正文与 state 都不受影响。
    """
    try:
        return get_stream_writer()
    except Exception:
        return None


def _emit(payload: dict[str, Any]) -> None:
    """发一个自定义流事件；失败只记 warning，绝不打断节点。"""
    writer = _writer()
    if writer is None:
        return
    try:
        writer(payload)
    except Exception as exc:
        logger.warning(
            "block_emit_failed",
            type=payload.get("type"),
            error=str(exc)[:_MAX_ERROR_CHARS],
        )


def start_block(kind: str, *, agent: str, step: int) -> None:
    """开始一个过程块。

    Args:
        kind: 块类型，见本模块的 ``BLOCK_*``。
        agent: 产出这个块的节点代号，必须与 ``graph.NODE_AGENT_SOURCE`` 的取值一致。
        step: 轮次，取自 ``state["step"]``。
    """
    _emit(
        {
            "type": BLOCK_START,
            "kind": kind,
            "step": step,
            "meta": {"agent": agent},
        }
    )


def delta_block(text: str, *, agent: str) -> None:
    """给当前的块追加一段增量。"""
    _emit({"type": BLOCK_DELTA, "agent": agent, "delta": text})


def stop_block(*, agent: str, status: str = STATUS_COMPLETE) -> None:
    """结束当前的块。"""
    _emit({"type": BLOCK_STOP, "agent": agent, "status": status})


def emit_text(text: str, *, agent: str, step: int) -> None:
    """把整段正文作为**一个**块发出去（非流式路径）。"""
    start_block(BLOCK_TEXT, agent=agent, step=step)
    delta_block(text, agent=agent)
    stop_block(agent=agent)
