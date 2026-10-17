"""聊天接口：SSE 流式对话。

上下文分两层：

* **账本**：``messages`` 表，全量原文，前端渲染与审计的唯一来源。
* **模型记忆**：LangGraph checkpoint，超阈值时被摘要压缩，只影响发给模型的内容。

因此前端拿到的永远是最新全量对话，而模型看到的是「摘要 + 最近若干条」。

流上还会走一路**过程块**（思维链，将来的工具调用）：它只服务展示，不进账本，
本轮期间以聚合态留在 Redis（``core/stream_state.py``），刷新页面可以回放。
事件与 ``kind`` 枚举见 ``document/中间过程可见-设计方案.md``。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.core import stream_state
from agent_app.core.database import async_session_maker
from agent_app.de import collect as de_collect
from agent_app.ds import collect
from agent_app.llm.agent import get_agent
from agent_app.llm.call_log import ModelCallLogger
from agent_app.memory.extract import spawn_memory_extraction
from agent_app.memory.middleware import MemoryContext
from agent_app.models.chat_session import ChatSession
from agent_app.models.message import Message
from agent_app.schemas.chat import (
    SESSION_TITLE_MAX_LEN,
    ChatRequest,
    MessageOut,
    SessionListOut,
    SessionMessagesOut,
    SessionOut,
    SessionSummaryOut,
    SessionUpdateIn,
    StreamStateOut,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])

logger = structlog.get_logger(__name__)

#: 多 Agent 路由上线前的占位来源标识，写入 ``messages.agent_source``。
AGENT_SOURCE = "orchestrator"

#: 错误事件中携带的异常文案上限。
_MAX_ERROR_LEN = 500

#: 产生模型输出的图节点名，只有它的流式分片才转发给前端。
_MODEL_NODE = "model"

#: 摘要中间件写回的消息带的标记（见 LangChain ``summarization.py``）。
_SUMMARY_SOURCE = "summarization"

#: LangGraph 流模式：模型分片走 ``messages``，工具自己发的消息走 ``custom``。
#: 写成 ``Literal`` 是为了 ``astream(stream_mode=[...])`` 能通过类型检查。
_STREAM_MESSAGES: Literal["messages"] = "messages"
_STREAM_CUSTOM: Literal["custom"] = "custom"

#: 工具在自定义通道里用的键（见 ``agent_app/ds/tool.py``）。
_NOTE_KEYS = ("ds_note", "de_note")
"""工具往外推工作说明时用的键。

DS 推 ``ds_note``（分析计划），DE 推 ``de_note``（正在检索什么、正在查哪个指标）。
两个子 Agent 共用同一个 ``custom`` 流，所以这里用元组匹配而不是单个键。
"""

#: 过程块类型，见 ``document/中间过程可见-设计方案.md`` §5.4。
_BLOCK_THINKING = "thinking"
_BLOCK_TEXT = "text"

#: 工具推来的一行工作说明。它与 ``thinking`` 不同：内容不在模型分片里，也不进账本。
_BLOCK_PLAN = "plan"

#: 过程块状态。
_BLOCK_RUNNING = "running"
_BLOCK_COMPLETE = "complete"
_BLOCK_INTERRUPTED = "interrupted"
_BLOCK_ERROR = "error"

#: 整轮生成状态。
_STATUS_RUNNING = "running"
_STATUS_DONE = "done"
_STATUS_ERROR = "error"
_STATUS_INTERRUPTED = "interrupted"

#: provider 用这个非标准字段承载思考过程（见 ``llm/client.py``）。
_REASONING_FIELD = "reasoning_content"


def _sse(payload: dict[str, Any]) -> dict[str, str]:
    """把事件负载编码为 sse-starlette 可消费的 chunk。"""
    return {"data": json.dumps(payload, ensure_ascii=False)}


def _short_error(exc: BaseException) -> str:
    """截断异常文案，避免超长内容灌进 SSE 事件。"""
    text = str(exc) or exc.__class__.__name__
    if len(text) > _MAX_ERROR_LEN:
        return text[:_MAX_ERROR_LEN] + "..."
    return text


def _chunk_text(chunk: BaseMessageChunk) -> str:
    """从流式分片中提取正文文本（思考另由 ``_chunk_reasoning`` 取）。"""
    content = chunk.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def _chunk_reasoning(chunk: BaseMessageChunk) -> str:
    """从流式分片中提取思考内容。

    ``ChatOpenAI`` 默认把 provider 的 ``reasoning_content`` 丢掉；
    ``llm/client.py`` 的 ``ChatOpenAIWithReasoning`` 把它捡回来放进
    ``additional_kwargs``，这里只是把它取出来。
    """
    value = chunk.additional_kwargs.get(_REASONING_FIELD)
    return value if isinstance(value, str) else ""


def _blocks_from(chunk: BaseMessageChunk) -> list[tuple[str, str]]:
    """把模型分片拆成 ``[(kind, delta), ...]``，``kind`` 见 ``_BLOCK_*``。

    同一个分片里可能既有思考又有正文（provider 一般把两者分在不同分片里，
    但不能假定），所以返回列表而不是单个值；正文排在后面，保证「思考块先
    于正文块」这一顺序在同一个分片内也成立。
    """
    deltas: list[tuple[str, str]] = []
    if reasoning := _chunk_reasoning(chunk):
        deltas.append((_BLOCK_THINKING, reasoning))
    if text := _chunk_text(chunk):
        deltas.append((_BLOCK_TEXT, text))
    return deltas


def _note_from(payload: object) -> str | None:
    """从工具推来的自定义负载里取出一行工作说明。

    DS 推 ``{"ds_note": "..."}``（见 ``ds/tool.py``），DE 推 ``{"de_note": "..."}``
    （见 ``de/tool.py``）。认不出来的形状一律忽略：这条通道只服务展示，不能影响生成。
    """
    if not isinstance(payload, dict):
        return None
    for key in _NOTE_KEYS:
        note = payload.get(key)
        if isinstance(note, str) and note:
            return note
    return None


def _open_block(state: stream_state.StreamState, kind: str) -> stream_state.StreamBlock:
    """在聚合态里追加一个新块并返回它。

    ``index`` 是本轮内的绝对槽位，但增量事件只引用 ``block_id``：``reset`` 之后
    槽位会复用，只有身份靠得住。
    """
    block = stream_state.StreamBlock(
        block_id=f"b{len(state.blocks)}",
        index=len(state.blocks),
        step=0,
        kind=kind,
    )
    state.blocks.append(block)
    return block


def _block_start_event(block: stream_state.StreamBlock) -> dict[str, str]:
    """``block-start`` 事件。"""
    return _sse(
        {
            "type": "block-start",
            "block_id": block.block_id,
            "index": block.index,
            "step": block.step,
            "kind": block.kind,
            "meta": block.meta,
        }
    )


def _block_stop_event(block: stream_state.StreamBlock) -> dict[str, str]:
    """``block-stop`` 事件。"""
    return _sse(
        {
            "type": "block-stop",
            "block_id": block.block_id,
            "status": block.status,
        }
    )


def _session_not_found(raw: str) -> HTTPException:
    """会话不存在与不属于当前用户返回同一个 404，避免暴露他人会话是否存在。"""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": f"会话不存在: {raw}", "code": "session_not_found"},
    )


def _parse_session_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise _session_not_found(raw) from exc


def _thread_config(session_id: uuid.UUID) -> RunnableConfig:
    """LangGraph 线程标识直接用会话 ID。

    会话归属在进入图之前已经校验过，所以线程 ID 不会跨用户串用。
    """
    return {"configurable": {"thread_id": str(session_id)}}


async def _resolve_session(
    db: AsyncSession,
    request: ChatRequest,
    *,
    user_id: uuid.UUID,
) -> ChatSession:
    """取回属于该用户的会话；``session_id`` 未提供时新建。"""
    if request.session_id:
        session_id = _parse_session_id(request.session_id)
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise _session_not_found(request.session_id)
        return session

    session = ChatSession(
        user_id=user_id,
        title=request.message.strip()[:SESSION_TITLE_MAX_LEN] or "新对话",
    )
    db.add(session)
    await db.flush()
    return session


@dataclass(frozen=True)
class _PendingTurn:
    """账本里已经写好的本轮用户消息。"""

    session_id: uuid.UUID
    user_message_id: uuid.UUID
    #: 本轮属于谁。记忆中间件要靠它去取该用户的长期记忆，
    #: 且必须在整个流式生成期间可用（那之后接的 HTTP 请求早已返回）。
    user_id: uuid.UUID


async def _store_user_message(
    request: ChatRequest,
    *,
    user_id: uuid.UUID,
) -> _PendingTurn:
    """落库用户消息，返回会话 ID 与该消息 ID。"""
    async with async_session_maker() as db:
        session = await _resolve_session(db, request, user_id=user_id)
        session.updated_at = datetime.now(UTC)
        message = Message(session_id=session.id, role="user", content=request.message)
        db.add(message)
        await db.flush()
        pending = _PendingTurn(
            session_id=session.id,
            user_message_id=message.id,
            user_id=user_id,
        )
        await db.commit()
        return pending


async def _retry_turn(
    request: ChatRequest,
    *,
    user_id: uuid.UUID,
) -> tuple[_PendingTurn, ChatRequest]:
    """重试本会话最后一条用户消息。

    不新增用户消息（账本与界面上都只有那一条提问），先把上一轮留下来的半截回复
    删掉，再用账本里的原文重新问一次。前端只在未完成的轮次上给重试入口，因此
    这里对已经答完的轮次直接报错，不去删一条完整的回答。
    """
    if request.session_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "重试需要带上 session_id",
                "code": "retry_requires_session",
            },
        )
    session = await _require_session(request.session_id, user_id=user_id)
    rows = await _load_ledger(session.id)
    last_user = max(
        (index for index, row in enumerate(rows) if row.role == "user"),
        default=None,
    )
    if last_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": "这个会话还没有可以重试的问题",
                "code": "nothing_to_retry",
            },
        )
    trailing = rows[last_user + 1 :]
    if any(row.role == "assistant" and not row.interrupted for row in trailing):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "这一轮已经答完了，不需要重试",
                "code": "turn_already_answered",
            },
        )

    unfinished = [row.id for row in trailing if row.interrupted]
    if unfinished:
        async with async_session_maker() as db:
            await db.execute(
                delete(Message)
                .where(Message.id.in_(unfinished))
                .execution_options(synchronize_session=False)
            )
            await db.commit()
        logger.info(
            "retry_dropped_partial_reply",
            session_id=str(session.id),
            dropped=len(unfinished),
        )

    question = rows[last_user]
    return (
        _PendingTurn(
            session_id=session.id,
            user_message_id=question.id,
            user_id=user_id,
        ),
        request.model_copy(update={"message": question.content}),
    )


async def _store_assistant_message(
    session_id: uuid.UUID,
    content: str,
    *,
    interrupted: bool = False,
    drop_trailing_partial: bool = False,
    charts: list[dict[str, Any]] | None = None,
) -> None:
    """落库助手回复；被终止时只存已经流出去的那部分并打上标记。

    ``charts`` 是这一轮里 DS 工具算出来的图表（已经翻译成 ECharts option）。它跟正文
    一起落库，因为前端渲染气泡时只读账本，不读 checkpoint。

    重试轮传 ``drop_trailing_partial``：落库前把这条提问后面残留的「已终止」
    回复删掉。被终止的那一轮传 ``interrupted``：若账本末尾已经是回复（说明
    这次重试已经答完），迟到的半截回复直接丢掉。两者都是为了让「重试不新增
    消息」在账本上真的成立：断流落库与重试请求谁先到不确定，两种顺序都得收尾。
    """
    async with async_session_maker() as db:
        if drop_trailing_partial:
            dropped = await _drop_trailing_partial_replies(db, session_id)
            if dropped:
                logger.info(
                    "dropped_trailing_partial_reply",
                    session_id=str(session_id),
                    dropped=dropped,
                )
        elif interrupted and await _ledger_ends_with_reply(db, session_id):
            logger.info("skipped_stale_interrupted_reply", session_id=str(session_id))
            return
        session = await db.get(ChatSession, session_id)
        if session is None:
            msg = f"会话不存在: {session_id}"
            raise ValueError(msg)
        session.updated_at = datetime.now(UTC)
        db.add(
            Message(
                session_id=session_id,
                role="assistant",
                content=content,
                agent_source=AGENT_SOURCE,
                charts=charts or [],
                interrupted=interrupted,
            )
        )
        await db.commit()


async def _recent_rows(db: AsyncSession, session_id: uuid.UUID) -> list[Message]:
    """在给定事务里读会话消息，按时间正序。"""
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(result.scalars().all())


async def _drop_trailing_partial_replies(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> int:
    """删掉最后一条用户消息之后残留的「已终止」回复，返回删除条数。"""
    rows = await _recent_rows(db, session_id)
    last_user = max(
        (index for index, row in enumerate(rows) if row.role == "user"),
        default=None,
    )
    if last_user is None:
        return 0
    stale = [row.id for row in rows[last_user + 1 :] if row.interrupted]
    if not stale:
        return 0
    await db.execute(
        delete(Message)
        .where(Message.id.in_(stale))
        .execution_options(synchronize_session=False)
    )
    return len(stale)


async def _ledger_ends_with_reply(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """账本最后一条是不是助手回复。"""
    rows = await _recent_rows(db, session_id)
    return bool(rows) and rows[-1].role == "assistant"


async def _load_ledger(session_id: uuid.UUID) -> list[Message]:
    """读取账本里该会话的全部消息，按时间正序。"""
    async with async_session_maker() as db:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at, Message.id)
        )
        return list(result.scalars().all())


def _to_model_messages(
    rows: Sequence[Message],
    *,
    skip_message_id: uuid.UUID,
) -> list[BaseMessage]:
    """账本行转模型消息。

    两条不参与模型上下文：

    * ``role='system'`` 的历史行：系统提示词由 Agent 统一注入，这里再喂一条
      只会重复或互相覆盖。
    * 被终止的半截回复（``interrupted``）：它没说完，喂进去会误导模型以为
      这轮已经答完了；被摘掉后该轮只剩下没有回答的提问，由
      ``_drop_trailing_questions`` 一并去掉。
    """
    converted: list[BaseMessage] = []
    for row in rows:
        if row.id == skip_message_id or row.interrupted:
            continue
        if row.role == "user":
            converted.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            converted.append(AIMessage(content=row.content))
    return converted


def _drop_trailing_questions(messages: list[BaseMessage]) -> list[BaseMessage]:
    """去掉末尾没有回答的提问。

    模型报错、超时、用户中途终止（点「新对话」、关页面）都会留下一条没有回答的
    提问；它只该留在账本与界面上，不该饵给模型：留在模型记忆里会打破「摘要 +
    完整问答对 + 1 条新提问」的条数规律，压缩的切点就会把一次问答劈开
    （提问被摘要、回答留在窗口里）。
    """
    end = len(messages)
    while end > 0 and isinstance(messages[end - 1], HumanMessage):
        end -= 1
    return messages if end == len(messages) else messages[:end]


async def _drop_unanswered_question(session_id: uuid.UUID) -> None:
    """把末尾没有回答的提问从 checkpoint（模型记忆）里摘掉。

    账本与界面完全不动，只是模型不再记得这个没答上来的问题。放在每轮开始时
    做，而不是在出错时做：用户主动终止时流会被取消，那时已经无法可靠地写库。
    """
    config = _thread_config(session_id)
    state = await get_agent().aget_state(config)
    messages: list[BaseMessage] = state.values.get("messages", [])
    if not messages:
        return
    dangling = messages[-1]
    if not isinstance(dangling, HumanMessage) or dangling.id is None:
        return
    await get_agent().aupdate_state(
        config, {"messages": [RemoveMessage(id=dangling.id)]}
    )
    logger.info(
        "dropped_unanswered_question",
        session_id=str(session_id),
        chars=len(str(dangling.content)),
    )


async def _initial_messages(
    turn: _PendingTurn,
    request: ChatRequest,
) -> list[BaseMessage]:
    """本轮送进图的消息。

    正常情况下 checkpoint 里已经有历史（含摘要），只补这一条新用户消息；
    只有 checkpoint 为空时（新会话、checkpoint 表被清、本功能上线前的老会话）
    才从账本重建一次，重建时排除刚写入的这条用户消息，避免重复。
    """
    new_message = HumanMessage(content=request.message)
    state = await get_agent().aget_state(_thread_config(turn.session_id))
    if state.values.get("messages"):
        return [new_message]
    rows = await _load_ledger(turn.session_id)
    history = _drop_trailing_questions(
        _to_model_messages(rows, skip_message_id=turn.user_message_id)
    )
    return [*history, new_message]


def _is_summary(message: BaseMessage) -> bool:
    """判断一条消息是不是摘要中间件写回的摘要。"""
    return message.additional_kwargs.get("lc_source") == _SUMMARY_SOURCE


async def _sync_compaction_marker(session_id: uuid.UUID) -> int:
    """把「已被摘要覆盖的消息条数」记到会话上供前端画分隔线，并返回该条数。

    覆盖条数 = 账本条数 - checkpoint 里保留的非摘要条数。它只服务展示，
    略有偏差不会影响正确性，因此不加事务保证（例如被用户终止的那一轮只留在
    账本里、不在 checkpoint 里，会让这个数字略微偏大）。
    """
    state = await get_agent().aget_state(_thread_config(session_id))
    messages: list[BaseMessage] = state.values.get("messages", [])
    summarized = sum(1 for message in messages if _is_summary(message))
    if summarized == 0:
        return 0
    kept = len(messages) - summarized
    async with async_session_maker() as db:
        session = await db.get(ChatSession, session_id)
        if session is None:
            return 0
        total = await db.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.session_id == session_id)
        )
        covered = max((total or 0) - kept, 0)
        logger.info(
            "compaction_marker",
            session_id=str(session_id),
            summarized=summarized,
            kept=kept,
            total=total or 0,
            covered=covered,
        )
        if session.compacted_message_count != covered:
            session.compacted_message_count = covered
            await db.commit()
        return covered


async def _persist_reply(
    session_id: uuid.UUID,
    reply: str,
    *,
    interrupted: bool = False,
    drop_trailing_partial: bool = False,
    charts: list[dict[str, Any]] | None = None,
) -> int:
    """落库助手回复，并同步压缩标记；返回当前「已被摘要覆盖的消息条数」。

    被终止的轮次不碰压缩标记：此时 checkpoint 里的状态是半截的（用户提问还在、
    回复没有），标记会在下一次正常轮次结束时自己算准。
    """
    if reply:
        await _store_assistant_message(
            session_id,
            reply,
            interrupted=interrupted,
            drop_trailing_partial=drop_trailing_partial,
            charts=charts,
        )
    if interrupted:
        return 0
    return await _sync_compaction_marker(session_id)


async def _persist_interrupted(session_id: uuid.UUID, reply: str) -> None:
    """客户端断开后的落库。

    此时响应通道已经没了，异常只能记日志：要么写成功，要么在日志里留下痕迹。
    """
    try:
        await _persist_reply(session_id, reply, interrupted=True)
    except Exception as exc:
        logger.warning(
            "persist_interrupted_failed",
            session_id=str(session_id),
            error=_short_error(exc),
        )


def _close_open_blocks(
    state: stream_state.StreamState,
    block_status: str,
) -> list[stream_state.StreamBlock]:
    """把还处于 ``running`` 的块置为终态，返回被关掉的块。"""
    closed: list[stream_state.StreamBlock] = []
    for block in state.blocks:
        if block.status == _BLOCK_RUNNING:
            block.status = block_status
            closed.append(block)
    return closed


async def _interrupt_cleanup(
    writer: stream_state.StreamStateWriter,
    state: stream_state.StreamState,
    session_id: uuid.UUID,
    reply: str,
) -> None:
    """断流收尾：过程块置为中断，并把半截正文落库。

    两个动作必须在同一个独立任务里完成：调用方此时正处于被取消的状态，直接
    ``await`` 会被取消信号立刻打断（见 ``_stream_chat`` 的 ``CancelledError`` 分支）。
    """
    _close_open_blocks(state, _BLOCK_INTERRUPTED)
    state.status = _STATUS_INTERRUPTED
    state.finished_at = time.time()
    await writer.flush()
    await _persist_interrupted(session_id, reply)


async def _stream_chat(
    turn: _PendingTurn,
    request: ChatRequest,
) -> AsyncIterator[dict[str, str]]:
    """生成过程块事件流。

    事件序列：``session`` → [``reset``] →（``block-start`` → ``block-delta``* →
    ``block-stop``）* → [``compaction``] → ``done``；失败时以 ``error`` 收尾。

    过程块只服务展示：它们不进 ``messages`` 账本，本轮期间以聚合态留在 Redis
    （``core/stream_state.py``）；正文照旧由 ``_persist_reply`` 落库。
    """
    session_id = turn.session_id
    run_id = uuid.uuid4().hex
    started_at = time.perf_counter()
    yield _sse({"type": "session", "session_id": str(session_id)})
    # agent_source 跟着 run-start 走：它描述「这一轮是谁答的」，不是某个块的属性
    yield _sse({"type": "run-start", "run_id": run_id, "agent_source": AGENT_SOURCE})
    if request.retry:
        # 重新生成：前端丢掉 index >= from_index 的过程块，气泡本身保留
        # （``_retry_turn`` 已经把上一轮没答完的回复作废了）。
        yield _sse({"type": "reset", "from_index": 0})

    collected: list[str] = []
    # 模型在对话中途调 ds_analysis 时，图表与调用计数放在这里。tool 进不了
    # LangGraph 的 state，而落库在这个函数的末尾，只能靠一个随调用作用域走的
    # ContextVar。退出时在 finally 里复位，异常与取消路径都不会留残留。
    ds_turn, ds_token = collect.begin()
    # DE 的记账（本轮调用次数、答案复用）同样需要跨 tool 调用存活，理由同上。
    # 工作说明的通道不在这里设：writer 只能在主 agent 的 tool 上下文里取到，
    # 见 ``de/tool.py`` 的 ``_stream_writer``。
    _de_turn, de_token = de_collect.begin()
    charts: list[dict[str, Any]] = []
    state = stream_state.StreamState(run_id=run_id)
    writer = stream_state.StreamStateWriter(session_id, state)
    # 先落一次 running 状态：生成期间刷新页面也能看到「正在生成」。
    await writer.flush()
    current: stream_state.StreamBlock | None = None
    first_delta_at: float | None = None
    try:
        await _drop_unanswered_question(turn.session_id)
        messages = await _initial_messages(turn, request)
        setup_at = time.perf_counter()
        # 挂上回调记录模型实际收到的消息（含摘要那一次调用），用于验收上下文压缩
        config = _thread_config(session_id)
        config["callbacks"] = [ModelCallLogger()]
        # 多个 ``stream_mode`` 时 LangGraph 产出 ``(mode, payload)`` 二元组，两种
        # mode 的 payload 形状不同，所以下面按 mode 分派，不能直接解包。
        stream = cast(
            "AsyncIterator[tuple[str, Any]]",
            get_agent().astream(
                {"messages": messages},
                config,
                # ``custom`` 是工具往外发消息的通道：DS 子 Agent 在算出分析计划后推
                # 一行「正在算什么」。没有它，一次数据分析的 60~80 秒里前端只显示
                # 「正在思考」，用户看不出它在干活。
                stream_mode=[_STREAM_MESSAGES, _STREAM_CUSTOM],
                # 长期记忆按用户注入，身份只能走 context（Agent 是 lru_cache 单例）。
                context=MemoryContext(user_id=turn.user_id),
            ),
        )
        async for mode, payload in stream:
            if mode == _STREAM_CUSTOM:
                note = _note_from(payload)
                if note is not None:
                    # 先收掉正在写的模型块：一行说明是独立的一块，混进 thinking 里
                    # 前端就分不出哪句是谁说的了。
                    if current is not None:
                        current.status = _BLOCK_COMPLETE
                        yield _block_stop_event(current)
                        current = None
                    note_block = _open_block(state, _BLOCK_PLAN)
                    note_block.text = note
                    note_block.status = _BLOCK_COMPLETE
                    yield _block_start_event(note_block)
                    yield _sse(
                        {
                            "type": "block-delta",
                            "block_id": note_block.block_id,
                            "delta": note,
                        }
                    )
                    yield _block_stop_event(note_block)
                    await writer.flush()
                continue
            chunk, metadata = cast("tuple[BaseMessageChunk, dict[str, Any]]", payload)
            # 摘要中间件在模型之外也会调用一次 LLM 生成摘要，那些分片同样会
            # 出现在流里；只转发模型节点的输出，否则摘要正文会漏进用户气泡。
            # 这条过滤与「要不要看中间过程」无关，不能删。
            if metadata.get("langgraph_node") != _MODEL_NODE:
                continue
            for kind, delta in _blocks_from(chunk):
                if current is None or current.kind != kind:
                    # kind 切换（thinking → text）就意味着上一块结束、新块开始。
                    if current is not None:
                        current.status = _BLOCK_COMPLETE
                        yield _block_stop_event(current)
                        await writer.flush()
                    current = _open_block(state, kind)
                    yield _block_start_event(current)
                    await writer.flush()
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()
                    logger.debug(
                        "chat_timing",
                        stage="first_delta",
                        at=round(first_delta_at, 6),
                        setup_ms=round((setup_at - started_at) * 1000, 1),
                        model_ms=round((first_delta_at - setup_at) * 1000, 1),
                    )
                current.text += delta
                if kind == _BLOCK_TEXT:
                    # 只有正文进账本；thinking 永不落库、也永不回喂模型。
                    collected.append(delta)
                yield _sse(
                    {
                        "type": "block-delta",
                        "block_id": current.block_id,
                        "delta": delta,
                    }
                )
                await writer.mark_dirty()
    except (asyncio.CancelledError, GeneratorExit) as exc:
        # 用户点了「停止」或者直接关掉页面：客户端断流会取消这个生成器
        # （LangGraph 的 astream 随之关闭，对上游模型的 HTTP 请求一并取消）。
        # 已经流出去的文字用户看过了，必须落库并标记为「已终止」，
        # 否则刷新页面后这段文字就消失了。
        reply = "".join(collected)
        logger.info(
            "chat_interrupted",
            session_id=str(session_id),
            chunks=len(collected),
            chars=len(reply),
            kind=type(exc).__name__,
        )
        # 独立 Task + shield：写库不会再被这次取消打断，
        # GeneratorExit（生成器被关闭）路径下同样只需要等到它结束。
        persist = asyncio.create_task(
            _interrupt_cleanup(writer, state, session_id, reply)
        )
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(persist)
        raise
    except Exception as exc:
        for block in _close_open_blocks(state, _BLOCK_ERROR):
            yield _block_stop_event(block)
        state.status = _STATUS_ERROR
        state.finished_at = time.time()
        await writer.flush()
        yield _sse({"type": "error", "content": f"模型调用失败：{_short_error(exc)}"})
        return
    finally:
        charts = list(ds_turn.charts)
        collect.finish(ds_token)
        de_collect.finish(de_token)

    if current is not None:
        current.status = _BLOCK_COMPLETE
        yield _block_stop_event(current)
        await writer.flush()

    try:
        covered = await _persist_reply(
            session_id,
            "".join(collected),
            charts=charts,
            drop_trailing_partial=request.retry,
        )
    except Exception as exc:
        state.status = _STATUS_ERROR
        state.finished_at = time.time()
        await writer.flush()
        yield _sse({"type": "error", "content": f"保存回复失败：{_short_error(exc)}"})
        return

    # 回复已经落库，这一轮才算成立，现在才起后台任务去抽取长期记忆。
    # 不 await：抽取要走一次模型调用，等它等于把延迟加在 done 之后。
    # 失败在抽取内部吞掉，不影响这次响应，也不影响下一轮。
    spawn_memory_extraction(
        user_id=turn.user_id,
        session_id=session_id,
        user_message=request.message,
        assistant_reply="".join(collected),
    )

    state.status = _STATUS_DONE
    state.finished_at = time.time()
    if covered:
        state.compaction = {"covered": covered}
    await writer.flush()
    if covered:
        yield _sse({"type": "compaction", "covered": covered})
    yield _sse({"type": "done", "session_id": str(session_id), "charts": charts})


@router.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> EventSourceResponse:
    """流式对话接口（SSE）。

    会话归属校验在流开始之前完成：``session_id`` 不存在或不属于当前用户时
    直接返回 404，不会先建好一个流再报错。``retry=true`` 时不再写入用户消息，
    而是重发该会话最后一条用户消息。
    """
    # TTFT 埋点：这条与 ``_stream_chat`` 里的 ``chat_timing`` 用同一套绝对时钟
    # （``time.perf_counter``），对得上就是 t0 → t1。只在 DEBUG=true 时输出。
    t0 = time.perf_counter()
    if request.retry:
        turn, effective_request = await _retry_turn(request, user_id=auth.user.id)
    else:
        turn = await _store_user_message(request, user_id=auth.user.id)
        effective_request = request
    logger.debug(
        "chat_timing",
        stage="turn_ready",
        at=round(time.perf_counter(), 6),
        ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return EventSourceResponse(_stream_chat(turn, effective_request))


@router.get("/chat/sessions", response_model=SessionListOut)
async def list_chat_sessions(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> SessionListOut:
    """返回当前用户的会话列表：置顶的在前，组内最近更新的在前。

    左侧「对话记录列表」的数据源。不带消息正文，消息仍由
    ``GET /chat/{session_id}/messages`` 拉取。
    """
    async with async_session_maker() as db:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == auth.user.id)
            # 置顶项整组排在前面，组内仍是「最近说过的最前」；``updated_at`` 精度
            # 可能相同（同一秒内新建多条），用 id 兜底保证顺序稳定
            .order_by(
                ChatSession.is_pinned.desc(),
                ChatSession.updated_at.desc(),
                ChatSession.id.desc(),
            )
        )
        sessions = result.scalars().all()
    return SessionListOut(sessions=[SessionOut.model_validate(row) for row in sessions])


async def _require_session_row(
    db: AsyncSession,
    raw: str,
    *,
    user_id: uuid.UUID,
) -> ChatSession:
    """在当前事务里按归属取回会话；不存在或不属于当前用户都返回 404。

    写接口（重命名 / 置顶 / 删除）用这个：拿到的是挂在 ``db`` 上的持久对象，
    可以直接改字段后 commit。只读接口走 ``_require_session``。
    """
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == _parse_session_id(raw),
            ChatSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise _session_not_found(raw)
    return session


async def _require_session(session_id: str, *, user_id: uuid.UUID) -> ChatSession:
    """按归属取回会话；不存在或不属于当前用户都返回 404。"""
    async with async_session_maker() as db:
        return await _require_session_row(db, session_id, user_id=user_id)


async def _drop_session_traces(session_id: uuid.UUID) -> None:
    """清理会话的两处派生数据：LangGraph checkpoint 与 Redis 过程块。

    两者都不进账本、丢了也能重建，所以整体 fail-soft：删会话在账本里已经完成，
    清理失败只记 warning。checkpoint 存的是给模型看的上下文（用户原文的副本），
    所以在删除会话时主动抹掉，而不是等它自己过期。
    """
    checkpointer = get_agent().checkpointer
    # langgraph 把 ``checkpointer`` 的类型写成 ``BaseCheckpointSaver | bool | None``
    # （``True`` 表示用默认内存实现），这里只处理真正的存储实例
    if isinstance(checkpointer, BaseCheckpointSaver):
        try:
            await checkpointer.adelete_thread(str(session_id))
        except Exception as exc:
            logger.warning(
                "session_checkpoint_clear_failed",
                session_id=str(session_id),
                error=str(exc),
            )
    await stream_state.clear(session_id)


@router.patch("/chat/{session_id}", response_model=SessionOut)
async def update_chat_session(
    session_id: str,
    payload: SessionUpdateIn,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> SessionOut:
    """重命名 / 置顶会话。

    只改请求体里显式给出的字段（见 ``SessionUpdateIn``），返回改完之后的会话：
    前端据此就地更新列表，不必再拉一次 ``GET /chat/sessions``。
    """
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "请求体里没有任何要修改的字段", "code": "empty_update"},
        )

    async with async_session_maker() as db:
        session = await _require_session_row(db, session_id, user_id=auth.user.id)
        for field, value in changes.items():
            setattr(session, field, value)
        # ``updated_at`` 由 TimestampMixin 的 onupdate 由数据库侧刷新；它在 flush
        # 之后处于过期状态，异步下访问会直接报 MissingGreenlet，所以显式 refresh
        await db.commit()
        await db.refresh(session)
        return SessionOut.model_validate(session)


@router.delete("/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: str,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> None:
    """删除会话。

    消息随外键级联删除。长期记忆不跟着删：记忆是用户级的事实，来源会话消失后
    ``memory_items.source_session_id`` 置空即可（见 ``models/memory.py``）。
    """
    async with async_session_maker() as db:
        session = await _require_session_row(db, session_id, user_id=auth.user.id)
        target = session.id
        # 直接发 DELETE 而不是 ORM ``db.delete``：``messages.chat_session_id`` 带
        # ondelete=CASCADE，交给数据库一次删干净，不必先把整段对话读进内存
        await db.execute(delete(ChatSession).where(ChatSession.id == target))
        await db.commit()
    await _drop_session_traces(target)


@router.get("/chat/{session_id}/messages", response_model=SessionMessagesOut)
async def list_chat_messages(
    session_id: str,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> SessionMessagesOut:
    """返回会话在账本里的全量消息。

    这是前端渲染历史对话的唯一来源，与模型看到的（可能已被压缩的）上下文无关。
    """
    session = await _require_session(session_id, user_id=auth.user.id)
    rows = await _load_ledger(session.id)
    return SessionMessagesOut(
        session_id=session.id,
        title=session.title,
        compacted_message_count=session.compacted_message_count,
        messages=[MessageOut.model_validate(row) for row in rows],
    )


@router.get("/chat/{session_id}/summary", response_model=SessionSummaryOut)
async def get_chat_summary(
    session_id: str,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> SessionSummaryOut:
    """【临时】返回当前 checkpoint 里给模型看的摘要。

    只为开发期肉眼验证压缩效果，不是正式产品能力。
    TODO(临时): 压缩功能验收完成后删除本接口与 ``SessionSummaryOut``。
    """
    session = await _require_session(session_id, user_id=auth.user.id)
    state = await get_agent().aget_state(_thread_config(session.id))
    messages: list[BaseMessage] = state.values.get("messages", [])
    parts = [
        message.content if isinstance(message.content, str) else str(message.content)
        for message in messages
        if _is_summary(message)
    ]
    return SessionSummaryOut(
        session_id=session.id,
        summary="\n\n".join(parts) or None,
    )


@router.get("/chat/{session_id}/stream-state", response_model=StreamStateOut | None)
async def get_chat_stream_state(
    session_id: str,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> StreamStateOut | None:
    """返回本轮（或最近一轮）过程块的聚合态，供刷新页面后回放。

    过程块只服务展示，不进账本，过期或被清理后返回 ``null``：前端拿不到就
    只渲染账本里的正文，不报错。
    """
    session = await _require_session(session_id, user_id=auth.user.id)
    state = await stream_state.load(session.id)
    if state is None:
        return None
    return StreamStateOut.model_validate(
        {
            "session_id": session.id,
            "run_id": state.run_id,
            "status": state.status,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "blocks": [block.to_dict() for block in state.blocks],
            "compaction": state.compaction,
            "usage": state.usage,
        }
    )
