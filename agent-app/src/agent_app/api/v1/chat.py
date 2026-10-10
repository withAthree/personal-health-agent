"""聊天接口：SSE 流式对话。

上下文分两层：

* **账本**：``messages`` 表，全量原文，前端渲染与审计的唯一来源。
* **模型记忆**：LangGraph checkpoint，超阈值时被摘要压缩，只影响发给模型的内容。

因此前端拿到的永远是最新全量对话，而模型看到的是「摘要 + 最近若干条」。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, cast

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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.core.database import async_session_maker
from agent_app.llm.agent import get_agent
from agent_app.llm.call_log import ModelCallLogger
from agent_app.models.chat_session import ChatSession
from agent_app.models.message import Message
from agent_app.schemas.chat import (
    ChatRequest,
    MessageOut,
    SessionMessagesOut,
    SessionSummaryOut,
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
    """从流式分片中提取纯文本内容（忽略 reasoning 等非正文分片）。"""
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
        title=request.message.strip()[:50] or "新对话",
    )
    db.add(session)
    await db.flush()
    return session


@dataclass(frozen=True)
class _PendingTurn:
    """账本里已经写好的本轮用户消息。"""

    session_id: uuid.UUID
    user_message_id: uuid.UUID


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
        )
        await db.commit()
        return pending


async def _store_assistant_message(session_id: uuid.UUID, content: str) -> None:
    """落库助手回复。"""
    async with async_session_maker() as db:
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
            )
        )
        await db.commit()


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

    ``role='system'`` 的历史行不参与模型上下文：系统提示词由 Agent 统一注入，
    这里再喂一条只会重复或互相覆盖。
    """
    converted: list[BaseMessage] = []
    for row in rows:
        if row.id == skip_message_id:
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


async def _sync_compaction_marker(session_id: uuid.UUID) -> None:
    """把「已被摘要覆盖的消息条数」记到会话上，供前端画分隔线。

    覆盖条数 = 账本条数 - checkpoint 里保留的非摘要条数。它只服务展示，
    略有偏差不会影响正确性，因此不加事务保证。
    """
    state = await get_agent().aget_state(_thread_config(session_id))
    messages: list[BaseMessage] = state.values.get("messages", [])
    summarized = sum(1 for message in messages if _is_summary(message))
    if summarized == 0:
        return
    kept = len(messages) - summarized
    async with async_session_maker() as db:
        session = await db.get(ChatSession, session_id)
        if session is None:
            return
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


async def _stream_chat(
    turn: _PendingTurn,
    request: ChatRequest,
) -> AsyncIterator[dict[str, str]]:
    """生成 ``session`` / ``chunk`` / ``done`` / ``error`` 事件。"""
    session_id = turn.session_id
    yield _sse({"type": "session", "session_id": str(session_id)})

    collected: list[str] = []
    try:
        await _drop_unanswered_question(turn.session_id)
        messages = await _initial_messages(turn, request)
        # 挂上回调记录模型实际收到的消息（含摘要那一次调用），用于验收上下文压缩
        config = _thread_config(session_id)
        config["callbacks"] = [ModelCallLogger()]
        # LangGraph 在 ``stream_mode="messages"`` 下实际产出 ``(message, metadata)``
        # 二元组，但类型标注给的是宽松联合，这里按运行时的真实形状收窄。
        stream = cast(
            "AsyncIterator[tuple[BaseMessageChunk, dict[str, Any]]]",
            get_agent().astream(
                {"messages": messages},
                config,
                stream_mode="messages",
            ),
        )
        async for chunk, metadata in stream:
            # 摘要中间件在模型之外也会调用一次 LLM 生成摘要，那些分片同样会
            # 出现在流里；只转发模型节点的输出，否则摘要正文会漏进用户气泡。
            if metadata.get("langgraph_node") != _MODEL_NODE:
                continue
            piece = _chunk_text(chunk)
            if not piece:
                continue
            collected.append(piece)
            yield _sse(
                {
                    "type": "chunk",
                    "content": piece,
                    "agent_source": AGENT_SOURCE,
                }
            )
    except Exception as exc:
        yield _sse({"type": "error", "content": f"模型调用失败：{_short_error(exc)}"})
        return

    try:
        reply = "".join(collected)
        if reply:
            await _store_assistant_message(session_id, reply)
        await _sync_compaction_marker(session_id)
    except Exception as exc:
        yield _sse({"type": "error", "content": f"保存回复失败：{_short_error(exc)}"})
        return

    yield _sse({"type": "done", "session_id": str(session_id)})


@router.post("/chat", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> EventSourceResponse:
    """流式对话接口（SSE）。

    会话归属校验在流开始之前完成：``session_id`` 不存在或不属于当前用户时
    直接返回 404，不会先建好一个流再报错。
    """
    turn = await _store_user_message(request, user_id=auth.user.id)
    return EventSourceResponse(_stream_chat(turn, request))


async def _require_session(session_id: str, *, user_id: uuid.UUID) -> ChatSession:
    """按归属取回会话；不存在或不属于当前用户都返回 404。"""
    async with async_session_maker() as db:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == _parse_session_id(session_id),
                ChatSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise _session_not_found(session_id)
        return session


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
