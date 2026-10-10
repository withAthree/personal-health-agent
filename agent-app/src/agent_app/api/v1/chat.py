"""聊天接口：SSE 流式对话。"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.core.database import async_session_maker
from agent_app.llm.client import SYSTEM_PROMPT, get_llm
from agent_app.models.chat_session import ChatSession
from agent_app.models.message import Message
from agent_app.schemas.chat import ChatRequest

router = APIRouter(prefix="/api/v1", tags=["chat"])

#: 多 Agent 路由上线前的占位来源标识，写入 ``messages.agent_source``。
AGENT_SOURCE = "orchestrator"

#: 错误事件中携带的异常文案上限。
_MAX_ERROR_LEN = 500


def _sse(payload: dict[str, Any]) -> dict[str, str]:
    """把事件负载编码为 sse-starlette 可消费的 chunk。"""
    return {"data": json.dumps(payload, ensure_ascii=False)}


def _short_error(exc: BaseException) -> str:
    """截断异常文案，避免超长内容灌进 SSE 事件。"""
    text = str(exc) or exc.__class__.__name__
    if len(text) > _MAX_ERROR_LEN:
        return text[:_MAX_ERROR_LEN] + "..."
    return text


def _chunk_text(chunk: AIMessageChunk) -> str:
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


async def _store_user_message(
    request: ChatRequest,
    *,
    user_id: uuid.UUID,
) -> uuid.UUID:
    """落库用户消息，返回会话 ID。"""
    async with async_session_maker() as db:
        session = await _resolve_session(db, request, user_id=user_id)
        session.updated_at = datetime.now(UTC)
        db.add(Message(session_id=session.id, role="user", content=request.message))
        await db.commit()
        return session.id


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


async def _stream_chat(
    session_id: uuid.UUID,
    request: ChatRequest,
) -> AsyncIterator[dict[str, str]]:
    """生成 ``session`` / ``chunk`` / ``done`` / ``error`` 事件。"""
    yield _sse({"type": "session", "session_id": str(session_id)})

    collected: list[str] = []
    try:
        history = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=request.message),
        ]
        async for chunk in get_llm().astream(history):
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
        await _store_assistant_message(session_id, "".join(collected))
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
    session_id = await _store_user_message(request, user_id=auth.user.id)
    return EventSourceResponse(_stream_chat(session_id, request))
