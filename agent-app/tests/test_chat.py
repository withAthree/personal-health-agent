"""功能单元 1：SSE 对话闭环集成测试。

同时覆盖功能单元 2 引入的鉴权与会话归属校验。
需要 ``docker compose up -d`` 中的 Postgres 可用，测试会真实读写数据库；
LLM 用替身替换，以便在未配置 ``LLM_API_KEY`` 时也能验证流式协议与落库逻辑。
"""

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest
from langchain_core.messages import AIMessageChunk, BaseMessage
from sqlalchemy import delete, func, select

import agent_app.api.v1.chat as chat_module
from agent_app.core.database import async_session_maker
from agent_app.models.chat_session import ChatSession
from agent_app.models.message import Message

_FAKE_PIECES = ("Cold is usually ", "viral. ", "Rest well.")
_FAKE_ANSWER = "".join(_FAKE_PIECES)


class _FakeLLM:
    """替身 LLM：按固定分片流式返回。"""

    async def astream(
        self,
        messages: list[BaseMessage],
    ) -> AsyncIterator[AIMessageChunk]:
        del messages
        for piece in _FAKE_PIECES:
            yield AIMessageChunk(content=piece)


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """把接口层使用的 LLM 替换为替身。"""
    monkeypatch.setattr(chat_module, "get_llm", lambda: _FakeLLM())


@pytest.fixture
async def created_sessions() -> AsyncIterator[list[str]]:
    """记录测试创建的会话，结束后删除（消息随外键级联删除）。"""
    session_ids: list[str] = []
    yield session_ids
    if not session_ids:
        return
    async with async_session_maker() as db:
        await db.execute(
            delete(ChatSession).where(
                ChatSession.id.in_([uuid.UUID(sid) for sid in session_ids])
            )
        )
        await db.commit()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _post_chat(
    client: httpx.AsyncClient,
    payload: dict[str, str],
    *,
    token: str,
) -> tuple[int, list[dict[str, str]]]:
    """发送一次对话请求；状态码为 200 时收集全部 SSE 事件。"""
    events: list[dict[str, str]] = []
    async with client.stream(
        "POST",
        "/api/v1/chat",
        json=payload,
        headers=_auth(token),
    ) as response:
        if response.status_code != 200:
            await response.aread()
            return response.status_code, events
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))
    return 200, events


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_chat_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={"message": "hi"})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "missing_token"


async def test_chat_streams_session_chunks_and_done(
    client: httpx.AsyncClient,
    account_token: str,
    fake_llm: None,
    created_sessions: list[str],
) -> None:
    status, events = await _post_chat(
        client,
        {"message": "How to deal with a cold?"},
        token=account_token,
    )
    assert status == 200
    created_sessions.append(events[0]["session_id"])

    kinds = [event["type"] for event in events]
    assert kinds[0] == "session"
    assert kinds[-1] == "done"
    assert kinds.count("error") == 0
    streamed = "".join(e["content"] for e in events if e["type"] == "chunk")
    assert streamed == _FAKE_ANSWER
    assert events[-1]["session_id"] == events[0]["session_id"]


async def test_chat_reuses_existing_session(
    client: httpx.AsyncClient,
    account_token: str,
    fake_llm: None,
    created_sessions: list[str],
) -> None:
    _, first = await _post_chat(
        client, {"message": "First question"}, token=account_token
    )
    session_id = first[0]["session_id"]
    created_sessions.append(session_id)

    _, second = await _post_chat(
        client,
        {"message": "Second question", "session_id": session_id},
        token=account_token,
    )
    assert second[0]["session_id"] == session_id
    assert second[-1]["type"] == "done"


async def test_chat_rejects_unknown_session(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "hi", "session_id": "00000000-0000-0000-0000-000000000000"},
        headers=_auth(account_token),
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


async def test_chat_rejects_malformed_session_id(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": "hi", "session_id": "not-a-uuid"},
        headers=_auth(account_token),
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


async def test_chat_rejects_blank_message(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"message": ""},
        headers=_auth(account_token),
    )
    assert response.status_code == 422


async def test_chat_isolates_sessions_between_users(
    client: httpx.AsyncClient,
    register_account: Callable[[], Awaitable[str]],
    fake_llm: None,
    created_sessions: list[str],
) -> None:
    """越权使用他人 session_id 必须拿不到内容，也不能往他人会话里写数据。"""
    alice = await register_account()
    bob = await register_account()

    _, events = await _post_chat(client, {"message": "Alice 的问题"}, token=alice)
    session_id = events[0]["session_id"]
    created_sessions.append(session_id)

    response = await client.post(
        "/api/v1/chat",
        json={"message": "Bob 的越权尝试", "session_id": session_id},
        headers=_auth(bob),
    )
    # 对 Bob 而言该会话不存在，返回 404 而不是 403，避免暴露会话是否存在
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"

    async with async_session_maker() as db:
        result = await db.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.session_id == uuid.UUID(session_id))
        )
        # 只有 Alice 的那一轮：1 条 user + 1 条 assistant
        assert result.scalar_one() == 2

    # Bob 自己起的新会话归他自己，消息数同样只算他这一轮
    _, bob_events = await _post_chat(client, {"message": "Bob 自己的问题"}, token=bob)
    bob_session_id = bob_events[0]["session_id"]
    created_sessions.append(bob_session_id)
    assert bob_session_id != session_id
