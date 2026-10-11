"""功能单元 1/3：SSE 对话闭环 + 多轮上下文管理集成测试。

需要 ``docker compose up -d`` 中的 Postgres 与 Redis 可用：测试会真实读写
数据库（账本），但模型与 checkpoint 用替身，不访问网络，也不写 checkpoint 表。

替身结构：

* ``_RecordingChatModel``：假模型，记录每次调用收到的消息，按固定分片返回。
* 每个用例自建一张图（``create_agent`` + ``InMemorySaver``），并把它挂到
  ``chat_module.get_agent``，因此断言的是「模型实际收到了什么消息」。
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field
from sqlalchemy import delete, func, select

import agent_app.api.v1.chat as chat_module
from agent_app.core.database import async_session_maker
from agent_app.llm.client import SYSTEM_PROMPT
from agent_app.llm.context import build_summarization_middleware
from agent_app.models.chat_session import ChatSession
from agent_app.models.message import Message

_FAKE_PIECES = ("Cold is usually ", "viral. ", "Rest well.")
_FAKE_ANSWER = "".join(_FAKE_PIECES)

#: 压缩用例专用阈值：第 3 轮必然触发摘要。
_COMPACT_TRIGGER = ("messages", 4)
_COMPACT_KEEP = ("messages", 2)

#: 非压缩用例的「几乎不会触发」阈值。
_NO_COMPACT_TRIGGER = ("messages", 100)


class _RecordingChatModel(BaseChatModel):
    """假模型：记录收到的消息，按固定分片返回内容。"""

    pieces: tuple[str, ...] = _FAKE_PIECES
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    @property
    def answer(self) -> str:
        return "".join(self.pieces)

    def _record(self, messages: list[BaseMessage]) -> None:
        self.calls.append(list(messages))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架签名
    ) -> ChatResult:
        del stop, kwargs
        self._record(messages)
        if run_manager is not None:
            for piece in self.pieces:
                run_manager.on_llm_new_token(piece)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架签名
    ) -> AsyncIterator[ChatGenerationChunk]:
        del stop, kwargs
        self._record(messages)
        for piece in self.pieces:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager is not None:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


@dataclass
class _Harness:
    """一张假模型驱动的图，以及它记录下来的调用。"""

    model: _RecordingChatModel
    graph: CompiledStateGraph[Any, Any, Any, Any]

    @property
    def last_call(self) -> list[BaseMessage]:
        """传给模型的最后一次调用（去掉系统提示词）。"""
        return [m for m in self.model.calls[-1] if not isinstance(m, SystemMessage)]


def _build_graph(
    model: BaseChatModel,
    *,
    trigger: tuple[str, int] | None,
    keep: tuple[str, int],
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_agent(
        model,
        tools=None,
        system_prompt=SYSTEM_PROMPT,
        middleware=[build_summarization_middleware(model, trigger=trigger, keep=keep)],
        checkpointer=InMemorySaver(),
    )


@pytest.fixture
def build_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., _Harness]:
    """返回一个工厂：造图并把它挂到接口层，供单个用例使用。"""

    def _build(
        *,
        trigger: tuple[str, int] | None = _NO_COMPACT_TRIGGER,
        keep: tuple[str, int] = _COMPACT_KEEP,
        pieces: tuple[str, ...] = _FAKE_PIECES,
    ) -> _Harness:
        model = _RecordingChatModel(pieces=pieces)
        graph = _build_graph(model, trigger=trigger, keep=keep)
        monkeypatch.setattr(chat_module, "get_agent", lambda: graph)
        return _Harness(model=model, graph=graph)

    return _build


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


async def _send(
    client: httpx.AsyncClient,
    *,
    token: str,
    message: str,
    session_id: str | None = None,
    sessions: list[str] | None = None,
    retry: bool = False,
) -> list[dict[str, str]]:
    """发一轮对话并断言成功，返回事件列表（顺手登记会话以便清理）。"""
    payload = {"message": message}
    if session_id is not None:
        payload["session_id"] = session_id
    if retry:
        payload["retry"] = True
    status, events = await _post_chat(client, payload, token=token)
    assert status == 200, events
    if session_id is None and sessions is not None:
        sessions.append(events[0]["session_id"])
    assert [e["type"] for e in events][-1] == "done", events
    return events


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [m.content for m in messages if isinstance(m.content, str)]


def _is_summary(message: BaseMessage) -> bool:
    return message.additional_kwargs.get("lc_source") == "summarization"


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
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    harness = build_agent()
    events = await _send(
        client,
        token=account_token,
        message="How to deal with a cold?",
        sessions=created_sessions,
    )

    kinds = [event["type"] for event in events]
    assert kinds[0] == "session"
    assert kinds[-1] == "done"
    assert kinds.count("error") == 0
    streamed = "".join(e["content"] for e in events if e["type"] == "chunk")
    assert streamed == _FAKE_ANSWER
    assert events[-1]["session_id"] == events[0]["session_id"]
    # 第一轮：模型只看到这一条用户消息
    assert _texts(harness.last_call) == ["How to deal with a cold?"]


async def test_chat_reuses_existing_session(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    build_agent()
    first = await _send(
        client, token=account_token, message="First question", sessions=created_sessions
    )
    session_id = first[0]["session_id"]

    second = await _send(
        client,
        token=account_token,
        message="Second question",
        session_id=session_id,
    )
    assert second[0]["session_id"] == session_id
    assert second[-1]["type"] == "done"


async def test_chat_sends_previous_turns_to_model(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """多轮上下文：第二轮必须带上第一轮的问与答。"""
    harness = build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]

    await _send(client, token=account_token, message="第二问", session_id=session_id)

    messages = harness.last_call
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage]
    assert _texts(messages) == ["第一问", _FAKE_ANSWER, "第二问"]


async def test_chat_compacts_old_turns_into_summary(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """超过阈值后，旧消息被摘要替换，摘要本身不能漏进 SSE 气泡。"""
    harness = build_agent(trigger=_COMPACT_TRIGGER, keep=_COMPACT_KEEP)
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]
    await _send(client, token=account_token, message="第二问", session_id=session_id)

    events = await _send(
        client,
        token=account_token,
        message="第三问",
        session_id=session_id,
    )

    messages = harness.last_call
    assert _is_summary(messages[0]), messages
    assert "第一问" not in "".join(_texts(messages))
    assert _texts(messages)[-1] == "第三问"
    # 摘要生成那次调用带的是历史全文，不是当前问题
    assert "第一问" in "".join(
        m.content for m in harness.model.calls[1] if isinstance(m.content, str)
    )

    # 关键回归点：摘要正文不能作为 chunk 事件漏给用户
    streamed = "".join(e["content"] for e in events if e["type"] == "chunk")
    assert streamed == _FAKE_ANSWER
    assert "summary of the conversation" not in streamed


async def test_compacted_message_count_is_recorded(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """压缩后，账本接口要告诉前端「前几条已被压缩」。"""
    build_agent(trigger=_COMPACT_TRIGGER, keep=_COMPACT_KEEP)
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]
    await _send(client, token=account_token, message="第二问", session_id=session_id)
    await _send(client, token=account_token, message="第三问", session_id=session_id)

    response = await client.get(
        f"/api/v1/chat/{session_id}/messages", headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["messages"]) == 6
    assert body["compacted_message_count"] > 0
    assert body["compacted_message_count"] < 6


async def test_chat_seeds_history_when_checkpoint_is_missing(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """checkpoint 丢了也能接上：新图必须从账本重建历史，且不重复当前问题。"""
    build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]

    # 换一张全新的图（等价于 checkpoint 表被清空 / 老会话）
    fresh = build_agent()
    await _send(client, token=account_token, message="第二问", session_id=session_id)

    messages = fresh.last_call
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage]
    assert _texts(messages) == ["第一问", _FAKE_ANSWER, "第二问"]


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
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """越权使用他人 session_id 必须拿不到内容，也不能往他人会话里写数据。"""
    build_agent()
    alice = await register_account()
    bob = await register_account()

    events = await _send(
        client, token=alice, message="Alice 的问题", sessions=created_sessions
    )
    session_id = events[0]["session_id"]

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

    bob_events = await _send(
        client, token=bob, message="Bob 自己的问题", sessions=created_sessions
    )
    assert bob_events[0]["session_id"] != session_id


async def test_list_messages_returns_full_ledger(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """前端渲染用的是账本：即使模型侧被压缩，全量原文也要在。"""
    build_agent(trigger=_COMPACT_TRIGGER, keep=_COMPACT_KEEP)
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]
    await _send(client, token=account_token, message="第二问", session_id=session_id)
    await _send(client, token=account_token, message="第三问", session_id=session_id)

    response = await client.get(
        f"/api/v1/chat/{session_id}/messages", headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == session_id
    assert body["title"] == "第一问"
    contents = [m["content"] for m in body["messages"]]
    assert contents == [
        "第一问",
        _FAKE_ANSWER,
        "第二问",
        _FAKE_ANSWER,
        "第三问",
        _FAKE_ANSWER,
    ]
    assert [m["role"] for m in body["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert body["messages"][1]["agent_source"] == "orchestrator"


async def test_list_messages_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get(
        f"/api/v1/chat/{uuid.uuid4()}/messages",
    )
    assert response.status_code == 401


async def test_list_messages_hides_other_users_session(
    client: httpx.AsyncClient,
    register_account: Callable[[], Awaitable[str]],
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    build_agent()
    alice = await register_account()
    bob = await register_account()
    events = await _send(
        client, token=alice, message="Alice 的问题", sessions=created_sessions
    )
    session_id = events[0]["session_id"]

    response = await client.get(
        f"/api/v1/chat/{session_id}/messages", headers=_auth(bob)
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


async def test_list_messages_rejects_unknown_session(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.get(
        f"/api/v1/chat/{uuid.uuid4()}/messages", headers=_auth(account_token)
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


async def test_summary_endpoint_returns_none_before_compaction(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """临时调试接口：没压缩时没有摘要。"""
    build_agent()
    events = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = events[0]["session_id"]

    response = await client.get(
        f"/api/v1/chat/{session_id}/summary", headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    assert response.json()["summary"] is None


async def test_summary_endpoint_returns_compaction_summary(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """临时调试接口：压缩后能看到模型实际拿到的摘要素材。"""
    build_agent(trigger=_COMPACT_TRIGGER, keep=_COMPACT_KEEP)
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]
    await _send(client, token=account_token, message="第二问", session_id=session_id)
    await _send(client, token=account_token, message="第三问", session_id=session_id)

    response = await client.get(
        f"/api/v1/chat/{session_id}/summary", headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary is not None
    # 摘要正文由框架内置提示词拼装，这里只验证拿到了内容而不是空壳
    assert "Cold is usually viral." in summary


async def test_summary_endpoint_hides_other_users_session(
    client: httpx.AsyncClient,
    register_account: Callable[[], Awaitable[str]],
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    build_agent()
    alice = await register_account()
    bob = await register_account()
    events = await _send(
        client, token=alice, message="Alice 的问题", sessions=created_sessions
    )
    session_id = events[0]["session_id"]

    response = await client.get(
        f"/api/v1/chat/{session_id}/summary", headers=_auth(bob)
    )
    assert response.status_code == 404


async def test_chat_drops_question_left_unanswered_by_previous_turn(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """上一轮没答上来的提问不再喂给模型，但仍然留在账本（页面上看得到）。"""
    harness = build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]

    # 模拟「上一轮报错 / 用户中途终止」：模型记忆里只多了提问，没有回答
    await harness.graph.aupdate_state(
        {"configurable": {"thread_id": session_id}},
        {"messages": [HumanMessage(content="没答上来的问题")]},
    )

    await _send(client, token=account_token, message="第二问", session_id=session_id)

    assert _texts(harness.last_call) == ["第一问", _FAKE_ANSWER, "第二问"]

    # 账本不受影响：界面上的历史仍是完整的问答对
    response = await client.get(
        f"/api/v1/chat/{session_id}/messages", headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    contents = [item["content"] for item in response.json()["messages"]]
    assert contents == ["第一问", _FAKE_ANSWER, "第二问", _FAKE_ANSWER]


def test_drop_trailing_questions_only_trims_unanswered_tail() -> None:
    """从账本重建上下文时，末尾没有回答的提问要被去掉。"""
    complete: list[BaseMessage] = [
        HumanMessage(content="问1"),
        AIMessage(content="答1"),
    ]
    assert chat_module._drop_trailing_questions(complete) == complete

    with_dangling: list[BaseMessage] = [
        *complete,
        HumanMessage(content="问2"),
        HumanMessage(content="问3"),
    ]
    trimmed = chat_module._drop_trailing_questions(with_dangling)
    assert [message.content for message in trimmed] == ["问1", "答1"]


async def _open_turn(
    client: httpx.AsyncClient,
    *,
    token: str,
    sessions: list[str],
    message: str = "睡眠不好怎么办",
    session_id: str | None = None,
) -> tuple[chat_module._PendingTurn, chat_module.ChatRequest]:
    """按接口的顺序建好本轮：先落库用户消息，再拿到流式生成器。"""
    me = (await client.get("/api/v1/auth/me", headers=_auth(token))).json()
    request = chat_module.ChatRequest(message=message, session_id=session_id)
    turn = await chat_module._store_user_message(request, user_id=uuid.UUID(me["id"]))
    if session_id is None:
        sessions.append(str(turn.session_id))
    return turn, request


async def _ledger(
    client: httpx.AsyncClient,
    *,
    token: str,
    session_id: uuid.UUID,
) -> list[dict[str, Any]]:
    response = await client.get(
        f"/api/v1/chat/{session_id}/messages", headers=_auth(token)
    )
    assert response.status_code == 200, response.text
    rows: list[dict[str, Any]] = response.json()["messages"]
    return rows


async def _read_until_chunks(
    stream: AsyncIterator[dict[str, str]],
    *,
    chunks: int,
) -> list[dict[str, Any]]:
    """读到指定数量的 chunk 事件就停下（模拟用户在这个时刻点了「停止」）。

    ``_stream_chat`` 产出的是 sse-starlette 的 chunk（``{"data": ...}``），
    这里顺手解出事件负载。
    """
    received: list[dict[str, Any]] = []
    async for chunk in stream:
        event = json.loads(chunk["data"])
        received.append(event)
        if sum(1 for item in received if item["type"] == "chunk") == chunks:
            break
    return received


@pytest.mark.parametrize("interrupt", ["close", "cancel"])
async def test_chat_persists_partial_reply_when_user_stops(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
    interrupt: str,
) -> None:
    """用户点「停止」：已经流出去的部分回复要落库并标记为已终止。

    ``close`` 复刻生成器被关闭（Starlette 断流后关闭迭代器），``cancel`` 复刻
    请求任务被取消（anyio 的 cancel scope），两条路都必须落盘。
    """
    build_agent()
    turn, request = await _open_turn(
        client, token=account_token, sessions=created_sessions
    )

    stream = chat_module._stream_chat(turn, request)
    received = await _read_until_chunks(stream, chunks=2)
    partial = "".join(e["content"] for e in received if e["type"] == "chunk")
    assert partial != ""
    assert _FAKE_ANSWER.startswith(partial)

    if interrupt == "close":
        await stream.aclose()
    else:
        with pytest.raises(asyncio.CancelledError):
            await stream.athrow(asyncio.CancelledError())

    rows = await _ledger(client, token=account_token, session_id=turn.session_id)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[1]["content"] == partial
    assert rows[1]["interrupted"] is True
    assert rows[0]["interrupted"] is False


async def test_chat_stores_no_reply_when_stopped_before_first_chunk(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """一个字都没流出去就停止：账本里只留用户提问，不留空的助手消息。"""
    build_agent()
    turn, request = await _open_turn(
        client, token=account_token, sessions=created_sessions
    )

    stream = chat_module._stream_chat(turn, request)
    first = json.loads((await anext(stream))["data"])
    assert first["type"] == "session"
    await stream.aclose()

    rows = await _ledger(client, token=account_token, session_id=turn.session_id)
    assert [row["role"] for row in rows] == ["user"]
    assert rows[0]["interrupted"] is False


async def test_chat_skips_interrupted_turn_when_seeding_from_ledger(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """从账本重建上下文时，被终止的那一轮（半截回复 + 无回答的提问）不进模型记忆。"""
    build_agent()
    turn, request = await _open_turn(
        client, token=account_token, sessions=created_sessions
    )

    stream = chat_module._stream_chat(turn, request)
    await _read_until_chunks(stream, chunks=2)
    await stream.aclose()

    # 换一张全新的图（等价于 checkpoint 被清空 / 老会话），走账本重建路径
    fresh = build_agent()
    await _send(
        client,
        token=account_token,
        message="第二问",
        session_id=str(turn.session_id),
    )

    assert _texts(fresh.last_call) == ["第二问"]


async def test_chat_retry_reuses_last_user_message(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """重试不新增用户消息：作废上一轮的半截回复，用账本里的原文重新回答。

    这里直接复刻「用户点了停止」之后的落盘状态（账本一条已终止的半截回复、
    模型记忆里那条没答完的提问），不依赖假模型被真切断的时机。
    """
    harness = build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]

    turn, _ = await _open_turn(
        client,
        token=account_token,
        sessions=created_sessions,
        message="第二问",
        session_id=session_id,
    )
    await chat_module._store_assistant_message(
        turn.session_id, "Cold is usually ", interrupted=True
    )
    await harness.graph.aupdate_state(
        chat_module._thread_config(turn.session_id),
        {"messages": [HumanMessage(content="第二问")]},
    )

    rows = await _ledger(client, token=account_token, session_id=turn.session_id)
    assert [row["role"] for row in rows] == ["user", "assistant", "user", "assistant"]
    assert rows[3]["interrupted"] is True

    await _send(
        client,
        token=account_token,
        message="第二问",
        session_id=session_id,
        retry=True,
    )

    rows = await _ledger(client, token=account_token, session_id=turn.session_id)
    # 账本没有变长：没有出现第二条「第二问」，半截回复被完整回答取代
    assert [row["role"] for row in rows] == ["user", "assistant", "user", "assistant"]
    assert rows[2]["content"] == "第二问"
    assert rows[3]["content"] == _FAKE_ANSWER
    assert rows[3]["interrupted"] is False
    # 模型看到的是历史 + 被重新提问的原文（悬空提问已被摘掉，不会出现两次）
    assert _texts(harness.last_call) == ["第一问", _FAKE_ANSWER, "第二问"]


async def test_chat_retry_ignores_payload_message(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """一个字都没流出去就停止的轮次：重试用账本里的原文，请求里的 message 被忽略。"""
    build_agent()
    turn, request = await _open_turn(
        client, token=account_token, sessions=created_sessions
    )

    stream = chat_module._stream_chat(turn, request)
    first = json.loads((await anext(stream))["data"])
    assert first["type"] == "session"
    await stream.aclose()

    events = await _send(
        client,
        token=account_token,
        message="这条文本应该被忽略",
        session_id=str(turn.session_id),
        retry=True,
    )
    assert [event["type"] for event in events][-1] == "done"

    rows = await _ledger(client, token=account_token, session_id=turn.session_id)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "睡眠不好怎么办"
    assert rows[1]["content"] == _FAKE_ANSWER
    assert rows[1]["interrupted"] is False


async def test_chat_retry_requires_session(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    """没有 session_id 就没有「最后一条用户消息」可重试。"""
    response = await client.post(
        "/api/v1/chat",
        json={"message": "第一问", "retry": True},
        headers=_auth(account_token),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "retry_requires_session"


async def test_chat_retry_rejects_finished_turn(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """已经答完的轮次不给重试，免得删掉一条完整的回答。"""
    build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )

    response = await client.post(
        "/api/v1/chat",
        json={
            "message": "第一问",
            "session_id": first[0]["session_id"],
            "retry": True,
        },
        headers=_auth(account_token),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "turn_already_answered"


async def test_chat_retry_rejects_session_without_questions(
    client: httpx.AsyncClient,
    account_token: str,
    created_sessions: list[str],
) -> None:
    """会话里一条用户消息都没有时，没有可重试的内容。"""
    me = (await client.get("/api/v1/auth/me", headers=_auth(account_token))).json()
    async with async_session_maker() as db:
        empty = ChatSession(user_id=uuid.UUID(me["id"]), title="空会话")
        db.add(empty)
        await db.flush()
        empty_id = empty.id
        await db.commit()
    created_sessions.append(str(empty_id))

    response = await client.post(
        "/api/v1/chat",
        json={"message": "第一问", "session_id": str(empty_id), "retry": True},
        headers=_auth(account_token),
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "nothing_to_retry"


async def test_chat_retry_drops_partial_reply_written_late(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """断流落库比重试请求还晚时，重试落库前仍要把那半截回复清掉。

    真实环境里服务端要等连接断开才发现用户点了停止，那条「已终止」回复可能比
    重试请求更晚落库，``_retry_turn`` 开头的清理就会扑空。
    """
    build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = first[0]["session_id"]

    turn, _ = await _open_turn(
        client,
        token=account_token,
        sessions=created_sessions,
        message="第二问",
        session_id=session_id,
    )
    # 按接口里的顺序复刻：先 _retry_turn（此时半截回复还没落库），半截回复随后才到
    me = (await client.get("/api/v1/auth/me", headers=_auth(account_token))).json()
    retry_request = chat_module.ChatRequest(
        message="第二问", session_id=session_id, retry=True
    )
    retry_turn, effective = await chat_module._retry_turn(
        retry_request, user_id=uuid.UUID(me["id"])
    )
    await chat_module._store_assistant_message(
        turn.session_id, "半截回复", interrupted=True
    )

    async for _chunk in chat_module._stream_chat(retry_turn, effective):
        pass

    rows = await _ledger(client, token=account_token, session_id=retry_turn.session_id)
    assert [row["role"] for row in rows] == ["user", "assistant", "user", "assistant"]
    assert [row["interrupted"] for row in rows] == [False, False, False, False]
    assert rows[3]["content"] == _FAKE_ANSWER


async def test_chat_pair_drops_stale_interrupted_reply(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """重试已经答完之后才落库的半截回复，直接丢掉。"""
    build_agent()
    first = await _send(
        client, token=account_token, message="第一问", sessions=created_sessions
    )
    session_id = uuid.UUID(first[0]["session_id"])
    before = await _ledger(client, token=account_token, session_id=session_id)

    await chat_module._store_assistant_message(
        session_id, "迟到的半截回复", interrupted=True
    )

    after = await _ledger(client, token=account_token, session_id=session_id)
    assert [row["role"] for row in after] == ["user", "assistant"]
    assert after == before
