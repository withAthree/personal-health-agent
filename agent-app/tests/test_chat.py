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
from agent_app.memory.middleware import MemoryContext
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
    """假模型：记录收到的消息，按固定分片返回内容。

    ``reasoning_pieces`` 复刻 ``ChatOpenAIWithReasoning`` 的行为：思考分片的
    ``content`` 是空的，思考正文放在 ``additional_kwargs["reasoning_content"]`` 里。
    """

    pieces: tuple[str, ...] = _FAKE_PIECES
    reasoning_pieces: tuple[str, ...] = ()
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    @property
    def answer(self) -> str:
        return "".join(self.pieces)

    def _reasoning_kwargs(self) -> dict[str, Any]:
        if not self.reasoning_pieces:
            return {}
        return {"reasoning_content": "".join(self.reasoning_pieces)}

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
            for piece in self.reasoning_pieces:
                run_manager.on_llm_new_token(piece)
            for piece in self.pieces:
                run_manager.on_llm_new_token(piece)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self.answer,
                        additional_kwargs=self._reasoning_kwargs(),
                    )
                )
            ]
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
        for piece in self.reasoning_pieces:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_content": piece},
                )
            )
            if run_manager is not None:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
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
        # 与 ``llm/agent.py`` 的 ``get_agent()`` 保持一致：接口层调用时会传
        # ``context=MemoryContext(...)``，图没有声明 context_schema 就接不住。
        # 这里不装记忆中间件：它会真去查库，而本文件的用例只关心消息序列。
        context_schema=MemoryContext,
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
        reasoning: tuple[str, ...] = (),
    ) -> _Harness:
        model = _RecordingChatModel(pieces=pieces, reasoning_pieces=reasoning)
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


async def _session_order(
    client: httpx.AsyncClient,
    token: str,
    ids: set[str],
) -> list[str]:
    """从 ``GET /chat/sessions`` 里按后端返回的顺序取出这几个会话。"""
    response = await client.get("/api/v1/chat/sessions", headers=_auth(token))
    assert response.status_code == 200, response.text
    return [s["id"] for s in response.json()["sessions"] if s["id"] in ids]


async def _post_chat(
    client: httpx.AsyncClient,
    payload: dict[str, str],
    *,
    token: str,
) -> tuple[int, list[dict[str, Any]]]:
    """发送一次对话请求；状态码为 200 时收集全部 SSE 事件。"""
    events: list[dict[str, Any]] = []
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
) -> list[dict[str, Any]]:
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


def _block_kinds(events: list[dict[str, Any]]) -> dict[str, str]:
    """从事件流里还原 ``block_id`` → ``kind``。"""
    return {
        str(event["block_id"]): str(event["kind"])
        for event in events
        if event["type"] == "block-start"
    }


def _text_of(events: list[dict[str, Any]], kind: str) -> str:
    """取某个 ``kind`` 的块拼起来的文本。"""
    kinds = _block_kinds(events)
    return "".join(
        str(event["delta"])
        for event in events
        if event["type"] == "block-delta"
        and kinds.get(str(event["block_id"])) == kind
    )


def _streamed_text(events: list[dict[str, Any]]) -> str:
    """拼出助手正文。

    只有 ``text`` 块的内容会进账本，``thinking`` 块只服务展示，不能算进正文。
    """
    return _text_of(events, "text")


def _deltas(events: list[dict[str, Any]]) -> list[str]:
    """全部块的增量分片（含 thinking），用于断言「流了几个分片」。"""
    return [str(event["delta"]) for event in events if event["type"] == "block-delta"]


def _block_statuses(events: list[dict[str, Any]]) -> list[str]:
    """按出现顺序取 ``block-stop`` 的状态。"""
    return [
        str(event["status"]) for event in events if event["type"] == "block-stop"
    ]


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
    streamed = _streamed_text(events)
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

    # 关键回归点：摘要正文不能作为过程块漏给用户
    streamed = _streamed_text(events)
    assert streamed == _FAKE_ANSWER
    assert "summary of the conversation" not in streamed
    assert not any(
        event["type"] == "block-delta"
        and "summary of the conversation" in event["delta"]
        for event in events
    )


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


async def test_list_sessions_returns_recently_updated_first(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """侧边栏「对话记录」：标题取首条提问，最近更新的排在最前。"""
    build_agent()
    older = (
        await _send(
            client, token=account_token, message="第一问", sessions=created_sessions
        )
    )[0]["session_id"]
    newer = (
        await _send(
            client, token=account_token, message="第二问", sessions=created_sessions
        )
    )[0]["session_id"]
    # 回到第一个会话再问一轮，它就成了最近更新的那个
    await _send(client, token=account_token, message="补问", session_id=older)

    response = await client.get("/api/v1/chat/sessions", headers=_auth(account_token))
    assert response.status_code == 200, response.text
    ours = [s for s in response.json()["sessions"] if s["id"] in {older, newer}]
    assert [s["id"] for s in ours] == [older, newer]
    assert [s["title"] for s in ours] == ["第一问", "第二问"]
    assert ours[0]["created_at"] <= ours[0]["updated_at"]


async def test_list_sessions_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/chat/sessions")
    assert response.status_code == 401


async def test_list_sessions_hides_other_users_sessions(
    client: httpx.AsyncClient,
    register_account: Callable[[], Awaitable[str]],
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    build_agent()
    alice = await register_account()
    bob = await register_account()
    await _send(client, token=alice, message="Alice 的问题", sessions=created_sessions)

    response = await client.get("/api/v1/chat/sessions", headers=_auth(bob))
    assert response.status_code == 200, response.text
    assert response.json()["sessions"] == []


async def test_rename_session_changes_title(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """重命名：返回改完的会话，列表里跟着变，前后空白由服务端剪掉。"""
    build_agent()
    session_id = (
        await _send(
            client,
            token=account_token,
            message="感冒了怎么办",
            sessions=created_sessions,
        )
    )[0]["session_id"]

    response = await client.patch(
        f"/api/v1/chat/{session_id}",
        json={"title": "  感冒护理  "},
        headers=_auth(account_token),
    )
    assert response.status_code == 200, response.text
    assert response.json()["title"] == "感冒护理"

    listing = await client.get("/api/v1/chat/sessions", headers=_auth(account_token))
    ours = [s for s in listing.json()["sessions"] if s["id"] == session_id]
    assert [s["title"] for s in ours] == ["感冒护理"]


async def test_update_session_rejects_blank_title(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """标题不能清空：只给了空白字符同样是非法请求。"""
    build_agent()
    session_id = (
        await _send(
            client, token=account_token, message="第一问", sessions=created_sessions
        )
    )[0]["session_id"]

    response = await client.patch(
        f"/api/v1/chat/{session_id}",
        json={"title": "   "},
        headers=_auth(account_token),
    )
    assert response.status_code == 422


async def test_update_session_rejects_empty_body(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """两个字段都没给就是空修改，直接拒绝而不走一遍无意义的写库。"""
    build_agent()
    session_id = (
        await _send(
            client, token=account_token, message="第一问", sessions=created_sessions
        )
    )[0]["session_id"]

    response = await client.patch(
        f"/api/v1/chat/{session_id}", json={}, headers=_auth(account_token)
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_update"


async def test_pinned_session_comes_first(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """置顶：被置顶的会话排在最新会话之前，即使别的会话刚说过话。"""
    build_agent()
    older = (
        await _send(
            client, token=account_token, message="第一问", sessions=created_sessions
        )
    )[0]["session_id"]
    newer = (
        await _send(
            client, token=account_token, message="第二问", sessions=created_sessions
        )
    )[0]["session_id"]
    assert await _session_order(client, account_token, {older, newer}) == [newer, older]

    response = await client.patch(
        f"/api/v1/chat/{older}",
        json={"is_pinned": True},
        headers=_auth(account_token),
    )
    assert response.status_code == 200, response.text
    assert response.json()["is_pinned"] is True

    # 给 newer 再说一轮：它是「最近更新」的那个，但置顶的 older 仍然在最前
    await _send(client, token=account_token, message="补问", session_id=newer)
    assert await _session_order(client, account_token, {older, newer}) == [older, newer]


async def test_unpin_session_clears_the_flag(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """取消置顶后列表里的标记跟着消失。"""
    build_agent()
    session_id = (
        await _send(
            client, token=account_token, message="第一问", sessions=created_sessions
        )
    )[0]["session_id"]
    await client.patch(
        f"/api/v1/chat/{session_id}",
        json={"is_pinned": True},
        headers=_auth(account_token),
    )

    response = await client.patch(
        f"/api/v1/chat/{session_id}",
        json={"is_pinned": False},
        headers=_auth(account_token),
    )
    assert response.status_code == 200, response.text
    assert response.json()["is_pinned"] is False

    listing = await client.get("/api/v1/chat/sessions", headers=_auth(account_token))
    ours = [s for s in listing.json()["sessions"] if s["id"] == session_id]
    assert [s["is_pinned"] for s in ours] == [False]


async def test_delete_session_removes_its_messages(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """删除会话：账本里的消息一起删掉，列表和消息接口都再也查不到。"""
    build_agent()
    session_id = (
        await _send(
            client, token=account_token, message="第一问", sessions=created_sessions
        )
    )[0]["session_id"]

    response = await client.delete(
        f"/api/v1/chat/{session_id}", headers=_auth(account_token)
    )
    assert response.status_code == 204, response.text
    assert response.content == b""

    listing = await client.get("/api/v1/chat/sessions", headers=_auth(account_token))
    assert session_id not in [s["id"] for s in listing.json()["sessions"]]

    messages = await client.get(
        f"/api/v1/chat/{session_id}/messages", headers=_auth(account_token)
    )
    assert messages.status_code == 404

    async with async_session_maker() as db:
        left = await db.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.session_id == uuid.UUID(session_id))
        )
    assert left.scalar_one() == 0


async def test_session_actions_hide_other_users_sessions(
    client: httpx.AsyncClient,
    register_account: Callable[[], Awaitable[str]],
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """别人的会话：重命名 / 置顶 / 删除一律 404，并且真的没被动过。"""
    build_agent()
    alice = await register_account()
    bob = await register_account()
    session_id = (
        await _send(
            client, token=alice, message="Alice 的问题", sessions=created_sessions
        )
    )[0]["session_id"]

    rename = await client.patch(
        f"/api/v1/chat/{session_id}",
        json={"title": "被改名了"},
        headers=_auth(bob),
    )
    assert rename.status_code == 404
    assert rename.json()["detail"]["code"] == "session_not_found"

    pin = await client.patch(
        f"/api/v1/chat/{session_id}",
        json={"is_pinned": True},
        headers=_auth(bob),
    )
    assert pin.status_code == 404
    assert pin.json()["detail"]["code"] == "session_not_found"

    remove = await client.delete(f"/api/v1/chat/{session_id}", headers=_auth(bob))
    assert remove.status_code == 404
    assert remove.json()["detail"]["code"] == "session_not_found"

    listing = await client.get("/api/v1/chat/sessions", headers=_auth(alice))
    ours = [s for s in listing.json()["sessions"] if s["id"] == session_id]
    assert [s["title"] for s in ours] == ["Alice 的问题"]
    assert [s["is_pinned"] for s in ours] == [False]


async def test_session_actions_require_authentication(
    client: httpx.AsyncClient,
) -> None:
    session_id = uuid.uuid4()

    rename = await client.patch(f"/api/v1/chat/{session_id}", json={"title": "x"})
    assert rename.status_code == 401

    remove = await client.delete(f"/api/v1/chat/{session_id}")
    assert remove.status_code == 401


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
    """读到指定数量的 ``block-delta`` 事件就停下（模拟用户在这个时刻点了「停止」）。

    ``_stream_chat`` 产出的是 sse-starlette 的 chunk（``{"data": ...}``），
    这里顺手解出事件负载。
    """
    received: list[dict[str, Any]] = []
    async for chunk in stream:
        event = json.loads(chunk["data"])
        received.append(event)
        if sum(1 for item in received if item["type"] == "block-delta") == chunks:
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
    partial = _streamed_text(received)
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


# ---------------------------------------------------------------------------
# 过程块可见：思维链分块 + 本轮聚合态回放
# 设计见 ``document/中间过程可见-设计方案.md``。
# ---------------------------------------------------------------------------

#: 假模型的思考分片。真实 provider 会把思考放在 ``reasoning_content`` 里。
_THINKING_PIECES = ("先排查常见原因。", "多半是病毒感染。")


def _stream_state_url(session_id: str | uuid.UUID) -> str:
    return f"/api/v1/chat/{session_id}/stream-state"


async def test_chat_streams_thinking_block_then_text_block(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """思考与正文各成一个块，且思考不落账本。"""
    build_agent(reasoning=_THINKING_PIECES)
    events = await _send(
        client,
        token=account_token,
        message="How to deal with a cold?",
        sessions=created_sessions,
    )

    assert [e["type"] for e in events][:2] == ["session", "run-start"]
    starts = [e for e in events if e["type"] == "block-start"]
    assert [e["kind"] for e in starts] == ["thinking", "text"]
    # index 从 0 起、连续递增
    assert [e["index"] for e in starts] == [0, 1]
    assert _text_of(events, "thinking") == "".join(_THINKING_PIECES)
    assert _streamed_text(events) == _FAKE_ANSWER
    assert _block_statuses(events) == ["complete", "complete"]
    assert [e["type"] for e in events][-1] == "done"

    rows = await _ledger(
        client,
        token=account_token,
        session_id=uuid.UUID(events[0]["session_id"]),
    )
    assert [row["role"] for row in rows] == ["user", "assistant"]
    # 账本里只有正文，思考不进 messages 表
    assert rows[1]["content"] == _FAKE_ANSWER
    assert _THINKING_PIECES[0] not in rows[1]["content"]


async def test_chat_without_thinking_keeps_single_text_block(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """模型不吐思考时，事件流与从前一样只有一个正文块。"""
    build_agent()
    events = await _send(
        client, token=account_token, message="你好", sessions=created_sessions
    )
    assert [e["kind"] for e in events if e["type"] == "block-start"] == ["text"]
    assert _streamed_text(events) == _FAKE_ANSWER
    assert _block_statuses(events) == ["complete"]


async def test_stream_state_replays_blocks_after_turn(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """刷新页面后能取回本轮的思考与正文。"""
    build_agent(reasoning=_THINKING_PIECES)
    events = await _send(
        client, token=account_token, message="你好", sessions=created_sessions
    )
    session_id = uuid.UUID(events[0]["session_id"])
    run_id = next(e["run_id"] for e in events if e["type"] == "run-start")

    response = await client.get(
        _stream_state_url(session_id), headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["session_id"] == str(session_id)
    assert payload["run_id"] == run_id
    assert payload["status"] == "done"
    assert payload["finished_at"] >= payload["started_at"]
    assert [block["kind"] for block in payload["blocks"]] == ["thinking", "text"]
    assert [block["status"] for block in payload["blocks"]] == [
        "complete",
        "complete",
    ]
    assert payload["blocks"][0]["text"] == "".join(_THINKING_PIECES)
    assert payload["blocks"][1]["text"] == _FAKE_ANSWER


async def test_stream_state_marks_interrupted_when_user_stops(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """用户点「停止」后，聚合态要落成中断态而不是一直 running。"""
    build_agent(reasoning=_THINKING_PIECES)
    turn, request = await _open_turn(
        client, token=account_token, sessions=created_sessions
    )

    stream = chat_module._stream_chat(turn, request)
    received = await _read_until_chunks(stream, chunks=1)
    await stream.aclose()

    response = await client.get(
        _stream_state_url(turn.session_id), headers=_auth(account_token)
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "interrupted"
    assert payload["finished_at"] is not None
    kinds = [block["kind"] for block in payload["blocks"]]
    assert kinds[0] == "thinking"
    assert all(block["status"] == "interrupted" for block in payload["blocks"])
    assert len(_deltas(received)) == 1
    assert payload["blocks"][0]["text"].startswith(_THINKING_PIECES[0][:1])


async def test_stream_state_is_per_run_after_retry(
    client: httpx.AsyncClient,
    account_token: str,
    build_agent: Callable[..., _Harness],
    created_sessions: list[str],
) -> None:
    """重新生成会开新的一轮：先发 ``reset``，run_id 变化，块从头开始。"""
    build_agent(reasoning=_THINKING_PIECES)
    turn, request = await _open_turn(
        client, token=account_token, sessions=created_sessions
    )

    # 先跑一轮并在中途停掉，制造一个「未答完」的会话——只有这种状态才允许重试。
    first = chat_module._stream_chat(turn, request)
    first_events: list[dict[str, Any]] = []
    async for chunk in first:
        first_events.append(json.loads(chunk["data"]))
        if len(_deltas(first_events)) == 1:
            break
    await first.aclose()
    first_run = next(e["run_id"] for e in first_events if e["type"] == "run-start")

    second = await _send(
        client,
        token=account_token,
        message="睡眠不好怎么办",
        session_id=str(turn.session_id),
        retry=True,
    )

    assert second[0]["type"] == "session"
    assert second[1]["type"] == "run-start"
    assert second[2]["type"] == "reset"
    assert second[2]["from_index"] == 0
    second_run = next(e["run_id"] for e in second if e["type"] == "run-start")
    assert first_run != second_run
    assert [e["index"] for e in second if e["type"] == "block-start"] == [0, 1]
    assert _streamed_text(second) == _FAKE_ANSWER

    response = await client.get(
        _stream_state_url(turn.session_id), headers=_auth(account_token)
    )
    payload = response.json()
    assert payload["run_id"] == second_run
    assert payload["status"] == "done"


async def test_stream_state_rejects_unknown_session(
    client: httpx.AsyncClient,
    account_token: str,
) -> None:
    response = await client.get(
        _stream_state_url(uuid.uuid4()), headers=_auth(account_token)
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


async def test_stream_state_requires_authentication(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(_stream_state_url(uuid.uuid4()))
    assert response.status_code == 401
