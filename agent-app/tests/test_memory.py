"""功能单元 1/3：长期记忆的渲染、裁剪、抽取落库与注入。

需要 ``docker compose up -d`` 中的 Postgres 可用：用例会真实读写
``memory_items``。模型全部用替身，不访问网络，也不写 checkpoint 表。

替身结构：

* ``_StubLLM`` / ``_StubStructured``：冒充 ``with_structured_output`` 的结果，
  让 ``extract_turn_memories`` 拿到固定的 ``MemoryOps``（或者抛异常）。
* ``_RecordingModel``：假对话模型，记录每次调用收到的消息，用来断言记忆确实
  被追加进了 system message、且没有进入 messages 账本。
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from sqlalchemy import select

import agent_app.memory.extract as extract_module
from agent_app.config import settings
from agent_app.core.database import async_session_maker
from agent_app.llm.client import SYSTEM_PROMPT
from agent_app.memory.extract import (
    MemoryOp,
    MemoryOps,
    apply_ops,
    extract_turn_memories,
)
from agent_app.memory.middleware import MemoryContext, MemoryMiddleware
from agent_app.memory.render import render_memories
from agent_app.memory.store import fetch_memories
from agent_app.models.memory import MemoryItem

#: 插入时显式给 ``created_at``，让「越新越优先」的断言不依赖插入速度。
_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def two_users(
    register_account: Callable[[], Awaitable[str]],
    created_users: list[uuid.UUID],
) -> tuple[uuid.UUID, uuid.UUID]:
    """两个互不相干的账号：记忆必须按 ``user_id`` 严格隔离。"""
    await register_account()
    await register_account()
    return created_users[0], created_users[1]


def _memory(category: str, content: str, *, offset: int = 0) -> MemoryItem:
    """构造一条未落库的记忆（只供纯渲染用例使用）。"""
    return MemoryItem(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        category=category,
        content=content,
        created_at=_BASE_TIME + timedelta(seconds=offset),
    )


async def _insert(
    user_id: uuid.UUID,
    category: str,
    content: str,
    *,
    offset: int = 0,
) -> uuid.UUID:
    """直接往表里插一条记忆（绕过抽取器）。"""
    item = _memory(category, content, offset=offset)
    item.user_id = user_id
    async with async_session_maker() as db:
        db.add(item)
        await db.commit()
    return item.id


async def _stored(user_id: uuid.UUID) -> dict[str, float | None]:
    """该用户的「内容 → 置信度」映射。"""
    async with async_session_maker() as db:
        rows = (
            await db.execute(select(MemoryItem).where(MemoryItem.user_id == user_id))
        ).scalars()
    return {row.content: row.confidence for row in rows}


async def _ids(user_id: uuid.UUID) -> list[uuid.UUID]:
    """该用户全部记忆的 id。"""
    async with async_session_maker() as db:
        rows = await db.execute(
            select(MemoryItem.id).where(MemoryItem.user_id == user_id)
        )
    return list(rows.scalars().all())


class _StubStructured:
    """冒充 ``with_structured_output`` 的返回值，记录收到的提示词。"""

    def __init__(self, result: MemoryOps | Exception) -> None:
        self._result = result
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> MemoryOps:
        self.prompts.append(prompt)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubLLM:
    """冒充旁路模型客户端。"""

    def __init__(self, result: MemoryOps | Exception) -> None:
        self.structured = _StubStructured(result)
        self.schema: type[BaseModel] | None = None

    def with_structured_output(self, schema: type[BaseModel]) -> _StubStructured:
        self.schema = schema
        return self.structured


class _RecordingModel(BaseChatModel):
    """假对话模型：只记录收到的消息。"""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "memory-recording"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架签名
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="嗯"))])


# --------------------------------------------------------------------------- 渲染


def test_render_returns_blank_without_items() -> None:
    assert render_memories([]) == ""


def test_render_groups_items_into_fixed_order() -> None:
    text = render_memories(
        [
            _memory("insight", "喜欢晨练"),
            _memory("health_fact", "青霉素过敏"),
            _memory("health_fact", "乳糖不耐受"),
            _memory("preference", "回答简短一点"),
        ]
    )
    assert "### 健康事实" in text
    assert "- 青霉素过敏" in text
    assert "- 乳糖不耐受" in text
    assert "### 其他了解" in text
    # 顺序固定为 _SECTIONS 声明的优先级，与传入顺序无关。
    assert text.index("### 健康事实") < text.index("### 沟通偏好")
    assert text.index("### 沟通偏好") < text.index("### 其他了解")
    # 没有内容的类别不出现标题。
    assert "### 目标" not in text


def test_render_collapses_newlines_and_truncates() -> None:
    text = render_memories([_memory("goal", "第一行\n第二行\n" + "长" * 900)])
    assert "第一行 第二行" in text
    assert "长" * 900 not in text


# --------------------------------------------------------------------------- 读取


async def test_fetch_memories_isolates_users(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_a, user_b = two_users
    await _insert(user_a, "health_fact", "A 的过敏")
    await _insert(user_b, "health_fact", "B 的过敏")

    assert {item.content for item in await fetch_memories(user_a)} == {"A 的过敏"}
    assert {item.content for item in await fetch_memories(user_b)} == {"B 的过敏"}


async def test_fetch_memories_caps_loose_categories(
    two_users: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _ = two_users
    monkeypatch.setattr(settings, "memory_max_per_loose_category", 2)
    monkeypatch.setattr(settings, "memory_max_injected", 50)
    # offset 越大越新：4 条 insight（宽松，只留最近 2 条）+ 4 条 health_fact（必留）。
    for index in range(4):
        await _insert(user_id, "insight", f"推测-{index}", offset=index)
    for index in range(4):
        await _insert(user_id, "health_fact", f"事实-{index}", offset=index)

    items = await fetch_memories(user_id)
    insights = {item.content for item in items if item.category == "insight"}
    facts = {item.content for item in items if item.category == "health_fact"}

    assert insights == {"推测-2", "推测-3"}
    assert facts == {"事实-0", "事实-1", "事实-2", "事实-3"}


async def test_fetch_memories_respects_total_cap(
    two_users: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _ = two_users
    monkeypatch.setattr(settings, "memory_max_per_loose_category", 20)
    monkeypatch.setattr(settings, "memory_max_injected", 3)
    for index in range(5):
        await _insert(user_id, "goal", f"目标-{index}", offset=index)

    items = await fetch_memories(user_id)
    assert [item.content for item in items] == ["目标-4", "目标-3", "目标-2"]


async def test_disabled_memory_is_not_read(
    two_users: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _ = two_users
    await _insert(user_id, "health_fact", "青霉素过敏")
    monkeypatch.setattr(settings, "memory_enabled", False)

    assert await fetch_memories(user_id) == []


# --------------------------------------------------------------------------- 落库


async def test_apply_ops_adds_updates_and_deletes(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_id, _ = two_users
    added = await apply_ops(
        user_id,
        None,
        [
            MemoryOp(
                op="add",
                category="goal",
                content="把体重降到 70kg",
                confidence=0.9,
            )
        ],
    )
    assert added == 1
    assert await _stored(user_id) == {"把体重降到 70kg": 0.9}

    (memory_id,) = await _ids(user_id)
    changed = await apply_ops(
        user_id,
        None,
        [
            MemoryOp(
                op="update",
                target_id=str(memory_id),
                content="把体重降到 68kg",
            )
        ],
    )
    assert changed == 1
    # 没给 confidence 时保留原值。
    assert await _stored(user_id) == {"把体重降到 68kg": 0.9}

    removed = await apply_ops(
        user_id, None, [MemoryOp(op="delete", target_id=str(memory_id))]
    )
    assert removed == 1
    assert await _stored(user_id) == {}


async def test_apply_ops_refuses_another_users_memory(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_a, user_b = two_users
    memory_id = await _insert(user_b, "health_fact", "B 的过敏")

    changed = await apply_ops(
        user_a,
        None,
        [
            MemoryOp(op="update", target_id=str(memory_id), content="被改了"),
            MemoryOp(op="delete", target_id=str(memory_id)),
        ],
    )

    assert changed == 0
    assert await _stored(user_b) == {"B 的过敏": None}


async def test_apply_ops_drops_invalid_operations(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_id, _ = two_users
    applied = await apply_ops(
        user_id,
        None,
        [
            MemoryOp(op="add", category="不是类别", content="x"),
            MemoryOp(op="add", category="goal", content="   "),
            MemoryOp(op="add", category=None, content="x"),
            MemoryOp(op="update", target_id="不是-uuid", content="x"),
            MemoryOp(op="update", target_id=str(uuid.uuid4()), content="x"),
            MemoryOp(op="delete", target_id=None),
        ],
    )

    assert applied == 0
    assert await _stored(user_id) == {}


async def test_apply_ops_skips_exact_duplicate(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_id, _ = two_users
    op = MemoryOp(op="add", category="health_fact", content="青霉素过敏")

    assert await apply_ops(user_id, None, [op]) == 1
    assert await apply_ops(user_id, None, [op]) == 0
    assert await _stored(user_id) == {"青霉素过敏": None}


# --------------------------------------------------------------------------- 抽取


async def test_extract_turn_memories_writes_model_output(
    two_users: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _ = two_users
    stub = _StubLLM(
        MemoryOps(
            ops=[
                MemoryOp(
                    op="add",
                    category="health_fact",
                    content="青霉素过敏",
                    confidence=0.8,
                )
            ]
        )
    )
    monkeypatch.setattr(extract_module, "get_memory_llm", lambda: stub)

    applied = await extract_turn_memories(
        user_id=user_id,
        session_id=None,
        user_message="我对青霉素过敏",
        assistant_reply="记下了。",
    )

    assert applied == 1
    assert stub.schema is MemoryOps
    assert await _stored(user_id) == {"青霉素过敏": 0.8}
    # 提示词里带上了本轮对话。
    assert "我对青霉素过敏" in stub.structured.prompts[0]
    assert "记下了。" in stub.structured.prompts[0]


async def test_extract_turn_memories_swallows_model_failure(
    two_users: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _ = two_users
    monkeypatch.setattr(
        extract_module, "get_memory_llm", lambda: _StubLLM(RuntimeError("上游 500"))
    )

    applied = await extract_turn_memories(
        user_id=user_id,
        session_id=None,
        user_message="我对青霉素过敏",
        assistant_reply="记下了。",
    )

    assert applied == 0
    assert await _stored(user_id) == {}


async def test_extract_turn_memories_ignores_empty_turn(
    two_users: tuple[uuid.UUID, uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, _ = two_users
    stub = _StubLLM(MemoryOps(ops=[]))
    monkeypatch.setattr(extract_module, "get_memory_llm", lambda: stub)

    applied = await extract_turn_memories(
        user_id=user_id,
        session_id=None,
        user_message="   ",
        assistant_reply="",
    )

    assert applied == 0
    assert stub.structured.prompts == []


# --------------------------------------------------------------------------- 注入


async def test_middleware_injects_memory_without_touching_checkpoint(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_id, _ = two_users
    await _insert(user_id, "health_fact", "青霉素过敏")

    model = _RecordingModel()
    graph = create_agent(
        model,
        tools=None,
        system_prompt=SYSTEM_PROMPT,
        middleware=[MemoryMiddleware()],
        checkpointer=InMemorySaver(),
        context_schema=MemoryContext,
    )
    config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="我能吃阿莫西林吗")]},
        config,
        context=MemoryContext(user_id=user_id),
    )

    system = model.calls[-1][0]
    assert isinstance(system, SystemMessage)
    assert SYSTEM_PROMPT in str(system.content)
    assert "青霉素过敏" in str(system.content)

    # 记忆只在模型调用时临时生效，不能进 checkpoint 的 messages 账本。
    state = await graph.aget_state(config)
    assert state.values["messages"]
    assert all(
        "青霉素过敏" not in str(message.content) for message in state.values["messages"]
    )


async def test_middleware_skips_memory_of_other_users(
    two_users: tuple[uuid.UUID, uuid.UUID],
) -> None:
    user_a, user_b = two_users
    await _insert(user_b, "health_fact", "B 的过敏")

    model = _RecordingModel()
    graph = create_agent(
        model,
        tools=None,
        system_prompt=SYSTEM_PROMPT,
        middleware=[MemoryMiddleware()],
        checkpointer=InMemorySaver(),
        context_schema=MemoryContext,
    )
    config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="我能吃阿莫西林吗")]},
        config,
        context=MemoryContext(user_id=user_a),
    )

    assert "B 的过敏" not in str(model.calls[-1][0].content)
