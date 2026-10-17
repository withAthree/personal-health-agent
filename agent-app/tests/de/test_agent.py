"""``de/agent.py``：模式选择、消息拼装、取最终文本。

不测真实的模型调用（那需要网络，且结果不稳定），只测三个纯逻辑决定：什么情况下走
``support``、``data_context`` 怎么进消息、最后一条 AI 消息怎么取。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_app.de import agent as de_agent
from agent_app.memory.middleware import MemoryContext

USER_ID = uuid.UUID("0b9c7d51-2e64-4f83-a1c5-7d3e8f9a0b21")


@dataclass
class _Graph:
    """假的 DE 图：记录收到的消息与 context，返回固定回复。"""

    reply: AIMessage = field(
        default_factory=lambda: AIMessage(
            content="血糖偏高，需要复查 [来源: 通用医学知识]。"
        )
    )
    payloads: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[MemoryContext] = field(default_factory=list)

    async def ainvoke(
        self,
        payload: dict[str, Any],
        *,
        context: MemoryContext,
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        self.contexts.append(context)
        return {"messages": [*payload["messages"], self.reply]}


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> _Graph:
    fake = _Graph()
    monkeypatch.setattr(de_agent, "build_de_agent", lambda _mode: fake)
    return fake


@pytest.fixture
def mode_recorder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """记录 ``run_de`` 请求了哪个模式。"""
    modes: list[str] = []

    def _build(mode: str) -> _Graph:
        modes.append(mode)
        return _Graph()

    monkeypatch.setattr(de_agent, "build_de_agent", _build)
    return modes


async def test_empty_data_context_runs_primary(mode_recorder: list[str]) -> None:
    await de_agent.run_de("二甲双胍有什么副作用？", user_id=USER_ID)

    assert mode_recorder == ["primary"]


async def test_non_empty_data_context_runs_support(mode_recorder: list[str]) -> None:
    await de_agent.run_de(
        "这些数字说明什么？",
        user_id=USER_ID,
        data_context="空腹血糖均值 6.3",
    )

    assert mode_recorder == ["support"]


async def test_whitespace_only_context_still_runs_primary(
    mode_recorder: list[str],
) -> None:
    await de_agent.run_de("这些数字说明什么？", user_id=USER_ID, data_context="  \n ")

    assert mode_recorder == ["primary"]


async def test_identity_travels_through_context(graph: _Graph) -> None:
    """记忆只能从 context 进来；写进消息会让它进账本。"""
    await de_agent.run_de("二甲双胍有什么副作用？", user_id=USER_ID)

    assert graph.contexts == [MemoryContext(user_id=USER_ID)]


async def test_context_is_appended_to_the_user_message(graph: _Graph) -> None:
    await de_agent.run_de(
        "这些数字说明什么？",
        user_id=USER_ID,
        data_context="空腹血糖均值 6.3",
    )

    sent = graph.payloads[0]["messages"][0]
    assert isinstance(sent, HumanMessage)
    assert "这些数字说明什么？" in sent.content
    assert "空腹血糖均值 6.3" in sent.content


async def test_answer_is_the_last_ai_text(graph: _Graph) -> None:
    out = await de_agent.run_de("二甲双胍有什么副作用？", user_id=USER_ID)

    assert out == "血糖偏高，需要复查 [来源: 通用医学知识]。"


def test_last_text_skips_trailing_tool_messages() -> None:
    """ReAct 收尾时最后一条不一定是 AI 消息，取到 ToolMessage 会得到空串。"""
    messages = [
        HumanMessage(content="问题"),
        AIMessage(content="先查一下"),
        ToolMessage(content="检索结果", tool_call_id="call-1"),
        AIMessage(content="最终回答"),
    ]

    assert de_agent._last_text(messages) == "最终回答"


def test_last_text_reads_content_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "第一段"},
            {"type": "text", "text": "第二段"},
        ]
    )

    assert de_agent._last_text([message]) == "第一段第二段"


def test_last_text_returns_empty_without_ai_message() -> None:
    assert de_agent._last_text([HumanMessage(content="只有提问")]) == ""


def test_compose_leaves_plain_question_untouched() -> None:
    assert de_agent._compose("原问题", "") == "原问题"


def test_compose_marks_the_context_block() -> None:
    """块首必须有一行说明，否则 DE 分不清哪段是用户说的、哪段是上游算的。"""
    composed = de_agent._compose("原问题", "均值 6.3")

    assert composed.startswith("原问题")
    assert "已有数据结果：" in composed
    assert composed.endswith("均值 6.3")
