"""``de/tool.py``：DE 与主 agent 之间的接缝。

不测 DE 内部的推理（那归检索工具自己的测试），只测接缝上的四个决定：身份从哪来、
一轮最多问几次、失败怎么变成话、成功后怎么记账。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain.tools import ToolRuntime

from agent_app.config import settings
from agent_app.de import collect
from agent_app.de import tool as de_tool
from agent_app.memory.middleware import MemoryContext

USER_ID = uuid.UUID("6f1d0a3e-8c4b-4a2f-9d1e-2b7c5a9e4f10")


def _runtime_with(context: Any) -> ToolRuntime[MemoryContext]:  # noqa: ANN401
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id="call-de-1",
        store=None,
    )


def _runtime() -> ToolRuntime[MemoryContext]:
    return _runtime_with(MemoryContext(user_id=USER_ID))


@dataclass
class _Runs:
    """假的 ``run_de``：记录收到什么，返回什么由用例定。"""

    questions: list[str] = field(default_factory=list)
    user_ids: list[uuid.UUID] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    answer: str = "空腹血糖 6.3 mmol/L 属于空腹血糖受损范围 [来源: 通用医学知识]。"
    raises: Exception | None = None
    delay: float = 0.0

    async def __call__(
        self,
        question: str,
        *,
        user_id: uuid.UUID,
        data_context: str = "",
    ) -> str:
        self.questions.append(question)
        self.user_ids.append(user_id)
        self.contexts.append(data_context)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        return self.answer


@pytest.fixture
def run_de(monkeypatch: pytest.MonkeyPatch) -> _Runs:
    runs = _Runs()
    monkeypatch.setattr(de_tool, "run_de", runs)
    return runs


async def _ask(
    question: str,
    *,
    data_context: str = "",
    runtime: ToolRuntime[MemoryContext] | None = None,
) -> str:
    payload = {
        "question": question,
        "data_context": data_context,
        "runtime": runtime or _runtime(),
    }
    out = await de_tool.de_expert.ainvoke(payload)
    assert isinstance(out, str)
    return out


async def test_identity_and_question_come_from_the_call(run_de: _Runs) -> None:
    await _ask("空腹血糖 6.3 是什么水平？")

    assert run_de.user_ids == [USER_ID]
    assert run_de.questions == ["空腹血糖 6.3 是什么水平？"]


async def test_data_context_reaches_the_sub_agent(run_de: _Runs) -> None:
    """支持模式的唯一开关就是这个参数，丢了它就退化成主模式。"""
    await _ask("这些数字说明什么？", data_context="空腹血糖均值 6.3，共 18 条记录")

    assert run_de.contexts == ["空腹血糖均值 6.3，共 18 条记录"]


async def test_answer_is_returned_verbatim(run_de: _Runs) -> None:
    """引用标记必须原样回到主 agent，不能被这里改写。"""
    out = await _ask("空腹血糖 6.3 是什么水平？")

    assert out == run_de.answer


async def test_repeated_question_is_reused(run_de: _Runs) -> None:
    with collect.collect() as turn:
        first = await _ask("空腹血糖 6.3 是什么水平？")
        second = await _ask("空腹血糖 6.3 是什么水平？")

    assert first == second
    assert len(run_de.questions) == 1
    assert turn.calls == 1


async def test_same_question_with_different_context_is_not_reused(
    run_de: _Runs,
) -> None:
    with collect.collect():
        await _ask("这些数字说明什么？", data_context="A")
        await _ask("这些数字说明什么？", data_context="B")

    assert len(run_de.questions) == 2


async def test_calls_are_capped_per_turn(run_de: _Runs) -> None:
    with collect.collect() as turn:
        for index in range(collect.MAX_CALLS_PER_TURN):
            await _ask(f"问题 {index}")

        out = await _ask("再来一个")

    assert turn.calls == collect.MAX_CALLS_PER_TURN
    assert len(run_de.questions) == collect.MAX_CALLS_PER_TURN
    assert "不要再问了" in out


async def test_works_without_a_collection_context(run_de: _Runs) -> None:
    """探针、单测、后台任务都不会开收集，这时不能报错也不能静默短路。"""
    assert collect.current_turn() is None

    out = await _ask("空腹血糖 6.3 是什么水平？")

    assert out == run_de.answer


async def test_timeout_becomes_readable_text(
    run_de: _Runs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "de_timeout_s", 0.01)
    run_de.delay = 1.0

    out = await _ask("二甲双胍有什么副作用？")

    assert "超时" in out
    assert "不要凭印象" in out


async def test_failure_becomes_readable_text(run_de: _Runs) -> None:
    run_de.raises = RuntimeError("connection reset")

    out = await _ask("二甲双胍有什么副作用？")

    assert "系统故障" in out
    assert "connection reset" not in out


async def test_empty_answer_does_not_invite_fabrication(run_de: _Runs) -> None:
    run_de.answer = ""

    out = await _ask("二甲双胍有什么副作用？")

    assert "没有返回内容" in out


async def test_missing_identity_skips_the_sub_agent(run_de: _Runs) -> None:
    out = await _ask("二甲双胍有什么副作用？", runtime=_runtime_with(None))

    assert "拿不到用户身份" in out
    assert run_de.questions == []


async def test_failed_call_does_not_consume_quota(run_de: _Runs) -> None:
    """超时/异常说明结果没拿到，占掉一次配额会让后续重试直接失败。"""
    run_de.raises = RuntimeError("boom")

    with collect.collect() as turn:
        await _ask("第一次")
        run_de.raises = None
        await _ask("第二次")

    assert turn.calls == 1
    assert len(run_de.questions) == 2
