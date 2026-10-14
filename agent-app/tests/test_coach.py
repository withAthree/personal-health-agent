"""HC 子图的契约测试（``agent_app.orchestration.coach``）。

不需要数据库、Redis 或网络：起草用的对话客户端与内部审核调用都换成替身。

不过**子图本身要真的跑一遍**：``coach_out`` 交回父状态时 ``contributions`` 只写自己
那一份（父状态这个 channel 带 ``operator.add``，多写就会重复），这条契约只有真跑才
验得到。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage

from agent_app.orchestration import blocks as blocks_module
from agent_app.orchestration import coach, nodes
from agent_app.orchestration.coach import (
    COACH_CHECK,
    COACH_OUT,
    COACH_REPLY,
    HC_MAX_STEPS,
    CoachState,
    CoachVerdict,
    build_coach_subgraph,
)
from agent_app.orchestration.state import Contribution


class _StubDraftLLM:
    """冒充 HC 的对话客户端：每次调用交出一份稿子，并记下收到的消息。

    剧本是**按调用次序**给的（第一项给第一次调用）；写完就一直重复最后一项，扔个
    异常进去就能演「这一次起草失败」。
    """

    def __init__(self, turns: Sequence[str | Exception]) -> None:
        self.turns = list(turns)
        self.prompts: list[list[AnyMessage]] = []

    async def astream(self, prompt: list[AnyMessage]) -> AsyncIterator[AIMessageChunk]:
        self.prompts.append(prompt)
        turn = self.turns[min(len(self.prompts) - 1, len(self.turns) - 1)]
        if isinstance(turn, Exception):
            raise turn
        yield AIMessageChunk(content=turn)


def _stub_draft_llm(
    monkeypatch: pytest.MonkeyPatch, *turns: str | Exception
) -> _StubDraftLLM:
    """把 HC 的对话客户端换成剧本，返回它本身（用来查看收到的消息）。"""
    llm = _StubDraftLLM(turns)
    monkeypatch.setattr(nodes, "get_agent_llm", lambda _agent: llm)
    return llm


def _verdict(
    verdict: str,
    *,
    questions: Sequence[str] = (),
    gaps: Sequence[str] = (),
    reason: str = "",
) -> CoachVerdict:
    return CoachVerdict(
        verdict=verdict,
        questions=list(questions),
        gaps=list(gaps),
        reason=reason,
    )


def _stub_judge(
    monkeypatch: pytest.MonkeyPatch, *results: CoachVerdict | Exception
) -> list[str]:
    """把内部审核换成剧本，返回它收到的提示词（最后一条一直重复）。"""
    seen: list[str] = []

    async def _judge(prompt: str) -> CoachVerdict:
        seen.append(prompt)
        result = results[min(len(seen) - 1, len(results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(coach, "_judge", _judge)
    return seen


def _spy_blocks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """把过程块换成记录器，返回（事件类型 + 载荷）列表。"""
    events: list[tuple[str, dict[str, Any]]] = []

    def _start(kind: str, **kwargs: object) -> None:
        events.append(("start", {"kind": kind, **kwargs}))

    def _delta(text: str, **kwargs: object) -> None:
        events.append(("delta", {"text": text, **kwargs}))

    def _stop(**kwargs: object) -> None:
        events.append(("stop", kwargs))

    monkeypatch.setattr(blocks_module, "start_block", _start)
    monkeypatch.setattr(blocks_module, "delta_block", _delta)
    monkeypatch.setattr(blocks_module, "stop_block", _stop)
    return events


def _kinds(events: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, Any]]:
    """把过程块事件压成（事件类型 + 块种类）。"""
    return [(event, payload.get("kind")) for event, payload in events]


def _state(**overrides: object) -> CoachState:
    base: dict[str, object] = {
        "messages": [HumanMessage(content="我总坚持不下来，怎么办")],
        "step": 0,
        "primary": "hc",
        "queries": {"hc": "用户问：坚持不下来，怎么调整"},
        "memory_digest": "## 用户画像\n- 目标：每周运动三次",
    }
    base.update(overrides)
    return cast("CoachState", base)


async def _run(state: CoachState) -> dict[str, Any]:
    return cast("dict[str, Any]", await build_coach_subgraph().ainvoke(state))


def _system(prompt: list[AnyMessage]) -> str:
    """取一次调用里那条系统消息的正文（提示词都拼在它里面）。"""
    return str(prompt[0].content)


def test_the_subgraph_has_the_designed_nodes_and_edges() -> None:
    graph = build_coach_subgraph().get_graph()
    assert set(graph.nodes) == {
        "__start__",
        "__end__",
        COACH_REPLY,
        COACH_CHECK,
        COACH_OUT,
    }
    plain = {(edge.source, edge.target) for edge in graph.edges if not edge.conditional}
    assert plain == {("__start__", COACH_REPLY), (COACH_OUT, "__end__")}
    conditional = {
        (edge.source, edge.target) for edge in graph.edges if edge.conditional
    }
    assert conditional == {
        (COACH_REPLY, COACH_CHECK),
        (COACH_REPLY, COACH_OUT),
        (COACH_CHECK, COACH_REPLY),
        (COACH_CHECK, COACH_OUT),
    }


def test_the_subgraph_does_not_redeclare_the_parent_accumulating_channel() -> None:
    """``tried_queries`` 在父状态里带 ``operator.add``。

    子图一旦声明同名 channel，就会把父状态已有的项读进来再交回去，父状态那份
    ``operator.add`` 于是又加一遍——重复。HC 本来也不自己取数，不声明是对的。
    """
    assert "tried_queries" not in CoachState.__annotations__


def test_route_after_reply_skips_the_judge_when_there_is_no_draft() -> None:
    assert coach.route_after_reply(_state()) == [COACH_OUT]
    assert coach.route_after_reply(_state(hc_draft="草案")) == [COACH_CHECK]
    assert coach.route_after_reply(_state(hc_draft="   ")) == [COACH_OUT]


def test_route_after_check_rewrites_only_while_there_is_room() -> None:
    revise = {"coach_verdict": "revise"}
    assert coach.route_after_check(_state(**revise, hc_steps=1)) == [COACH_REPLY]
    assert coach.route_after_check(_state(**revise, hc_steps=HC_MAX_STEPS)) == [
        COACH_OUT
    ]
    assert coach.route_after_check(_state(coach_verdict="deliver", hc_steps=1)) == [
        COACH_OUT
    ]


async def test_deliver_hands_back_one_contribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_draft_llm(monkeypatch, "这周先把目标降到两次，每次二十分钟。")
    prompts = _stub_judge(monkeypatch, _verdict("deliver"))
    events = _spy_blocks(monkeypatch)

    result = await _run(_state())

    assert len(llm.prompts) == 1
    assert len(prompts) == 1
    contributions = result["contributions"]
    assert len(contributions) == 1
    assert contributions[0].content == "这周先把目标降到两次，每次二十分钟。"
    assert contributions[0].agent == "hc"
    assert contributions[0].role == "primary"
    assert contributions[0].step == 0
    # HC 不取外部数据：稿子里的数字来自其他 Agent 的材料，来源记在那份材料上。
    assert contributions[0].sources == []
    assert result["user_gaps"] == []
    assert result["context_gaps"] == []
    # 没有重写就不该有 plan 块。
    assert _kinds(events) == []


async def test_revise_rewrites_once_with_the_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_draft_llm(
        monkeypatch, "第一版", "第二版：这周先两次，另外想确认你的膝盖情况。"
    )
    prompts = _stub_judge(
        monkeypatch,
        _verdict(
            "revise",
            questions=["你的膝盖有旧伤吗？"],
            reason="没了解旧伤就给了训练强度",
        ),
        _verdict("deliver"),
    )
    events = _spy_blocks(monkeypatch)

    result = await _run(_state())

    assert len(llm.prompts) == 2
    # 每写出一次就审核一次：第一次要求改，改完第二次通过。
    assert len(prompts) == 2
    first, second = (_system(prompt) for prompt in llm.prompts)
    assert "没了解旧伤就给了训练强度" not in first
    assert "没了解旧伤就给了训练强度" in second
    assert "你的膝盖有旧伤吗？" in second
    assert [c.content for c in result["contributions"]] == [
        "第二版：这周先两次，另外想确认你的膝盖情况。"
    ]
    # 重写要多花一次模型调用，这段等待要有过程块可看。
    assert ("start", "plan") in _kinds(events)


async def test_the_rewrite_gives_up_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _stub_draft_llm(monkeypatch, "第一版", "第二版")
    _stub_judge(
        monkeypatch,
        _verdict(
            "revise",
            questions=["你的膝盖有旧伤吗？"],
            reason="还没确认旧伤",
        ),
    )

    result = await _run(_state())

    assert len(llm.prompts) == HC_MAX_STEPS
    assert [c.content for c in result["contributions"]] == ["第二版"]
    # 没改掉的缺口照样上报，交给反思节点决定是问用户还是派别的 Agent 补。
    assert result["user_gaps"] == ["你的膝盖有旧伤吗？"]


async def test_a_failed_check_still_delivers_the_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_draft_llm(monkeypatch, "评估的稿子")
    _stub_judge(monkeypatch, RuntimeError("审核模型挂了"))

    result = await _run(_state())

    assert [c.content for c in result["contributions"]] == ["评估的稿子"]
    assert result["user_gaps"] == []


async def test_a_revise_without_usable_notes_is_treated_as_deliver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_draft_llm(monkeypatch, "只有一版")
    _stub_judge(monkeypatch, _verdict("revise"))

    result = await _run(_state())

    assert len(llm.prompts) == 1
    assert [c.content for c in result["contributions"]] == ["只有一版"]


async def test_a_failed_draft_produces_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_draft_llm(monkeypatch, RuntimeError("对话模型挂了"))
    prompts = _stub_judge(monkeypatch, _verdict("deliver"))

    result = await _run(_state())

    assert result["contributions"] == []
    # 没有草稿就不该去审核空稿。
    assert prompts == []


async def test_a_failed_rewrite_keeps_the_first_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_draft_llm(monkeypatch, "第一版", RuntimeError("重写挂了"))
    _stub_judge(
        monkeypatch,
        _verdict("revise", reason="语气太说教", questions=["你平时几点下班？"]),
    )

    result = await _run(_state())

    assert [c.content for c in result["contributions"]] == ["第一版"]
    assert result["user_gaps"] == ["你平时几点下班？"]


async def test_system_gaps_go_to_the_context_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_draft_llm(monkeypatch, "第一版", "第二版")
    _stub_judge(
        monkeypatch,
        _verdict(
            "revise",
            gaps=["近 14 天血糖记录"],
            reason="材料里没有近两周的血糖",
        ),
    )

    result = await _run(_state())

    assert result["context_gaps"] == ["近 14 天血糖记录"]
    assert result["user_gaps"] == []


async def test_the_prompt_carries_memory_material_and_the_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_draft_llm(monkeypatch, "草案")
    _stub_judge(monkeypatch, _verdict("deliver"))

    await _run(
        _state(
            sub_questions=["最近的运动量", "用户的旧伤情况"],
            contributions=[
                Contribution(
                    agent="ds",
                    step=0,
                    role="support",
                    content="最近 7 天日均 7300 步",
                    sources=["health_metrics"],
                )
            ],
        )
    )

    system = _system(llm.prompts[0])
    assert "每周运动三次" in system
    assert "最近 7 天日均 7300 步" in system
    assert "用户问：坚持不下来，怎么调整" in system
    assert "最近的运动量" in system


async def test_the_role_follows_who_leads(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_draft_llm(monkeypatch, "草案")
    _stub_judge(monkeypatch, _verdict("deliver"))

    leading = await _run(_state(primary="hc"))
    supporting = await _run(_state(primary="ds"))

    assert leading["contributions"][0].role == "primary"
    assert supporting["contributions"][0].role == "support"
