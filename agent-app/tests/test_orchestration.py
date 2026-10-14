"""图编排的契约测试。

绝大多数用例不需要数据库、Redis 或网络：图级用例把节点内部的两个 IO 名字换成桩，
节点级用例把编排器客户端换成桩。**唯一例外**是文件清单那一条——它要验证的是 SQL
与用户隔离，只能连库跑。

分工：

* 拓扑与路由（骨架）——「连线不许被顺手改掉」；
* ``intake`` / ``orchestrator``——决策怎么被校验、降级与摊平进 state。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent_app.core.database import async_session_maker
from agent_app.memory.middleware import MemoryContext
from agent_app.models.file import FileMetadata
from agent_app.orchestration import blocks as blocks_module
from agent_app.orchestration import coach, nodes
from agent_app.orchestration import graph as g
from agent_app.orchestration.coach import CoachState
from agent_app.orchestration.state import (
    MAX_ROUNDS,
    Contribution,
    OrchestrationState,
    RoutingDecision,
)

_AGENT_NODES = {g.DATA_SCIENCE, g.DOMAIN_EXPERT, g.HEALTH_COACH}

_EXPECTED_NODES = {
    g.INTAKE,
    g.ORCHESTRATOR,
    g.DATA_SCIENCE,
    g.DOMAIN_EXPERT,
    g.HEALTH_COACH,
    g.INTEGRATE,
    g.REFLECT,
    g.NEXT_ROUND,
    g.FINALIZE,
    g.MEMORY,
}

_EXPECTED_PLAIN_EDGES = {
    ("__start__", g.INTAKE),
    (g.INTAKE, g.ORCHESTRATOR),
    (g.INTEGRATE, g.REFLECT),
    (g.NEXT_ROUND, g.ORCHESTRATOR),
    (g.FINALIZE, "__end__"),
    (g.MEMORY, "__end__"),
}

_EXPECTED_CONDITIONAL_EDGES = (
    {(g.ORCHESTRATOR, target) for target in (g.REFLECT, *_AGENT_NODES)}
    | {
        (source, target)
        for source in _AGENT_NODES
        for target in (_AGENT_NODES - {source}) | {g.INTEGRATE}
    }
    | {(g.REFLECT, target) for target in (g.FINALIZE, g.MEMORY, g.NEXT_ROUND)}
)


# =====================================================================
# 测试替身
# =====================================================================


class _StubStructured:
    """冒充 ``with_structured_output`` 的返回值，记录收到的提示词。"""

    def __init__(self, result: RoutingDecision | dict[str, Any] | Exception) -> None:
        self._result = result
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> RoutingDecision | dict[str, Any]:
        self.prompts.append(prompt)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubLLM:
    """冒充编排器客户端：``.schema`` 与 ``.structured.prompts`` 供断言。"""

    def __init__(self, result: RoutingDecision | dict[str, Any] | Exception) -> None:
        self.structured = _StubStructured(result)
        self.schema: type[BaseModel] | None = None

    def with_structured_output(self, schema: type[BaseModel]) -> _StubStructured:
        self.schema = schema
        return self.structured


class _StubStreamLLM:
    """冒充对话客户端：``astream`` 把准备好的分片吐出来，并记下收到的消息。"""

    def __init__(self, chunks: list[str], *, error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.prompts: list[list[AnyMessage]] = []

    async def astream(self, prompt: list[AnyMessage]) -> AsyncIterator[AIMessageChunk]:
        self.prompts.append(prompt)
        for text in self.chunks:
            yield AIMessageChunk(content=text)
        if self.error is not None:
            raise self.error


class _StubToolLLM:
    """冒充子 Agent 客户端：按剧本依次给出工具调用或正文。

    剧本走完后一直重复最后一项——这正是「模型盯着同一份数据反复取」的形态，
    用来测 ``_MAX_TOOL_ROUNDS`` 的兜底。
    """

    def __init__(
        self, turns: list[AIMessageChunk], *, error: Exception | None = None
    ) -> None:
        self.turns = turns
        self.error = error
        self.messages: list[list[AnyMessage]] = []
        self.bound: list[int] = []
        self._index = 0

    def bind_tools(self, tools: list[BaseTool]) -> _StubToolLLM:
        self.bound.append(len(tools))
        return self

    async def astream(
        self, messages: list[AnyMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        self.messages.append(list(messages))
        if self.error is not None:
            raise self.error
        turn = self.turns[min(self._index, len(self.turns) - 1)]
        self._index += 1
        yield turn


def _tool_turn(
    name: str, args: dict[str, Any], call_id: str = "call-1"
) -> AIMessageChunk:
    """一个「模型要调工具」的分片。"""
    return AIMessageChunk(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _stub_agent_llm(monkeypatch: pytest.MonkeyPatch, llm: _StubToolLLM) -> _StubToolLLM:
    """把子 Agent 客户端换成桩，并让 ``_current_user_id`` 返回一个真身份。"""
    monkeypatch.setattr(nodes, "get_agent_llm", lambda _agent: llm)
    monkeypatch.setattr(nodes, "_current_user_id", uuid.uuid4)
    return llm


def _stub_health_tools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: str = "2026-10-01  6.7",
    sources: list[str] | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """把真实健康数据工具换成假的，返回它被调用的记录。"""
    seen: list[dict[str, Any]] = []

    @tool
    async def read_health_metric(zh_name: str) -> str:
        """读取某个指标的逐次测量值。"""
        seen.append({"zh_name": zh_name})
        if error is not None:
            raise error
        return result

    def _build(_user_id: uuid.UUID, used: list[str]) -> list[BaseTool]:
        used.extend(sources or [])
        return [read_health_metric]

    monkeypatch.setattr(nodes, "build_health_tools", _build)
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


def _stub_model(
    monkeypatch: pytest.MonkeyPatch,
    result: RoutingDecision | dict[str, Any] | Exception,
) -> _StubLLM:
    """把编排器客户端换成桩，返回它供断言。"""
    stub = _StubLLM(result)
    monkeypatch.setattr(nodes, "get_orchestrator_llm", lambda: stub)
    return stub


async def _blank(_user_id: uuid.UUID) -> str:
    """旁路查询的桩：什么都不返回。"""
    return ""


def _stub_router(monkeypatch: pytest.MonkeyPatch, decision: RoutingDecision) -> None:
    """挡掉图级用例里的一切真实 IO：不查库、不问模型。

    ``nodes.orchestrator`` 是图编译时按引用绑定的，测试改不动它本身；改得动的是它
    内部按名字去查的那几个。
    """
    monkeypatch.setattr(nodes, "_current_user_id", uuid.uuid4)
    monkeypatch.setattr(nodes, "_load_memory_digest", _blank)
    monkeypatch.setattr(nodes, "_load_files_digest", _blank)

    async def _decide(prompt: str) -> RoutingDecision:
        del prompt
        return decision

    monkeypatch.setattr(nodes, "_decide", _decide)


def _stub_graph(
    monkeypatch: pytest.MonkeyPatch,
    decision: RoutingDecision,
    *,
    reflect: Callable[[OrchestrationState], dict[str, Any]] | None = None,
) -> dict[str, int]:
    """给整张图打桩，返回各收尾节点被调用的次数。

    节点函数是 ``add_node`` 时按引用绑定的，所以桩必须在构图**之前**打好：图级
    用例不能复用 ``get_orchestration_graph()`` 缓存的那一张，要自己 build 一次。
    ``reflect`` 留空表示用真的（专门测它的空产出短路）。
    """
    calls = {"reflect": 0, "finalize": 0, "memory": 0, "agent": 0}
    real_reflect = nodes.reflect
    _stub_router(monkeypatch, decision)

    def _counting_reflect(state: OrchestrationState) -> dict[str, Any]:
        calls["reflect"] += 1
        return (real_reflect if reflect is None else reflect)(state)

    def _counting_finalize(state: OrchestrationState) -> dict[str, Any]:
        # 真的 ``finalize`` 在没有 draft 时会调对话模型，图级用例一律换成桩。
        del state
        calls["finalize"] += 1
        return {}

    def _counting_memory(state: OrchestrationState) -> dict[str, Any]:
        del state
        calls["memory"] += 1
        return {}

    def _counting_agent(state: OrchestrationState) -> dict[str, Any]:
        # 三个子 Agent 现在真会取数、调模型（单位 4 起），图级用例一律换成桩。
        del state
        calls["agent"] += 1
        return {}

    def _counting_coach(state: CoachState) -> dict[str, Any]:
        # 子图节点必须把参数注解成 ``CoachState``：langgraph 是拿函数签名的注解推
        # ``input_schema`` 的，拿 ``OrchestrationState`` 去建一张 ``CoachState`` 的图，
        # 会在 ``add_node`` 时因同名 channel 类型不一致报错。
        del state
        calls["agent"] += 1
        return {}

    def _stub_coach() -> CompiledStateGraph[Any, Any, Any, Any]:
        # HC 已经是一个子图（``coach.build_coach_subgraph``）：图级用例只关心它作为
        # 一个节点被调度了几次，用一张只有一个节点的最小图顶上。
        stub = StateGraph(CoachState)
        stub.add_node("coach_stub", _counting_coach)
        stub.add_edge(START, "coach_stub")
        stub.add_edge("coach_stub", END)
        return stub.compile()

    monkeypatch.setattr(nodes, "reflect", _counting_reflect)
    monkeypatch.setattr(nodes, "finalize", _counting_finalize)
    monkeypatch.setattr(nodes, "memory", _counting_memory)
    monkeypatch.setattr(nodes, "data_science", _counting_agent)
    monkeypatch.setattr(nodes, "domain_expert", _counting_agent)
    monkeypatch.setattr(coach, "build_coach_subgraph", _stub_coach)
    return calls


def _state(**overrides: object) -> OrchestrationState:
    """一个最小可用的 state：一条用户消息，其余字段按需覆盖。"""
    base: dict[str, object] = {"messages": [HumanMessage(content="我最近血糖怎么样")]}
    base.update(overrides)
    return cast("OrchestrationState", base)


def _edges(*, conditional: bool) -> set[tuple[str, str]]:
    """取出 ``get_graph()`` 里指定类型的边。"""
    return {
        (edge.source, edge.target)
        for edge in g.get_orchestration_graph().get_graph().edges
        if edge.conditional is conditional
    }


# =====================================================================
# 拓扑
# =====================================================================


def test_graph_compiles_with_expected_nodes() -> None:
    drawable = g.get_orchestration_graph().get_graph()
    assert set(drawable.nodes) >= _EXPECTED_NODES


def test_plain_edges_are_exactly_as_designed() -> None:
    # 只断言「设计里的边都在」：LangGraph 自己会补 __start__ / __end__ 相关的连线，
    # 逐条相等会让测试随框架版本碎掉。
    assert _edges(conditional=False) >= _EXPECTED_PLAIN_EDGES


def test_conditional_edges_are_exactly_as_designed() -> None:
    assert _edges(conditional=True) == _EXPECTED_CONDITIONAL_EDGES


def test_general_node_is_gone() -> None:
    # 不分配任务的路径改成直接进反思，通用对话的正文由 ``finalize`` 自己生成。
    drawable = g.get_orchestration_graph().get_graph()
    assert "general" not in drawable.nodes
    assert "general" not in g.NODE_AGENT_SOURCE


def test_agent_edges_never_loop_back_to_themselves() -> None:
    # 路由永不给 Agent 节点派自己，path_map 里也就不该有自环。
    for node in _AGENT_NODES:
        assert (node, node) not in _EXPECTED_CONDITIONAL_EDGES


# =====================================================================
# 路由
# =====================================================================


def test_route_from_orchestrator_without_assignment_goes_to_reflection() -> None:
    assert g.route_from_orchestrator({"primary": "general"}) == [g.REFLECT]
    # primary 缺失等同于「不分配任务」。
    assert g.route_from_orchestrator({}) == [g.REFLECT]


def test_route_from_orchestrator_primary_only_goes_straight_to_primary() -> None:
    assert g.route_from_orchestrator({"primary": "ds", "supports": []}) == [
        g.DATA_SCIENCE
    ]


def test_route_from_orchestrator_dispatches_supports_only() -> None:
    # 先辅后主：编排器只派支持 Agent，主 Agent 等支持 Agent 的出边来触发。
    assert g.route_from_orchestrator({"primary": "hc", "supports": ["ds", "de"]}) == [
        g.DATA_SCIENCE,
        g.DOMAIN_EXPERT,
    ]


def test_agent_route_sends_support_to_primary_and_primary_to_integrate() -> None:
    state: OrchestrationState = {"primary": "hc", "supports": ["ds", "de"]}
    assert g.make_route_from_agent("ds")(state) == [g.HEALTH_COACH]
    assert g.make_route_from_agent("de")(state) == [g.HEALTH_COACH]
    assert g.make_route_from_agent("hc")(state) == [g.INTEGRATE]


def test_agent_route_goes_integrate_when_it_is_the_only_one() -> None:
    # ``primary`` 存的是代号（``ds``），不是节点名，路由里比的就是代号。
    for code in g.AGENT_NODE:
        assert g.make_route_from_agent(code)({"primary": code, "supports": []}) == [
            g.INTEGRATE
        ]


def test_route_from_reflect_stops_on_every_stop_condition() -> None:
    stop = [g.FINALIZE, g.MEMORY]
    assert g.route_from_reflect({"sufficient": True}) == stop
    assert g.route_from_reflect({"stalled": True}) == stop
    assert g.route_from_reflect({"step": MAX_ROUNDS}) == stop


def test_route_from_reflect_loops_back_when_information_is_missing() -> None:
    assert g.route_from_reflect({"step": 0}) == [g.NEXT_ROUND]
    assert g.route_from_reflect({"step": MAX_ROUNDS - 1}) == [g.NEXT_ROUND]


# =====================================================================
# 端到端：最坏路径必须跑得完
# =====================================================================


async def test_worst_case_path_finishes_within_recursion_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主 + 两个支持 Agent、反思一直说不够，跑满 MAX_ROUNDS 次回环。

    这是步数最多的一条路径，也是 RECURSION_LIMIT 存在的理由：默认的 25 在这里
    会抛 GraphRecursionError。
    """

    def _never_satisfied(state: OrchestrationState) -> dict[str, Any]:
        del state
        return {}

    calls = _stub_graph(
        monkeypatch,
        RoutingDecision(primary="hc", supports=["ds", "de"]),
        reflect=_never_satisfied,
    )
    graph = g.build_orchestration_graph().compile()
    result = await graph.ainvoke(
        {},
        {"recursion_limit": g.RECURSION_LIMIT},
        context=MemoryContext(user_id=uuid.uuid4()),
    )

    assert result["step"] == MAX_ROUNDS
    # 每轮都要过反思，最后一轮不再回环；收尾只跑一次。
    assert calls["reflect"] == MAX_ROUNDS + 1
    assert calls["finalize"] == 1
    assert calls["memory"] == 1


async def test_general_branch_goes_through_reflection_to_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 不分配任务 → 跳过整合，但反思照过（真的 ``reflect`` 对空产出直接放行）。
    calls = _stub_graph(monkeypatch, RoutingDecision(primary="general", cuj="闲聊"))
    graph = g.build_orchestration_graph().compile()
    result = await graph.ainvoke(
        {},
        {"recursion_limit": g.RECURSION_LIMIT},
        context=MemoryContext(user_id=uuid.uuid4()),
    )

    assert result.get("step") is None
    assert calls["reflect"] == 1
    assert calls["finalize"] == 1


# =====================================================================
# intake：只组装上下文
# =====================================================================


async def test_intake_returns_nothing_without_a_user_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 非图执行环境（或 context 里没有身份）时不能查库，也不能把整轮对话弄挂。
    monkeypatch.setattr(nodes, "_current_user_id", lambda: None)
    assert await nodes.intake(_state()) == {}


async def test_intake_fills_both_digests_from_the_context_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = uuid.uuid4()
    seen: list[uuid.UUID] = []

    async def _memory(uid: uuid.UUID) -> str:
        seen.append(uid)
        return "## 关于这位用户\n- 目标：控制血糖"

    async def _files(uid: uuid.UUID) -> str:
        seen.append(uid)
        return "- 血糖记录.csv（已解析）"

    monkeypatch.setattr(nodes, "_current_user_id", lambda: user_id)
    monkeypatch.setattr(nodes, "_load_memory_digest", _memory)
    monkeypatch.setattr(nodes, "_load_files_digest", _files)

    update = await nodes.intake(_state())
    assert update == {
        "memory_digest": "## 关于这位用户\n- 目标：控制血糖",
        "files_digest": "- 血糖记录.csv（已解析）",
    }
    # 身份只能来自 context，而且两个查询用的是同一个 id。
    assert seen == [user_id, user_id]


async def test_files_digest_reads_only_the_users_own_files(
    register_account: Callable[[], Awaitable[str]],
    created_users: list[uuid.UUID],
) -> None:
    """这一条要连库：验证的是 SQL 与用户隔离，渲染另有纯函数用例。"""
    await register_account()
    user_id = created_users[-1]
    assert await nodes._load_files_digest(user_id) == ""

    async with async_session_maker() as db:
        db.add(
            FileMetadata(
                user_id=user_id,
                filename="a.csv",
                original_name="血糖记录.csv",
                file_path=f"{user_id}/a.csv",
                file_size=1,
                mime_type="text/csv",
                extension="csv",
                status="parsed",
            )
        )
        await db.commit()

    assert await nodes._load_files_digest(user_id) == "- 血糖记录.csv（已解析）"
    # 另一个用户看不到它。
    assert await nodes._load_files_digest(uuid.uuid4()) == ""


def test_render_files_labels_the_status() -> None:
    assert nodes._render_files([]) == ""
    assert nodes._render_files(
        [("血糖记录.csv", "parsed"), ("体检报告.pdf", "failed")]
    ) == ("- 血糖记录.csv（已解析）\n- 体检报告.pdf（解析失败）")
    # 库里出现没见过的状态时原样带出去，不吞掉。
    assert nodes._render_files([("y.csv", "weird")]) == "- y.csv（weird）"


def test_render_history_keeps_only_the_recent_turns() -> None:
    messages: list[AnyMessage] = [SystemMessage(content="系统提示")]
    for index in range(4):
        messages += [
            HumanMessage(content=f"问 {index}"),
            AIMessage(content=f"答 {index}"),
        ]

    lines = nodes._render_history(messages).splitlines()
    # 只留最后 6 条，system 不进提示词。
    assert lines[0] == "用户：问 1"
    assert lines[-1] == "助手：答 3"
    assert "系统提示" not in "\n".join(lines)


# =====================================================================
# orchestrator：决策的校验、降级与摊平
# =====================================================================


async def test_orchestrator_writes_the_decision_into_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _stub_model(
        monkeypatch,
        RoutingDecision(
            cuj="数据趋势分析",
            primary="ds",
            queries={"ds": "取最近 14 天的血糖记录"},
            sub_questions=["最近血糖趋势如何"],
            reason="要看数据才能答",
        ),
    )

    update = await nodes.orchestrator(_state())

    assert update["primary"] == "ds"
    assert update["supports"] == []
    assert update["cuj"] == "数据趋势分析"
    assert update["queries"] == {"ds": "取最近 14 天的血糖记录"}
    assert update["sub_questions"] == ["最近血糖趋势如何"]
    assert update["routing_reason"] == "要看数据才能答"
    # 派出去的查询进 tried_queries，回环时才不会重复问一遍。
    assert update["tried_queries"] == ["取最近 14 天的血糖记录"]
    assert stub.schema is RoutingDecision


async def test_orchestrator_prompt_carries_memory_files_and_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _stub_model(monkeypatch, RoutingDecision(primary="general"))

    await nodes.orchestrator(
        _state(
            memory_digest="## 关于这位用户\n- 目标：控制血糖",
            files_digest="- 血糖记录.csv（已解析）",
        )
    )

    prompt = stub.structured.prompts[0]
    assert "控制血糖" in prompt
    assert "血糖记录.csv（已解析）" in prompt
    assert "用户：我最近血糖怎么样" in prompt
    # 首轮没有「上一轮」段落。
    assert "上一轮" not in prompt


async def test_orchestrator_prompt_carries_the_gaps_on_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _stub_model(monkeypatch, RoutingDecision(primary="ds"))

    await nodes.orchestrator(
        _state(
            step=1,
            gaps=["缺用户近 14 天的血糖记录"],
            tried_queries=["取血糖"],
        )
    )

    prompt = stub.structured.prompts[0]
    assert "只补缺口" in prompt
    assert "- 取血糖" in prompt
    assert "- 缺用户近 14 天的血糖记录" in prompt


async def test_orchestrator_clears_supports_when_primary_is_general(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # general 分支根本不看 supports：留着它既不会生效，又会误导后面的反思。
    _stub_model(
        monkeypatch,
        RoutingDecision(primary="general", supports=["ds"], queries={"ds": "查血糖"}),
    )

    update = await nodes.orchestrator(_state())

    assert update["primary"] == "general"
    assert update["supports"] == []
    # ds 不再是参与者，给它重写的查询也一并丢掉。
    assert update["queries"] == {}


async def test_orchestrator_removes_primary_from_supports_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # supports 里留着 primary 会让它在「先辅后主」路径上被跑两遍。
    _stub_model(monkeypatch, RoutingDecision(primary="hc", supports=["ds", "ds", "hc"]))

    update = await nodes.orchestrator(_state())

    assert update["supports"] == ["ds"]


async def test_orchestrator_drops_queries_of_non_participants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_model(
        monkeypatch,
        RoutingDecision(
            primary="de",
            supports=["ds"],
            queries={
                "de": "解释二甲双胍的常见副作用",
                "ds": "汇总最近 30 天的血糖",
                "hc": "聊聊坚持不下去怎么办",
                "general": "你好",
                "tj": "越界的键",
            },
        ),
    )

    update = await nodes.orchestrator(_state())

    assert update["queries"] == {
        "de": "解释二甲双胍的常见副作用",
        "ds": "汇总最近 30 天的血糖",
    }


async def test_orchestrator_accumulates_sub_questions_across_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 子问题描述的是「用户这一句需要被回答什么」，回环不会改变它：第 2 轮覆盖
    # 第 1 轮只会让反思漏判。
    _stub_model(
        monkeypatch,
        RoutingDecision(primary="ds", sub_questions=["血糖趋势如何", "要不要调药"]),
    )

    update = await nodes.orchestrator(_state(sub_questions=["血糖趋势如何"]))

    assert update["sub_questions"] == ["血糖趋势如何", "要不要调药"]


async def test_orchestrator_falls_back_to_general_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_model(monkeypatch, RuntimeError("上游 500"))

    update = await nodes.orchestrator(_state())

    assert update["primary"] == "general"
    assert update["supports"] == []
    assert update["queries"] == {}


async def test_orchestrator_falls_back_when_primary_is_out_of_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 闭集由 ``RoutingDecision`` 的字面量类型守住：模型给了集合外的角色，这一份
    # 输出就整体不可用，降级而不是猜一个最近的角色。
    _stub_model(monkeypatch, {"primary": "营养师", "cuj": "咨询"})

    update = await nodes.orchestrator(_state())

    assert update["primary"] == "general"


# =====================================================================
# 收尾：反思短路与 finalize 的两种形态
# =====================================================================


def test_reflect_passes_the_general_path_without_asking_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 没有产出可评 = 这一轮不需要专业 Agent，也就没有缺口，直接放行：用一个会炸的
    # 客户端证明这里根本没问模型。
    monkeypatch.setattr(
        nodes, "get_orchestrator_llm", lambda: pytest.fail("不该问模型")
    )

    assert nodes.reflect(_state()) == {"sufficient": True}


async def test_finalize_forwards_the_draft_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 走过整合的路径上稿子就是正文，再让模型润一遍只会更慢、还可能引入新的幻觉。
    monkeypatch.setattr(nodes, "get_general_llm", lambda: pytest.fail("不该调模型"))

    assert await nodes.finalize(_state(draft="整合好的正文")) == {}


async def test_finalize_writes_the_answer_when_there_is_no_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StubStreamLLM(["你", "好"])
    monkeypatch.setattr(nodes, "get_general_llm", lambda: stub)

    update = await nodes.finalize(_state(primary="general"))

    assert update["draft"] == "你好"
    prompt = stub.prompts[0]
    assert isinstance(prompt[0], SystemMessage)
    assert "不要假装做过分析" in str(prompt[0].content)
    assert [message.content for message in prompt[1:]] == ["我最近血糖怎么样"]


async def test_finalize_puts_the_memory_digest_into_the_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _StubStreamLLM(["好的"])
    monkeypatch.setattr(nodes, "get_general_llm", lambda: stub)
    state = _state(primary="general", memory_digest="## 长期记忆\n- 有糖尿病")

    await nodes.finalize(state)

    assert "有糖尿病" in str(stub.prompts[0][0].content)


async def test_finalize_raises_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 一个字都没生成时不能编兜底话术，让 ``chat.py`` 走它既有的 error 分支。
    monkeypatch.setattr(
        nodes,
        "get_general_llm",
        lambda: _StubStreamLLM([], error=RuntimeError("上游 500")),
    )

    with pytest.raises(RuntimeError, match="上游 500"):
        await nodes.finalize(_state(primary="general"))


# =====================================================================
# 数据科学 Agent（单位 4：健康数据工具 + 小工具循环）
# =====================================================================


async def test_data_science_hits_no_model_without_a_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nodes, "_current_user_id", lambda: None)
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda _agent: pytest.fail("没有身份就不该问模型")
    )

    assert await nodes.data_science(_state(primary="ds")) == {}


async def test_data_science_turns_a_tool_call_into_a_contribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_agent_llm(
        monkeypatch,
        _StubToolLLM(
            [
                _tool_turn("read_health_metric", {"zh_name": "空腹血糖"}),
                AIMessageChunk(content="近一个月空腹血糖从 7.0 降到 6.4。"),
            ]
        ),
    )
    seen = _stub_health_tools(monkeypatch, sources=["体检报告.pdf"])
    events = _spy_blocks(monkeypatch)

    update = await nodes.data_science(
        _state(primary="ds", step=0, queries={"ds": "看近一个月的空腹血糖走势"})
    )

    contribution = update["contributions"][0]
    assert (contribution.agent, contribution.step, contribution.role) == (
        "ds",
        0,
        "primary",
    )
    assert contribution.content == "近一个月空腹血糖从 7.0 降到 6.4。"
    assert contribution.sources == ["体检报告.pdf"]
    assert seen == [{"zh_name": "空腹血糖"}]
    assert (
        "start",
        {
            "kind": "tool_call",
            "agent": "ds",
            "step": 0,
            "meta": {"tool": "read_health_metric"},
        },
    ) in events
    assert (
        "delta",
        {"text": 'read_health_metric({"zh_name": "空腹血糖"})', "agent": "ds"},
    ) in events
    assert events[-1] == ("stop", {"agent": "ds", "status": "complete"})
    # 工具返回的原文既交给模型、也发给用户，不二次加工
    tool_message = llm.messages[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "2026-10-01  6.7"
    assert "看近一个月的空腹血糖走势" in str(llm.messages[0][0].content)


async def test_data_science_falls_back_to_the_user_message_as_the_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_agent_llm(monkeypatch, _StubToolLLM([AIMessageChunk(content="好的")]))
    _stub_health_tools(monkeypatch)

    await nodes.data_science(_state(primary="ds", queries={}))

    assert "我最近血糖怎么样" in str(llm.messages[0][0].content)


async def test_the_sub_agent_gets_the_user_message_as_its_own_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 只发 system 的请求会被部分 provider（实测 DeepSeek-V3.2）回成空正文且不调工具。
    llm = _stub_agent_llm(monkeypatch, _StubToolLLM([AIMessageChunk(content="好的")]))
    _stub_health_tools(monkeypatch)

    await nodes.data_science(_state(primary="ds"))

    system, question = llm.messages[0]
    assert isinstance(system, SystemMessage)
    assert isinstance(question, HumanMessage)
    assert question.content == "我最近血糖怎么样"


async def test_data_science_hands_material_back_when_it_is_the_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_agent_llm(
        monkeypatch, _StubToolLLM([AIMessageChunk(content="近 7 天日均 7300 步。")])
    )
    _stub_health_tools(monkeypatch)

    update = await nodes.data_science(
        _state(
            primary="hc",
            step=1,
            contributions=[
                Contribution(
                    agent="hc",
                    step=0,
                    role="primary",
                    content="先聊睡眠。",
                    sources=[],
                )
            ],
            gaps=["缺最近两周的步数"],
        )
    )

    contribution = update["contributions"][0]
    assert (contribution.role, contribution.step) == ("support", 1)
    system = str(llm.messages[0][0].content)
    assert "先聊睡眠。" in system  # 局部通信：从共享 state 读主 Agent 的产物
    assert "缺最近两周的步数" in system


async def test_a_failing_tool_is_handed_back_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_agent_llm(
        monkeypatch,
        _StubToolLLM(
            [
                _tool_turn("read_health_metric", {"zh_name": "尿酸"}),
                AIMessageChunk(content="没取到尿酸数据。"),
            ]
        ),
    )
    _stub_health_tools(monkeypatch, error=RuntimeError("数据库连接超时"))
    events = _spy_blocks(monkeypatch)

    update = await nodes.data_science(_state(primary="ds"))

    assert "工具执行失败：数据库连接超时" in str(llm.messages[1][-1].content)
    assert update["contributions"][0].content == "没取到尿酸数据。"
    assert ("stop", {"agent": "ds", "status": "error"}) in events


async def test_an_unknown_tool_is_reported_back_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _stub_agent_llm(
        monkeypatch,
        _StubToolLLM(
            [
                _tool_turn("read_blood_pressure", {}),
                AIMessageChunk(content="手上没有这个工具。"),
            ]
        ),
    )
    _stub_health_tools(monkeypatch)

    update = await nodes.data_science(_state(primary="ds"))

    assert "没有名为 read_blood_pressure 的工具" in str(llm.messages[1][-1].content)
    assert update["contributions"][0].content == "手上没有这个工具。"


async def test_the_tool_loop_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _stub_agent_llm(
        monkeypatch,
        _StubToolLLM([_tool_turn("read_health_metric", {"zh_name": "空腹血糖"})]),
    )
    seen = _stub_health_tools(monkeypatch)

    assert await nodes.data_science(_state(primary="ds")) == {}
    assert len(seen) == nodes._MAX_TOOL_ROUNDS
    assert len(llm.messages) == nodes._MAX_TOOL_ROUNDS


async def test_data_science_produces_nothing_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_agent_llm(
        monkeypatch,
        _StubToolLLM([AIMessageChunk(content="x")], error=RuntimeError("上游 500")),
    )
    _stub_health_tools(monkeypatch)

    assert await nodes.data_science(_state(primary="ds")) == {}


async def test_data_science_produces_nothing_on_an_empty_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_agent_llm(monkeypatch, _StubToolLLM([AIMessageChunk(content="   ")]))
    _stub_health_tools(monkeypatch)

    assert await nodes.data_science(_state(primary="ds")) == {}


def test_render_gaps_is_empty_on_the_first_round() -> None:
    assert nodes._render_gaps(_state()) == ""


def test_material_carries_the_other_agents_and_the_gaps() -> None:
    material = nodes.render_material(
        _state(
            contributions=[
                Contribution(
                    agent="ds",
                    step=0,
                    role="support",
                    content="日均 7300 步。",
                    sources=["体检报告.pdf"],
                ),
                Contribution(
                    agent="hc",
                    step=1,
                    role="primary",
                    content="先从睡眠开始。",
                    sources=[],
                ),
            ]
        ),
        "hc",
    )

    assert "### 来自 ds（第 1 轮，支持 Agent）" in material
    assert "日均 7300 步。" in material
    assert "（依据：体检报告.pdf）" in material
    assert "先从睡眠开始。" not in material  # 自己不重复喂回去
    assert "我最近血糖怎么样" in material


def test_material_says_none_when_there_is_nothing() -> None:
    state = _state(messages=[], contributions=[])

    assert nodes.render_material(state, "ds") == "（暂无）"


def test_agent_source_covers_the_coach_subgraph_nodes() -> None:
    """子图里的模型分片带的是**内层**节点名，不登记就会被 ``chat.py`` 丢掉。"""
    inner = set(coach.build_coach_subgraph().get_graph().nodes) - {
        "__start__",
        "__end__",
    }

    assert inner
    for name in inner:
        assert g.NODE_AGENT_SOURCE[name] == "hc"


async def test_the_coach_subgraph_contributes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HC 子图接进外层图后，父状态里的材料只多这一份。

    子图把 ``contributions`` 声明成了不带 reducer 的普通 channel（见 ``coach`` 的模块
    docstring）：真接进来跑一遍，才能确认这份产出没有被 ``operator.add`` 加两遍。
    """
    _stub_router(
        monkeypatch, RoutingDecision(primary="hc", queries={"hc": "坚持不下来怎么调"})
    )
    llm = _StubStreamLLM(["这周先两次，每次二十分钟。"])
    monkeypatch.setattr(nodes, "get_agent_llm", lambda _agent: llm)

    async def _deliver(prompt: str) -> coach.CoachVerdict:
        del prompt
        return coach.CoachVerdict(verdict="deliver")

    def _noop(state: OrchestrationState) -> dict[str, Any]:
        del state
        return {}

    monkeypatch.setattr(coach, "_judge", _deliver)
    monkeypatch.setattr(nodes, "reflect", lambda _state: {"sufficient": True})
    monkeypatch.setattr(nodes, "finalize", _noop)
    monkeypatch.setattr(nodes, "memory", _noop)

    state = (
        await g.build_orchestration_graph()
        .compile()
        .ainvoke(
            {},
            {"recursion_limit": g.RECURSION_LIMIT},
            context=MemoryContext(user_id=uuid.uuid4()),
        )
    )

    contributions = list(state["contributions"])
    assert [item.agent for item in contributions] == ["hc"]
    assert contributions[0].content == "这周先两次，每次二十分钟。"
