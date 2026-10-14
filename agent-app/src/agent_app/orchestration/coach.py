"""健康教练（HC）子图：一轮教练回复的内部加工。

外层图只看到 ``health_coach`` 这一个节点，里面是三步：

    START → coach_reply → coach_check ─┬─ [可以交付 / 缺系统数据] ─→ coach_out → END
                 ↑                     └─ [要改 / 需要用户补充] ─┘
                 └──────── 重写（最多一次）─────────────┘

对应设计图（``C:\\Users\\huang\\Desktop\\pha orgin\\HC内部决策逻辑与工作流程图.png``）
的映射：``coach_reply`` = Open-ended Conversational Flow；``coach_check`` = 把图里的
Conclusion Module 与 Recommendation Module 合并成一次判断（都要看同一份草稿，分两次
调用是把同一份材料喂两遍）；图里的 Structured Context 读的是长期记忆、写的是回复落库
后的记忆抽取（``memory/extract.py``），Context Gathering / Feedback Incorporation
因此都不建节点。

三个设计要点：

1. **审核是显式的一步**，不是让起草节点自己保证质量。图里最值钱的那个判断是
   「现在该不该给建议」：用户情况不明（旧伤、疾病史、可支配时间、器械）就甩一套
   训练计划，是教练类 Agent 最常见的失败模式。
2. **需要用户补充的信息不在这里向用户提问**。HC 只把它写进 ``user_gaps``，由反射
   节点决定怎么问（``interrupt`` + ``Command`` 在那一侧，用户已定：第 4 步反思）。
3. **本子图的 channel 与父状态同名时必须当心 reducer**：父状态的 ``contributions``
   带 ``operator.add``（按轮累积），而子图把它声明成**不带 reducer** 的普通 channel，
   由 ``coach_out`` **只写自己那一份**——子图的输出是它全部 channel 的最终值，原样
   交给父节点；若在这里把读进来的材料一起写回，父状态里的材料会被累加第二遍。
   ``tried_queries`` 同理，所以子图**不声明**它（读不到就不渲染「已经取过什么」，
   HC 本来也不靠自己取数）。

硬上限 ``HC_MAX_STEPS = 2``：初次起草 + 最多一次重写。宁可交一份不完整但诚实的稿子，
也不要在一轮里反复自我修改，把用户的首字等待时间越拉越长。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict, cast

import structlog
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from agent_app.orchestration import blocks, nodes
from agent_app.orchestration.llm import get_agent_judge_llm
from agent_app.orchestration.prompts import (
    COACH_CHECK_PROMPT,
    COACH_REVISE_PROMPT,
    EMPTY_PLACEHOLDER,
    HC_PROMPT,
)
from agent_app.orchestration.state import (
    AgentCode,
    Contribution,
    OrchestrationState,
    PrimaryCode,
)

logger = structlog.get_logger(__name__)

#: 子图内部节点名。它们必须登记进 ``graph.NODE_AGENT_SOURCE``（值都是 ``"hc"``）：
#: 子图里模型分片与过程块带的是**内层**节点名，``chat.py`` 靠那张表把它们归到 HC。
COACH_REPLY = "coach_reply"
COACH_CHECK = "coach_check"
COACH_OUT = "coach_out"

#: 本子图负责的 Agent 代号。
_AGENT: AgentCode = "hc"

#: 一轮最多起草几次：初次 + 最多一次重写。
HC_MAX_STEPS = 2

#: 审核结论的两个取值。
_DELIVER = "deliver"
_REVISE = "revise"

#: 出错信息进日志前的截断长度（与 ``nodes`` 同一口径）。
_MAX_ERROR_CHARS = 200

#: 显式 path_map：少了它，画的图会认为「可能去任意节点」。
_REPLY_TARGETS: list[str] = [COACH_CHECK, COACH_OUT]
_CHECK_TARGETS: list[str] = [COACH_REPLY, COACH_OUT]


class CoachState(TypedDict, total=False):
    """HC 子图的状态。

    声明的键要覆盖它**读**的父状态字段：子图只拿得到自己 schema 里有的键，没声明的
    读进来就是空的。多出来的键是子图自己的中间结果。
    """

    # 从父状态读进来的（写成不带 reducer 的普通 channel，理由见模块 docstring）
    messages: Annotated[list[AnyMessage], add_messages]
    step: int
    primary: PrimaryCode | None
    queries: dict[str, str]
    sub_questions: list[str]
    memory_digest: str
    contributions: list[Contribution]
    gaps: list[str]

    # 子图的中间结果
    hc_draft: str
    hc_steps: int
    coach_verdict: str
    coach_questions: list[str]
    coach_gaps: list[str]
    coach_note: str

    # 交回父状态的两类缺口
    user_gaps: list[str]
    context_gaps: list[str]


class CoachVerdict(BaseModel):
    """审核节点一次调用的结论。

    ``questions`` 与 ``gaps`` 是**两个出口**，不要混：``questions`` 是只有用户本人
    答得上、只能靠问他的事（进 ``user_gaps``）；``gaps`` 是系统里能拿到、只是还没给到
    HC 的数据（进 ``context_gaps``，由外层回环派 DS / DE 补）。
    """

    verdict: str = Field(
        description="deliver：草稿可以直接交付；revise：按 reason 重写一次"
    )
    questions: list[str] = Field(
        default_factory=list,
        description="只有用户本人能答的问题，最多 2 条；没有就留空",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="系统里能补的数据（例如「近 14 天血糖记录」）；没有就留空",
    )
    reason: str = Field(default="", description="判 revise 的原因，用来指导重写")


def _view(state: CoachState) -> OrchestrationState:
    """把子图状态当成父状态交给 ``nodes`` 的共用函数。

    子图 schema 是父状态的子集（多出来的键都是子图自己的），运行时形状一致；这里
    显式断言只是为了让 ``nodes`` 保持原有类型，不必为一处复用放宽它的入参注解。
    """
    return cast("OrchestrationState", state)


def _clean(items: Sequence[str]) -> list[str]:
    """去掉空白项，保持原顺序。"""
    return [item.strip() for item in items if item.strip()]


def _query(state: CoachState) -> str:
    """编排器给 HC 重写过的查询；没有就回落到用户原话。"""
    return (
        (state.get("queries") or {}).get(_AGENT)
        or nodes.last_user_text(_view(state))
        or EMPTY_PLACEHOLDER
    )


def _render_revise_note(state: CoachState) -> str:
    """重写那轮塞进生成提示词的修改要求；首轮返回空串。"""
    reason = (state.get("coach_note") or "").strip()
    questions = _clean(state.get("coach_questions") or [])
    if not reason and not questions:
        return ""
    return COACH_REVISE_PROMPT.format(
        reason=reason or EMPTY_PLACEHOLDER,
        questions="\n".join(f"- {item}" for item in questions) or EMPTY_PLACEHOLDER,
    )


def _emit_revise_plan(state: CoachState, rounds: int) -> None:
    """重写时发一个 ``plan`` 块。

    重写要多花一次模型调用（用户要多等一段），这段时间要有东西可看；顺便让「模型
    自己说要不要改」这件事在过程展示里是可见的。
    """
    questions = _clean(state.get("coach_questions") or [])
    blocks.start_block(
        blocks.BLOCK_PLAN,
        agent=_AGENT,
        step=state.get("step", 0),
        meta={"phase": "revise", "round": rounds, "questions": len(questions)},
    )
    blocks.delta_block(f"第 {rounds} 次起草：按审核意见修改上一版。", agent=_AGENT)
    blocks.stop_block(agent=_AGENT)


async def coach_reply(state: CoachState) -> dict[str, Any]:
    """起草（或重写）这一轮的教练内容。

    模型报错或返回空正文时**不覆盖已有草稿**：重写失败就退回首版（``coach_out`` 会
    交回它），而不是把手里的稿子换成空。
    """
    step = state.get("step", 0)
    rounds = state.get("hc_steps", 0) + 1
    revise_note = ""
    if rounds > 1:
        revise_note = _render_revise_note(state)
        _emit_revise_plan(state, rounds)
    prompt = HC_PROMPT.format(
        memory=state.get("memory_digest") or EMPTY_PLACEHOLDER,
        role_hint=nodes.role_hint(nodes.agent_role(_AGENT, _view(state)), _view(state)),
        query=_query(state),
        sub_questions="\n".join(
            f"- {item}" for item in state.get("sub_questions") or []
        )
        or EMPTY_PLACEHOLDER,
        material=nodes.render_material(_view(state), _AGENT),
        revise_note=revise_note,
    )
    try:
        content = await nodes.run_agent(
            agent=_AGENT,
            prompt=prompt,
            question=nodes.last_user_text(_view(state)) or EMPTY_PLACEHOLDER,
            tools=[],
            step=step,
        )
    except Exception as exc:
        logger.warning(
            "coach_reply_failed",
            step=step,
            rounds=rounds,
            error=str(exc)[:_MAX_ERROR_CHARS],
        )
        return {"hc_steps": rounds}
    if not content.strip():
        logger.info("coach_reply_empty", step=step, rounds=rounds)
        return {"hc_steps": rounds}
    logger.info("coach_drafted", step=step, rounds=rounds, chars=len(content))
    return {"hc_steps": rounds, "hc_draft": content}


async def coach_check(state: CoachState) -> dict[str, Any]:
    """审核草稿：现在能不能直接交付用户，还缺什么。

    审核失败（模型报错、解析不出结论）一律当作「可以交付」：宁可直接交一版没改过的
    稿子，也不要因为审核环节出问题让用户拿不到回复。``revise`` 但没给出任何可用的
    修改信息也按同样口径降级——没有信息可用的重写只会把同一份稿子再写一遍。
    """
    step = state.get("step", 0)
    prompt = COACH_CHECK_PROMPT.format(
        query=_query(state),
        material=nodes.render_material(_view(state), _AGENT),
        draft=state.get("hc_draft") or "",
    )
    try:
        raw = await _judge(prompt)
        verdict = (
            raw if isinstance(raw, CoachVerdict) else CoachVerdict.model_validate(raw)
        )
    except Exception as exc:
        logger.warning(
            "coach_check_failed", step=step, error=str(exc)[:_MAX_ERROR_CHARS]
        )
        return {
            "coach_verdict": _DELIVER,
            "coach_questions": [],
            "coach_gaps": [],
            "coach_note": "",
        }
    questions = _clean(verdict.questions)
    gaps = _clean(verdict.gaps)
    reason = verdict.reason.strip()
    effective = verdict.verdict
    if effective == _REVISE and not (reason or questions or gaps):
        logger.info("coach_verdict_repaired", step=step, reason="empty_revise")
        effective = _DELIVER
    logger.info(
        "coach_checked",
        step=step,
        verdict=verdict.verdict,
        effective=effective,
        questions=len(questions),
        gaps=len(gaps),
    )
    return {
        "coach_verdict": effective,
        "coach_questions": questions,
        "coach_gaps": gaps,
        "coach_note": reason,
    }


async def coach_out(state: CoachState) -> dict[str, Any]:
    """交回父图：一份 ``Contribution`` + 两类缺口。

    ``contributions`` **只写自己这一份**（理由见模块 docstring）。没有草稿就交空列表：
    HC 这一轮没有产出，而不是编一段出来。

    两类缺口不管审核结论是什么都照实上报——``questions`` / ``gaps`` 是模型指出的真实
    缺口，即使它同时说了「可以交付」也不该丢掉。
    """
    step = state.get("step", 0)
    draft = (state.get("hc_draft") or "").strip()
    questions = _clean(state.get("coach_questions") or [])
    gaps = _clean(state.get("coach_gaps") or [])
    if state.get("coach_verdict") == _REVISE:
        # 次数用完了：交回未修改的版本，缺口由反射节点决定怎么补。
        logger.info(
            "coach_delivered_unrevised", step=step, reason=state.get("coach_note")
        )
    if not draft:
        logger.info("coach_produced_nothing", step=step)
        return {"contributions": [], "user_gaps": questions, "context_gaps": gaps}
    role = nodes.agent_role(_AGENT, _view(state))
    logger.info(
        "coach_produced",
        step=step,
        role=role,
        chars=len(draft),
        questions=len(questions),
        gaps=len(gaps),
    )
    return {
        "contributions": [
            nodes.make_contribution(
                agent=_AGENT,
                step=step,
                role=role,
                content=draft,
                # HC 不取外部数据：稿子里的数字来自其他 Agent 的材料，来源已经记在
                # 那份材料上，这里再抄一份只会让整合节点看到两份同样的依据。
                sources=[],
            )
        ],
        "user_gaps": questions,
        "context_gaps": gaps,
    }


async def _judge(prompt: str) -> CoachVerdict:
    """问一次审核模型。单独抽出来是为了让测试能换掉它（节点函数本身改不动）。"""
    model = get_agent_judge_llm(_AGENT).with_structured_output(CoachVerdict)
    raw = await model.ainvoke(prompt)
    return raw if isinstance(raw, CoachVerdict) else CoachVerdict.model_validate(raw)


def route_after_reply(state: CoachState) -> list[str]:
    """没有草稿就直接交回，不让审核节点去评一份空稿。"""
    if (state.get("hc_draft") or "").strip():
        return [COACH_CHECK]
    return [COACH_OUT]


def route_after_check(state: CoachState) -> list[str]:
    """要改、而且还有重写次数 → 回起草；否则交回。"""
    verdict = state.get("coach_verdict")
    if verdict == _REVISE and state.get("hc_steps", 0) < HC_MAX_STEPS:
        return [COACH_REPLY]
    return [COACH_OUT]


def build_coach_subgraph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """编译 HC 子图。

    不带 checkpointer：持久化由外层图统一管（``graph.py`` 接入 ``chat.py`` 时才把
    ``get_checkpointer()`` 传进 ``compile()``），子图跟着外层走就行。
    """
    graph: StateGraph[CoachState, None, CoachState, CoachState] = StateGraph(CoachState)
    graph.add_node(COACH_REPLY, coach_reply)
    graph.add_node(COACH_CHECK, coach_check)
    graph.add_node(COACH_OUT, coach_out)
    graph.add_edge(START, COACH_REPLY)
    graph.add_conditional_edges(COACH_REPLY, route_after_reply, _REPLY_TARGETS)
    graph.add_conditional_edges(COACH_CHECK, route_after_check, _CHECK_TARGETS)
    graph.add_edge(COACH_OUT, END)
    return graph.compile()
