"""图编排的节点实现。

**当前进度**：``intake`` / ``orchestrator`` 已实现（单位 2，编排决策）；``finalize``
已实现两种形态（转发 ``draft`` / 现生成正文），``reflect`` 只实现了「没有产出可评时
直接判够」的短路（流程调整），三条判据待单位 7；三个子 Agent 与 ``integrate`` /
``memory`` 仍是空操作。

实现某个节点时**只改它自己的函数体**，不要动 ``graph.py`` 里的连线；唯一例外是
``next_round``——它在骨架阶段就实现完了（理由见该函数）。

每个节点遵守同一条分工：能用确定性代码可靠完成的事（取数、渲染、校验、降级）
放在 Python 里做完，模型只负责判断与生成。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.runtime import get_runtime
from sqlalchemy import select

from agent_app.core.database import async_session_maker
from agent_app.memory.middleware import MemoryContext
from agent_app.memory.render import render_memories
from agent_app.memory.store import fetch_memories
from agent_app.models.file import FileMetadata
from agent_app.orchestration import blocks
from agent_app.orchestration.llm import get_general_llm, get_orchestrator_llm
from agent_app.orchestration.prompts import (
    EMPTY_PLACEHOLDER,
    GENERAL_PROMPT,
    ORCHESTRATOR_PROMPT,
    RETRY_CONTEXT,
)
from agent_app.orchestration.state import (
    AgentCode,
    OrchestrationState,
    RoutingDecision,
)

logger = structlog.get_logger(__name__)

#: 本模块节点写进过程块 ``meta.agent`` 的代号。必须与 ``graph.NODE_AGENT_SOURCE``
#: 里这些节点的取值一致——``graph`` 导入 ``nodes``，所以这里不能反向 import 那张表。
_AGENT_SOURCE = "orchestrator"

#: 出错信息进日志前的截断长度（与 ``memory/extract.py`` 同一口径）。
_MAX_ERROR_CHARS = 200

#: 编排器能看到的历史消息条数。路由是局部判断，喂全量历史只会被稀释。
_HISTORY_MESSAGES = 6

#: 单条历史进提示词的字符上限。一轮里贴一大段数据的情况很常见，不截断会吃掉整个窗口。
_HISTORY_MESSAGE_CHARS = 400

#: 文件清单最多列几个（最近上传的优先）。
_FILES_LIMIT = 10

#: 文件状态的中文标注。编排器需要知道那份文件到底能不能用（``failed`` 等于没数据）。
#: 键写成 ``str`` 而不是 ``FileStatus``：库里读回来的就是普通字符串，声明成字面量
#: 反而要在取值处到处 cast。
_FILE_STATUS_LABELS: dict[str, str] = {
    "pending": "待解析",
    "parsing": "解析中",
    "parsed": "已解析",
    "failed": "解析失败",
    "confirmed": "已确认",
}


# =====================================================================
# 公共工具
# =====================================================================


def _current_user_id() -> uuid.UUID | None:
    """从图运行的 context 里取 ``user_id``。

    ``user_id`` 只能从 context 来，绝不从模型输出或用户输入来——与
    ``memory/middleware.py``、``memory/store.py`` 是同一套纪律。非图执行环境
    （单元测试直接调节点函数）拿不到 runtime，返回 ``None``：记忆与文件清单都是
    旁路，取不到就当没有，不能让整轮对话挂掉。
    """
    try:
        context = get_runtime(MemoryContext).context
    except Exception:
        return None
    return None if context is None else context.user_id


async def _load_memory_digest(user_id: uuid.UUID) -> str:
    """渲染要注入编排器的长期记忆；查不到就当没有。"""
    try:
        return render_memories(await fetch_memories(user_id))
    except Exception as exc:
        logger.warning(
            "orchestrator_memory_failed",
            user_id=str(user_id),
            error=str(exc)[:_MAX_ERROR_CHARS],
        )
        return ""


async def _load_files_digest(user_id: uuid.UUID) -> str:
    """取该用户最近上传的文件清单。

    按 ``user_id`` 过滤是硬要求，不是优化：漏掉这个条件就是跨用户泄漏
    （见 ``memory/store.py`` 的模块 docstring）。
    """
    statement = (
        select(FileMetadata.original_name, FileMetadata.status)
        .where(FileMetadata.user_id == user_id)
        .order_by(FileMetadata.created_at.desc())
        .limit(_FILES_LIMIT)
    )
    try:
        async with async_session_maker() as db:
            rows = (await db.execute(statement)).all()
    except Exception as exc:
        logger.warning(
            "orchestrator_files_failed",
            user_id=str(user_id),
            error=str(exc)[:_MAX_ERROR_CHARS],
        )
        return ""
    return _render_files([(row[0], row[1]) for row in rows])


def _render_files(rows: Sequence[tuple[str, str]]) -> str:
    """把 ``(原始文件名, 状态)`` 渲染成提示词里的一段清单；没有文件时返回空串。"""
    if not rows:
        return ""
    return "\n".join(
        f"- {name}（{_FILE_STATUS_LABELS.get(status, status)}）"
        for name, status in rows
    )


def _render_history(messages: Sequence[AnyMessage]) -> str:
    """把最近几轮对话压成几行文本。

    只取最后 ``_HISTORY_MESSAGES`` 条，并丢掉 system（它进不了这里：图里传的是
    账本重建出来的 user / assistant 正文）。
    """
    recent = [m for m in messages if not isinstance(m, SystemMessage)][
        -_HISTORY_MESSAGES:
    ]
    lines: list[str] = []
    for message in recent:
        text = _message_text(message)
        if not text:
            continue
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        lines.append(f"{role}：{text}")
    return "\n".join(lines)


def _message_text(message: AnyMessage) -> str:
    """取消息正文并压成单行；多模态分块只保留文本部分。"""
    content = message.content
    if isinstance(content, str):
        text = content
    else:
        text = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return " ".join(text.split())[:_HISTORY_MESSAGE_CHARS]


def _render_attempt(state: OrchestrationState) -> str:
    """回环重入时补上一段「上一轮试过什么、还缺什么」；首轮返回空串。"""
    tried = state.get("tried_queries") or []
    gaps = state.get("gaps") or []
    if not tried and not gaps:
        return ""
    return RETRY_CONTEXT.format(
        tried="\n".join(f"- {query}" for query in tried) or "（无）",
        gaps="\n".join(f"- {gap}" for gap in gaps) or "（无）",
    )


def _repair(decision: RoutingDecision) -> RoutingDecision:
    """修掉决策里的自相矛盾，**不改语义**。

    只做三件确定性的事：``supports`` 去重、踢掉 primary 自己；``primary`` 是
    ``general`` 时清空 ``supports``（否则它会静默失效——``route_from_orchestrator``
    在 ``general`` 分支根本不看它——却会误导后面的反思）；``queries`` 只留参与者
    的键，并丢掉空串。

    「这个任务该不该给 ds」这类语义判断一律不做：那是模型的活。模型判错了，改的
    是提示词，不是在代码里补规则。
    """
    supports: list[AgentCode] = list(dict.fromkeys(decision.supports))
    supports = [agent for agent in supports if agent != decision.primary]
    if decision.primary == "general":
        supports = []
    participants = {decision.primary, *supports}
    queries = {
        key: value.strip()
        for key, value in decision.queries.items()
        if key in participants and value.strip()
    }
    if supports == decision.supports and queries == decision.queries:
        return decision
    logger.info(
        "orchestrator_decision_repaired",
        primary=decision.primary,
        raw_supports=list(decision.supports),
        supports=supports,
        raw_query_keys=list(decision.queries),
        query_keys=list(queries),
    )
    return decision.model_copy(update={"supports": supports, "queries": queries})


async def _decide(prompt: str) -> RoutingDecision:
    """问一次编排器，拿回一个合法决策。

    单独抽出来是为了让单元测试有地方下桩：``nodes.orchestrator`` 是图编译时按引用
    绑定的，测试改不动它，但改得动它调的这条路径。
    """
    model = get_orchestrator_llm().with_structured_output(RoutingDecision)
    raw = await model.ainvoke(prompt)
    decision = (
        raw if isinstance(raw, RoutingDecision) else RoutingDecision.model_validate(raw)
    )
    return _repair(decision)


def _decision_state(
    decision: RoutingDecision, state: OrchestrationState
) -> dict[str, Any]:
    """把决策摊平成 state 更新。

    ``sub_questions`` 取并集：它描述的是「用户这一句需要被回答什么」，回环不会改变
    它，让第 2 轮覆盖第 1 轮只会让反思漏判。``tried_queries`` 带 ``operator.add``
    的 reducer，返回列表就是追加。
    """
    sub_questions = list(
        dict.fromkeys([*(state.get("sub_questions") or []), *decision.sub_questions])
    )
    return {
        "cuj": decision.cuj,
        "primary": decision.primary,
        "supports": decision.supports,
        "queries": decision.queries,
        "sub_questions": sub_questions,
        "routing_reason": decision.reason,
        "tried_queries": list(decision.queries.values()),
    }


def _general_only() -> dict[str, Any]:
    """降级：编排失败时按通用对话处理，这一轮至少还能答上一句话。

    编排是控制节点，它挂掉不该让整轮对话挂掉。错误只进日志——用户看到的应该是
    一个正常回答，而不是一条内部栈。
    """
    return {
        "cuj": None,
        "primary": "general",
        "supports": [],
        "queries": {},
        "routing_reason": "",
    }


# =====================================================================
# 阶段 1：接收数据
# =====================================================================


async def intake(state: OrchestrationState) -> dict[str, Any]:
    """接收用户消息，把本轮要用的上下文准备好。

    **只做组装，不做判断**：是否需要数据、需要哪些数据都由编排器与各个 Agent 自己
    决定。这一步能确定的事只有两件——长期记忆和已上传文件清单——它们与意图无关，
    查一次就能给后面所有轮次复用（回环是 ``next_round → orchestrator``，不会再经过
    这里）。

    真正的取数不在这里，而是做成 Agent 可调用的工具（``user_id`` 由 ``context``
    注入、SQL 强制按 ``user_id`` 过滤，与 ``memory/store.py`` 同一套纪律）：
    「该取什么」依赖意图理解，前置盲取要么取全量撞压缩阈值，要么取少了返工。
    """
    user_id = _current_user_id()
    if user_id is None:
        logger.warning("orchestrator_context_missing")
        return {}
    return {
        "memory_digest": await _load_memory_digest(user_id),
        "files_digest": await _load_files_digest(user_id),
    }


# =====================================================================
# 阶段 2：编排器
# =====================================================================


async def orchestrator(state: OrchestrationState) -> dict[str, Any]:
    """编排器：推断主 / 支持 Agent、确定协作工作流、为每个 Agent 重写查询。

    一次结构化输出吐出整个决策（``RoutingDecision``），不解析自由文本；失败、
    超时、输出不合 schema 一律降级成通用对话，只记 warning。
    """
    prompt = ORCHESTRATOR_PROMPT.format(
        memories=state.get("memory_digest") or EMPTY_PLACEHOLDER,
        files=state.get("files_digest") or EMPTY_PLACEHOLDER,
        history=_render_history(state.get("messages") or []) or EMPTY_PLACEHOLDER,
        attempt=_render_attempt(state),
    )
    try:
        decision = await _decide(prompt)
    except Exception as exc:
        logger.warning(
            "orchestrator_failed",
            step=state.get("step", 0),
            error=str(exc)[:_MAX_ERROR_CHARS],
        )
        return _general_only()
    logger.info(
        "orchestrator_decided",
        step=state.get("step", 0),
        cuj=decision.cuj,
        primary=decision.primary,
        supports=decision.supports,
        reason=decision.reason,
    )
    return _decision_state(decision, state)


# =====================================================================
# 阶段 2：三个子 Agent（主 / 辅由 ``primary`` / ``supports`` 决定，不写死在节点上）
# =====================================================================


def data_science(state: OrchestrationState) -> dict[str, Any]:
    """数据科学 Agent：多模态健康数据分析、统计计算。

    落地时要做的：

    * 用 ``state["queries"]`` 里重写后的查询去查个人健康数据
      （``health_metrics`` / ``health_records``，**没有读接口，要新写工具**）；
    * 产出追加进 ``contributions``（返回 ``{"contributions": [...]}``）；
    * 第一版**不做 Docker 沙箱、不出图表**（用户已定：先把流程跑通），
      只做「查库 + 基于查询结果分析」。
    """
    return {}


def domain_expert(state: OrchestrationState) -> dict[str, Any]:
    """健康领域专家 Agent：医学知识 + 个人健康记录，给出循证解释。

    落地时要做的：

    * 数据不够时**联网搜索**（用户已定：暂时不做 RAG，改为增加联网检索能力）；
    * 产出**必须带 ``sources``**（命中的链接），整合节点要靠它核对「每个数值断言
      都能追到一次真实命中」；
    * 提示词硬约束：不做临床诊断、必须建议就医确诊、所有建议附医疗免责声明。
    """
    return {}


def health_coach(state: OrchestrationState) -> dict[str, Any]:
    """健康教练 Agent：行为改变、动机性访谈、制定长期计划。

    落地时要做的：对话状态分析 → 教练策略选择（MI / CBT / SMART）→ 共情回应生成
    → 目标跟踪更新（新目标 / 里程碑 / 障碍）。语气温暖、支持性、非评判性。
    """
    return {}


# =====================================================================
# 阶段 3：结果整合
# =====================================================================


def integrate(state: OrchestrationState) -> dict[str, Any]:
    """结果整合：把本轮所有 ``contributions`` 合成一份稿子，并给出缺口清单。

    落地时要做的：

    * 全局语气统一 + 冲突消解（多份产出互相矛盾时以证据更强的为准，并在稿子里
      说明分歧）；
    * 一次调用同时产出 ``draft`` 与 ``gaps``——「整合」和「反思」在职责上本来就
      重叠，分两次调用只会把同一份材料喂两遍；
    * ``gaps`` 要具体到可执行（「缺用户近 14 天血糖记录」而不是「信息不足」）。
    """
    return {}


# =====================================================================
# 阶段 4：反思
# =====================================================================


def reflect(state: OrchestrationState) -> dict[str, Any]:
    """反思：信息是否足够回复用户；不够就把缺口带回编排器重跑。

    落地时要做的（三条判据，用户已定要「轮数 + 判据」双控）：

    * **意图覆盖**：编排器拆出的每个子问题在稿子里都有对应结论；
    * **数据可溯**：稿子里每个数值断言都能追到本轮的一次真实工具返回——
      对不上就算缺口。这一条**至少一半靠规则**（正则抓「数值 + 单位」比对本轮
      工具结果集合），不能全靠模型自评；
    * **无未解决冲突**：多份产出之间没有互相矛盾的结论。

    写回的字段：``sufficient``（够不够）、``gaps``（缺口）、``stalled``
    （本轮缺口与上一轮重合 → 再问也拿不到，立即停）。

    **不要由主 Agent 自评**：那样它永远说自己够了。
    """
    if not state.get("contributions"):
        # 没有产出可评：说明这一轮编排器判「不需要分配任务」，没有稿子也没有缺口，
        # 「不需要专业 Agent」本身就等于「现有信息够答」，所以直接放行，**不调模型**。
        return {"sufficient": True}
    return {}


def next_round(state: OrchestrationState) -> dict[str, Any]:
    """回环推进：轮次 +1，并清掉上一轮的结论。

    这是**唯一**改 ``state["step"]`` 的地方，所以它不属于「待实现的节点细节」，
    骨架阶段就实现掉——否则路由函数读不到新轮次，回环会退化成死循环。

    清 ``sufficient`` / ``stalled`` 是必要的：它们是上一轮反思的结论，留着会让
    新的一轮在还没反思之前就走到结束分支。
    """
    return {
        "step": state.get("step", 0) + 1,
        "sufficient": False,
        "stalled": False,
    }


# =====================================================================
# 阶段 5 / 6：记忆与回复
# =====================================================================


def memory(state: OrchestrationState) -> dict[str, Any]:
    """记忆：总结本次任务，决定是否写入 / 更新长期记忆。

    与 ``finalize`` **并行**跑，读的是 ``state["draft"]``（整合节点已经产出的最终稿），
    所以不需要等回复发完。

    **注意**：同一步被调度的两个节点互相看不见对方的产物，所以「不需要分配任务」
    那条路径上 ``finalize`` 现生成的 ``draft`` 这个节点读不到（那时 ``draft`` 是空的）。

    落地时**必须先解决一个冲突**：``api/v1/chat.py`` 现在已经在回复落库后调
    ``spawn_memory_extraction(...)`` 做同一件事。二选一——把抽取挪进来、或者删掉
    这个节点——否则每轮会抽两次。
    """
    return {}


async def finalize(state: OrchestrationState) -> dict[str, Any]:
    """回复用户：产出这一轮唯一一份用户可见的正文。

    两种形态，区别只在有没有走过整合节点：

    * ``state["draft"]`` 有值 → **只转发**，不再调模型（稿子已经是写好的正文）；
    * 没有 ``draft``（``orchestrator`` 判「不需要分配任务」→ ``reflect`` → 这里）
      → 用流式对话模型现生成，见 ``_answer_directly``。

    这是全图**唯一**产出用户可见正文（``text`` 块）的节点。其余节点只产过程块
    （``thinking`` / ``plan`` / ``tool_call`` / ``tool_result``），它们的内容不进账本、
    不回喂模型。
    """
    draft = state.get("draft")
    if draft:
        blocks.emit_text(draft, agent=_AGENT_SOURCE, step=state.get("step", 0))
        return {}
    return {"draft": await _answer_directly(state)}


async def _answer_directly(state: OrchestrationState) -> str:
    """由通用对话模型直接作答，边生成边把分片发成正文块。

    ``state["messages"]`` 里已经有当前这条用户消息（``chat.py`` 调图时传进来的），
    所以这里只补一条 system，不再重复贴历史。

    模型失败时直接抛出：正文一个字都没生成，应该让 ``chat.py`` 走它既有的
    ``error`` 分支，而不是编一段兜底话术糊弄用户。块要按 ``error`` 收尾，
    否则前端会留一个永远转圈的块。
    """
    step = state.get("step", 0)
    system = GENERAL_PROMPT
    if digest := state.get("memory_digest"):
        # 长期记忆只能从 context 进来（见 ``memory/middleware.py``），这里用的是
        # ``intake`` 已经渲染好的那一份，不重复查库。
        system = f"{system}\n\n{digest}"
    prompt: list[AnyMessage] = [SystemMessage(content=system)]
    prompt.extend(state.get("messages") or [])
    parts: list[str] = []
    blocks.start_block(blocks.BLOCK_TEXT, agent=_AGENT_SOURCE, step=step)
    try:
        async for chunk in get_general_llm().astream(prompt):
            if text := chunk.text:
                parts.append(text)
                blocks.delta_block(text, agent=_AGENT_SOURCE)
    except Exception:
        blocks.stop_block(agent=_AGENT_SOURCE, status=blocks.STATUS_ERROR)
        raise
    blocks.stop_block(agent=_AGENT_SOURCE)
    return "".join(parts)
