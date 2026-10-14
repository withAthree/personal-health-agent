"""图编排的骨架：节点、边与路由。

拓扑（对齐设计文档的六步）::

    START → intake → orchestrator
      orchestrator ─┬─ [不需要分配任务] ─────────────────→ reflect
                    ├─ [只给主 Agent] ────────────────────→ integrate
                    └─ [支持 Agents 先跑，再跑主 Agent] ──→ integrate
      integrate → reflect ─┬─ 足够 / 到上限 / 缺口不收敛 ─┬─→ finalize ─┐
                           │                              └─→ memory  ─┤
                           └─ 不够 ─→ next_round ─→ orchestrator      └→ END

「不需要分配任务」时跳过第 3 步（没有产出要合），但**第 4 步反思照过**：它是每一轮
都要过的判定点，``reflect`` 对空产出直接判「够」（不发模型调用），于是进 ``finalize``
现生成正文。

    START → intake → orchestrator
      orchestrator ─┬─ [不需要分配任务] ─────────────────→ reflect
                    ├─ [只给主 Agent] ────────────────────→ integrate
                    └─ [支持 Agents 先跑，再跑主 Agent] ──→ integrate
      integrate → reflect ─┬─ 足够 / 到上限 / 缺口不收敛 ─┬─→ finalize ─┐
                           │                              └─→ memory  ─┤
                           └─ 不够 ─→ next_round ─→ orchestrator      └→ END

其中 ``health_coach`` 不是普通节点而是一个**子图**（``orchestration/coach.py``，见该模块
docstring 的内环拓扑）：一轮教练回复内部有「起草 → 审核 → 最多重写一次」的多步与
自己的停止条件。它的内部节点名也要登记进 ``NODE_AGENT_SOURCE``。

三条贯穿全图的约定（改 ``api/v1/chat.py`` 时按这个来）：

1. **节点名 → agent_source**：见 ``NODE_AGENT_SOURCE``。现在 ``chat.py`` 用
   ``if metadata["langgraph_node"] != "model"`` 硬过滤流式分片，多 Agent 之后
   每个节点都在吐分片，那个常量必然要换成这张表；
2. **轮次**：``state["step"]``，由 ``nodes.next_round`` 唯一维护，写进过程块的
   ``block-start.step``，前端按它把回环的每一轮分开渲染；
3. **用户可见正文只有一处**：``finalize``。其余节点只产过程块。

调用图时必须传 ``config={"recursion_limit": RECURSION_LIMIT, ...}``，理由见该常量。

骨架阶段可以真编译、真跑（节点本体是空操作，路由是完整的），所以拓扑可以直接用
``get_orchestration_graph().get_graph()`` 渲染出来核对。
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_app.memory.middleware import MemoryContext
from agent_app.orchestration import coach, nodes
from agent_app.orchestration.state import (
    MAX_ROUNDS,
    AgentCode,
    OrchestrationState,
    PrimaryCode,
)

# =====================================================================
# 节点名
# =====================================================================

INTAKE = "intake"
ORCHESTRATOR = "orchestrator"
DATA_SCIENCE = "data_science"
DOMAIN_EXPERT = "domain_expert"
HEALTH_COACH = "health_coach"
INTEGRATE = "integrate"
REFLECT = "reflect"
NEXT_ROUND = "next_round"
FINALIZE = "finalize"
MEMORY = "memory"

#: Agent 代号 → 节点名。
AGENT_NODE: dict[AgentCode, str] = {
    "ds": DATA_SCIENCE,
    "de": DOMAIN_EXPERT,
    "hc": HEALTH_COACH,
}

#: 节点名 → 写进账本 ``messages.agent_source`` 与过程块 ``meta.agent`` 的代号。
#:
#: 非 Agent 节点（编排、整合、反思、收尾）都归到 ``orchestrator``：它们是同一条
#: 控制链的实现细节，对用户而言就是「系统自己在做事」。``memory`` 单独留一个值，
#: 它跑在后台、和回复并行，出问题时需要能把它和用户的等待路径分开看。
NODE_AGENT_SOURCE: dict[str, str] = {
    INTAKE: "orchestrator",
    ORCHESTRATOR: "orchestrator",
    DATA_SCIENCE: "ds",
    DOMAIN_EXPERT: "de",
    HEALTH_COACH: "hc",
    INTEGRATE: "orchestrator",
    REFLECT: "orchestrator",
    NEXT_ROUND: "orchestrator",
    FINALIZE: "orchestrator",
    MEMORY: "memory",
    # HC 子图的**内部**节点：子图里模型分片与过程块带的是内层节点名，
    # 也要在这张表里各占一条，否则 ``chat.py`` 会把它们当成不认识的节点丢掉。
    coach.COACH_REPLY: "hc",
    coach.COACH_CHECK: "hc",
    coach.COACH_OUT: "hc",
}


# =====================================================================
# 运行参数
# =====================================================================

#: 图执行的步数上限，调用时通过 ``config={"recursion_limit": RECURSION_LIMIT}`` 传入。
#:
#: 不能省。LangGraph 默认 25，而最坏路径（主 + 两个支持 Agent，反思一直说不够，
#: 跑满 ``MAX_ROUNDS`` 次回环）在节点全是空操作时就实测要 26 步——4 轮 × 6 个
#: superstep（编排、支持 Agent、主 Agent、整合、反思、next_round，最后一轮没有
#: next_round）+ finalize + memory，**默认值正好差一步**。真实节点内部还要调工具，
#: 每次工具往返都要再走一步；HC 子图的每个节点也计一步（一轮教练最多 3 步），
#: 所以留一倍余量。
RECURSION_LIMIT = 50


# =====================================================================
# 路由
# =====================================================================


def route_from_orchestrator(state: OrchestrationState) -> list[str]:
    """编排器之后的分发，对应「推断主 / 支持 Agent」的三种情况。

    1. 不分配任务（``primary`` 是 ``general``）→ **跳过整合、直接进反思**：没有产出
       要合，但第 4 步是每轮都要过的判定点，由它给出结论再去 ``finalize``；
    2. 只给主 Agent（``supports`` 为空）→ 直接进主 Agent；
    3. 主 + 支持 → **只把支持 Agent 派出去**，主 Agent 由支持 Agent 的出边
       （``make_route_from_agent``）在它们全部就绪后才触发。

    第 3 条就是「先辅后主」，代价是主 Agent 的启动要等支持 Agent，首字延迟变长；
    换来的是支持 Agent 的产物在主 Agent 开工前就已经在 state 里——否则主 Agent
    只能凭残缺上下文开工，产物再交给整合节点融合，主 / 辅关系就退化成
    「多个平等的 Agent + 一个整合器」，两步并成一步。
    """
    primary: PrimaryCode = state.get("primary") or "general"
    if primary == "general":
        return [REFLECT]
    supports = [AGENT_NODE[code] for code in state.get("supports") or []]
    return supports or [AGENT_NODE[primary]]


def make_route_from_agent(
    agent: AgentCode,
) -> Callable[[OrchestrationState], list[str]]:
    """给某个 Agent 节点造出边路由：自己不是主 Agent 就去主 Agent，是主就去整合。

    支持 Agent 全部在执行完的同一步里触发主 Agent，LangGraph 一个节点在一步里
    只会被调度一次，所以两个支持 Agent 不会让主 Agent 跑两遍。

    ``path_map`` 里刻意不含自己（``data_science`` 的出边不会再回到
    ``data_science``），因为本函数永远不会返回自己。
    """

    def route(state: OrchestrationState) -> list[str]:
        primary = state.get("primary")
        if primary is not None and primary != "general" and primary != agent:
            return [AGENT_NODE[primary]]
        return [INTEGRATE]

    return route


def route_from_reflect(state: OrchestrationState) -> list[str]:
    """反思之后的去向：结束（回复 ‖ 记忆）还是回环补缺口。

    三个结束条件任何一个成立就停，**不是只数轮数**：

    * ``sufficient``：信息够了；
    * ``stalled``：本轮缺口和上一轮重合，说明这份信息根本拿不到（比如用户压根
      没上传那份数据），再循环只是白烧 token，该走降级回答；
    * ``step >= MAX_ROUNDS``：硬上限兜底。
    """
    step = state.get("step", 0)
    if state.get("sufficient") or state.get("stalled") or step >= MAX_ROUNDS:
        return [FINALIZE, MEMORY]
    return [NEXT_ROUND]


# 每条条件边的可能去向。显式给出来，否则图可视化会认为「可能去任意节点」。
_ORCHESTRATOR_TARGETS = [REFLECT, DATA_SCIENCE, DOMAIN_EXPERT, HEALTH_COACH]
_REFLECT_TARGETS = [FINALIZE, MEMORY, NEXT_ROUND]


# =====================================================================
# 组装
# =====================================================================


def build_orchestration_graph() -> StateGraph[
    OrchestrationState, MemoryContext, OrchestrationState, OrchestrationState
]:
    """搭出图结构（未编译）。

    ``context_schema=MemoryContext`` 现在就声明：记忆只能从 ``context`` 进来
    （理由见 ``memory/middleware.py``），图的调用方必须传当前用户。

    **不加 checkpointer**：``get_checkpointer()`` 构造时会绑定当前运行的事件循环，
    不能在 import / 构图阶段调用。等接入 ``api/v1/chat.py`` 时再在 ``compile()``
    里传。
    """
    graph = StateGraph(OrchestrationState, context_schema=MemoryContext)

    graph.add_node(INTAKE, nodes.intake)
    graph.add_node(ORCHESTRATOR, nodes.orchestrator)
    graph.add_node(DATA_SCIENCE, nodes.data_science)
    graph.add_node(DOMAIN_EXPERT, nodes.domain_expert)
    graph.add_node(HEALTH_COACH, coach.build_coach_subgraph())
    graph.add_node(INTEGRATE, nodes.integrate)
    graph.add_node(REFLECT, nodes.reflect)
    graph.add_node(NEXT_ROUND, nodes.next_round)
    graph.add_node(FINALIZE, nodes.finalize)
    graph.add_node(MEMORY, nodes.memory)

    graph.add_edge(START, INTAKE)
    graph.add_edge(INTAKE, ORCHESTRATOR)
    graph.add_conditional_edges(
        ORCHESTRATOR, route_from_orchestrator, _ORCHESTRATOR_TARGETS
    )

    # 三个 Agent 的出边：先辅后主，主 Agent 直接进整合。
    for code, node in AGENT_NODE.items():
        targets = [INTEGRATE] + [
            other for other in AGENT_NODE.values() if other != node
        ]
        graph.add_conditional_edges(node, make_route_from_agent(code), targets)

    graph.add_edge(INTEGRATE, REFLECT)
    graph.add_conditional_edges(REFLECT, route_from_reflect, _REFLECT_TARGETS)
    graph.add_edge(NEXT_ROUND, ORCHESTRATOR)
    graph.add_edge(FINALIZE, END)
    graph.add_edge(MEMORY, END)

    return graph


@lru_cache(maxsize=1)
def get_orchestration_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """返回编译后的图单例。

    与 ``llm/agent.py`` 的 ``get_agent()`` 同样是 ``lru_cache`` 单例：骨架阶段
    没有 checkpointer、没有按用户变化的状态，单例是安全的。**接入 ``chat.py``
    时要把 ``get_checkpointer()`` 传进 ``compile()``，届时这个单例仍然成立**
    ——checkpointer 本身就是进程级单例，图本身不持有用户数据。
    """
    return build_orchestration_graph().compile()
