"""图编排的共享状态。

多 Agent 协作的中间产物（编排决策、各 Agent 的产出、缺口清单）全部经 LangGraph
的 channel 在节点之间传递，**不落库、不进 checkpoint**。

三条边界必须先分清，动手改 ``api/v1/chat.py`` 之前再读一遍：

* **这里的 state**：只活在当前这一轮的图执行里。跨轮没有语义——下一轮从账本
  重建 ``messages``，其余字段一律是空的；
* **checkpoint**（``llm/agent.py`` 的 ``AsyncPostgresSaver``）：只存 ``messages``，
  是模型记忆，可压缩、可重建；
* **账本**（``messages`` 表）：只存 user / assistant 正文，是前端唯一的渲染来源。

所以「子 Agent 的过程不进上下文、结果进上下文」落到这里就是：节点往 state 里写的
中间产物**只在图内流转**，真正回到模型的仍然只有最终正文那一条 assistant 消息。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

#: Agent 代号。全程只用这三个短码：账本的 ``messages.agent_source`` 列、流式事件的
#: ``meta.agent``、前端的配色都按它取值（``String(20)`` 装得下）。
AgentCode = Literal["ds", "de", "hc"]

#: 主 Agent 的取值：三个 Agent 之一，或 ``general``（不需要分配任务时的通用对话）。
PrimaryCode = Literal["ds", "de", "hc", "general"]

#: 反思回环的硬上限。``step`` 从 0 起，走到 ``MAX_ROUNDS`` 时无论反思结论如何都
#: 强制进整合、并降级回答（能确定的部分 + 不能确定的部分 + 需要用户补什么）。
#:
#: 取 3 是用户定的。代价很直接：每多一轮就要重跑一遍 Agent，而这些时间全部加在
#: 用户等待的首字延迟上。
MAX_ROUNDS = 3


class Contribution(BaseModel):
    """一个 Agent 在本轮交出的一份产出。

    支持 Agent 交出的也是这个类型，区别只在 ``role``。主 / 辅是**本次任务的角色**，
    不是 Agent 的固有属性：同一个 DS 在「分析血糖趋势」里是主，在「失眠怎么办」里
    是支持。所以这里记 ``role`` 而不给 Agent 本身贴标签。
    """

    agent: AgentCode
    step: int
    role: Literal["primary", "support"]
    content: str
    sources: list[str] = Field(
        default_factory=list,
        description="这条结论的依据出处（个人记录 / 联网检索命中），供整合节点核对",
    )


class RoutingDecision(BaseModel):
    """编排器一次调用吐出的决策。

    字段与编排器写回 ``OrchestrationState`` 的那几个一一对应（见
    ``nodes._decision_state``），所以改这里就要回去改 ``prompts.py`` 的提示词：
    提示词里少写一个字段时，模型会给它填默认值，而且**不会报错**。

    刻意**没有** ``need_data``：取数已经做成 Agent 自己的工具，「这轮要不要取数」
    由 Agent 在执行中决定，编排器只负责分工。
    """

    cuj: str = Field(default="", description="这条消息属于哪类用例，两三个词")
    primary: PrimaryCode = Field(description="对这条回复负最终责任的角色，只能有一个")
    supports: list[AgentCode] = Field(
        default_factory=list, description="只补 primary 缺的能力，可以为空"
    )
    queries: dict[str, str] = Field(
        default_factory=dict,
        description="给每个参与角色重写后的查询，键只能用 primary 与 supports 里的代号",
    )
    sub_questions: list[str] = Field(
        default_factory=list, description="用户这一句需要被回答的子问题（1～4 条）"
    )
    reason: str = Field(default="", description="为什么这样分工；只进日志，不给用户看")


class OrchestrationState(TypedDict, total=False):
    """图编排的共享状态。

    ``total=False`` 是有意的：除 ``messages`` 外都可能缺省，函数一律用
    ``state.get(...)`` 取值，别假设它一定存在。

    带 reducer 的字段（返回时必须给**列表**，不能给单个元素，否则会拼接错位）：

    * ``messages``：``add_messages``，按消息追加与去重；
    * ``contributions``：``operator.add``，**按轮累积**（回环时第 2 轮的产出追加在
      第 1 轮后面，所以每条都带 ``step`` 标明轮次）。
    """

    #: 对话历史。图内节点读它，checkpoint 负责压缩与跨轮持久化。
    messages: Annotated[list[AnyMessage], add_messages]

    #: 当前轮次，从 0 起。**唯一写它的是 ``nodes.next_round``**。
    #: 前端过程块的分轮显示（``block-start`` 的 ``step``）读的就是这个值。
    step: int

    #: 编排器给出的用例分类（设计文档 §8.2 的 CUJ），用于选主 / 辅与后续统计。
    cuj: str | None

    #: 本轮的主 Agent（对最终回答负最终责任的那个）。
    primary: PrimaryCode | None

    #: 本轮的 supports（补主 Agent 缺的能力，产物交给主 Agent 与整合节点消费）。
    supports: list[AgentCode]

    #: 编排器为每个参与角色重写后的查询语句，键是 ``primary`` 与 ``supports`` 里的代号。
    queries: dict[str, str]

    #: 编排器拆出的子问题。反思的「意图覆盖」判据要用它，**少一条就会漏判**，
    #: 所以回环时取并集而不是覆盖（见 ``nodes._decision_state``）。
    sub_questions: list[str]

    #: 编排器为什么这样分工。只用于排障与人工评审：不进提示词、不给用户看。
    routing_reason: str

    #: ``intake`` 渲染好的长期记忆段（``memory/render.py`` 的格式）。
    #: 回环重入时直接复用，不重复查库。
    memory_digest: str

    #: ``intake`` 渲染好的已上传文件清单。回环重入时同样复用。
    files_digest: str

    #: 各 Agent 的产出，按轮累积（见类 docstring）。
    contributions: Annotated[list[Contribution], operator.add]

    #: 整合节点产出的最终稿。``finalize`` 把它作为正文发给用户，``memory`` 拿它
    #: 做记忆抽取，所以两者才能真的并行。
    draft: str | None

    #: 反思节点给出的缺口清单（每条要具体到「缺什么、谁能补」）。
    gaps: list[str]

    #: 反思节点的结论：信息是否足够回复用户。
    sufficient: bool

    #: 反思节点的结论：本轮缺口与上一轮重合，再问也拿不到 → 立即停止回环。
    stalled: bool

    #: 已经试过的查询，回环时带上，避免第 2 轮把第 1 轮的动作再做一遍。
    tried_queries: Annotated[list[str], operator.add]
