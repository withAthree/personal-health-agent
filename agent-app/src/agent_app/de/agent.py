"""DE 子 Agent 的装配与一次性调用。

DE 不需要手写 ``StateGraph``：它的循环是 ReAct（思考 → 调工具 → 用结果再思考），
``create_agent`` 已经提供了。DS 之所以手写图，是因为它有 Plan / Code / Execute /
Check 四个固定阶段和一个沙箱，不是 ReAct。

**没有 checkpointer，也没有摘要中间件**：DE 一次调用就是一轮，第二轮不再需要上一轮
的中间步骤（用户看到的是主助手转述后的结论）。挂 checkpointer 只会往库里写没人读的
状态。``MemoryMiddleware`` 要挂——它让 DE 知道用户的既往病史、用药和过敏，这对判断
「这个数值要不要紧」是必要输入。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent_app.de.prompts import PRIMARY_PROMPT, SUPPORT_PROMPT
from agent_app.de.tools import compare_health_measure, web_search
from agent_app.llm.client import get_de_llm
from agent_app.memory.middleware import MemoryContext, MemoryMiddleware

Mode = Literal["primary", "support"]
"""DE 的两种调用形态。

``primary``：用户直接问了一个医学问题，DE 独立取证，回答完整问题。
``support``：上游已经算好一段内容（通常是 DS 的分析结果），DE 只解释它。
"""


def _tools_for(mode: Mode) -> list[BaseTool]:
    """当前模式下可用的工具。

    两种模式的工具集**完全一样**：查本人指标（``compare_health_measure``）和查外部
    资料（``web_search``）。早期设计里支持模式禁用检索，理由是网页结果会带来时效性
    内容、把回答拉长；实际验证下来这个理由不成立——支持模式要解释的恰恰是「这个数
    意味着什么」，没有外部资料只能用参数化记忆，而记忆是会过期的。约束改成放在
    提示词里：支持模式只在必要时检索、且限 300 字。

    ``mode`` 参数保留是为了将来真的出现按模式分化的工具时改一处即可。
    """
    del mode
    return [compare_health_measure, web_search]


def _prompt_for(mode: Mode) -> str:
    """按模式选提示词。"""
    return PRIMARY_PROMPT if mode == "primary" else SUPPORT_PROMPT


@lru_cache(maxsize=2)
def build_de_agent(mode: Mode) -> CompiledStateGraph[Any, Any, Any, Any]:
    """装配一个 DE 子 Agent。

    按 ``mode`` 缓存：图无状态（没有 checkpointer），同一模式的实例可以复用于所有
    用户。用户身份走 ``context``，与 ``llm/agent.py`` 的 ``get_agent()`` 同一套理由
    ——任何按用户变化的东西都不能写进构造参数，否则会被第一个用户钉死。

    Args:
        mode: 调用形态，决定提示词与工具集。

    Returns:
        编译好的图。调用时用 ``ainvoke({"messages": [...]}, context=...)``。
    """
    return create_agent(
        get_de_llm(),
        tools=_tools_for(mode),
        system_prompt=_prompt_for(mode),
        middleware=[MemoryMiddleware()],
        context_schema=MemoryContext,
    )


async def run_de(
    question: str,
    *,
    user_id: uuid.UUID,
    data_context: str = "",
) -> str:
    """跑一轮 DE，返回最终回答的文本。

    Args:
        question: 要问 DE 的问题原文。
        user_id: 当前用户；DE 靠它读长期记忆。
        data_context: 上游算好的内容。非空时进入 ``support`` 模式，DE 只解释它。

    Returns:
        DE 的最后一条回复文本。图跑完却没有文本时返回空串，由调用方决定怎么兜底。
    """
    mode: Mode = "support" if data_context.strip() else "primary"
    agent = build_de_agent(mode)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=_compose(question, data_context))]},
        context=MemoryContext(user_id=user_id),
    )
    return _last_text(result["messages"])


def _compose(question: str, data_context: str) -> str:
    """把问题与上游内容拼成一条用户消息。

    不拼进 system prompt：提示词是按模式固定的常量，按调用变化的内容进去会让
    ``build_de_agent`` 的缓存失效，也会把数据混进「规则」里。
    """
    context = data_context.strip()
    if not context:
        return question
    return f"{question}\n\n已有数据结果：\n{context}"


def _last_text(messages: Sequence[BaseMessage]) -> str:
    """取最后一条带文本的 AI 消息。

    不直接取 ``messages[-1]``：ReAct 结束时最后一条一定是 AI 消息，但某些模型会在
    工具调用之后补一条纯 ``tool_calls`` 的空内容消息，取到它会得到空串。
    """
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _message_text(message)
        if text:
            return text
    return ""


def _message_text(message: AIMessage) -> str:
    """把 AI 消息的 ``content`` 归一成字符串。"""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()
