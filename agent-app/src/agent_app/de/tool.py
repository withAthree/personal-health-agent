"""把 DE 接成主 agent 的一个 tool。

这是 DE 与产品之间唯一的接缝，与 ``ds/tool.py`` 的三件事同构：选模型调用入口、
把结构化结果渲染成文本、在 ``de/collect.py`` 上记账。差别在失败处理——DS 的失败
是「数据算不出来」，DE 的失败是「证据查不到」，后者可以直接用模型自身的知识降级，
前者不行。

**为什么超时设在这里而不是子 agent 内部**：DE 内部可能串起多次检索，单次 HTTP 超时
管不住总时长。整轮加一个墙钟上限，才能保证用户不会无限等。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import structlog
from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer

from agent_app.config import settings
from agent_app.de.agent import run_de
from agent_app.de.collect import (
    MAX_CALLS_PER_TURN,
    announce,
    current_turn,
    notify_via,
)
from agent_app.memory.middleware import MemoryContext

logger = structlog.get_logger(__name__)


def _stream_writer() -> Callable[[dict[str, str]], None] | None:
    """取外层的流式写入口；没人消费事件流时返回 ``None``。

    必须在 **``de_expert`` 自己的上下文里**取：``get_stream_writer()`` 读的是当前
    LangGraph 运行的配置，而 DE 子 agent 是另一次 ``ainvoke``，从它内部取拿到的是子
    agent 那一层的运行时，消息出不了门。

    取不到并不是错误：探针、单测、后台任务都不会开事件流。
    """
    try:
        return get_stream_writer()
    except Exception:
        return None


@tool
async def de_expert(
    question: str,
    runtime: ToolRuntime[MemoryContext],
    data_context: str = "",
) -> str:
    """向医学证据分析员（DE）提问，拿回带来源标注的医学解释。

    用它：
    - 用户问某个指标的医学含义、某个症状的可能原因、某项干预有没有证据；
    - 你刚用 ``ds_analysis`` 拿到了数字，需要解释这些数字意味着什么
      （此时把 DS 返回的文本原样放进 ``data_context``）。

    不用它：
    - 只需要判断单个数值是否在参考区间，用 ``compare_health_measure`` 就够；
    - 用户只问数据本身（多少、涨没涨），``ds_analysis`` 已经能回答；
    - 和健康无关的闲聊。

    ``question`` 写用户想问的完整意思。``data_context`` 只在需要解释上游结果时填，
    填了之后 DE 会只解释这段内容、不再扩展话题；不填则 DE 独立回答整个问题。
    """
    context = runtime.context
    if not isinstance(context, MemoryContext):
        return "拿不到用户身份，没法读取长期记忆。"

    turn = current_turn()
    key = (question, data_context.strip())
    if turn is not None:
        remembered = turn.answers.get(key)
        if remembered is not None:
            return remembered
        if turn.calls >= MAX_CALLS_PER_TURN:
            return (
                f"这一轮已经问过 {MAX_CALLS_PER_TURN} 次医学分析，不要再问了。"
                "用已经拿到的内容回答用户；还缺什么就直说缺什么。"
            )

    try:
        async with asyncio.timeout(settings.de_timeout_s):
            with notify_via(_stream_writer()):
                # 检索前的思考与检索本身都不产生可见输出，先推一行让用户知道在跑。
                announce("正在查证这个医学问题")
                text = await run_de(
                    question,
                    user_id=context.user_id,
                    data_context=data_context,
                )
    except TimeoutError:
        logger.warning("de_timeout", seconds=settings.de_timeout_s)
        return (
            "医学检索超时了，这次没有拿到结果。"
            "如实告诉用户现在查不到、稍后再试，不要凭印象补一个答案。"
        )
    except Exception as exc:
        logger.warning("de_failed", error=str(exc))
        return (
            "医学分析没能完成：检索服务出错，这是系统故障。"
            "如实说明现在查不到，不要推测结论。"
        )

    if not text:
        return "医学分析没有返回内容。如实告诉用户这次没查到，不要替它编一个答案。"

    if turn is not None:
        turn.calls += 1
        turn.answers[key] = text
    return text
