"""把长期记忆注入 system prompt 的中间件。

**为什么必须在中间件里注入**：``llm/agent.py`` 的 ``get_agent()`` 是
``lru_cache`` 单例，``llm/client.py`` 的 ``SYSTEM_PROMPT`` 是模块级常量。于是

* 按用户拼 system prompt，会让**第一个用户的记忆被所有用户看到**；
* 把记忆当 ``SystemMessage`` 塞进 ``messages``，会被 LangGraph checkpoint
  持久化进会话状态——之后每一轮它都在，而且每轮再塞一次会叠成重复副本。

放在 ``awrap_model_call`` 里注入是「每次模型调用时临时生效」：既不进
checkpoint、也不进 ``messages`` 账本，前端与回放接口都看不到它。

用户身份走 ``context``（``MemoryContext``）而不是中间件实例字段，理由
同样是中间件是单例。
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from agent_app.memory.render import render_memories
from agent_app.memory.store import fetch_memories

logger = structlog.get_logger(__name__)

#: 出错信息进日志前的截断长度。
_MAX_ERROR_CHARS = 200


@dataclass(frozen=True)
class MemoryContext:
    """一次对话运行的上下文。

    由 ``create_agent(..., context_schema=MemoryContext)`` 声明，
    调用时用 ``astream(..., context=MemoryContext(user_id=...))`` 传入。
    """

    user_id: uuid.UUID


class MemoryMiddleware(AgentMiddleware[Any, MemoryContext, Any]):
    """把该用户的长期记忆追加到本次模型调用的 system message 之后。"""

    async def awrap_model_call(
        self,
        request: ModelRequest[MemoryContext],
        handler: Callable[
            [ModelRequest[MemoryContext]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelResponse[Any]:
        """注入记忆后继续调用链；取不到记忆时原样放行。"""
        context = request.runtime.context
        if not isinstance(context, MemoryContext):
            # 只会在装配漏传 context 时发生，属于编程错误；但为了不把
            # 整轮对话拖垮，这里降级为「没有记忆」。
            logger.warning("memory_context_missing")
            return await handler(request)
        try:
            block = render_memories(await fetch_memories(context.user_id))
        except Exception as exc:
            # 记忆是尽力而为的旁路：查库失败就当这次没有记忆。
            logger.warning(
                "memory_fetch_failed",
                user_id=str(context.user_id),
                error=str(exc)[:_MAX_ERROR_CHARS],
            )
            return await handler(request)
        if not block:
            return await handler(request)
        return await handler(
            request.override(system_message=_with_memory(request.system_message, block))
        )


def _with_memory(system_message: SystemMessage | None, block: str) -> SystemMessage:
    """把记忆段拼到原 system message 之后。"""
    if system_message is None:
        return SystemMessage(content=block)
    raw = system_message.content
    base = raw if isinstance(raw, str) else str(raw)
    return SystemMessage(content=f"{base}\n\n{block}" if base else block)
