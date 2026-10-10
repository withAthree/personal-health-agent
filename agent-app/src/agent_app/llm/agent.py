"""LangGraph Agent 组装：checkpointer + 摘要中间件。

这里有两套数据，职责必须分清：

* **账本**：数据库里的 ``messages`` 表，全量原文，前端唯一的渲染来源。
* **模型记忆**：LangGraph checkpoint，只用来决定「这次给模型发什么」。

checkpoint 是可重建的缓存：它丢了、被清空、或者本次改动之前就存在的老会话，
都会在下一轮从账本重建一次。反过来不成立——账本永远不会被压缩影响。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent_app.config import settings
from agent_app.core.database import to_libpq_dsn
from agent_app.llm.client import SYSTEM_PROMPT, get_llm
from agent_app.llm.context import build_summarization_middleware

#: checkpoint 连接池上限。
CHECKPOINT_POOL_MAX_SIZE = 5

#: checkpoint 连接池类型：显式写出 ``dict_row`` 的行类型，
#: 否则 ``psycopg_pool`` 会推断成 ``AsyncConnection[tuple[Any, ...]]``。
_CheckpointPool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]]

_pool: _CheckpointPool | None = None


@lru_cache(maxsize=1)
def get_checkpointer() -> AsyncPostgresSaver:
    """返回 checkpoint 存储单例。

    必须在事件循环内首次调用：``AsyncPostgresSaver`` 构造时会绑定
    当前运行中的事件循环（``asyncio.get_running_loop()``），跨循环复用会出错。
    """
    global _pool
    pool: _CheckpointPool = AsyncConnectionPool(
        conninfo=to_libpq_dsn(settings.database_url),
        max_size=CHECKPOINT_POOL_MAX_SIZE,
        open=False,
        # 与官方 from_conn_string 保持一致：autocommit + 关闭 prepared statement，
        # 这样在事务模式的连接池（如 pgbouncer）后面也能工作。
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    _pool = pool
    return AsyncPostgresSaver(pool)


@lru_cache(maxsize=1)
def get_agent() -> CompiledStateGraph[Any, Any, Any, Any]:
    """返回带 checkpointer 与摘要中间件的 Agent 单例。"""
    return create_agent(
        get_llm(),
        tools=None,
        system_prompt=SYSTEM_PROMPT,
        middleware=[build_summarization_middleware()],
        checkpointer=get_checkpointer(),
    )


async def open_checkpointer() -> None:
    """打开连接池并建好 checkpoint 表（幂等，启动时调用）。"""
    checkpointer = get_checkpointer()
    pool = _pool
    if pool is not None:
        await pool.open()
    await checkpointer.setup()


async def close_checkpointer() -> None:
    """关闭连接池并清空缓存（应用退出时调用）。"""
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        await pool.close()
    get_checkpointer.cache_clear()
    get_agent.cache_clear()
