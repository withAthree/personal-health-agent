"""FastAPI 应用入口。"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_app.api.v1.auth import router as auth_router
from agent_app.api.v1.chat import router as chat_router
from agent_app.api.v1.upload import router as upload_router
from agent_app.config import settings
from agent_app.core.logging import configure_logging
from agent_app.llm.agent import close_checkpointer, open_checkpointer
from agent_app.parsing.worker import run_parse_worker

configure_logging(debug=settings.debug)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """启动时建好 checkpoint 表、拉起解析 worker，退出时都收干净。

    checkpoint 表由 LangGraph 自己管理（``CREATE TABLE IF NOT EXISTS``，幂等），
    Alembic 管不到它，所以必须在这里初始化。

    worker 用 ``asyncio.create_task`` 与 API 同进程：它大部分时间在等 Redis 队列，
    真正占 CPU 的只有 ONNX 推理那一下。退出时必须显式 cancel 并等它收尾，否则
    事件循环关掉时会在 ``BRPOP`` 上抛一串「Task was destroyed but it is pending」。
    """
    await open_checkpointer()
    worker = asyncio.create_task(run_parse_worker())
    try:
        yield
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        await close_checkpointer()


app = FastAPI(
    title="Personal Health Agent API",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # 开发阶段先全放开。allow_credentials 必须为 False：浏览器不接受
    # Access-Control-Allow-Origin: "*" 与凭据同时生效。
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}
