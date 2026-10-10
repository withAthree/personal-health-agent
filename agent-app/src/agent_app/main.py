"""FastAPI 应用入口。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_app.api.v1.auth import router as auth_router
from agent_app.api.v1.chat import router as chat_router
from agent_app.config import settings
from agent_app.core.logging import configure_logging
from agent_app.llm.agent import close_checkpointer, open_checkpointer

configure_logging(debug=settings.debug)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """启动时建好 checkpoint 表，退出时释放连接池。

    checkpoint 表由 LangGraph 自己管理（``CREATE TABLE IF NOT EXISTS``，幂等），
    Alembic 管不到它，所以必须在这里初始化。
    """
    await open_checkpointer()
    try:
        yield
    finally:
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


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}
