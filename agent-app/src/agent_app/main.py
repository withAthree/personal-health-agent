"""FastAPI 应用入口。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_app.api.v1.auth import router as auth_router
from agent_app.api.v1.chat import router as chat_router
from agent_app.config import settings

app = FastAPI(
    title="Personal Health Agent API",
    version="0.1.0",
    debug=settings.debug,
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
