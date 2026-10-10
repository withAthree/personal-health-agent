"""Personal Health Agent 后端应用包。"""

import uvicorn

from agent_app.config import settings


def main() -> None:
    """启动开发服务器（``agent-app`` 命令入口）。"""
    uvicorn.run(
        "agent_app.main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=True,
        # Windows 上 psycopg 异步驱动需要 SelectorEventLoop
        loop="agent_app.core.event_loop:loop_factory",
    )
