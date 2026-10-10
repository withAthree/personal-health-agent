"""数据库引擎与会话工厂。"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent_app.config import settings

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

async_session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def to_libpq_dsn(url: str) -> str:
    """把 SQLAlchemy 连接串转成 psycopg 能直接使用的 DSN。

    ``postgresql+psycopg://`` 是 SQLAlchemy 的方言写法，交给 psycopg
    自己解析会报错（psycopg 的连接池与 LangGraph checkpointer 都走 psycopg）。
    """
    prefix = "postgresql+psycopg://"
    if url.startswith(prefix):
        return "postgresql://" + url[len(prefix) :]
    return url


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI 依赖：提供一个请求级数据库会话。"""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
