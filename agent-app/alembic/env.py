"""Alembic 迁移环境（异步引擎）。"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from agent_app.config import settings
from agent_app.core.event_loop import loop_factory
from agent_app.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

#: 不属于 Alembic 管理的对象名前缀。LangGraph 的 checkpointer 会在启动时
#: 自己建表与索引（见 agent_app.llm.agent.open_checkpointer），必须排除，
#: 否则 autogenerate 会生成 DROP TABLE，把模型记忆连同表一起删掉。
_IGNORED_NAME_PREFIXES = ("checkpoint",)


def include_object(
    obj: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object,
) -> bool:
    """告诉 Alembic 哪些对象不归它管。"""
    del obj, type_, reflected, compare_to
    if name is None:
        return True
    return not name.startswith(_IGNORED_NAME_PREFIXES)


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连接数据库。"""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在已建立的连接上执行迁移。"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：用异步引擎连接数据库后执行迁移。"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式入口。"""
    asyncio.run(run_async_migrations(), loop_factory=loop_factory)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
