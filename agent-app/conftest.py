"""pytest 全局配置。"""

import asyncio
from collections.abc import Callable

import pytest

from agent_app.core.event_loop import loop_factory


@pytest.fixture(scope="session")
def _asyncio_loop_factory() -> Callable[[], asyncio.AbstractEventLoop]:
    """让 pytest-asyncio 使用项目自定义事件循环。

    Windows 默认的 ``ProactorEventLoop`` 无法驱动 psycopg 异步连接，
    不覆盖该 fixture 会导致所有涉及数据库的测试报 ``InterfaceError``。
    """
    return loop_factory
