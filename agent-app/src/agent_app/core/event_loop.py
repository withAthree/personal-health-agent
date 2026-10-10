"""事件循环工厂。

psycopg 的异步驱动依赖事件循环的 ``add_reader`` / ``add_writer``，
而 Windows 默认的 ``ProactorEventLoop`` 不提供这两个方法，
因此在该平台上强制使用 ``SelectorEventLoop``。
"""

import asyncio
import sys


def loop_factory(*, use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    """创建事件循环。

    ``use_subprocess`` 仅为兼容 uvicorn 调用 ``loop_factory(use_subprocess=...)``
    的约定，自身不使用。
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
