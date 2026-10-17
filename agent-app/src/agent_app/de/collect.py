"""DE 一轮对话里的记账：调用次数与结果复用。

与 ``ds/collect.py`` 同一套机制（``ContextVar`` + ``begin`` / ``finish``），但**分开
一个对象**：两边的量不同（DS 记图表与运行次数，DE 只记调用次数），合并成一个类会让
``api/v1/chat.py`` 在落库时多一层分支判断。

为什么需要次数上限：``de_expert`` 内部要再跑一整轮 ReAct（可能带检索），一次提问
被主 agent 拆成三次调用就会成倍烧模型和外部接口。``ds_analysis`` 同样的理由设了
``MAX_RUNS_PER_TURN``。

取不到轮次对象时 tool 照常工作，只是不计数也不复用——探针、单测、后台任务都不会
开收集。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

MAX_CALLS_PER_TURN = 2
"""一轮对话里最多真跑几次 DE。

比 ``ds`` 的 3 次少：一次 DE 调用内部还可能带检索工具，成本高于一次数据分析。
主 agent 通常也只需要一次「解释这些数字」加一次「补充某个知识点」。
"""


@dataclass
class DeTurn:
    """本轮对话里 DE 的落点。"""

    calls: int = 0
    """已经真跑了几次。上限见 ``MAX_CALLS_PER_TURN``。"""

    answers: dict[tuple[str, str], str] = field(default_factory=dict)
    """``(question, data_context)`` → 返回的文本。同一轮里重复问同一句直接复用。"""


_current: ContextVar[DeTurn | None] = ContextVar("de_turn", default=None)


def begin() -> tuple[DeTurn, Token[DeTurn | None]]:
    """开始收集，返回轮次对象与复位令牌。"""
    turn = DeTurn()
    return turn, _current.set(turn)


def finish(token: Token[DeTurn | None]) -> None:
    """复位。放在 ``finally`` 里，异常与取消路径都会走到。"""
    _current.reset(token)


@contextlib.contextmanager
def collect() -> Iterator[DeTurn]:
    """测试用的上下文管理器版本，等价于 ``begin`` + ``finish``。"""
    turn, token = begin()
    try:
        yield turn
    finally:
        finish(token)


def current_turn() -> DeTurn | None:
    """当前轮次的落点，没有人在收集时返回 ``None``。"""
    return _current.get()


_notifier: ContextVar[Callable[[str], None] | None] = ContextVar(
    "de_notifier", default=None
)
"""工作说明通往 SSE 的通道。

**必须由 ``de/tool.py`` 的 ``de_expert`` 设置，不能由 ``api/v1/chat.py`` 设置。**
``get_stream_writer()`` 读的是当前 LangGraph 运行的配置，只有在主 agent 的 tool 上下文
里取到的才是外层那个；``chat.py`` 在运行时之外，取不到。

与 ``_current`` 分开是刻意的：``DeTurn`` 的生命周期是一轮对话（``chat.py`` 开、
``chat.py`` 关），通道的生命周期是一次 ``de_expert`` 调用。合并之后「一轮里的第二次
调用沿用第一次的 writer」会变成一个需要额外清理的状态。
"""


@contextlib.contextmanager
def notify_via(writer: Callable[[dict[str, str]], None] | None) -> Iterator[None]:
    """把工作说明接到 ``writer``。``writer`` 为 ``None`` 时是空操作。"""
    if writer is None:
        yield
        return

    def notifier(text: str) -> None:
        writer({"de_note": text})

    token = _notifier.set(notifier)
    try:
        yield
    finally:
        _notifier.reset(token)


def announce(text: str) -> None:
    """往当前通道推一行工作说明；没有人在监听时什么也不做。

    只做展示，不影响生成：推不出去照样返回文本。
    """
    notifier = _notifier.get()
    if notifier is not None:
        notifier(text)
