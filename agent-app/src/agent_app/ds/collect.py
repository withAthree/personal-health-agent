"""DS 产物离开 tool 的通道，以及一轮里能跑几次。

``ds_analysis`` 是个普通 tool 函数，进不了 LangGraph 的 state；而图表要落到
``messages.charts``，只有 ``agent_app/api/v1/chat.py`` 知道会话 id 和那条消息的落库
时机。所以 tool 把翻译好的 ECharts option 放在这里，流式结束后由持久化层取走。

用 ``ContextVar`` 而不是模块级列表：``astream`` 里跑的节点与 tool 在同一个上下文树
里，ContextVar 才不会被同一进程里并发的另一轮对话串到；而且它随 ``with`` 退出自动
复位，没有跨请求残留。

取不到轮次对象时 tool 照常返回文本，只是图表丢掉。图表是展示，不该让它把一次已经
算完的分析弄失败。
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

MAX_RUNS_PER_TURN = 3
"""一轮对话里最多真跑几次分析。

主 agent 可能把一个问题拆成几次工具调用（先看一年趋势，再补一个近 30 天的对比）。
不设上限时，一次提问可以按次数烧模型调用。到顶后 tool 直接返回提示，不再跑。
"""


@dataclass
class DsTurn:
    """本轮对话里 DS 的落点。"""

    charts: list[dict[str, Any]] = field(default_factory=list)
    """渲染好的 ECharts option，落库时写进 ``messages.charts``。"""

    answers: dict[str, str] = field(default_factory=dict)
    """问题原文 → 渲染好的文本。同一轮里重复问同一句时直接复用，不再跑一次。"""

    runs: int = 0


_current: ContextVar[DsTurn | None] = ContextVar("ds_turn", default=None)


def begin() -> tuple[DsTurn, Token[DsTurn | None]]:
    """开始收集，返回轮次对象与复位令牌。

    用显式的 ``begin``/``finish`` 而不是 ``with``：调用方是个在 ``try`` 里横跨几十行
    的异步生成器，包一层 ``with`` 要为整段重新缩进。
    """
    turn = DsTurn()
    return turn, _current.set(turn)


def finish(token: Token[DsTurn | None]) -> None:
    """复位。放在 ``finally`` 里，异常与取消路径都会走到。"""
    _current.reset(token)


@contextlib.contextmanager
def collect() -> Iterator[DsTurn]:
    """测试用的上下文管理器版本，等价于 ``begin`` + ``finish``。"""
    turn, token = begin()
    try:
        yield turn
    finally:
        finish(token)


def current_turn() -> DsTurn | None:
    """当前轮次的落点，没有人在收集时返回 ``None``。"""
    return _current.get()
