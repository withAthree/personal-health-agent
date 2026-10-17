"""``de/collect.py``：本轮记账与工作说明通道。

两条独立的状态：``_current``（一轮对话的落点）与 ``_notifier``（一次 ``de_expert``
调用的 SSE 通道）。这里分别测它们的进出与恢复。
"""

from __future__ import annotations

from agent_app.de.collect import (
    announce,
    begin,
    collect,
    current_turn,
    finish,
    notify_via,
)


def test_announce_without_channel_is_noop() -> None:
    """探针、单测、后台任务都没有 writer，不能因此抛异常。"""
    announce("没有人在听")


def test_notify_via_forwards_to_writer() -> None:
    written: list[dict[str, str]] = []

    with notify_via(written.append):
        announce("查你的「空腹血糖」记录")

    assert written == [{"de_note": "查你的「空腹血糖」记录"}]


def test_notify_via_none_is_noop() -> None:
    with notify_via(None):
        announce("丢弃")

    # 没有 writer 时 announce 不该留下任何痕迹；下一行的正常使用也不受影响。
    written: list[dict[str, str]] = []
    with notify_via(written.append):
        announce("正常")
    assert written == [{"de_note": "正常"}]


def test_notify_via_restores_outer_channel() -> None:
    """内层退出后必须回到外层，否则第二次 ``de_expert`` 会把消息推到已关闭的流。"""
    outer: list[dict[str, str]] = []
    inner: list[dict[str, str]] = []

    with notify_via(outer.append):
        with notify_via(inner.append):
            announce("内层")
        announce("外层还在")

    assert inner == [{"de_note": "内层"}]
    assert outer == [{"de_note": "外层还在"}]


def test_notify_via_none_does_not_shadow_outer() -> None:
    """取不到 writer 是「没有通道可用」，不是「主动静默」：不能反过来切断外层。"""
    outer: list[dict[str, str]] = []

    with notify_via(outer.append), notify_via(None):
        announce("外层照常收到")

    assert outer == [{"de_note": "外层照常收到"}]


def test_notify_via_resets_on_exception() -> None:
    written: list[dict[str, str]] = []

    try:
        with notify_via(written.append):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    announce("异常之后不该再推")
    assert written == []


def test_begin_installs_turn_until_finish() -> None:
    turn, token = begin()
    try:
        assert current_turn() is turn
    finally:
        finish(token)

    assert current_turn() is None


def test_collect_context_manager_yields_turn() -> None:
    with collect() as turn:
        assert current_turn() is turn

    assert current_turn() is None
