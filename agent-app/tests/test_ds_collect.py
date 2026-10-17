"""``ds/collect.py``：轮次作用域。

它存在的原因是 tool 进不了 LangGraph 的 state。这里钉两件事：取用时拿到的就是当前
那一轮，退出后不残留（否则下一轮对话会拿到上一轮的图表）。
"""

from __future__ import annotations

from agent_app.ds.collect import begin, collect, current_turn, finish


def test_no_turn_before_begin() -> None:
    assert current_turn() is None


def test_begin_exposes_the_turn_until_finish() -> None:
    turn, token = begin()
    try:
        assert current_turn() is turn
        turn.charts.append({"series": []})
        assert current_turn() is not None
        assert current_turn().charts == [{"series": []}]
    finally:
        finish(token)

    assert current_turn() is None


def test_nested_begin_restores_the_outer_turn() -> None:
    outer, outer_token = begin()
    inner, inner_token = begin()
    try:
        assert current_turn() is inner
    finally:
        finish(inner_token)

    assert current_turn() is outer
    finish(outer_token)


def test_context_manager_matches_begin_and_finish() -> None:
    with collect() as turn:
        assert current_turn() is turn

    assert current_turn() is None


def test_context_manager_restores_state_on_error() -> None:
    try:
        with collect():
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert current_turn() is None


def test_each_turn_starts_empty() -> None:
    with collect() as first:
        first.runs = 3
        first.answers["问题"] = "答案"
        first.charts.append({"series": []})

    with collect() as second:
        assert second.runs == 0
        assert second.answers == {}
        assert second.charts == []
