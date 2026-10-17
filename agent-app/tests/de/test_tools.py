"""``de/tools.py``：DE 子 Agent 拿到的两个工具。

测的是工具契约——参数怎么进来、失败怎么变成话、结果怎么交给模型。真实的数据库查询
与网络检索在这里都被替身顶掉：``load_series`` 是纯 SQL + 解析，``search`` 是外部
服务，两者的细节各有自己的测试。
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from langchain.tools import ToolRuntime

from agent_app.de import tools as de_tools
from agent_app.de.measures import Lookup, Reading, Series
from agent_app.de.search import SearchHit
from agent_app.memory.middleware import MemoryContext

USER_ID = uuid.UUID("9d2f5c71-4a38-4e6b-8f10-3c7b6a2e9d14")


def _runtime_with(context: Any) -> ToolRuntime[MemoryContext]:  # noqa: ANN401
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id="call-de-2",
        store=None,
    )


def _runtime() -> ToolRuntime[MemoryContext]:
    return _runtime_with(MemoryContext(user_id=USER_ID))


def _lookup() -> Lookup:
    reading = Reading(
        value_text="6.3",
        value=6.3,
        unit="mmol/L",
        reference_text="3.9-6.1",
        measured_on=date(2026, 10, 7),
        position="above",
        delta=0.2,
    )
    return Lookup(
        query="血糖",
        series=(Series(name="空腹血糖", unit="mmol/L", readings=(reading,), total=21),),
        omitted=0,
    )


async def test_compare_reads_the_current_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """身份只能来自 ``runtime.context``；工具参数里不能有 user_id。"""
    seen: list[tuple[uuid.UUID, str]] = []

    async def _fake_load(user_id: uuid.UUID, metric: str) -> Lookup:
        seen.append((user_id, metric))
        return _lookup()

    monkeypatch.setattr(de_tools, "load_series", _fake_load)

    out = await de_tools.compare_health_measure.coroutine(  # type: ignore[misc]
        metric="血糖", runtime=_runtime()
    )

    assert seen == [(USER_ID, "血糖")]
    assert "高于上限 0.2" in out


async def test_compare_refuses_without_identity() -> None:
    out = await de_tools.compare_health_measure.coroutine(  # type: ignore[misc]
        metric="血糖", runtime=_runtime_with(None)
    )

    assert "拿不到用户身份" in out


async def test_web_search_renders_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_search(query: str, max_results: int) -> list[SearchHit]:
        return [SearchHit("空腹血糖正常值", "https://a", "3.9-6.1 mmol/L")]

    monkeypatch.setattr(de_tools, "search", _fake_search)

    out = await de_tools.web_search.coroutine(  # type: ignore[misc]
        query="空腹血糖 正常值", max_results=3
    )

    assert "https://a" not in out
    assert "3.9-6.1" in out


async def test_web_search_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _timeout(query: str, max_results: int) -> list[SearchHit]:
        raise TimeoutError

    monkeypatch.setattr(de_tools, "search", _timeout)

    out = await de_tools.web_search.coroutine(  # type: ignore[misc]
        query="空腹血糖", max_results=3
    )

    assert "超时" in out


async def test_web_search_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """检索失败不能让 DE 崩掉，要变成一句它可以转述的话。"""

    async def _boom(query: str, max_results: int) -> list[SearchHit]:
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(de_tools, "search", _boom)

    out = await de_tools.web_search.coroutine(  # type: ignore[misc]
        query="空腹血糖", max_results=3
    )

    assert "检索失败" in out
    assert "429" in out
