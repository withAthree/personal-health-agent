"""对话客户端（``get_llm``）的思考模式参数。

DeepSeek 官方接口默认开启思考（默认档 ``high``），硅基流动相反。所以关闭时必须**显式**
传 ``{"thinking": {"type": "disabled"}}``，不能依赖 provider 的缺省值——这两家的缺省方向
正好相反，靠缺省值就会在换 provider 时静默变成另一种行为。

``reasoning_effort`` 放在 ``extra_body`` 里而不是顶层关键字参数：两者最终都落在请求体
根上，但 ``extra_body`` 不依赖 langchain 是否认得这个字段名。
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from agent_app.config import settings
from agent_app.llm.client import get_llm


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> ChatOpenAI:
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_enable_thinking", enabled)
    get_llm.cache_clear()
    return get_llm()


def test_thinking_off_is_sent_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        client = _client(monkeypatch, enabled=False)

        assert client.extra_body == {"thinking": {"type": "disabled"}}
    finally:
        get_llm.cache_clear()


def test_thinking_on_carries_the_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """强度档只在开启思考时出现。

    写一个「思考关着但还带 reasoning_effort」的请求体是自相矛盾的，DeepSeek 会忽略
    它，而人读代码时会以为思考开着。
    """
    try:
        client = _client(monkeypatch, enabled=True)

        assert client.extra_body == {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
        }
    finally:
        get_llm.cache_clear()
