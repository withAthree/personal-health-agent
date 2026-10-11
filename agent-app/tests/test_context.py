"""上下文管理配置的契约测试。

不需要数据库、Redis 或网络：只校验摘要提示词的公有契约与中间件装配。
"""

from typing import Any

import pytest
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_app.config import settings
from agent_app.llm.client import get_utility_llm
from agent_app.llm.context import (
    SUMMARY_KEEP_FRACTION,
    SUMMARY_KEEP_TOKENS,
    SUMMARY_PROMPT,
    SUMMARY_TRIGGER_FRACTION,
    SUMMARY_TRIGGER_TOKENS,
    build_summarization_middleware,
    chinese_token_counter,
)


class _StubChatModel(BaseChatModel):
    """占位模型：只用于构造中间件，不会被调用。"""

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架签名
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


def _stub_model() -> _StubChatModel:
    """带 ``profile`` 的占位模型。

    百分比口径的阈值要求框架能从 ``model.profile["max_input_tokens"]`` 读到窗口
    大小，否则中间件在构造期就抛 ``ValueError``；占位模型必须补上这个字段，
    否则测不到中间件本身。
    """
    return _StubChatModel(profile={"max_input_tokens": 128000})


def test_summary_prompt_keeps_framework_contract() -> None:
    """``<messages>`` 标记与 ``{messages}`` 占位符是框架的公开契约。"""
    assert "<messages>" in SUMMARY_PROMPT.splitlines()
    assert "</messages>" in SUMMARY_PROMPT.splitlines()
    assert SUMMARY_PROMPT.count("{messages}") == 1


def test_summary_prompt_renders_without_leftover_braces() -> None:
    """框架用 ``format(messages=...)`` 填充，译文中不能有别的花括号。"""
    rendered = SUMMARY_PROMPT.format(messages="SENTINEL")

    assert rendered.count("SENTINEL") == 1
    assert "{" not in rendered
    assert "}" not in rendered


def test_summary_prompt_headings_are_chinese() -> None:
    for heading in ("## 会话意图", "## 摘要", "## 产物", "## 后续步骤"):
        assert heading in SUMMARY_PROMPT


def test_middleware_uses_chinese_prompt_and_counter() -> None:
    middleware = build_summarization_middleware(_stub_model())

    assert isinstance(middleware, SummarizationMiddleware)
    assert middleware.summary_prompt is SUMMARY_PROMPT
    assert middleware.token_counter is chinese_token_counter


def test_fractional_context_sizes_need_model_profile() -> None:
    """百分比口径靠 ``profile`` 取窗口，取不到就是构造期错误、每次对话都会失败。"""
    with pytest.raises(ValueError, match="profile"):
        build_summarization_middleware(_StubChatModel())

    middleware = build_summarization_middleware(_stub_model())

    assert middleware.trigger == SUMMARY_TRIGGER_FRACTION
    assert middleware.keep == SUMMARY_KEEP_FRACTION
    assert SUMMARY_KEEP_FRACTION[0] == "fraction"
    assert 0 < SUMMARY_KEEP_FRACTION[1] <= 1
    assert SUMMARY_TRIGGER_FRACTION[0] == "fraction"
    assert 0 < SUMMARY_TRIGGER_FRACTION[1] <= 1


def test_summary_client_carries_model_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """摘要客户端必须带 ``profile``，否则上面那两个百分比口径全部失效。

    不接网：只构造客户端，不发出请求。未配置密钥时补一个占位值，构造后立刻
    清掉 ``lru_cache``，避免占位密钥的客户端泄给其他用例。
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    get_utility_llm.cache_clear()
    try:
        client = get_utility_llm()

        assert client.profile == {"max_input_tokens": settings.llm_max_input_tokens}
    finally:
        get_utility_llm.cache_clear()


def test_compaction_keeps_turn_alignment() -> None:
    """``SUMMARY_KEEP_TOKENS`` 必须是奇数条消息，否则压缩会把一次问答劈开。

    压缩总是发生在「新用户消息已追加、助手回复还没生成」的时刻，此时消息列表
    长度是偶数（摘要 + 若干完整问答对 + 1 条新提问）；保留奇数条，切点才会落在
    完整问答对的边界上（提问和它的回答一起被摘要，或一起被保留）。

    当前启用的是百分比口径，本用例守着这个仍可切换的备选口径。
    """
    kind, value = SUMMARY_KEEP_TOKENS

    assert kind == "messages"
    assert value % 2 == 1


def test_compaction_triggers_by_tokens() -> None:
    """绝对口径的 ``SUMMARY_TRIGGER_TOKENS`` 仍是一个正数阈值（当前未启用）。"""
    kind, value = SUMMARY_TRIGGER_TOKENS

    assert kind == "tokens"
    assert value > 0
