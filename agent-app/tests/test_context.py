"""上下文管理配置的契约测试。

不需要数据库、Redis 或网络：只校验摘要提示词的公有契约与中间件装配。
"""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_app.llm.context import (
    SUMMARY_KEEP,
    SUMMARY_PROMPT,
    SUMMARY_TRIGGER,
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
    middleware = build_summarization_middleware(_StubChatModel())

    assert isinstance(middleware, SummarizationMiddleware)
    assert middleware.summary_prompt is SUMMARY_PROMPT
    assert middleware.token_counter is chinese_token_counter


def test_compaction_keeps_turn_alignment() -> None:
    """``SUMMARY_KEEP`` 必须是奇数条消息，否则压缩会把一次问答从中间劈开。

    压缩总是发生在「新用户消息已追加、助手回复还没生成」的时刻，此时消息列表
    长度是偶数（摘要 + 若干完整问答对 + 1 条新提问）；保留奇数条，切点才会落在
    完整问答对的边界上（提问和它的回答一起被摘要，或一起被保留）。
    """
    kind, value = SUMMARY_KEEP

    assert kind == "messages"
    assert value % 2 == 1


def test_compaction_triggers_by_tokens() -> None:
    """按 token 触发：消息条数口径误差太大，百分比口径模型又拿不到窗口。"""
    kind, value = SUMMARY_TRIGGER

    assert kind == "tokens"
    assert value > 0
