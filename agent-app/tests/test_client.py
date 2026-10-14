"""LLM 客户端的契约测试。

不需要数据库、Redis 或网络：只校验 ``ChatOpenAIWithReasoning`` 对流式分片的修正，
以及多 Agent 的模型选型是否能按角色隔离（``orchestration/llm.py``）。
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from agent_app.config import settings
from agent_app.llm.client import (
    ChatOpenAIWithReasoning,
    _is_terminal_stream_chunk,
    get_dialog_llm,
    get_llm,
    get_summary_llm,
    get_utility_llm,
)
from agent_app.orchestration.llm import get_agent_llm, get_orchestrator_llm


def _client() -> ChatOpenAIWithReasoning:
    """构造一个不会发起真实请求的客户端。"""
    return ChatOpenAIWithReasoning(
        model="stub-model",
        api_key="stub-key",
        base_url="http://127.0.0.1:1/v1",
    )


def _usage(total_tokens: int) -> dict[str, Any]:
    """复刻硅基流动的上报方式：每个分片都带一份**累积** usage。"""
    return {
        "prompt_tokens": 10,
        "completion_tokens": total_tokens - 10,
        "total_tokens": total_tokens,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def _convert(client: ChatOpenAIWithReasoning, chunk: dict[str, Any]) -> AIMessageChunk:
    converted = client._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert converted is not None
    assert isinstance(converted.message, AIMessageChunk)
    return converted.message


def test_terminal_chunk_detection() -> None:
    """收尾分片 = 有 choices 且 finish_reason 非空，或没有 choices。"""
    assert _is_terminal_stream_chunk([]) is True
    assert _is_terminal_stream_chunk([{"finish_reason": "stop"}]) is True
    assert _is_terminal_stream_chunk([{"finish_reason": None}]) is False


def test_intermediate_chunk_drops_usage_but_keeps_reasoning() -> None:
    """过程分片丢掉 usage，思考内容照常保留。"""
    message = _convert(
        _client(),
        {
            "choices": [
                {
                    "delta": {"content": "血", "reasoning_content": "先想一下"},
                    "finish_reason": None,
                }
            ],
            "usage": _usage(11),
        },
    )

    assert message.usage_metadata is None
    assert message.additional_kwargs["reasoning_content"] == "先想一下"


def test_terminal_chunk_keeps_usage() -> None:
    """收尾分片保留 usage，它就是本轮的真实总量。"""
    message = _convert(
        _client(),
        {
            "choices": [{"delta": {"content": "压"}, "finish_reason": "stop"}],
            "usage": _usage(17),
        },
    )

    assert message.usage_metadata is not None
    assert message.usage_metadata["total_tokens"] == 17


def test_usage_only_chunk_keeps_usage() -> None:
    """OpenAI 规范的收尾分片不带 choices，只带 usage，同样要保留。"""
    message = _convert(_client(), {"choices": [], "usage": _usage(17)})

    assert message.usage_metadata is not None
    assert message.usage_metadata["total_tokens"] == 17


def test_merged_usage_is_not_multiplied_by_chunk_count() -> None:
    """回归：累积 usage 被逐分片相加是 TTFT 事故的根因。

    未修复时四个分片会加成 ``11 + 12 + 13 + 20 = 56``；修复后只剩下收尾分片的
    20。这个数字会被 ``SummarizationMiddleware`` 拿去比压缩阈值。
    """
    client = _client()
    merged: AIMessageChunk | None = None
    for index, total_tokens in enumerate([11, 12, 13, 20]):
        message = _convert(
            client,
            {
                "choices": [
                    {
                        "delta": {"content": "血"},
                        "finish_reason": "stop" if index == 3 else None,
                    }
                ],
                "usage": _usage(total_tokens),
            },
        )
        merged = message if merged is None else merged + message

    assert merged is not None
    assert merged.usage_metadata is not None
    assert merged.usage_metadata["total_tokens"] == 20
    assert str(merged.content) == "血血血血"


# =====================================================================
# 多 Agent 的模型选型：按角色隔离，留空回落 ``llm_model``
# =====================================================================
#
# 下面每个用例都自己 ``cache_clear()``：客户端是**进程级** ``lru_cache`` 单例，
# 不清就会把 monkeypatch 过的模型名泄给其他用例。


def _use_placeholder_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """补一个占位密钥，避免构造客户端时依赖环境里的真实配置。"""
    monkeypatch.setattr(settings, "llm_api_key", "test-key")


def test_dialog_client_is_cached_by_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一模型名恒返回同一实例，不同模型名互不影响。"""
    _use_placeholder_key(monkeypatch)
    get_dialog_llm.cache_clear()
    try:
        first = get_dialog_llm("model-a")

        assert first is get_dialog_llm("model-a")
        assert first is not get_dialog_llm("model-b")
        assert first.model_name == "model-a"
        assert first.temperature == 0.7
        assert first.streaming is True
    finally:
        get_dialog_llm.cache_clear()


def test_plain_llm_shares_the_dialog_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_llm`` 就是「模型名＝``llm_model``的对话客户端」，两者共缓存。"""
    _use_placeholder_key(monkeypatch)
    monkeypatch.setattr(settings, "llm_model", "base-model")
    get_llm.cache_clear()
    get_dialog_llm.cache_clear()
    try:
        assert get_llm() is get_dialog_llm("base-model")
    finally:
        get_llm.cache_clear()
        get_dialog_llm.cache_clear()


def test_agents_take_their_own_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """三个业务 Agent 各自选型，换一个不影响另一个。"""
    _use_placeholder_key(monkeypatch)
    monkeypatch.setattr(settings, "llm_model", "base-model")
    monkeypatch.setattr(settings, "llm_ds_model", "ds-model")
    monkeypatch.setattr(settings, "llm_de_model", "de-model")
    monkeypatch.setattr(settings, "llm_hc_model", "hc-model")
    get_dialog_llm.cache_clear()
    try:
        ds = get_agent_llm("ds")
        de = get_agent_llm("de")
        hc = get_agent_llm("hc")

        assert ds.model_name == "ds-model"
        assert de.model_name == "de-model"
        assert hc.model_name == "hc-model"
        assert len({id(ds), id(de), id(hc)}) == 3
    finally:
        get_dialog_llm.cache_clear()


def test_agents_fall_back_to_the_base_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三个 Agent 的配置都留空时回落 ``llm_model``（默认状态）。"""
    _use_placeholder_key(monkeypatch)
    monkeypatch.setattr(settings, "llm_model", "base-model")
    monkeypatch.setattr(settings, "llm_ds_model", "")
    monkeypatch.setattr(settings, "llm_de_model", "")
    monkeypatch.setattr(settings, "llm_hc_model", "")
    get_dialog_llm.cache_clear()
    try:
        assert get_agent_llm("ds") is get_dialog_llm("base-model")
        assert get_agent_llm("de") is get_dialog_llm("base-model")
        assert get_agent_llm("hc") is get_dialog_llm("base-model")
    finally:
        get_dialog_llm.cache_clear()


def test_orchestrator_uses_a_utility_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """控制节点走非流式、温度 0.0 的客户端，并且能回落 ``llm_model``。"""
    _use_placeholder_key(monkeypatch)
    monkeypatch.setattr(settings, "llm_model", "base-model")
    monkeypatch.setattr(settings, "llm_orchestrator_model", "orch-model")
    get_utility_llm.cache_clear()
    try:
        client = get_orchestrator_llm()

        assert client.model_name == "orch-model"
        assert client.streaming is False
        # 编排是分类任务，不能沿用旁路默认的 0.2。
        assert client.temperature == 0.0
        assert client.profile == {"max_input_tokens": settings.llm_max_input_tokens}

        monkeypatch.setattr(settings, "llm_orchestrator_model", "")
        fallback = get_orchestrator_llm()
        assert fallback.model_name == "base-model"
        assert fallback.temperature == 0.0
    finally:
        get_utility_llm.cache_clear()


def test_utility_factories_share_the_model_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """摘要与记忆抽取落到同一个按模型名的缓存上，不会各造一份客户端。"""
    _use_placeholder_key(monkeypatch)
    monkeypatch.setattr(settings, "llm_summary_model", "summary-model")
    monkeypatch.setattr(settings, "llm_memory_model", "summary-model")
    get_summary_llm.cache_clear()
    get_utility_llm.cache_clear()
    try:
        assert get_summary_llm() is get_utility_llm("summary-model")
    finally:
        get_summary_llm.cache_clear()
        get_utility_llm.cache_clear()
