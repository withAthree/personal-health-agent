"""LLM 客户端的契约测试。

不需要数据库、Redis 或网络：只校验 ``ChatOpenAIWithReasoning`` 对流式分片的修正。
"""

from typing import Any

from langchain_core.messages import AIMessageChunk

from agent_app.llm.client import ChatOpenAIWithReasoning, _is_terminal_stream_chunk


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
