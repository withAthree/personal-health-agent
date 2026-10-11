"""LLM 客户端（硅基流动，OpenAI 兼容接口）。"""

from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent_app.config import settings

#: 功能单元 1 使用的系统提示词，后续由多 Agent 路由替换。
SYSTEM_PROMPT = (
    "你是一名专业的个人健康助手，负责回答一般健康知识问题，"
    "提供准确、有依据的健康信息。请使用简体中文，回答简洁清晰。"
    "你不是医生，涉及诊断、用药或急症时，请明确提示用户咨询专业医疗人员。"
)


#: provider 用来承载思考过程的非标准字段；``ChatOpenAI`` 默认把它丢掉。
_REASONING_FIELD = "reasoning_content"


def _is_terminal_stream_chunk(choices: list[dict[str, Any]]) -> bool:
    """判断流式分片是不是本轮最后一个携带 usage 的分片。

    只给「收尾分片」保留 usage 是为了绕开硅基流动的用法上报方式：它**每个分片都
    带一份累积 usage**（第一个分片 ``total_tokens=10``，之后 11、12、…，最后一个
    等于真实总量），而 ``AIMessageChunk`` 合并时对 ``usage_metadata`` 是**相加**
    （``langchain_core.messages.ai.add_usage``），于是 N 个分片被加成一个约为真实值
    N/2 倍的数——分片越多越离谱，几百个分片就到了百万级。

    收尾分片有两种形态：
    - 有 ``choices`` 且 ``finish_reason`` 非空（硅基流动走这条，它不发纯 usage 分片）；
    - ``choices`` 为空（OpenAI 规范里 ``include_usage`` 的收尾分片，只有 usage）。

    只保留收尾分片的 usage 后，相加结果恰好等于最终的累积值，因此对上面两种
    provider 都成立；完全不上报 usage 的 provider 不受影响。

    Args:
        choices: 该分片的 ``choices`` 列表。

    Returns:
        是否应保留该分片的 ``usage_metadata``。
    """
    if not choices:
        return True
    return bool(choices[0].get("finish_reason"))


class ChatOpenAIWithReasoning(ChatOpenAI):
    """修正 provider 两处非标准行为的 ``ChatOpenAI``：思考内容与流式 usage。

    **思考内容**：基类的 ``_convert_delta_to_message_chunk`` 是**模块级函数**，只取
    ``content`` / ``function_call`` / ``tool_calls``，思考内容在那里被丢掉；能覆盖的
    是本方法（实例方法）。提取出来的内容放进 ``additional_kwargs``，而
    ``_convert_message_to_dict`` 只转发白名单 key（``name`` / ``tool_calls`` /
    ``function_call`` / ``audio`` / ``__openai_role__``），因此它**不会**被回喂上游。
    ``chunk.content_blocks`` 在这里帮不上忙：chat completions 的流式分片走
    ``_convert_to_v1_from_chat_completions_chunk``，那个函数不读 ``additional_kwargs``。

    **流式 usage**：见 ``_is_terminal_stream_chunk``。这条不只是数字好看——
    ``SummarizationMiddleware._should_summarize_based_on_reported_tokens`` 会拿
    最后一条 ``AIMessage`` 的 ``usage_metadata["total_tokens"]`` 去比压缩阈值，
    放大后的值恒大于任何阈值，会让摘要**每轮都触发**，而摘要是非流式调用、整段
    阻塞在用户首字之前。
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """在基类转换结果上补一份被丢掉的思考内容。"""
        converted = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if converted is None:
            return None
        # 与基类同样的取值口径：真正的 chat completions 分片在 ``choices``，
        # ``beta.chat.completions.stream`` 的分片嵌在 ``chunk`` 里。
        choices: list[dict[str, Any]] = chunk.get("choices") or (
            chunk.get("chunk", {}).get("choices") or []
        )
        if (
            isinstance(converted.message, AIMessageChunk)
            and converted.message.usage_metadata is not None
            and not _is_terminal_stream_chunk(choices)
        ):
            converted.message.usage_metadata = None
        if not choices:
            return converted
        delta: dict[str, Any] = choices[0].get("delta") or {}
        reasoning = delta.get(_REASONING_FIELD)
        if isinstance(reasoning, str) and reasoning:
            converted.message.additional_kwargs[_REASONING_FIELD] = reasoning
        return converted


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回对话用的 LLM 客户端单例。

    是否输出思考过程由 ``settings.llm_enable_thinking`` 决定，关闭时请求体与
    改动前完全一致（不传 ``extra_body``）。
    """
    return ChatOpenAIWithReasoning(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=0.7,
        streaming=True,
        extra_body={"enable_thinking": True} if settings.llm_enable_thinking else None,
    )


@lru_cache(maxsize=1)
def get_summary_llm() -> ChatOpenAI:
    """返回摘要专用的 LLM 客户端单例。

    与对话客户端分开：摘要需要一次性拿到完整文本，不需要流式，
    温度也调低，避免摘要本身被重新演绎。

    ``profile`` 不能省：压缩阈值用的是百分比口径（``llm/context.py`` 的
    ``SUMMARY_TRIGGER_FRACTION`` / ``SUMMARY_KEEP_FRACTION``），而
    ``SummarizationMiddleware`` 只从 ``model.profile["max_input_tokens"]`` 取
    窗口大小，取不到就在构造期直接抛 ``ValueError``。中间件由
    ``llm/agent.py`` 的 ``get_agent()`` 每次请求重新装配，所以少这一个字段
    会让每次对话都返回错误、而不是等额度用满才出问题。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=0.2,
        streaming=False,
        profile={"max_input_tokens": settings.llm_max_input_tokens},
    )
