"""LLM 客户端（硅基流动，OpenAI 兼容接口）。"""

from functools import cache, lru_cache
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


@cache
def get_dialog_llm(model: str) -> ChatOpenAI:
    """返回**按模型名缓存**的对话客户端。

    与 ``get_llm`` 只有一点不同：模型名是参数。参数化是为了多 Agent——三个业务
    Agent 各自选型后互不影响（角色 → 模型名的映射在 ``orchestration/llm.py``）。
    ``lru_cache`` 是**进程级**单例，所以换模型要重启进程（操作见
    ``document/模型使用清单.md``）。

    是否输出思考过程由 ``settings.llm_enable_thinking`` 决定，关闭时请求体与
    改动前完全一致（不传 ``extra_body``）。

    Args:
        model: 模型名。**本函数不做回落**，由调用方决定回落目标。

    Returns:
        该模型名对应的对话客户端（同一模型名恒返回同一个实例）。
    """
    return ChatOpenAIWithReasoning(
        model=model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=0.7,
        streaming=True,
        extra_body={"enable_thinking": True} if settings.llm_enable_thinking else None,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回单 Agent 时代那个对话客户端的单例（模型名＝``settings.llm_model``）。

    多 Agent 上线后业务 Agent 走 ``orchestration/llm.py`` 的 ``get_agent_llm``；
    这个函数保留给还未改造的调用点，与 ``get_dialog_llm`` **共用同一份缓存**，
    所以同一个模型名不会构造出两个客户端。
    """
    return get_dialog_llm(settings.llm_model)


@lru_cache(maxsize=1)
def get_summary_llm() -> ChatOpenAI:
    """返回**上下文摘要**专用的 LLM 客户端单例。

    摘要是这个项目里唯一在用户等待路径上的旁路调用（``_stream_chat`` 要在
    回复后触发压缩），所以它单独一个配置项 ``settings.llm_summary_model``。
    """
    return get_utility_llm(settings.llm_summary_model or settings.llm_model)


@lru_cache(maxsize=1)
def get_memory_llm() -> ChatOpenAI:
    """返回**长期记忆抽取**专用的 LLM 客户端单例。

    抽取在后台任务里跑（``memory/extract.py``），对延迟不敏感，但对结构化输出
    的可靠性敏感，因此与摘要分开选型：``settings.llm_memory_model``。
    """
    return get_utility_llm(settings.llm_memory_model or settings.llm_model)


@lru_cache(maxsize=1)
def get_parse_llm() -> ChatOpenAI:
    """返回**文档解析抽取**专用的 LLM 客户端单例。

    与对话主流程解耦：``get_llm`` 是流式、高温、可带思考的对话客户端，而文档
    解析只需要把 OCR 原文里的信息抄成结构化 JSON，两者选型标准不同（用户要求
    这个模型将来可能单独更换），所以走独立配置 ``settings.llm_parse_model``。

    温度 0.0：抽取任务不允许重新演绎，只要求忠于原文。
    """
    return _build_utility_llm(
        settings.llm_parse_model or settings.llm_model,
        temperature=0.0,
    )


def _build_utility_llm(model: str, *, temperature: float = 0.2) -> ChatOpenAI:
    """旁路任务客户端的共同构造。

    旁路任务 = 上下文摘要 + 长期记忆抽取 + 文档解析抽取：都不与用户直接对话，
    都需要一次性拿到完整结果（非流式），都不希望内容被重新演绎（低温）。

    ``profile`` 不能省：压缩阈值用的是百分比口径（``llm/context.py`` 的
    ``SUMMARY_TRIGGER_FRACTION`` / ``SUMMARY_KEEP_FRACTION``），而
    ``SummarizationMiddleware`` 只从 ``model.profile["max_input_tokens"]`` 取
    窗口大小，取不到就在**构造期**直接抛 ``ValueError``。
    """
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=temperature,
        streaming=False,
        profile={"max_input_tokens": settings.llm_max_input_tokens},
    )


@cache
def get_utility_llm(model: str, *, temperature: float = 0.2) -> ChatOpenAI:
    """返回**按模型名缓存**的旁路客户端（非流式、低温）。

    多 Agent 的控制节点（编排 / 整合 / 反思 / 收尾）用它：不与用户直接对话、要
    一次拿到完整结果、要能 ``with_structured_output``。与 ``get_dialog_llm`` 的
    缓存口径相同（**进程级**单例，换模型要重启进程）。

    Args:
        model: 模型名。**本函数不做回落**，由调用方决定回落目标。
        temperature: 采样温度。缓存键包含它，所以同一模型的不同温度会各建一个
            实例；默认沿用旁路任务共同的 0.2，只有在调用方明确要求时才改。

    Returns:
        该模型名对应的旁路客户端（同一模型名 + 同一温度恒返回同一个实例）。
    """
    return _build_utility_llm(model, temperature=temperature)
