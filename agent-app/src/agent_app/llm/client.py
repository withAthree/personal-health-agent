"""LLM 客户端（OpenAI 兼容接口，当前指向 DeepSeek 官方）。"""

from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from agent_app.config import settings

#: 主助手的系统提示词。三个工具的分工写在这里：工具自己的 docstring 只能说明
#: 「我怎么用」，说不清「什么时候该用我而不是另外两个」。
SYSTEM_PROMPT = (
    "你是一名专业的个人健康助手，负责回答一般健康知识问题，"
    "提供准确、有依据的健康信息。请使用简体中文，回答简洁清晰。"
    "你不是医生，涉及诊断、用药或急症时，请明确提示用户咨询专业医疗人员。"
    "\n\n"
    "你有三个工具，按问题的性质选：\n"
    "- 问某个指标的当前值、在不在参考范围、比上次高还是低 "
    "→ compare_health_measure；\n"
    "- 问趋势、变化、多个指标之间的关系，需要算 → ds_analysis；\n"
    "- 其余所有医学问题 → de_expert。\n"
    "\n"
    "「其余所有医学问题」包括：某个指标偏高偏低意味着什么、可能的原因、要不要紧、"
    "有没有研究证据、生活方式怎么调整、某项检查是做什么的、报告上某项异常怎么看。\n"
    "这类问题**必须**交给 de_expert，不要用自己的知识直接回答。\n"
    "唯一的例外是既不涉及判断也不涉及用户数据的纯定义问题（例如「血糖的单位有哪些」），"
    "这种可以直接答。\n"
    "既要算数据又要解释时，先用 ds_analysis 拿到结果，再把它的文本原样放进 "
    "de_expert 的 data_context。\n"
    "\n"
    "用户有自己的健康记录。涉及用户数据的数字只能来自工具返回，"
    "不要自己推算、估算或补全；工具说数据不足就是不足，如实告诉用户缺什么。"
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

    思考模式**必须显式传**：DeepSeek 官方接口默认开启思考（默认档 ``high``），不传
    等于开着；硅基流动则相反，不传就是关。所以两个分支都写出来，不依赖 provider
    缺省值。

    ``reasoning_effort`` 放在 ``extra_body`` 里而不是当作顶层关键字参数：两者最终都
    落在请求体根上，但 ``extra_body`` 不依赖 langchain 是否认得这个字段名。
    """
    if settings.llm_enable_thinking:
        extra_body: dict[str, object] = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": settings.llm_thinking_effort,
        }
    else:
        extra_body = {"thinking": {"type": "disabled"}}

    return ChatOpenAIWithReasoning(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=0.7,
        streaming=True,
        extra_body=extra_body,
    )


@lru_cache(maxsize=1)
def get_summary_llm() -> ChatOpenAI:
    """返回**上下文摘要**专用的 LLM 客户端单例。

    摘要是这个项目里唯一在用户等待路径上的旁路调用（``_stream_chat`` 要在
    回复后触发压缩），所以它单独一个配置项 ``settings.llm_summary_model``。
    """
    return _build_utility_llm(settings.llm_summary_model or settings.llm_model)


@lru_cache(maxsize=1)
def get_memory_llm() -> ChatOpenAI:
    """返回**长期记忆抽取**专用的 LLM 客户端单例。

    抽取在后台任务里跑（``memory/extract.py``），对延迟不敏感，但对结构化输出
    的可靠性敏感，因此与摘要分开选型：``settings.llm_memory_model``。
    """
    return _build_utility_llm(settings.llm_memory_model or settings.llm_model)


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


@lru_cache(maxsize=1)
def get_ds_llm() -> ChatOpenAI:
    """返回**数据分析子 Agent（DS）**专用的 LLM 客户端单例。

    一个客户端服务两个调用点（``ds/graph.py`` 的 Planner 与 Coder）：两者的技术形态
    完全一致（非流式、结构化输出、同一份输入：问题 + 指标目录摘要），差别只在 prompt
    与 schema。分两个客户端只会多出一处需要同步的配置，没有任何区分度。

    温度取 **0.2 而不是 0.0**：Coder 会重试，而重试的前提是这一次能写出与上一次不同的
    代码——0.0 下模型很容易把同一份代码逐字再交一次，把重试预算白烧掉。0.2 仍远低于
    对话客户端的 0.7，不会让模型偏离“忠于数据”这条要求。

    它与 ``_build_utility_llm`` 的另外三个客户端**在任务位置上不同**：DS 在用户等待
    路径上（用户的问题就是在等这个分析结果），不是旁路任务。共用的是技术形态，不是
    优先级。
    """
    return _build_utility_llm(
        settings.llm_ds_model or settings.llm_model,
        temperature=0.2,
    )


@lru_cache(maxsize=1)
def get_de_llm() -> ChatOpenAI:
    """返回**医学证据分析员（DE）**专用的 LLM 客户端单例。

    与 ``get_ds_llm`` 同构：DE 也是子 Agent，非流式、低温。温度取 **0.3** 而不是
    0.2：DE 要写的是给人读的医学解释，不止是抽取和计算；但也不能高到让它在证据
    不足时编出一个像样的答案。

    单独一个配置项 ``settings.llm_de_model``：DE 的选型标准与 DS 不同——DS 的要求
    是「代码写得对、数字算得准」，DE 的要求是「引文找得准、解释不越界」。
    """
    return _build_utility_llm(
        settings.llm_de_model or settings.llm_model,
        temperature=0.3,
    )


def _build_utility_llm(model: str, *, temperature: float = 0.2) -> ChatOpenAI:
    """非流式、低温的客户端工厂。

    共用它的五个调用点里，三个是旁路任务（上下文摘要 + 长期记忆抽取 + 文档解析抽取），
    两个是子 Agent（``get_ds_llm`` 与 ``get_de_llm``）。五者的共同点是：不与用户直接
    对话、需要一次性拿到完整结果（非流式）、不希望内容被重新演绎（低温）。子 Agent
    不在旁路（用户的问题就在等它们的结果），但要的正是同一种技术形态。

    ``profile`` 不能省：压缩阈值用的是百分比口径（``llm/context.py`` 的
    ``SUMMARY_TRIGGER_FRACTION`` / ``SUMMARY_KEEP_FRACTION``），而
    ``SummarizationMiddleware`` 只从 ``model.profile["max_input_tokens"]`` 取
    窗口大小，取不到就在**构造期**直接抛 ``ValueError``。

    ``extra_body`` 显式关闭思考：DeepSeek 官方接口默认开启思考（默认档 ``high``），
    不传等于开着。四个调用点都不需要思考；结构化输出还必须关掉它，理由见
    ``structured_output()``。
    """
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        temperature=temperature,
        streaming=False,
        extra_body={"thinking": {"type": "disabled"}},
        profile={"max_input_tokens": settings.llm_max_input_tokens},
    )


def structured_output[ModelT: BaseModel](
    llm: ChatOpenAI, schema: type[ModelT]
) -> Runnable[Any, ModelT | dict[str, Any]]:
    """让模型按 ``schema`` 输出结构化结果。

    显式指定 ``method="function_calling"``，不用 langchain 的默认值——默认走
    ``response_format={"type": "json_schema"}``，DeepSeek 官方接口回
    ``400 This response_format type is unavailable now``；``method="json_mode"``
    则要求提示词里出现 "json" 这个词，本项目的提示词里没有。

    前提是思考模式已关（``_build_utility_llm()`` 负责）：开着思考时 DeepSeek 回
    ``400 Thinking mode does not support this tool_choice``。

    **只能用 ``_build_utility_llm()`` 造出来的客户端调它**，不要传 ``get_llm()``
    那个流式高温客户端。
    """
    return llm.with_structured_output(schema, method="function_calling")
