"""上下文管理：摘要中间件与中文 token 估算。

模型看到的上下文由 LangGraph checkpoint 承载：超过阈值时，
``SummarizationMiddleware`` 会把较早的消息换成一段摘要写回 checkpoint。
账本（``messages`` 表）始终保留全量原文，不受这里影响。
"""

from collections.abc import Iterable

from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.summarization import ContextSize
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.messages.utils import count_tokens_approximately

from agent_app.config import settings
from agent_app.llm.client import get_summary_llm

#: 触发摘要的阈值。**当前生效的是百分比口径**（``build_summarization_middleware``
#: 的默认参数），按模型窗口的 ``settings.summary_trigger_fraction`` 触发。
#: 不用消息条数是因为单条消息长度差很多。
#: 走百分比要求模型带 ``profile``（框架从 ``model.profile["max_input_tokens"]``
#: 取窗口），取不到会在构造期抛 ``ValueError``、让每次对话都失败；摘要客户端的
#: ``profile`` 由 ``llm/client.py`` 的 ``get_summary_llm`` 提供。
#: 数值直接决定「摘要那次非流式调用」多久出现一次，而这个调用会整个挡在用户的
#: 首字等待里。``SUMMARY_TRIGGER_TOKENS`` 是等价的绝对口径，当前未被使用。
SUMMARY_TRIGGER_TOKENS: ContextSize = ("tokens", settings.summary_trigger_tokens)
SUMMARY_TRIGGER_FRACTION: ContextSize = ("fraction", settings.summary_trigger_fraction)

#: 摘要后保留多少最近内容。**当前生效的也是百分比口径**，保留窗口的
#: ``settings.summary_keep_fraction``（同样是构造期就要 ``profile``）。
#: ``SUMMARY_KEEP_TOKENS`` 是条数口径，当前未被使用；用它时必须是奇数：
#: 压缩总是发生在「新用户消息已追加、助手回复还没生成」的时刻，此时列表长度是
#: 偶数（摘要 + 若干完整问答对 + 1 条新提问），保留奇数条才能让切点落在完整
#: 问答对的边界上；保留偶数条会把一次问答从中间劈开（提问被摘要、回答留在
#: 窗口里）。
SUMMARY_KEEP_TOKENS: ContextSize = ("messages", settings.summary_keep_messages)
SUMMARY_KEEP_FRACTION: ContextSize = ("fraction", settings.summary_keep_fraction)

#: 中文 token 估算的每个 token 对应字符数。
#: 框架默认值是按英文校准的 4.0，中文会被低估约 4 倍（实测 200 个汉字
#: 只算出 54 token），必须显式传值，否则 token 阈值形同虚设。
CHINESE_CHARS_PER_TOKEN = 2.0


#: 官方 ``DEFAULT_SUMMARY_PROMPT`` 的中文译文（逐句直译，不改结构）。
#: 保留框架的公有契约：单独成行的 ``<messages>`` 标记与 ``{messages}`` 占位符
#: （框架用 ``summary_prompt.format(messages=...)`` 填充），因此译文中不能出现
#: 其他花括号。
SUMMARY_PROMPT = """<role>
上下文提取助手
</role>

<primary_objective>
你在本任务中的唯一目标：从下方对话历史中提取质量最高、最相关的上下文。
</primary_objective>

<objective_information>
你即将达到可以接受的输入 token 总数上限，因此必须从你的对话历史中提取
质量最高、最相关的信息片段。
这段上下文随后会覆盖下方呈现的对话历史。正因如此，请确保你提取的上下文
只包含为了继续推进总体目标所必需的最重要信息。
</objective_information>

<instructions>
下方的对话历史将被你在这一步提取的上下文取代。
你要确保不重复任何你已经完成的操作，因此从对话历史中提取的上下文
应当聚焦于对你的总体目标最重要的信息。

请按以下小节组织你的摘要。每个小节相当于一份清单——你必须在其中填入
相关信息；如果某个小节确实没有可报告的内容，就明确写“无”：

## 会话意图

用户的主要目标或请求是什么？你试图完成的总体任务是什么？
这一部分应当简洁，但完整到足以让人理解整个会话的目的。

## 摘要

提取并记录对话历史中所有最重要的上下文。包括本次对话中确定的重要选择、
结论或策略，也包括关键决策背后的理由。记录任何被否决的方案，
以及它们为什么没有被采用。

## 产物

本次对话中创建、修改或访问了哪些产物、文件或资源？对于文件修改，
请列出具体文件路径，并简要说明对每个文件所做的改动。
这一部分用于防止产物信息被悄悄丢失。

## 后续步骤

为了实现会话意图，还有哪些具体任务需要完成？你接下来应该做什么？

</instructions>

用户会把完整的消息历史发给你，你要从中提取上下文以生成替换内容。
请仔细完整地阅读，并深入思考哪些信息对你的总体目标最重要、应当被保存：

基于以上所有内容，请仔细通读整段对话历史，提取最重要、最相关的上下文，
用它来替换原文，以便为对话历史腾出空间。
只回复提取出的上下文。不要包含任何额外信息，
也不要在这段上下文之前或之后添加任何文字。

<messages>
需要摘要的消息：
{messages}
</messages>"""


def chinese_token_counter(messages: Iterable[MessageLikeRepresentation]) -> int:
    """中文场景下的 token 估算器（签名对齐框架的 ``TokenCounter``）。"""
    return count_tokens_approximately(
        messages,
        chars_per_token=CHINESE_CHARS_PER_TOKEN,
    )


def build_summarization_middleware(
    model: BaseChatModel | None = None,
    *,
    trigger: ContextSize | None = SUMMARY_TRIGGER_FRACTION,
    keep: ContextSize = SUMMARY_KEEP_FRACTION,
) -> SummarizationMiddleware:
    """构造摘要中间件。

    ``model`` 为空时用摘要客户端 ``get_summary_llm()``；测试可注入假模型，避免真实调用。
    ``trigger`` / ``keep`` 默认是百分比口径，要求该模型带 ``profile``
    （摘要客户端已带；注入假模型时要自己补，否则这里就抛 ``ValueError``）。
    摘要提示词是框架内置 ``DEFAULT_SUMMARY_PROMPT`` 的中文译文（``SUMMARY_PROMPT``）。
    """
    return SummarizationMiddleware(
        model=model if model is not None else get_summary_llm(),
        trigger=trigger,
        keep=keep,
        token_counter=chinese_token_counter,
        summary_prompt=SUMMARY_PROMPT,
    )
