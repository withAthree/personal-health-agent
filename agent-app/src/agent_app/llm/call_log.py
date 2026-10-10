"""模型调用日志：把每次真正发给模型的消息写进日志。

用途是验收上下文压缩：日志里应当出现「1 条摘要 + 最近若干条」，而不是
全量聊天记录；摘要那一次调用会带 ``node=SummarizationMiddleware.before_model``。
消息内容只在 ``DEBUG=true`` 时打印，避免生产日志里落健康隐私数据。
"""

from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

logger = structlog.get_logger(__name__)

#: 摘要消息的标记，与框架 ``summarization.py`` 写入的 ``lc_source`` 保持一致。
_SUMMARY_SOURCE = "summarization"

#: debug 日志里单条消息内容的最大字符数。
_PREVIEW_CHARS = 400


def _is_summary(message: BaseMessage) -> bool:
    """判断这条消息是不是摘要中间件写回的摘要。"""
    return message.additional_kwargs.get("lc_source") == _SUMMARY_SOURCE


def _role(message: BaseMessage) -> str:
    """取消息角色名（human / ai / system / tool）。"""
    return str(message.type)


def _preview(message: BaseMessage) -> str:
    """取消息正文的截断预览。"""
    text = str(message.content)
    if len(text) <= _PREVIEW_CHARS:
        return text
    return f"{text[:_PREVIEW_CHARS]}…（共 {len(text)} 字符）"


class ModelCallLogger(BaseCallbackHandler):
    """记录每次 chat 模型调用的入参，用于核对模型实际看到的上下文。"""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架回调签名
    ) -> None:
        """打印这一次调用收到的消息清单。"""
        node = str((metadata or {}).get("langgraph_node", "unknown"))
        batches = [list(batch) for batch in messages]
        summaries = sum(1 for batch in batches for item in batch if _is_summary(item))
        logger.info(
            "llm_call",
            node=node,
            messages=sum(len(batch) for batch in batches),
            summaries=summaries,
            compacted=summaries > 0,
            roles=[_role(item) for batch in batches for item in batch],
        )
        for batch_index, batch in enumerate(batches):
            for position, item in enumerate(batch):
                logger.debug(
                    "llm_message",
                    node=node,
                    batch=batch_index,
                    position=position,
                    role=_role(item),
                    summary=_is_summary(item),
                    chars=len(str(item.content)),
                    content=_preview(item),
                )
