"""模型调用日志：把每次真正发给模型的消息写进日志。

用途有两个：

1. 验收上下文压缩：日志里应当出现「1 条摘要 + 最近若干条」，而不是
   全量聊天记录；摘要那一次调用会带 ``node=SummarizationMiddleware.before_model``。
2. 定位首 token 延迟（TTFT）：``llm_first_token`` 给出每个节点从发起调用到
   收到第一个 token 的毫秒数，``llm_completion`` 给出总耗时，两者按 ``node``
   就能把「摘要那次额外往返」与「模型节点本身」拆开量。

消息内容只在 ``DEBUG=true`` 时打印，避免生产日志里落健康隐私数据；两个计时
事件同样是 ``logger.debug``，生产环境（INFO）不输出。
"""

import time
from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

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

    def __init__(self) -> None:
        """记录每个 run 的起点，用于算首 token 延迟。"""
        self._runs: dict[UUID, tuple[str, float]] = {}
        self._first_token_logged: set[UUID] = set()

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
        """打印这一次调用收到的消息清单，并记下起点时间。"""
        node = str((metadata or {}).get("langgraph_node", "unknown"))
        # 在回调内部自己记住起止点，不依赖框架把 metadata 转发到
        # on_llm_new_token / on_llm_end（这两个回调拿不到 langgraph_node）。
        self._runs[run_id] = (node, time.perf_counter())
        self._first_token_logged.discard(run_id)
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

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架回调签名
    ) -> None:
        """只记每个 run 的第一个 token：它定义了用户感知到的等待时长。"""
        if run_id in self._first_token_logged:
            return
        self._first_token_logged.add(run_id)
        started = self._runs.get(run_id)
        if started is None:
            return
        node, at = started
        now = time.perf_counter()
        logger.debug(
            "llm_first_token",
            node=node,
            at=round(now, 6),
            ttft_ms=round((now - at) * 1000, 1),
            chars=len(token) if isinstance(token, str) else None,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架回调签名
    ) -> None:
        """记录一次调用的总耗时，然后忘掉这个 run。"""
        started = self._runs.pop(run_id, None)
        self._first_token_logged.discard(run_id)
        if started is None:
            return
        node, at = started
        now = time.perf_counter()
        logger.debug(
            "llm_completion",
            node=node,
            at=round(now, 6),
            ms=round((now - at) * 1000, 1),
            generations=len(response.generations),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401 - 复刻框架回调签名
    ) -> None:
        """失败或被打断时也清掉记录，避免字典随会话数无限增长。"""
        self._runs.pop(run_id, None)
        self._first_token_logged.discard(run_id)
