"""聊天接口的请求与响应模型。"""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """``POST /api/v1/chat`` 请求体。"""

    message: str = Field(min_length=1, description="用户消息内容")
    session_id: str | None = Field(
        default=None,
        description="已有会话 ID；为空时新建会话",
    )
    retry: bool = Field(
        default=False,
        description=(
            "重试本会话最后一条用户消息：不新增用户消息，先作废上一轮未完成的回复，"
            "再重新回答。必须与 session_id 一起使用，此时 message 会被忽略。"
        ),
    )


class MessageOut(BaseModel):
    """会话中的一条账本消息。"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    agent_source: str | None = None
    #: 该助手回复是否被用户主动终止（前端据此显示「已终止」并给出重试入口）。
    interrupted: bool = False
    created_at: datetime


class SessionOut(BaseModel):
    """左侧「对话记录列表」里的一条会话。

    只带列表渲染需要的元信息，消息正文仍按需走
    ``GET /chat/{session_id}/messages``。
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None = None
    created_at: datetime
    #: 会话内最后一次写入的时间；列表按它倒序，最近说过的排最前。
    updated_at: datetime


class SessionListOut(BaseModel):
    """``GET /api/v1/chat/sessions`` 响应体。"""

    sessions: list[SessionOut]


class SessionMessagesOut(BaseModel):
    """``GET /api/v1/chat/{session_id}/messages`` 响应体。

    ``messages`` 是账本全量消息（发给模型的摘要不会出现在这里），
    ``compacted_message_count`` 表示前多少条已经被摘要覆盖。
    """

    session_id: uuid.UUID
    title: str | None = None
    compacted_message_count: int = 0
    messages: list[MessageOut]


class SessionSummaryOut(BaseModel):
    """【临时】``GET /api/v1/chat/{session_id}/summary`` 响应体。

    TODO(临时): 压缩功能验收完成后，连同该接口一起删除。
    """

    session_id: uuid.UUID
    summary: str | None = None


#: 过程块的类型。``thinking`` / ``text`` 是本版实现，其余为协议预留（见
#: ``document/中间过程可见-设计方案.md`` §5.4）。
StreamKind = Literal[
    "thinking",
    "text",
    "tool_call",
    "tool_result",
    "plan",
    "image",
    "error",
]

#: 单个过程块的状态。
StreamBlockStatus = Literal["running", "complete", "interrupted", "error"]

#: 整轮生成的状态。
StreamStatus = Literal["running", "done", "error", "interrupted"]


class StreamBlockOut(BaseModel):
    """一个过程块（思考 / 正文，将来的工具调用）。"""

    #: 块身份。增量事件只引用它，不引用 ``index``（``reset`` 会让 ``index`` 复用）。
    block_id: str
    #: 本次生成内的绝对槽位，从 0 起，跨 step 不重置。
    index: int
    #: 轮次，从 0 起；多 Agent 上线前恒为 0。
    step: int
    kind: StreamKind
    text: str
    status: StreamBlockStatus
    #: 按 ``kind`` 定义。``thinking`` 预留 ``{"signature": ..., "redacted": ...}``。
    meta: dict[str, Any] | None = None


class StreamStateOut(BaseModel):
    """``GET /api/v1/chat/{session_id}/stream-state`` 响应体。

    这是本轮对话过程（思维链，将来的工具调用）的聚合态，只服务展示：存在
    Redis 里、有 TTL，刷新页面后可以回放，**不进 ``messages`` 账本**。
    """

    session_id: uuid.UUID
    run_id: str
    status: StreamStatus
    started_at: float
    finished_at: float | None = None
    blocks: list[StreamBlockOut]
    #: 本轮结束时算出的「已被摘要覆盖的消息条数」，与账本上的压缩标记同源。
    compaction: dict[str, Any] | None = None
    #: 预留，本版不填。
    usage: dict[str, Any] | None = None
