"""聊天接口的请求与响应模型。"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """``POST /api/v1/chat`` 请求体。"""

    message: str = Field(min_length=1, description="用户消息内容")
    session_id: str | None = Field(
        default=None,
        description="已有会话 ID；为空时新建会话",
    )


class MessageOut(BaseModel):
    """会话中的一条账本消息。"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    agent_source: str | None = None
    created_at: datetime


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
