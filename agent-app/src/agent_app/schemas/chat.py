"""聊天接口的请求模型。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """``POST /api/v1/chat`` 请求体。"""

    message: str = Field(min_length=1, description="用户消息内容")
    session_id: str | None = Field(
        default=None,
        description="已有会话 ID；为空时新建会话",
    )
