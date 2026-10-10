"""对话会话模型。"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agent_app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from agent_app.models.message import Message


class ChatSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """一次对话会话，包含多条消息。"""

    __tablename__ = "chat_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cuj_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    #: 已被摘要覆盖的消息条数（前端据此画「已压缩」分隔线，仅用于展示）。
    compacted_message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
