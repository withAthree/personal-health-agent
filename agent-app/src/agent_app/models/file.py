"""上传文件的元信息与解析状态。"""

from __future__ import annotations

import uuid
from typing import Literal

from sqlalchemy import BigInteger, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from agent_app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

#: 解析流水线的状态。
#:
#: ``pending`` 刚落盘、还在队列里；``parsing`` worker 正在读它；``parsed`` 已经拿到
#: 草稿 JSON；``confirmed`` 用户点了确定、内容已进正式表；``failed`` 见
#: ``error_message``。``parsed`` 和 ``confirmed`` 的唯一差别就是有没有写正式表。
FileStatus = Literal["pending", "parsing", "parsed", "failed", "confirmed"]


class FileMetadata(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """一个上传的文件。

    ``parsed_content`` 是**草稿**：模型抽出来什么样就原样存进来，用户在页面上改的也是
    它。正式数据在 ``health_records`` / ``health_metrics``，由用户确认后才写入——
    草稿表可以随便改，正式表只存放用户认可的事实。
    """

    __tablename__ = "file_metadata"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    #: 磁盘上的文件名（``{id}.{ext}``），不是用户上传时的名字。
    filename: Mapped[str] = mapped_column(String(255))
    #: 用户上传时的原始文件名，只用于展示。
    original_name: Mapped[str] = mapped_column(String(255))
    #: 相对 ``settings.upload_dir`` 的路径，形如 ``{user_id}/{id}.{ext}``。
    file_path: Mapped[str] = mapped_column(String(1024))
    file_size: Mapped[int] = mapped_column(BigInteger)
    mime_type: Mapped[str] = mapped_column(String(128))
    extension: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    #: 解析草稿，形状见 ``agent_app.parsing.schema.ParsedReport``。
    parsed_content: Mapped[dict[str, object] | None] = mapped_column(
        JSONB, nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
