"""正式表：用户确认之后才写入的健康数据。

拆成「报告头 + 指标条目」两层，是为了让指标能被查询和聚合（「最近三个月血红蛋白
多少」），而不是把整份 JSON 塞进一个字段了事。``findings`` 例外——它是自由文本结论，
当前没有检索需求，先用 JSONB 存着。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agent_app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class HealthRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """一次「用户确认入库」产生的报告。

    每条记录来自一个上传文件，但 ``file_id`` 可以为空：将来支持手工录入时就没有文件。
    """

    __tablename__ = "health_records"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("file_metadata.id", ondelete="SET NULL"),
        nullable=True,
    )
    #: 测量时间。解析出日期就用解析出来的，解析不出用上传时间兜底。
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    #: 原文里的日期写法（如 ``10月12日，周一``），保留下来便于人工核对。
    date_raw: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("''")
    )
    #: ``otherInfoList``：结论性的自由文本，如「肝内钙化灶」。
    findings: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    metrics: Mapped[list[HealthMetric]] = relationship(
        back_populates="record",
        cascade="all, delete-orphan",
        order_by="HealthMetric.position",
    )


class HealthMetric(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """报告里的一条指标，对应草稿 ``dataList`` 的一项。"""

    __tablename__ = "health_metrics"

    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("health_records.id", ondelete="CASCADE"),
        index=True,
    )
    #: 原文里的英文缩写（``Hb``）。多数场景没有，允许空串。
    en_name: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default=text("''")
    )
    zh_name: Mapped[str] = mapped_column(String(128))
    #: 一律字符串：``2.69`` / ``10小时33分钟`` / ``52-171`` 都是合法值。
    value: Mapped[str] = mapped_column(String(128))
    unit: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("''")
    )
    reference_range: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default=text("''")
    )
    #: 在 ``dataList`` 里的原始顺序，保证回显时指标顺序不乱。
    position: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    record: Mapped[HealthRecord] = relationship(back_populates="metrics")
