"""正式表：用户确认之后才写入的健康数据。

两张表分工不同：

``health_metrics`` 是**当前有效的指标值**，身份是 ``(user_id, measured_date,
zh_name)``——同一个指标的同一天只有一个值，后来者覆盖先前的。这样「今天中午传了
步数 3000、晚上传了 8000」留下的是 8000，「今天传过、明天重新截图今天的数据又传
一遍」也不会多出一行。

``health_records`` 是**只增不改的流水账**，记录每次确认入库的日期与结论。它不再
向下挂指标：指标已经按天去重，再挂一次层级只会让「这次确认改了哪几条」变得没有
答案。

``findings`` 留在流水账上而不是指标表里——它是自由文本结论（「肝内钙化灶」），
没有「某一天的值」这种语义，去重反而会丢掉「哪次体检发现的」。
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import (
    Date,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from agent_app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class HealthRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """一次「用户确认入库」产生的流水账。

    每条记录来自一个上传文件，但 ``file_id`` 可以为空：将来支持手工录入时就没有文件。
    确认之后这份记录不再改动——用户想修正只能重新上传一份（见
    ``agent_app.api.v1.upload.confirm_file``）。
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
    #: 测量日期。粒度到天，一律北京时间——解析出日期就用解析出来的，解析不出用上传
    #: 时间兜底（同样换算到北京时间再取日期）。
    measured_date: Mapped[date] = mapped_column(Date)
    #: 原文里的日期写法（如 ``10月12日，周一``），保留下来便于人工核对。
    date_raw: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("''")
    )
    #: ``otherInfoList``：结论性的自由文本，如「肝内钙化灶」。
    findings: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )


class HealthMetric(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """某一天某个指标的一个值，对应草稿 ``dataList`` 的一项。

    ``(user_id, measured_date, zh_name)`` 唯一。写入走 upsert：同一指标同一天重复
    出现时更新值，而不是插一条新的。
    """

    __tablename__ = "health_metrics"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "measured_date",
            "zh_name",
            name="uq_health_metrics_series",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    #: 测量日期。粒度到天，一律北京时间；与 ``zh_name`` 一起构成一条时间序列的身份。
    measured_date: Mapped[date] = mapped_column(Date, index=True)
    #: 最后一次写入这一行的文件。用于追溯「这个值是哪次上传带来的」。
    source_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("file_metadata.id", ondelete="SET NULL"),
        nullable=True,
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
