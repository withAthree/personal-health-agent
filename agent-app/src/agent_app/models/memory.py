"""长期记忆条目。

一行 = 一条关于**该用户本人**的、长期有效的说法。这些行会被直接注入
system prompt（不走向量召回，见 ``memory/store.py``），所以：

* 表很小、行很少，但每一行都会消耗每次对话的窗口；
* ``user_id`` 上的过滤条件漏一个就是跨用户泄漏，不是「多取了几行」。
"""

from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from agent_app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

#: 允许的类别。这五个的划分依据是**注入策略**而非语义纯度：
#: ``health_fact`` / ``obstacle`` / ``goal`` 漏掉一次就是一次错答，必须无条件在场；
#: ``preference`` / ``insight`` 属于锦上添花，超量时按新旧裁剪。
MEMORY_CATEGORIES = ("insight", "goal", "obstacle", "preference", "health_fact")

#: CHECK 约束的 SQL 文本。刻意不排序，与迁移里的字面量保持一致，
#: 免得 alembic 把同一约束看成两处定义。
_CATEGORY_SQL = ", ".join(f"'{category}'" for category in MEMORY_CATEGORIES)


class MemoryItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """一条长期记忆。"""

    __tablename__ = "memory_items"
    __table_args__ = (
        CheckConstraint(
            f"category IN ({_CATEGORY_SQL})",
            name="ck_memory_items_category",
        ),
        Index("idx_memory_items_user_category", "user_id", "category"),
        Index("idx_memory_items_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: 抽取器自报的把握（0~1）。**没有校准过**，只用于排序与事后分析，
    #: 当前不作为写入门槛（见 ``settings.memory_min_confidence``）。
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: 这条记忆是从哪次会话抽出来的。会话被删时置空，记忆本身保留。
    source_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
