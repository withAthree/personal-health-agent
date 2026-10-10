"""add session compaction marker

Revision ID: 6627af21c4cc
Revises: 21e09caeda22
Create Date: 2026-10-11 01:15:09.855479

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6627af21c4cc"
down_revision: str | None = "21e09caeda22"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 自动生成时把 LangGraph 的 checkpoint_* 表也识别成了「待删除」——那些表由
    # checkpointer 自己创建与管理（见 alembic/env.py 的 include_object），
    # 这里手动删掉相关命令，只保留真正属于本次改动的列。
    op.add_column(
        "chat_sessions",
        sa.Column(
            "compacted_message_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "compacted_message_count")
