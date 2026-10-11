"""add chat_sessions pinned flag

Revision ID: e4b1c7d29f30
Revises: 54fc94e7071e
Create Date: 2026-10-11 12:30:12.104885

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4b1c7d29f30"
down_revision: str | None = "54fc94e7071e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 自动生成时同时比对出了 LangGraph 的 checkpoint_* 表（由 checkpointer 自己
    # 管理，见 alembic/env.py 的 include_object），此处只保留本次真正新增的列。
    op.add_column(
        "chat_sessions",
        sa.Column(
            "is_pinned", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "is_pinned")
