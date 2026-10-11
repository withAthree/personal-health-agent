"""add messages interrupted flag

Revision ID: bbc4e5fae92a
Revises: 6627af21c4cc
Create Date: 2026-10-11 02:25:44.904114

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "bbc4e5fae92a"
down_revision: str | None = "6627af21c4cc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 自动生成时同时比对出了 LangGraph 的 checkpoint_* 表（由 checkpointer 自己
    # 管理，见 alembic/env.py 的 include_object），此处只保留本次真正新增的列。
    op.add_column(
        "messages",
        sa.Column(
            "interrupted", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_column("messages", "interrupted")
