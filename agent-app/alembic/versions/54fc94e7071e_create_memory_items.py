"""create memory items

Revision ID: 54fc94e7071e
Revises: bbc4e5fae92a
Create Date: 2026-10-11 19:19:40.813028

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "54fc94e7071e"
down_revision: str | None = "bbc4e5fae92a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source_session_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "category IN ('insight', 'goal', 'obstacle', 'preference', 'health_fact')",
            name="ck_memory_items_category",
        ),
        sa.ForeignKeyConstraint(
            ["source_session_id"],
            ["chat_sessions.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_memory_items_user_category",
        "memory_items",
        ["user_id", "category"],
        unique=False,
    )
    op.create_index(
        "idx_memory_items_user_created",
        "memory_items",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_memory_items_user_created", table_name="memory_items")
    op.drop_index("idx_memory_items_user_category", table_name="memory_items")
    op.drop_table("memory_items")
