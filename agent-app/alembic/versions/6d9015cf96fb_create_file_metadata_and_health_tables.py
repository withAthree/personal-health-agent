"""create file metadata and health tables

Revision ID: 6d9015cf96fb
Revises: e4b1c7d29f30
Create Date: 2026-10-13 23:42:59.091043

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "6d9015cf96fb"
down_revision: str | None = "e4b1c7d29f30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 上传的文件：元信息 + 解析状态 + 草稿 JSON。
    op.create_table(
        "file_metadata",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("extension", sa.String(length=32), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column(
            "parsed_content", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_file_metadata_user_id"), "file_metadata", ["user_id"], unique=False
    )

    # 正式数据：报告头 + 指标条目，用户确认草稿后才写入。
    op.create_table(
        "health_records",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("file_id", sa.UUID(), nullable=True),
        sa.Column("measured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "date_raw",
            sa.String(length=64),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "findings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
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
        sa.ForeignKeyConstraint(["file_id"], ["file_metadata.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_health_records_user_id"), "health_records", ["user_id"], unique=False
    )

    op.create_table(
        "health_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("record_id", sa.UUID(), nullable=False),
        sa.Column(
            "en_name",
            sa.String(length=128),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column("zh_name", sa.String(length=128), nullable=False),
        sa.Column("value", sa.String(length=128), nullable=False),
        sa.Column(
            "unit", sa.String(length=64), server_default=sa.text("''"), nullable=False
        ),
        sa.Column(
            "reference_range",
            sa.String(length=128),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), server_default=sa.text("0"), nullable=False),
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
        sa.ForeignKeyConstraint(["record_id"], ["health_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_health_metrics_record_id"), "health_metrics", ["record_id"], unique=False
    )

    # memory_items.source_file_id 是 wip/upload-parse 分支留下的列：外键指向上面那张
    # 被重建的旧 file_metadata，当前模型没有对应字段，存量数据全为 NULL。一并移除，
    # 让数据库与模型保持一致；将来真的需要「记忆来自哪个文件」时再加一次迁移。
    op.drop_index(op.f("idx_memory_items_source_file"), table_name="memory_items")
    op.drop_column("memory_items", "source_file_id")


def downgrade() -> None:
    op.add_column(
        "memory_items",
        sa.Column("source_file_id", sa.UUID(), autoincrement=False, nullable=True),
    )
    op.create_index(
        op.f("idx_memory_items_source_file"),
        "memory_items",
        ["source_file_id"],
        unique=False,
    )

    op.drop_index(op.f("ix_health_metrics_record_id"), table_name="health_metrics")
    op.drop_table("health_metrics")
    op.drop_index(op.f("ix_health_records_user_id"), table_name="health_records")
    op.drop_table("health_records")
    op.drop_index(op.f("ix_file_metadata_user_id"), table_name="file_metadata")
    op.drop_table("file_metadata")
