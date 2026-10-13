"""add metric series uniqueness and file content hash

Revision ID: b3f7c21a9d05
Revises: 6d9015cf96fb
Create Date: 2026-10-16 10:12:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b3f7c21a9d05"
down_revision: str | None = "6d9015cf96fb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 上传文件加内容哈希：同一用户再传一次同一个文件时可以直接拦下，不必重跑解析。
    op.add_column(
        "file_metadata",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_file_metadata_user_content_hash",
        "file_metadata",
        ["user_id", "content_hash"],
        unique=False,
    )

    # 指标表从「挂在某次确认下面」改成「按天去重的时间序列」：身份由 record_id 换成
    # (user_id, measured_date, zh_name)，写入走 upsert。流水账不再向下挂指标。
    # 两表一起重建——唯一约束和列语义都变了，存量数据本来就是开发期的测试数据，
    # 直接清空重来比逐列搬迁更省事，也不会留下对不上新主键的孤儿行。
    op.drop_table("health_metrics")
    op.drop_table("health_records")

    op.create_table(
        "health_records",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("file_id", sa.UUID(), nullable=True),
        sa.Column("measured_date", sa.Date(), nullable=False),
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
        op.f("ix_health_records_user_id"),
        "health_records",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "health_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("measured_date", sa.Date(), nullable=False),
        sa.Column("source_file_id", sa.UUID(), nullable=True),
        sa.Column(
            "en_name",
            sa.String(length=128),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column("zh_name", sa.String(length=128), nullable=False),
        sa.Column("value", sa.String(length=128), nullable=False),
        sa.Column(
            "unit",
            sa.String(length=64),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "reference_range",
            sa.String(length=128),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "position",
            sa.Integer(),
            server_default=sa.text("0"),
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
        sa.ForeignKeyConstraint(
            ["source_file_id"], ["file_metadata.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "measured_date",
            "zh_name",
            name="uq_health_metrics_series",
        ),
    )
    op.create_index(
        op.f("ix_health_metrics_user_id"),
        "health_metrics",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_health_metrics_measured_date"),
        "health_metrics",
        ["measured_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_health_metrics_measured_date"), table_name="health_metrics")
    op.drop_index(op.f("ix_health_metrics_user_id"), table_name="health_metrics")
    op.drop_table("health_metrics")
    op.drop_index(op.f("ix_health_records_user_id"), table_name="health_records")
    op.drop_table("health_records")

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
        op.f("ix_health_records_user_id"),
        "health_records",
        ["user_id"],
        unique=False,
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
            "unit",
            sa.String(length=64),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "reference_range",
            sa.String(length=128),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "position",
            sa.Integer(),
            server_default=sa.text("0"),
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
        sa.ForeignKeyConstraint(
            ["record_id"], ["health_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_health_metrics_record_id"),
        "health_metrics",
        ["record_id"],
        unique=False,
    )

    op.drop_index("ix_file_metadata_user_content_hash", table_name="file_metadata")
    op.drop_column("file_metadata", "content_hash")
