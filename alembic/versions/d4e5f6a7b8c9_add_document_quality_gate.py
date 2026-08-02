"""Add document ingestion quality state.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-08-01 22:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("quality_status", sa.Text(), server_default="unchecked", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column(
            "quality_report",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("documents", sa.Column("quality_checked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "documents",
        sa.Column("ingestion_version", sa.Text(), server_default="1", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column("indexing_status", sa.Text(), server_default="idle", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column("duplicate_of_document_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_documents_duplicate_of_document_id",
        "documents",
        "documents",
        ["duplicate_of_document_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_documents_quality_status", "documents", ["quality_status"])
    op.create_index("ix_documents_indexing_status", "documents", ["indexing_status"])
    op.create_index(
        "ix_documents_duplicate_of_document_id",
        "documents",
        ["duplicate_of_document_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_duplicate_of_document_id", table_name="documents")
    op.drop_index("ix_documents_indexing_status", table_name="documents")
    op.drop_index("ix_documents_quality_status", table_name="documents")
    op.drop_constraint("fk_documents_duplicate_of_document_id", "documents", type_="foreignkey")
    op.drop_column("documents", "duplicate_of_document_id")
    op.drop_column("documents", "indexing_status")
    op.drop_column("documents", "ingestion_version")
    op.drop_column("documents", "quality_checked_at")
    op.drop_column("documents", "quality_report")
    op.drop_column("documents", "quality_status")
