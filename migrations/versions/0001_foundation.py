"""Enable pgvector and persist immutable index recipes.

Revision ID: 0001_foundation
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "index_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("embedding_provider", sa.String(40), nullable=False),
        sa.Column("embedding_model", sa.String(120), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("tokenizer", sa.String(80), nullable=False),
        sa.Column("normalization_version", sa.String(40), nullable=False),
        sa.Column("chunker_version", sa.String(80), nullable=False),
        sa.Column("chunk_size_tokens", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap_tokens", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fingerprint", name="uq_index_profiles_fingerprint"),
        sa.CheckConstraint("dimensions = 1536", name="ck_index_profiles_dimensions"),
        sa.CheckConstraint("chunk_size_tokens > 0", name="ck_index_profiles_chunk_size"),
        sa.CheckConstraint(
            "chunk_overlap_tokens >= 0 AND chunk_overlap_tokens < chunk_size_tokens",
            name="ck_index_profiles_overlap",
        ),
    )


def downgrade() -> None:
    op.drop_table("index_profiles")
    # Extension may be shared by later schemas; do not drop it implicitly.
