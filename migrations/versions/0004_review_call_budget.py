"""Allow the bounded evidence review alongside one draft repair.

Revision ID: 0004_review_budget
Revises: 0003_conversation
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_review_budget"
down_revision = "0003_conversation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_turn_counts", "chat_turns", type_="check")
    op.create_check_constraint(
        "ck_turn_counts",
        "chat_turns",
        "turn_number > 0 AND ai_calls_reserved >= 0 AND ai_calls_reserved <= 5",
    )


def downgrade() -> None:
    # Retain honest call history; never rewrite completed turns to fit an older limit.
    if op.get_bind().scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM chat_turns WHERE ai_calls_reserved > 4)")
    ):
        raise RuntimeError("Cannot downgrade while turns retain five-call history.")
    op.drop_constraint("ck_turn_counts", "chat_turns", type_="check")
    op.create_check_constraint(
        "ck_turn_counts",
        "chat_turns",
        "turn_number > 0 AND ai_calls_reserved >= 0 AND ai_calls_reserved <= 4",
    )
