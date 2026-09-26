"""Persist versioned interactive optimization sessions.

Revision ID: 0004_optimization_sessions
Revises: 0003_preparation
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_optimization_sessions"
down_revision = "0003_preparation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "optimization_sessions",
        sa.Column(
            "task_id", sa.String(length=36), sa.ForeignKey("tasks.task_id"), primary_key=True
        ),
        sa.Column(
            "run_id", sa.String(length=36), sa.ForeignKey("experiment_runs.run_id"), unique=True
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("optimization_sessions")
