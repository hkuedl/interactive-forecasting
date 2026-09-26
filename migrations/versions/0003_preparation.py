"""Persist versioned Preparation aggregate.

Revision ID: 0003_preparation
Revises: 0002_experiment_version
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_preparation"
down_revision = "0002_experiment_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "preparations",
        sa.Column(
            "task_id", sa.String(length=36), sa.ForeignKey("tasks.task_id"), primary_key=True
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("preparations")
