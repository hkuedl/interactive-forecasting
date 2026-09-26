"""Add optimistic version to persisted experiment runs.

Revision ID: 0002_experiment_version
Revises: 0001_foundation
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_experiment_version"
down_revision = "0001_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("experiment_runs") as batch:
        batch.add_column(sa.Column("version", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("experiment_runs") as batch:
        batch.drop_column("version")
