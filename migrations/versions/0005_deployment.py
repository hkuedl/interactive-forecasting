"""Persist deployment sessions, forecasts and immutable forecast versions.

Revision ID: 0005_deployment
Revises: 0004_optimization_sessions
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_deployment"
down_revision = "0004_optimization_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deployment_sessions",
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.task_id"), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.create_table(
        "forecasts",
        sa.Column("forecast_id", sa.String(36), primary_key=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.task_id"), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.create_index("ix_forecasts_task_id", "forecasts", ["task_id"])
    op.create_table(
        "forecast_versions",
        sa.Column("version_id", sa.String(36), primary_key=True),
        sa.Column(
            "forecast_id", sa.String(36), sa.ForeignKey("forecasts.forecast_id"), nullable=False
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.UniqueConstraint("forecast_id", "version_number"),
    )
    op.create_index("ix_forecast_versions_forecast_id", "forecast_versions", ["forecast_id"])


def downgrade() -> None:
    op.drop_index("ix_forecast_versions_forecast_id", table_name="forecast_versions")
    op.drop_table("forecast_versions")
    op.drop_index("ix_forecasts_task_id", table_name="forecasts")
    op.drop_table("forecasts")
    op.drop_table("deployment_sessions")
