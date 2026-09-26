"""Enforce one original forecast per deployment session.

Revision ID: 0007_one_original
Revises: 0006_deployment_workflow
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_one_original"
down_revision = "0006_deployment_workflow"
branch_labels = None
depends_on = None


def upgrade() -> None:
    duplicates = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT session_id FROM forecasts WHERE session_id IS NOT NULL "
                "GROUP BY session_id HAVING COUNT(*) > 1"
            )
        )
        .first()
    )
    if duplicates is not None:
        raise RuntimeError(
            "Multiple original forecasts exist for a deployment session; "
            "resolve ownership explicitly before upgrading. No records were removed."
        )
    op.drop_index("ix_forecasts_session_id", table_name="forecasts")
    op.create_index("ix_forecasts_session_id", "forecasts", ["session_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_forecasts_session_id", table_name="forecasts")
    op.create_index("ix_forecasts_session_id", "forecasts", ["session_id"])
