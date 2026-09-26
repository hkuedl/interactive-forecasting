"""Add multi-session deployment and versioned reference/adjustment records.

Revision ID: 0006_deployment_workflow
Revises: 0005_deployment
"""

import json
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision = "0006_deployment_workflow"
down_revision = "0005_deployment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    op.create_table(
        "deployment_sessions_next",
        sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(36), nullable=False, unique=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.task_id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.add_column("forecasts", sa.Column("session_id", sa.String(36), nullable=True))
    for row in bind.execute(sa.text("SELECT task_id, version, document FROM deployment_sessions")):
        document = json.loads(row.document) if isinstance(row.document, str) else dict(row.document)
        session_id = str(document.get("session_id") or uuid4())
        document["session_id"] = session_id
        if document.get("forecast_id"):
            version = bind.execute(
                sa.text(
                    "SELECT version_id FROM forecast_versions "
                    "WHERE forecast_id=:forecast_id AND version_number=0"
                ),
                {"forecast_id": str(document["forecast_id"])},
            ).first()
            if version is not None:
                document["current_version_id"] = version.version_id
            forecast_row = bind.execute(
                sa.text("SELECT document FROM forecasts WHERE forecast_id=:forecast_id"),
                {"forecast_id": str(document["forecast_id"])},
            ).first()
            if forecast_row is not None:
                forecast_doc = (
                    json.loads(forecast_row.document)
                    if isinstance(forecast_row.document, str)
                    else dict(forecast_row.document)
                )
                forecast_doc["deployment_session_id"] = session_id
                bind.execute(
                    sa.text(
                        "UPDATE forecasts SET session_id=:session_id, document=:document "
                        "WHERE forecast_id=:forecast_id"
                    ).bindparams(sa.bindparam("document", type_=sa.JSON())),
                    {
                        "session_id": session_id,
                        "document": forecast_doc,
                        "forecast_id": str(document["forecast_id"]),
                    },
                )
        bind.execute(
            sa.text(
                "INSERT INTO deployment_sessions_next "
                "(session_id, task_id, version, document) VALUES "
                "(:session_id, :task_id, :version, :document)"
            ).bindparams(sa.bindparam("document", type_=sa.JSON())),
            {
                "session_id": session_id,
                "task_id": row.task_id,
                "version": row.version,
                "document": document,
            },
        )
    op.drop_table("deployment_sessions")
    op.rename_table("deployment_sessions_next", "deployment_sessions")
    op.create_index("ix_deployment_sessions_task_id", "deployment_sessions", ["task_id"])
    op.create_index("ix_forecasts_session_id", "forecasts", ["session_id"])
    op.create_table(
        "reference_analyses",
        sa.Column(
            "forecast_id",
            sa.String(36),
            sa.ForeignKey("forecasts.forecast_id"),
            primary_key=True,
        ),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.create_table(
        "adjustments",
        sa.Column("adjustment_id", sa.String(36), primary_key=True),
        sa.Column(
            "forecast_id",
            sa.String(36),
            sa.ForeignKey("forecasts.forecast_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.create_index("ix_adjustments_forecast_id", "adjustments", ["forecast_id"])


def downgrade() -> None:
    raise RuntimeError(
        "5B multi-session downgrade would discard forecast sessions and version lineage"
    )
