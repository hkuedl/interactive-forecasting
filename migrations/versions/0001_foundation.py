"""Foundational metadata tables.

Revision ID: 0001_foundation
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("task_id", sa.String(36), primary_key=True),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("substate", sa.String(80)),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "messages",
        sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("message_id", sa.String(36), nullable=False, unique=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.task_id"), nullable=False),
        sa.Column("correlation_id", sa.String(36), nullable=False),
        sa.Column("parent_message_id", sa.String(36)),
        sa.Column("topic", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_task_id", "messages", ["task_id"])
    op.create_index("ix_messages_correlation_id", "messages", ["correlation_id"])
    op.create_table(
        "message_deliveries",
        sa.Column("delivery_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "message_id", sa.String(36), sa.ForeignKey("messages.message_id"), nullable=False
        ),
        sa.Column("handler_name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_message_deliveries_message_id", "message_deliveries", ["message_id"])
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.task_id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_task_id", "jobs", ["task_id"])
    op.create_table(
        "experiment_runs",
        sa.Column("run_id", sa.String(36), primary_key=True),
        sa.Column("spec_id", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.create_table(
        "llm_calls",
        sa.Column("call_id", sa.String(36), primary_key=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.task_id"), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
    )
    op.create_index("ix_llm_calls_task_id", "llm_calls", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_calls_task_id", table_name="llm_calls")
    op.drop_table("llm_calls")
    op.drop_table("experiment_runs")
    op.drop_index("ix_jobs_task_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_message_deliveries_message_id", table_name="message_deliveries")
    op.drop_table("message_deliveries")
    op.drop_index("ix_messages_correlation_id", table_name="messages")
    op.drop_index("ix_messages_task_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("tasks")
