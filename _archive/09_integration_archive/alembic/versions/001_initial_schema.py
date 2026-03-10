"""
資料庫初始化遷移

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""

from alembic import op  # type: ignore[import-not-found]

from sqlalchemy.dialects import postgresql  # type: ignore[import-not-found]

# revision identifiers
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """創建初始資料庫表結構"""

    # 創建 scans 表
    op.create_table(
        "scans",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column("targets", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("error_info", sa.Text(), nullable=True),
        sa.Column("total_tasks", sa.Integer(), server_default="0", nullable=True),
        sa.Column("completed_tasks", sa.Integer(), server_default="0", nullable=True),
        sa.Column("failed_tasks", sa.Integer(), server_default="0", nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # 創建 findings 表
    op.create_table(
        "findings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("finding_id", sa.String(255), nullable=False),
        sa.Column("scan_id", sa.String(255), nullable=False),
        sa.Column("task_id", sa.String(255), nullable=False),
        sa.Column("vulnerability_name", sa.String(255), nullable=False),
        sa.Column("severity", sa.String(50), nullable=False),
        sa.Column("confidence", sa.String(50), nullable=False),
        sa.Column("cwe", sa.String(50), nullable=True),
        sa.Column("target_url", sa.Text(), nullable=False),
        sa.Column("target_parameter", sa.String(255), nullable=True),
        sa.Column("target_method", sa.String(10), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("raw_data", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["scan_id"], ["scans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("finding_id"),
    )

    # 創建 reports 表
    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("report_id", sa.String(255), nullable=False),
        sa.Column("scan_id", sa.String(255), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("format", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), server_default="generating", nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("file_path", sa.String(1000), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["scan_id"], ["scans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("report_id"),
    )

    # 創建 task_executions 表
    op.create_table(
        "task_executions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(255), nullable=False),
        sa.Column("scan_id", sa.String(255), nullable=False),
        sa.Column("worker_id", sa.String(255), nullable=True),
        sa.Column("task_type", sa.String(100), nullable=False),
        sa.Column("target_url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(50), server_default="pending", nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Numeric(10, 3), nullable=True),
        sa.Column("findings_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["scan_id"], ["scans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # 創建索引
    op.create_index("idx_scans_status", "scans", ["status"])
    op.create_index("idx_scans_created_at", "scans", ["created_at"])

    op.create_index("idx_findings_finding_id", "findings", ["finding_id"])
    op.create_index("idx_findings_scan_id", "findings", ["scan_id"])
    op.create_index("idx_findings_task_id", "findings", ["task_id"])
    op.create_index("idx_findings_severity", "findings", ["severity"])
    op.create_index("idx_findings_status", "findings", ["status"])
    op.create_index("idx_findings_created_at", "findings", ["created_at"])

    op.create_index("idx_reports_report_id", "reports", ["report_id"])
    op.create_index("idx_reports_scan_id", "reports", ["scan_id"])
    op.create_index("idx_reports_status", "reports", ["status"])
    op.create_index("idx_reports_format", "reports", ["format"])
    op.create_index("idx_reports_created_at", "reports", ["created_at"])

    op.create_index("idx_task_executions_task_id", "task_executions", ["task_id"])
    op.create_index("idx_task_executions_scan_id", "task_executions", ["scan_id"])
    op.create_index("idx_task_executions_status", "task_executions", ["status"])
    op.create_index("idx_task_executions_task_type", "task_executions", ["task_type"])
    op.create_index("idx_task_executions_started_at", "task_executions", ["started_at"])


def downgrade() -> None:
    """撤銷遷移"""
    op.drop_table("task_executions")
    op.drop_table("reports")
    op.drop_table("findings")
    op.drop_table("scans")
