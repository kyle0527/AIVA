from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (  # type: ignore[import-not-found]
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import (  # type: ignore[import-not-found]
    declarative_base,
)
from sqlalchemy.orm import Session, sessionmaker  # type: ignore[import-not-found]

from services.aiva_common.schemas import FindingPayload

from .test_result_database import TestResultDatabase  # type: ignore[import-not-found]

Base = declarative_base()  # type: ignore[misc]


class FindingRecord(Base):  # type: ignore[misc, valid-type]
    """漏洞發現記錄表"""

    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    finding_id = Column(String(255), unique=True, nullable=False, index=True)
    scan_id = Column(String(255), nullable=False, index=True)
    task_id = Column(String(255), nullable=False, index=True)

    # 漏洞信息
    vulnerability_name = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False, index=True)
    confidence = Column(String(50), nullable=False)
    cwe = Column(String(50))

    # 目標信息
    target_url = Column(Text, nullable=False)
    target_parameter = Column(String(255))
    target_method = Column(String(10))

    # 狀態
    status = Column(String(50), nullable=False, index=True)

    # 完整的 JSON 數據
    raw_data = Column(Text, nullable=False)

    # 時間戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_finding_payload(self) -> FindingPayload:
        """轉換為 FindingPayload 對象"""
        # 從資料庫欄位取得字串值
        raw_data_str: str = str(self.raw_data)  # type: ignore[attr-defined]
        return FindingPayload.model_validate_json(raw_data_str)


class SqlResultDatabase(TestResultDatabase):
    """SQL 資料庫實現的結果存儲"""

    def __init__(
        self,
        database_url: str,
        auto_migrate: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
    ):
        self.database_url = database_url
        self.auto_migrate = auto_migrate

        # 創建引擎，處理不同資料庫類型的參數
        engine_args: dict[str, Any] = {"echo": False}  # 生產環境不輸出 SQL

        # 只為非 SQLite 資料庫添加連接池參數
        if not database_url.startswith("sqlite://"):
            engine_args.update({
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout,
                "pool_recycle": pool_recycle,
            })

        self.engine = create_engine(database_url, **engine_args)

        # 創建 session factory
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # 自動創建表（如果啟用）
        if auto_migrate:
            self._create_tables()

    def _create_tables(self) -> None:
        """創建資料庫表"""
        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        """獲取資料庫 session"""
        return self.SessionLocal()

    async def save_finding(self, finding: FindingPayload) -> None:
        """保存漏洞發現到資料庫"""
        session = self._get_session()
        try:
            # 檢查是否已存在
            existing = (
                session.query(FindingRecord)
                .filter_by(finding_id=finding.finding_id)
                .first()
            )

            if existing:
                # 更新現有記錄
                existing.status = finding.status  # type: ignore[misc]
                existing.raw_data = finding.model_dump_json()  # type: ignore[misc]
                existing.updated_at = datetime.utcnow()  # type: ignore[misc]
            else:
                # 創建新記錄
                record = FindingRecord(
                    finding_id=finding.finding_id,
                    scan_id=finding.scan_id,
                    task_id=finding.task_id,
                    vulnerability_name=finding.vulnerability.name,
                    severity=finding.vulnerability.severity,
                    confidence=finding.vulnerability.confidence,
                    cwe=finding.vulnerability.cwe,
                    target_url=str(finding.target.url),
                    target_parameter=finding.target.parameter,
                    target_method=finding.target.method,
                    status=finding.status,
                    raw_data=finding.model_dump_json(),
                )
                session.add(record)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    async def get_finding(self, finding_id: str) -> FindingPayload | None:
        """根據 ID 獲取漏洞發現"""
        session = self._get_session()
        try:
            record = (
                session.query(FindingRecord).filter_by(finding_id=finding_id).first()
            )

            if record:
                return record.to_finding_payload()
            return None
        finally:
            session.close()

    async def list_findings(
        self,
        scan_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FindingPayload]:
        """列出漏洞發現"""
        session = self._get_session()
        try:
            query = session.query(FindingRecord)

            if scan_id:
                query = query.filter_by(scan_id=scan_id)

            if severity:
                query = query.filter_by(severity=severity)

            records = (
                query.order_by(FindingRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return [record.to_finding_payload() for record in records]
        finally:
            session.close()

    async def count_findings(
        self,
        scan_id: str | None = None,
        severity: str | None = None,
    ) -> int:
        """統計漏洞發現數量"""
        session = self._get_session()
        try:
            query = session.query(func.count(FindingRecord.id))

            if scan_id:
                query = query.filter(FindingRecord.scan_id == scan_id)

            if severity:
                query = query.filter(FindingRecord.severity == severity)

            return query.scalar() or 0
        finally:
            session.close()

    async def get_scan_summary(self, scan_id: str) -> dict[str, Any]:
        """獲取掃描摘要統計"""
        session = self._get_session()
        try:
            # 按嚴重程度統計
            severity_stats = (
                session.query(FindingRecord.severity, func.count(FindingRecord.id))
                .filter_by(scan_id=scan_id)
                .group_by(FindingRecord.severity)
                .all()
            )

            # 按漏洞類型統計
            vuln_type_stats = (
                session.query(
                    FindingRecord.vulnerability_name, func.count(FindingRecord.id)
                )
                .filter_by(scan_id=scan_id)
                .group_by(FindingRecord.vulnerability_name)
                .all()
            )

            # 總數
            total = (
                session.query(func.count(FindingRecord.id))
                .filter_by(scan_id=scan_id)
                .scalar()
                or 0
            )

            return {
                "scan_id": scan_id,
                "total_findings": total,
                "by_severity": {row[0]: row[1] for row in severity_stats},  # type: ignore[misc]
                "by_vulnerability_type": {row[0]: row[1] for row in vuln_type_stats},  # type: ignore[misc]
            }
        finally:
            session.close()
