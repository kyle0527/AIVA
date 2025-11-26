"""
Finding Repository - 漏洞發現資料庫存取層

提供統一的介面來存取和管理漏洞發現記錄（PostgreSQL）
遵循 aiva_common 規範
"""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine, desc, func, and_, or_
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

from services.aiva_common.enums import Severity, Confidence
from services.aiva_common.schemas import FindingPayload

logger = logging.getLogger(__name__)

Base = declarative_base()


class FindingRecord(Base):  # type: ignore[misc, valid-type]
    """漏洞發現記錄模型（PostgreSQL）"""

    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    finding_id = Column(String(255), unique=True, nullable=False, index=True)
    scan_id = Column(String(255), nullable=False, index=True)
    task_id = Column(String(255), nullable=False, index=True)
    vulnerability_name = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False, index=True)
    confidence = Column(String(50), nullable=False)
    cwe = Column(String(50), nullable=True)
    target_url = Column(Text, nullable=False)
    target_parameter = Column(String(255), nullable=True)
    target_method = Column(String(10), nullable=True)
    status = Column(String(50), nullable=False, default="active", index=True)
    raw_data = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "id": self.id,
            "finding_id": self.finding_id,
            "scan_id": self.scan_id,
            "task_id": self.task_id,
            "vulnerability_name": self.vulnerability_name,
            "severity": self.severity,
            "confidence": self.confidence,
            "cwe": self.cwe,
            "target_url": self.target_url,
            "target_parameter": self.target_parameter,
            "target_method": self.target_method,
            "status": self.status,
            "raw_data": json.loads(self.raw_data) if self.raw_data else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class FindingRepository:
    """漏洞發現資料庫存取層
    
    提供對漏洞發現記錄的 CRUD 操作
    """

    def __init__(self, database_url: str) -> None:
        """初始化資料庫連接

        Args:
            database_url: PostgreSQL 連接字串
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # 創建資料表（如果不存在）
        Base.metadata.create_all(self.engine)

        logger.info(f"FindingRepository initialized with DB: {database_url}")

    def get_session(self) -> Session:
        """獲取資料庫會話"""
        return self.SessionLocal()

    def save_finding(
        self,
        finding: FindingPayload,
        scan_id: str,
        task_id: str | None = None,
    ) -> FindingRecord:
        """保存漏洞發現記錄

        Args:
            finding: FindingPayload 對象
            scan_id: 掃描 ID
            task_id: 任務 ID（可選）

        Returns:
            儲存的漏洞記錄
        """
        session = self.get_session()
        try:
            # 準備原始資料
            raw_data = {
                "finding_id": finding.finding_id,
                "affected_url": finding.affected_url,
                "vulnerability_type": finding.vulnerability_type,
                "description": finding.description,
                "evidence": finding.evidence,
                "remediation": finding.remediation,
                "references": finding.references,
            }

            # 創建記錄
            record = FindingRecord(
                finding_id=finding.finding_id,
                scan_id=scan_id,
                task_id=task_id or f"task_{uuid4().hex[:8]}",
                vulnerability_name=finding.vulnerability_type or "unknown",
                severity=finding.severity.value,
                confidence=finding.confidence.value,
                cwe=finding.cwe,
                target_url=finding.affected_url or "",
                target_parameter=None,  # 可以從 evidence 中提取
                target_method=None,  # 可以從 evidence 中提取
                status="active",
                raw_data=json.dumps(raw_data, ensure_ascii=False),
            )

            session.add(record)
            session.commit()
            session.refresh(record)

            logger.info(
                f"Saved finding: {record.finding_id} "
                f"({record.severity}, {record.confidence})"
            )

            return record

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save finding: {e}")
            raise
        finally:
            session.close()

    def get_finding(self, finding_id: str) -> FindingRecord | None:
        """獲取單一漏洞記錄

        Args:
            finding_id: 漏洞 ID

        Returns:
            漏洞記錄或 None
        """
        session = self.get_session()
        try:
            record = (
                session.query(FindingRecord)
                .filter(FindingRecord.finding_id == finding_id)
                .first()
            )
            return record
        finally:
            session.close()

    def query_findings(
        self,
        scan_id: str | None = None,
        severity: Severity | None = None,
        confidence: Confidence | None = None,
        vulnerability_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FindingRecord]:
        """查詢漏洞發現記錄

        Args:
            scan_id: 掃描 ID 過濾
            severity: 嚴重程度過濾
            confidence: 信心度過濾
            vulnerability_type: 漏洞類型過濾
            status: 狀態過濾
            limit: 返回數量限制
            offset: 偏移量（分頁）

        Returns:
            漏洞記錄列表
        """
        session = self.get_session()
        try:
            query = session.query(FindingRecord)

            # 應用過濾條件
            if scan_id:
                query = query.filter(FindingRecord.scan_id == scan_id)
            if severity:
                query = query.filter(FindingRecord.severity == severity.value)
            if confidence:
                query = query.filter(FindingRecord.confidence == confidence.value)
            if vulnerability_type:
                query = query.filter(
                    FindingRecord.vulnerability_name == vulnerability_type
                )
            if status:
                query = query.filter(FindingRecord.status == status)

            # 排序（按嚴重性和時間）
            severity_order = {
                "CRITICAL": 0,
                "HIGH": 1,
                "MEDIUM": 2,
                "LOW": 3,
                "INFORMATIONAL": 4,
            }

            # 按創建時間降序
            query = query.order_by(desc(FindingRecord.created_at))

            # 應用分頁
            query = query.limit(limit).offset(offset)

            results = query.all()

            logger.info(
                f"Queried {len(results)} findings "
                f"(scan_id={scan_id}, severity={severity})"
            )

            return results

        finally:
            session.close()

    def update_finding_status(
        self,
        finding_id: str,
        new_status: str,
    ) -> bool:
        """更新漏洞狀態

        Args:
            finding_id: 漏洞 ID
            new_status: 新狀態（active, fixed, false_positive, accepted_risk）

        Returns:
            是否成功更新
        """
        session = self.get_session()
        try:
            record = (
                session.query(FindingRecord)
                .filter(FindingRecord.finding_id == finding_id)
                .first()
            )

            if not record:
                logger.warning(f"Finding not found: {finding_id}")
                return False

            record.status = new_status
            record.updated_at = datetime.utcnow()

            session.commit()

            logger.info(f"Updated finding {finding_id} status to {new_status}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update finding status: {e}")
            return False
        finally:
            session.close()

    def delete_finding(self, finding_id: str) -> bool:
        """刪除漏洞記錄

        Args:
            finding_id: 漏洞 ID

        Returns:
            是否成功刪除
        """
        session = self.get_session()
        try:
            record = (
                session.query(FindingRecord)
                .filter(FindingRecord.finding_id == finding_id)
                .first()
            )

            if not record:
                logger.warning(f"Finding not found: {finding_id}")
                return False

            session.delete(record)
            session.commit()

            logger.info(f"Deleted finding: {finding_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete finding: {e}")
            return False
        finally:
            session.close()

    def get_statistics(
        self,
        scan_id: str | None = None,
    ) -> dict[str, Any]:
        """獲取漏洞統計資訊

        Args:
            scan_id: 掃描 ID 過濾（可選）

        Returns:
            統計資訊字典
        """
        session = self.get_session()
        try:
            query = session.query(FindingRecord)

            if scan_id:
                query = query.filter(FindingRecord.scan_id == scan_id)

            # 總數
            total_count = query.count()

            # 按嚴重性分組統計
            severity_stats = (
                query.with_entities(
                    FindingRecord.severity,
                    func.count(FindingRecord.id).label("count"),
                )
                .group_by(FindingRecord.severity)
                .all()
            )

            # 按信心度分組統計
            confidence_stats = (
                query.with_entities(
                    FindingRecord.confidence,
                    func.count(FindingRecord.id).label("count"),
                )
                .group_by(FindingRecord.confidence)
                .all()
            )

            # 按狀態分組統計
            status_stats = (
                query.with_entities(
                    FindingRecord.status,
                    func.count(FindingRecord.id).label("count"),
                )
                .group_by(FindingRecord.status)
                .all()
            )

            # 按漏洞類型分組統計
            vuln_type_stats = (
                query.with_entities(
                    FindingRecord.vulnerability_name,
                    func.count(FindingRecord.id).label("count"),
                )
                .group_by(FindingRecord.vulnerability_name)
                .order_by(desc("count"))
                .limit(10)
                .all()
            )

            return {
                "total_findings": total_count,
                "by_severity": {
                    severity: count for severity, count in severity_stats
                },
                "by_confidence": {
                    confidence: count for confidence, count in confidence_stats
                },
                "by_status": {status: count for status, count in status_stats},
                "top_vulnerability_types": [
                    {"type": vuln_type, "count": count}
                    for vuln_type, count in vuln_type_stats
                ],
            }

        finally:
            session.close()

    def export_findings(
        self,
        scan_id: str | None = None,
        severity: Severity | None = None,
        format: str = "json",
    ) -> list[dict[str, Any]]:
        """匯出漏洞發現記錄

        Args:
            scan_id: 掃描 ID 過濾
            severity: 嚴重程度過濾
            format: 匯出格式（json, csv）

        Returns:
            漏洞記錄列表
        """
        findings = self.query_findings(
            scan_id=scan_id,
            severity=severity,
            limit=10000,  # 大量匯出
        )

        return [finding.to_dict() for finding in findings]

    def bulk_update_status(
        self,
        finding_ids: list[str],
        new_status: str,
    ) -> int:
        """批次更新漏洞狀態

        Args:
            finding_ids: 漏洞 ID 列表
            new_status: 新狀態

        Returns:
            更新數量
        """
        session = self.get_session()
        try:
            updated_count = (
                session.query(FindingRecord)
                .filter(FindingRecord.finding_id.in_(finding_ids))
                .update(
                    {
                        "status": new_status,
                        "updated_at": datetime.utcnow(),
                    },
                    synchronize_session=False,
                )
            )

            session.commit()

            logger.info(f"Bulk updated {updated_count} findings to status {new_status}")
            return updated_count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to bulk update findings: {e}")
            return 0
        finally:
            session.close()
