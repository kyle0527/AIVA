"""
Asset and Vulnerability Lifecycle Manager

提供資產與漏洞生命週期管理的核心服務，包括：
- 資產註冊與更新
- 漏洞去重與合併
- 狀態追蹤與歷史記錄
- 風險評分計算
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_  # type: ignore[import-not-found]
from sqlalchemy.orm import Session  # type: ignore[import-not-found]

from services.aiva_common.schemas import FindingPayload
from services.aiva_common.utils import get_logger

from .models_enhanced import (
    Asset,
    AssetStatus,
    BusinessCriticality,
    Vulnerability,
    VulnerabilityHistory,
    VulnerabilityStatus,
    VulnerabilityTag,
)

logger = get_logger(__name__)


class AssetVulnerabilityManager:
    """資產與漏洞生命週期管理器"""

    def __init__(self, session: Session):
        self.session = session

    def register_asset(
        self,
        asset_value: str,
        asset_type: str,
        name: str | None = None,
        business_criticality: str = "medium",
        environment: str = "development",
        owner: str | None = None,
        tags: list[str] | None = None,
        technology_stack: dict[str, Any] | None = None,
    ) -> Asset:
        """
        註冊或更新資產

        Args:
            asset_value: 資產值 (URL, IP, 儲存庫路徑等)
            asset_type: 資產類型
            name: 資產名稱
            business_criticality: 業務重要性
            environment: 環境
            owner: 負責人
            tags: 標籤列表
            technology_stack: 技術堆疊

        Returns:
            Asset: 資產物件
        """
        # 生成資產 ID (基於類型和值的雜湊)
        asset_id = self._generate_asset_id(asset_type, asset_value)

        # 查詢是否已存在
        asset = self.session.query(Asset).filter_by(asset_id=asset_id).first()

        if asset:
            # 更新現有資產
            asset.last_scanned_at = datetime.utcnow()  # type: ignore[misc]
            if name:
                asset.name = name  # type: ignore[misc]
            if owner:
                asset.owner = owner  # type: ignore[misc]
            if tags:
                asset.tags = tags  # type: ignore[misc]
            if technology_stack:
                asset.technology_stack = technology_stack  # type: ignore[misc]
            logger.info(f"Updated existing asset: {asset_id}")
        else:
            # 創建新資產
            asset = Asset(
                asset_id=asset_id,
                name=name or asset_value,
                type=asset_type,
                value=asset_value,
                business_criticality=business_criticality,
                environment=environment,
                owner=owner,
                tags=tags or [],
                technology_stack=technology_stack or {},
                status=AssetStatus.ACTIVE.value,
                first_discovered_at=datetime.utcnow(),
                last_scanned_at=datetime.utcnow(),
            )
            self.session.add(asset)
            logger.info(f"Registered new asset: {asset_id}")

        self.session.commit()
        return asset

    def process_finding(
        self, finding: FindingPayload, asset_id: str | None = None
    ) -> tuple[Vulnerability, bool]:
        """
        處理 Finding，進行漏洞去重和生命週期管理

        Args:
            finding: Finding 資料
            asset_id: 關聯的資產 ID (如果未提供，會嘗試從 finding 推斷)

        Returns:
            tuple[Vulnerability, bool]: (漏洞物件, 是否為新漏洞)
        """
        # 如果沒有提供 asset_id，嘗試從 finding 推斷
        if not asset_id:
            asset_id = self._infer_asset_id_from_finding(finding)

        # 生成漏洞 ID (用於去重)
        vulnerability_id = self._generate_vulnerability_id(finding, asset_id)

        # 查詢是否已存在相同的漏洞
        vulnerability = (
            self.session.query(Vulnerability)
            .filter_by(vulnerability_id=vulnerability_id)
            .first()
        )

        is_new = False
        if vulnerability:
            # 更新現有漏洞
            vulnerability.last_detected_at = datetime.utcnow()  # type: ignore[misc]

            # 如果之前已修復，現在又發現了，重新開啟
            if vulnerability.status == VulnerabilityStatus.FIXED.value:  # type: ignore[attr-defined]
                old_status = vulnerability.status  # type: ignore[attr-defined]
                vulnerability.status = VulnerabilityStatus.OPEN.value  # type: ignore[misc]
                vulnerability.fixed_at = None  # type: ignore[misc]
                vulnerability.verified_fixed_at = None  # type: ignore[misc]

                # 記錄狀態變更
                self._log_status_change(
                    vulnerability_id, old_status, VulnerabilityStatus.OPEN.value
                )

                logger.warning(
                    f"Previously fixed vulnerability reappeared: {vulnerability_id}"
                )

            logger.info(f"Updated existing vulnerability: {vulnerability_id}")
        else:
            # 創建新漏洞
            is_new = True
            vulnerability = Vulnerability(
                vulnerability_id=vulnerability_id,
                name=finding.vulnerability.name.value,
                vulnerability_type=self._extract_vuln_type(finding),
                severity=finding.vulnerability.severity.value,
                confidence=finding.vulnerability.confidence.value,
                cwe=finding.vulnerability.cwe,
                asset_id=asset_id,
                location=self._extract_location(finding),
                description=finding.vulnerability.description,
                impact=getattr(finding.vulnerability, "impact", None),
                remediation=getattr(finding.vulnerability, "remediation", None),
                risk_score=self._calculate_initial_risk_score(finding, asset_id),
                exploitability=self._assess_exploitability(finding),
                status=VulnerabilityStatus.NEW.value,
                first_detected_at=datetime.utcnow(),
                last_detected_at=datetime.utcnow(),
            )
            self.session.add(vulnerability)
            logger.info(f"Created new vulnerability: {vulnerability_id}")

        self.session.commit()
        return vulnerability, is_new

    def update_vulnerability_status(
        self,
        vulnerability_id: str,
        new_status: str,
        changed_by: str | None = None,
        comment: str | None = None,
    ) -> Vulnerability | None:
        """
        更新漏洞狀態

        Args:
            vulnerability_id: 漏洞 ID
            new_status: 新狀態
            changed_by: 變更者
            comment: 變更說明

        Returns:
            Vulnerability: 更新後的漏洞物件
        """
        vulnerability = (
            self.session.query(Vulnerability)
            .filter_by(vulnerability_id=vulnerability_id)
            .first()
        )

        if not vulnerability:
            logger.warning(f"Vulnerability not found: {vulnerability_id}")
            return None

        old_status = vulnerability.status  # type: ignore[attr-defined]

        if old_status == new_status:
            logger.debug(f"Status unchanged for vulnerability: {vulnerability_id}")
            return vulnerability

        # 更新狀態
        vulnerability.status = new_status  # type: ignore[misc]

        # 如果狀態變更為 fixed，記錄修復時間
        if new_status == VulnerabilityStatus.FIXED.value:
            vulnerability.fixed_at = datetime.utcnow()  # type: ignore[misc]

        # 記錄歷史
        self._log_status_change(
            vulnerability_id, old_status, new_status, changed_by, comment
        )

        self.session.commit()
        logger.info(
            f"Updated vulnerability status: {vulnerability_id} from {old_status} to {new_status}"
        )

        return vulnerability

    def assign_vulnerability(
        self,
        vulnerability_id: str,
        assigned_to: str,
        changed_by: str | None = None,
    ) -> Vulnerability | None:
        """
        指派漏洞給特定人員

        Args:
            vulnerability_id: 漏洞 ID
            assigned_to: 指派給
            changed_by: 變更者

        Returns:
            Vulnerability: 更新後的漏洞物件
        """
        vulnerability = (
            self.session.query(Vulnerability)
            .filter_by(vulnerability_id=vulnerability_id)
            .first()
        )

        if not vulnerability:
            return None

        old_assigned = vulnerability.assigned_to  # type: ignore[attr-defined]
        vulnerability.assigned_to = assigned_to  # type: ignore[misc]

        # 記錄歷史
        history = VulnerabilityHistory(
            vulnerability_id=vulnerability_id,
            changed_by=changed_by,
            change_type="assignment_change",
            old_value=old_assigned,
            new_value=assigned_to,
        )
        self.session.add(history)
        self.session.commit()

        logger.info(f"Assigned vulnerability {vulnerability_id} to {assigned_to}")
        return vulnerability

    def add_vulnerability_tag(
        self, vulnerability_id: str, tag: str
    ) -> VulnerabilityTag | None:
        """
        為漏洞添加標籤

        Args:
            vulnerability_id: 漏洞 ID
            tag: 標籤名稱

        Returns:
            VulnerabilityTag: 標籤物件
        """
        # 檢查是否已存在
        existing = (
            self.session.query(VulnerabilityTag)
            .filter_by(vulnerability_id=vulnerability_id, tag=tag)
            .first()
        )

        if existing:
            logger.debug(f"Tag already exists: {vulnerability_id} - {tag}")
            return existing

        tag_obj = VulnerabilityTag(vulnerability_id=vulnerability_id, tag=tag)
        self.session.add(tag_obj)
        self.session.commit()

        logger.info(f"Added tag to vulnerability: {vulnerability_id} - {tag}")
        return tag_obj

    def get_asset_vulnerabilities(
        self, asset_id: str, include_fixed: bool = False
    ) -> list[Vulnerability]:
        """
        獲取資產的所有漏洞

        Args:
            asset_id: 資產 ID
            include_fixed: 是否包含已修復的漏洞

        Returns:
            list[Vulnerability]: 漏洞列表
        """
        query = self.session.query(Vulnerability).filter_by(asset_id=asset_id)

        if not include_fixed:
            query = query.filter(
                Vulnerability.status.in_(  # type: ignore[attr-defined]
                    [
                        VulnerabilityStatus.NEW.value,
                        VulnerabilityStatus.OPEN.value,
                        VulnerabilityStatus.IN_PROGRESS.value,
                    ]
                )
            )

        return query.order_by(Vulnerability.risk_score.desc()).all()  # type: ignore[attr-defined]

    def get_overdue_vulnerabilities(self) -> list[Vulnerability]:
        """
        獲取所有逾期的漏洞

        Returns:
            list[Vulnerability]: 逾期漏洞列表
        """
        return (
            self.session.query(Vulnerability)
            .filter(
                and_(
                    Vulnerability.sla_deadline < datetime.utcnow(),  # type: ignore[attr-defined]
                    Vulnerability.status.in_(  # type: ignore[attr-defined]
                        [
                            VulnerabilityStatus.NEW.value,
                            VulnerabilityStatus.OPEN.value,
                            VulnerabilityStatus.IN_PROGRESS.value,
                        ]
                    ),
                )
            )
            .order_by(Vulnerability.sla_deadline.asc())  # type: ignore[attr-defined]
            .all()
        )

    def calculate_mttr(
        self, severity: str | None = None, days: int = 30
    ) -> dict[str, float]:
        """
        計算平均修復時間 (Mean Time To Resolve)

        Args:
            severity: 嚴重程度篩選 (可選)
            days: 統計天數

        Returns:
            dict: MTTR 統計資料
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        query = self.session.query(Vulnerability).filter(
            and_(
                Vulnerability.status == VulnerabilityStatus.FIXED.value,  # type: ignore[attr-defined]
                Vulnerability.fixed_at.isnot(None),  # type: ignore[attr-defined]
                Vulnerability.first_detected_at >= cutoff_date,  # type: ignore[attr-defined]
            )
        )

        if severity:
            query = query.filter(Vulnerability.severity == severity)  # type: ignore[attr-defined]

        vulnerabilities = query.all()

        if not vulnerabilities:
            return {"count": 0, "avg_hours": 0, "min_hours": 0, "max_hours": 0}

        resolution_times = []
        for vuln in vulnerabilities:
            if vuln.fixed_at and vuln.first_detected_at:  # type: ignore[attr-defined]
                delta = vuln.fixed_at - vuln.first_detected_at  # type: ignore[attr-defined]
                hours = delta.total_seconds() / 3600
                resolution_times.append(hours)

        return {
            "count": len(resolution_times),
            "avg_hours": sum(resolution_times) / len(resolution_times),
            "min_hours": min(resolution_times),
            "max_hours": max(resolution_times),
        }

    # 私有輔助方法

    def _generate_asset_id(self, asset_type: str, asset_value: str) -> str:
        """生成資產唯一 ID"""
        content = f"{asset_type}:{asset_value}"
        return f"asset_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def _generate_vulnerability_id(
        self, finding: FindingPayload, asset_id: str
    ) -> str:
        """
        生成漏洞唯一 ID（用於去重）

        基於：資產 ID + 漏洞類型 + 位置
        """
        vuln_type = self._extract_vuln_type(finding)
        location = self._extract_location_key(finding)

        content = f"{asset_id}:{vuln_type}:{location}"
        return f"vuln_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def _infer_asset_id_from_finding(self, finding: FindingPayload) -> str:
        """從 Finding 推斷資產 ID"""
        # 簡單實現：基於目標 URL
        target_url = str(finding.target.url)
        return self._generate_asset_id("url", target_url)

    def _extract_vuln_type(self, finding: FindingPayload) -> str:
        """提取漏洞類型"""
        return finding.vulnerability.name.value.lower().replace(" ", "_")

    def _extract_location(self, finding: FindingPayload) -> dict[str, Any]:
        """提取漏洞位置資訊"""
        return {
            "url": str(finding.target.url),
            "parameter": finding.target.parameter,
            "method": finding.target.method,
        }

    def _extract_location_key(self, finding: FindingPayload) -> str:
        """生成位置唯一鍵（用於去重）"""
        url = str(finding.target.url)
        param = finding.target.parameter or ""
        method = finding.target.method or ""
        return f"{url}:{param}:{method}"

    def _calculate_initial_risk_score(
        self, finding: FindingPayload, asset_id: str
    ) -> float:
        """計算初始風險分數"""
        # 基礎分數
        severity_scores = {
            "CRITICAL": 10.0,
            "HIGH": 7.0,
            "MEDIUM": 4.0,
            "LOW": 1.0,
            "INFO": 0.5,
        }

        base_score = severity_scores.get(
            finding.vulnerability.severity.value.upper(), 1.0
        )

        # 信心度調整
        confidence_multipliers = {"HIGH": 1.0, "MEDIUM": 0.8, "LOW": 0.6}

        confidence_multiplier = confidence_multipliers.get(
            finding.vulnerability.confidence.value.upper(), 0.8
        )

        # 獲取資產的業務重要性
        asset = self.session.query(Asset).filter_by(asset_id=asset_id).first()
        if asset:
            business_multipliers = {
                BusinessCriticality.CRITICAL.value: 2.0,
                BusinessCriticality.HIGH.value: 1.5,
                BusinessCriticality.MEDIUM.value: 1.0,
                BusinessCriticality.LOW.value: 0.7,
            }
            business_multiplier = business_multipliers.get(
                asset.business_criticality, 1.0  # type: ignore[attr-defined]
            )
        else:
            business_multiplier = 1.0

        return round(base_score * confidence_multiplier * business_multiplier, 2)

    def _assess_exploitability(self, finding: FindingPayload) -> str:
        """評估可利用性"""
        # 簡單實現：基於信心度
        confidence = finding.vulnerability.confidence.value.upper()
        if confidence == "HIGH":
            return "high"
        elif confidence == "MEDIUM":
            return "medium"
        else:
            return "low"

    def _log_status_change(
        self,
        vulnerability_id: str,
        old_status: str,
        new_status: str,
        changed_by: str | None = None,
        comment: str | None = None,
    ) -> None:
        """記錄狀態變更歷史"""
        history = VulnerabilityHistory(
            vulnerability_id=vulnerability_id,
            changed_by=changed_by or "system",
            change_type="status_change",
            old_value=old_status,
            new_value=new_status,
            comment=comment,
        )
        self.session.add(history)
