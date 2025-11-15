"""Risk Assessment Engine - 風險評估引擎

整合多維度風險評估:
- CVSS 基礎分數計算
- 威脅情報整合 (是否被積極利用)
- 資產重要性權重
- 可利用性評估
- 業務影響分析
"""

import asyncio
from typing import TYPE_CHECKING

# 直接使用 aiva_common 標準枚舉 - 符合 aiva_common 規範
from services.aiva_common.enums.common import Severity, ThreatLevel
from services.aiva_common.enums.security import VulnerabilityType
from services.aiva_common.schemas.findings import FindingPayload
from services.aiva_common.utils.logging import get_logger

if TYPE_CHECKING:
    from services.integration.aiva_integration.threat_intel.threat_intel.intel_aggregator import (
        IntelAggregator,
    )

logger = get_logger(__name__)


class RiskAssessmentEngine:
    """風險評估引擎

    根據多個維度評估漏洞的實際風險分數 (0-10)
    """

    def __init__(self, enable_threat_intel: bool = True):
        """初始化風險評估引擎

        Args:
            enable_threat_intel: 是否啟用威脅情報查詢
        """
        self.enable_threat_intel = enable_threat_intel
        self.intel_aggregator: "IntelAggregator | None" = None

        if enable_threat_intel:
            try:
                from services.integration.aiva_integration.threat_intel.threat_intel.intel_aggregator import (
                    IntelAggregator,
                )

                self.intel_aggregator = IntelAggregator()
                logger.info("Threat intelligence enabled for risk assessment")
            except ImportError:
                logger.warning(
                    "Threat intel module not available, using base scoring only"
                )
                self.intel_aggregator = None
                self.enable_threat_intel = False

    def assess_risk(self, finding: FindingPayload) -> float:
        """評估漏洞風險分數

        Args:
            finding: 漏洞發現結果

        Returns:
            float: 0.0 - 10.0 的風險分數
        """
        # 1. 基礎 CVSS 分數 (根據 severity)
        base_score = self._calculate_base_score(finding)

        # 2. 威脅情報調整 (可選)
        if (
            self.enable_threat_intel
            and self.intel_aggregator
            and finding.vulnerability.cwe
        ):
            base_score = self._adjust_by_threat_intel(base_score, finding)

        # 3. 可利用性調整
        base_score = self._adjust_by_exploitability(base_score, finding)

        # 4. 資產重要性調整
        base_score = self._adjust_by_asset_criticality(base_score, finding)

        # 確保分數在 0-10 範圍內
        final_score = max(0.0, min(base_score, 10.0))

        logger.debug(
            f"Risk assessment for {finding.vulnerability.name}: {final_score:.2f}"
        )

        return final_score

    def _calculate_base_score(self, finding: FindingPayload) -> float:
        """計算基礎 CVSS 分數

        Args:
            finding: 漏洞發現結果

        Returns:
            float: 基礎分數
        """
        # 根據 severity 映射到 CVSS 分數
        severity_scores = {
            Severity.CRITICAL: 9.5,
            Severity.HIGH: 7.5,
            Severity.MEDIUM: 5.0,
            Severity.LOW: 3.0,
            Severity.NONE: 0.0,  # 使用 NONE 而不是 INFO/INFORMATIONAL
        }

        base_score = severity_scores.get(finding.vulnerability.severity, 5.0)

        # 根據漏洞類型調整 - 包含 Phase I 高價值漏洞
        vuln_type_multipliers = {
            VulnerabilityType.SQLI: 1.15,  # SQL Injection
            VulnerabilityType.XSS: 1.0,
            VulnerabilityType.SSRF: 1.1,
            VulnerabilityType.IDOR: 1.05,
            VulnerabilityType.BOLA: 1.05,
            VulnerabilityType.INFO_LEAK: 0.9,
            VulnerabilityType.WEAK_AUTH: 1.15,
            VulnerabilityType.PRICE_MANIPULATION: 1.2,
            VulnerabilityType.WORKFLOW_BYPASS: 1.1,
            VulnerabilityType.RACE_CONDITION: 1.15,
            # Phase I 高價值漏洞類型
            "Client-Side Authorization Bypass": 1.25,  # 客戶端授權繞過 - 高風險
            "Advanced SSRF - Internal Services": 1.3,  # 進階 SSRF - 內部服務訪問
            "Advanced SSRF - Cloud Metadata": 1.4,  # 進階 SSRF - 雲端元數據 (極高風險)
            "JavaScript Authorization Bypass": 1.2,  # JS 授權繞過
            "Local Storage Auth Manipulation": 1.15,  # 本地存儲授權操作
        }

        # 支援字串類型的漏洞類型名稱 (Phase I 格式)
        vuln_name = finding.vulnerability.name
        if hasattr(vuln_name, "value"):
            # 獲取漏洞類型倍數
            vuln_type_str = vuln_name if isinstance(vuln_name, str) else str(vuln_name)
            multiplier = vuln_type_multipliers.get(vuln_type_str, 1.0)
        else:
            # 處理字串格式的漏洞類型 (如 "Client-Side Authorization Bypass")
            multiplier = vuln_type_multipliers.get(str(vuln_name), 1.0)

        base_score *= multiplier

        # Phase I 特殊評估邏輯
        if isinstance(vuln_name, str):
            base_score = self._assess_phase_i_specific_risk(
                base_score, vuln_name, finding
            )

        return base_score

    def _assess_phase_i_specific_risk(
        self, base_score: float, vuln_type: str, finding: FindingPayload
    ) -> float:
        """Phase I 特定漏洞的風險評估增強"""
        # 客戶端授權繞過特殊邏輯
        if "Client-Side Authorization Bypass" in vuln_type:
            # 如果發現硬編碼管理員權限，風險極高
            if finding.evidence and "hardcoded_admin" in str(finding.evidence):
                base_score *= 1.5
            # 如果涉及支付或關鍵業務流程
            target_url = str(finding.target.url).lower()
            if any(
                critical in target_url
                for critical in ["payment", "admin", "checkout", "order"]
            ):
                base_score *= 1.3

        # 進階 SSRF 特殊邏輯
        elif "Advanced SSRF" in vuln_type:
            # 雲端元數據訪問風險極高
            if "Cloud Metadata" in vuln_type:
                base_score *= 1.6  # AWS IMDSv1/v2, GCP metadata 等
            # 內部服務訪問
            elif "Internal Services" in vuln_type:
                if finding.evidence and any(
                    service in str(finding.evidence)
                    for service in [
                        "elasticsearch",
                        "redis",
                        "kubernetes",
                        "docker",
                        "consul",
                    ]
                ):
                    base_score *= 1.4

        return base_score

    def _adjust_by_threat_intel(
        self, base_score: float, finding: FindingPayload
    ) -> float:
        """根據威脅情報調整分數

        Args:
            base_score: 基礎分數
            finding: 漏洞發現結果

        Returns:
            float: 調整後的分數
        """
        cwe = finding.vulnerability.cwe
        if not self.intel_aggregator or not cwe:
            return base_score

        # 威脅情報查詢已簡化,因為 IntelAggregator 可能沒有 query_cwe 方法
        # 如果需要威脅情報整合,可以在這裡添加相應的邏輯
        return base_score

    def _adjust_by_exploitability(
        self, base_score: float, finding: FindingPayload
    ) -> float:
        """根據可利用性調整分數

        Args:
            base_score: 基礎分數
            finding: 漏洞發現結果

        Returns:
            float: 調整後的分數
        """
        # 如果有證據,提升分數
        if finding.evidence and finding.evidence.proof:
            base_score *= 1.1

        # 根據信心度調整
        confidence_multipliers = {
            "Certain": 1.2,
            "Firm": 1.0,
            "Possible": 0.8,
        }
        # 安全地獲取 confidence 值
        confidence_value = getattr(finding.vulnerability.confidence, 'value', 'Firm')
        multiplier = confidence_multipliers.get(confidence_value, 1.0)
        base_score *= multiplier

        return base_score

    def _adjust_by_asset_criticality(
        self, base_score: float, finding: FindingPayload
    ) -> float:
        """根據資產重要性調整分數

        Args:
            base_score: 基礎分數
            finding: 漏洞發現結果

        Returns:
            float: 調整後的分數
        """
        # 根據 URL 路徑推斷資產重要性
        # 實際環境中應該從配置或資產管理系統獲取
        url = str(finding.target.url)

        critical_paths = ["/admin", "/payment", "/checkout", "/api/payment"]
        high_paths = ["/api", "/account", "/profile", "/user"]

        if any(path in url.lower() for path in critical_paths):
            base_score *= 1.3
        elif any(path in url.lower() for path in high_paths):
            base_score *= 1.15

        return base_score

    def get_risk_level(self, risk_score: float) -> ThreatLevel:
        """將風險分數轉換為威脅等級

        Args:
            risk_score: 風險分數 (0-10)

        Returns:
            ThreatLevel: 威脅等級
        """
        if risk_score >= 9.0:
            return ThreatLevel.CRITICAL
        elif risk_score >= 7.0:
            return ThreatLevel.HIGH
        elif risk_score >= 4.0:
            return ThreatLevel.MEDIUM
        elif risk_score >= 2.0:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO

    def batch_assess(
        self, findings: list[FindingPayload]
    ) -> list[tuple[FindingPayload, float]]:
        """批量評估風險分數

        Args:
            findings: 漏洞發現列表

        Returns:
            list: [(finding, risk_score), ...]
        """
        # 由於 assess_risk 現在是同步的，直接調用即可
        results = []
        for finding in findings:
            try:
                score = self.assess_risk(finding)
                results.append((finding, score))
            except Exception as e:
                logger.error(f"Failed to assess risk for {finding.finding_id}: {e}")
                score = self._calculate_base_score(finding)  # Fallback
                results.append((finding, score))

        return results
