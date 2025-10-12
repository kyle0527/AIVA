from __future__ import annotations

from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class CompliancePolicyChecker:
    """
    合規政策檢查器

    根據不同的安全標準（如 OWASP Top 10、PCI DSS、GDPR 等）
    檢查發現的漏洞是否符合合規要求，並生成合規報告。
    """

    def __init__(self) -> None:
        self._policies: dict[str, dict[str, Any]] = {
            "owasp_top10": {
                "name": "OWASP Top 10",
                "version": "2021",
                "critical_vulns": ["injection", "broken_auth", "sensitive_exposure"],
            },
            "pci_dss": {
                "name": "PCI DSS",
                "version": "3.2.1",
                "required_checks": ["input_validation", "encryption", "access_control"],
            },
            "gdpr": {
                "name": "GDPR",
                "focus": "data_protection",
                "requirements": ["data_encryption", "access_logging"],
            },
        }

    def check_compliance(
        self, findings: list[dict[str, Any]], policy: str = "owasp_top10"
    ) -> dict[str, Any]:
        """
        檢查漏洞發現是否符合指定的合規政策

        Args:
            findings: 漏洞發現列表
            policy: 合規政策名稱

        Returns:
            合規檢查結果
        """
        if policy not in self._policies:
            return {"error": f"Unknown policy: {policy}"}

        policy_config = self._policies[policy]
        compliance_result = {
            "policy": policy_config["name"],
            "version": policy_config.get("version", "latest"),
            "total_findings": len(findings),
            "critical_violations": 0,
            "high_violations": 0,
            "medium_violations": 0,
            "low_violations": 0,
            "compliance_score": 0.0,
            "recommendations": [],
        }

        for finding in findings:
            severity = finding.get("severity", "low").lower()
            vuln_type = finding.get("vulnerability_type", "").lower()

            # 檢查是否為關鍵違規
            if (
                policy == "owasp_top10"
                and any(
                    critical in vuln_type
                    for critical in policy_config["critical_vulns"]
                )
                and severity in ["critical", "high"]
            ):
                compliance_result["critical_violations"] += 1

            # 統計各級別違規
            if severity == "critical":
                compliance_result["critical_violations"] += 1
            elif severity == "high":
                compliance_result["high_violations"] += 1
            elif severity == "medium":
                compliance_result["medium_violations"] += 1
            else:
                compliance_result["low_violations"] += 1

        # 計算合規分數 (100分滿分)
        total_violations = (
            compliance_result["critical_violations"] * 4
            + compliance_result["high_violations"] * 3
            + compliance_result["medium_violations"] * 2
            + compliance_result["low_violations"] * 1
        )

        max_possible_score = max(len(findings) * 4, 1)  # 避免除零
        compliance_result["compliance_score"] = max(
            0, 100 - (total_violations / max_possible_score * 100)
        )

        # 生成建議
        if compliance_result["critical_violations"] > 0:
            compliance_result["recommendations"].append(
                "立即修復所有關鍵級別的安全漏洞"
            )
        if compliance_result["compliance_score"] < 80:
            compliance_result["recommendations"].append(
                "建議加強安全控制措施以提高合規分數"
            )

        logger.info(
            f"Compliance check completed for {policy}: "
            f"score={compliance_result['compliance_score']:.1f}"
        )
        return compliance_result

    def get_available_policies(self) -> list[str]:
        """獲取可用的合規政策列表"""
        return list(self._policies.keys())

    def add_custom_policy(
        self, policy_name: str, policy_config: dict[str, Any]
    ) -> None:
        """添加自定義合規政策"""
        self._policies[policy_name] = policy_config
        logger.info(f"Added custom compliance policy: {policy_name}")
