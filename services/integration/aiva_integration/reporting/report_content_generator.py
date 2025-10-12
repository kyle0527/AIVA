from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ReportContentGenerator:
    """
    報告內容生成器

    根據漏洞發現、風險評估、合規檢查等結果
    生成結構化的安全評估報告內容。
    """

    def __init__(self) -> None:
        # 報告模板映射可以在未來擴展使用
        pass

    def generate_report_content(
        self,
        findings: list[dict[str, Any]],
        risk_assessment: dict[str, Any],
        compliance_result: dict[str, Any],
        correlation_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        生成完整的報告內容

        Args:
            findings: 漏洞發現列表
            risk_assessment: 風險評估結果
            compliance_result: 合規檢查結果
            correlation_analysis: 相關性分析結果

        Returns:
            結構化的報告內容
        """
        report_content = {
            "metadata": self._generate_metadata(len(findings)),
            "executive_summary": self._generate_executive_summary(
                findings, risk_assessment, compliance_result
            ),
            "risk_overview": self._generate_risk_overview(risk_assessment),
            "findings_summary": self._generate_findings_summary(findings),
            "technical_details": self._generate_technical_details(findings),
            "correlation_analysis": self._format_correlation_analysis(
                correlation_analysis
            ),
            "compliance_status": self._format_compliance_result(compliance_result),
            "recommendations": self._generate_recommendations(
                findings, risk_assessment, compliance_result
            ),
            "appendices": self._generate_appendices(findings),
        }

        logger.info(f"Report content generated with {len(findings)} findings")
        return report_content

    def _generate_metadata(self, findings_count: int) -> dict[str, Any]:
        """生成報告元數據"""
        return {
            "report_id": f"AIVA-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.now(UTC).isoformat(),
            "version": "1.0",
            "findings_count": findings_count,
            "report_type": "Security Assessment",
        }

    def _generate_executive_summary(
        self,
        findings: list[dict[str, Any]],
        risk_assessment: dict[str, Any],
        compliance_result: dict[str, Any],
    ) -> dict[str, Any]:
        """生成執行摘要"""
        total_findings = len(findings)
        risk_level = risk_assessment.get("risk_level", "unknown")
        compliance_score = compliance_result.get("compliance_score", 0)

        severity_counts: dict[str, int] = {}
        for finding in findings:
            severity = finding.get("severity", "unknown").lower()
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        critical_count = severity_counts.get("critical", 0)
        high_count = severity_counts.get("high", 0)

        # 生成關鍵發現
        key_findings = []
        if critical_count > 0:
            key_findings.append(f"發現 {critical_count} 個關鍵級別安全漏洞")
        if high_count > 0:
            key_findings.append(f"發現 {high_count} 個高風險安全漏洞")
        if compliance_score < 70:
            key_findings.append(f"合規分數偏低 ({compliance_score:.1f}/100)")

        # 風險等級說明
        risk_descriptions = {
            "critical": "需要立即處理的嚴重安全風險",
            "high": "需要優先處理的高風險問題",
            "medium": "需要關注的中等風險問題",
            "low": "建議修復的低風險問題",
        }

        return {
            "total_findings": total_findings,
            "overall_risk_level": risk_level,
            "risk_description": risk_descriptions.get(risk_level, "未知風險等級"),
            "compliance_score": compliance_score,
            "severity_breakdown": severity_counts,
            "key_findings": key_findings,
            "immediate_actions_required": critical_count > 0
            or risk_level == "critical",
        }

    def _generate_risk_overview(
        self, risk_assessment: dict[str, Any]
    ) -> dict[str, Any]:
        """生成風險概覽"""
        return {
            "overall_score": risk_assessment.get("overall_risk_score", 0),
            "risk_level": risk_assessment.get("risk_level", "unknown"),
            "environment": risk_assessment.get("environment", "unknown"),
            "priority_findings_count": len(
                risk_assessment.get("priority_findings", [])
            ),
            "risk_factors": self._extract_risk_factors(risk_assessment),
        }

    def _generate_findings_summary(
        self, findings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """生成發現摘要"""
        if not findings:
            return {"message": "未發現安全漏洞", "details": {}}

        # 按類型分組
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_location: dict[str, int] = {}

        for finding in findings:
            vuln_type = finding.get("vulnerability_type", "unknown")
            severity = finding.get("severity", "unknown")
            location = finding.get("location", {}).get("url", "unknown")

            by_type[vuln_type] = by_type.get(vuln_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_location[location] = by_location.get(location, 0) + 1

        return {
            "by_vulnerability_type": sorted(
                by_type.items(), key=lambda x: x[1], reverse=True
            ),
            "by_severity": by_severity,
            "by_location": sorted(
                by_location.items(), key=lambda x: x[1], reverse=True
            )[:10],  # 前10個
            "most_common_type": (
                max(by_type, key=lambda x: by_type[x]) if by_type else "none"
            ),
            "highest_severity": self._get_highest_severity(by_severity),
        }

    def _generate_technical_details(
        self, findings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """生成技術細節"""
        technical_details = []

        for finding in findings:
            detail = {
                "finding_id": finding.get("finding_id", "unknown"),
                "vulnerability_type": finding.get("vulnerability_type", "unknown"),
                "severity": finding.get("severity", "unknown"),
                "location": finding.get("location", {}),
                "description": finding.get("description", "無描述"),
                "evidence": finding.get("evidence", {}),
                "impact": finding.get("impact", "未評估"),
                "recommendation": finding.get("recommendation", "無建議"),
                "references": finding.get("references", []),
            }
            technical_details.append(detail)

        # 按嚴重性排序
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        technical_details.sort(
            key=lambda x: severity_order.get(x["severity"].lower(), 4)
        )

        return technical_details

    def _format_correlation_analysis(
        self, correlation_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """格式化相關性分析"""
        if not correlation_analysis:
            return {"message": "無相關性分析數據"}

        return {
            "has_correlations": correlation_analysis.get("has_correlations", False),
            "correlation_groups": correlation_analysis.get("correlation_groups", []),
            "attack_chains": correlation_analysis.get("attack_chains", []),
            "risk_amplification": correlation_analysis.get("risk_amplification", 1.0),
            "summary": correlation_analysis.get("summary", {}),
        }

    def _format_compliance_result(
        self, compliance_result: dict[str, Any]
    ) -> dict[str, Any]:
        """格式化合規結果"""
        if not compliance_result:
            return {"message": "無合規檢查數據"}

        return {
            "policy": compliance_result.get("policy", "unknown"),
            "score": compliance_result.get("compliance_score", 0),
            "violations": {
                "critical": compliance_result.get("critical_violations", 0),
                "high": compliance_result.get("high_violations", 0),
                "medium": compliance_result.get("medium_violations", 0),
                "low": compliance_result.get("low_violations", 0),
            },
            "recommendations": compliance_result.get("recommendations", []),
        }

    def _generate_recommendations(
        self,
        findings: list[dict[str, Any]],
        risk_assessment: dict[str, Any],
        compliance_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """生成修復建議"""
        recommendations = []

        # 從風險評估中提取建議
        risk_recommendations = risk_assessment.get("recommendations", [])
        for rec in risk_recommendations:
            recommendations.append(
                {
                    "category": "risk_mitigation",
                    "priority": "high",
                    "description": rec,
                }
            )

        # 從合規檢查中提取建議
        compliance_recommendations = compliance_result.get("recommendations", [])
        for rec in compliance_recommendations:
            recommendations.append(
                {
                    "category": "compliance",
                    "priority": "medium",
                    "description": rec,
                }
            )

        # 通用安全建議
        if findings:
            recommendations.append(
                {
                    "category": "general_security",
                    "priority": "medium",
                    "description": "建議定期進行安全掃描和評估",
                }
            )

        return recommendations

    def _generate_appendices(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """生成附錄"""
        return {
            "vulnerability_classifications": self._get_vuln_classifications(),
            "severity_definitions": self._get_severity_definitions(),
            "scan_methodology": self._get_scan_methodology(),
            "raw_findings_count": len(findings),
        }

    def _extract_risk_factors(self, risk_assessment: dict[str, Any]) -> list[str]:
        """提取風險因素"""
        factors = []

        env_multiplier = risk_assessment.get("environment_multiplier", 1.0)
        if env_multiplier > 1.5:
            factors.append("生產環境風險放大")

        severity_breakdown = risk_assessment.get("severity_breakdown", {})
        if severity_breakdown.get("critical", 0) > 0:
            factors.append("存在關鍵級別漏洞")

        return factors

    def _get_highest_severity(self, by_severity: dict[str, int]) -> str:
        """獲取最高嚴重性等級"""
        severity_order = ["critical", "high", "medium", "low"]
        for severity in severity_order:
            if by_severity.get(severity, 0) > 0:
                return severity
        return "none"

    def _get_vuln_classifications(self) -> dict[str, str]:
        """獲取漏洞分類說明"""
        return {
            "xss": "跨站腳本攻擊",
            "sqli": "SQL注入",
            "ssrf": "伺服器端請求偽造",
            "csrf": "跨站請求偽造",
            "lfi": "本地文件包含",
            "rfi": "遠程文件包含",
        }

    def _get_severity_definitions(self) -> dict[str, str]:
        """獲取嚴重性等級定義"""
        return {
            "critical": "可直接導致系統完全妥協的漏洞",
            "high": "可能導致重大安全影響的漏洞",
            "medium": "存在安全風險但影響有限的漏洞",
            "low": "安全最佳實踐相關的問題",
        }

    def _get_scan_methodology(self) -> str:
        """獲取掃描方法說明"""
        return (
            "本次安全評估使用 AIVA 自動化漏洞評估平台進行，"
            "結合靜態分析、動態測試和人工驗證等多種技術。"
        )
