from __future__ import annotations

from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class RiskAssessmentEngine:
    """
    風險評估引擎

    基於發現的漏洞、環境資訊、業務影響等因素
    進行綜合風險評估和優先級排序。
    """

    def __init__(self) -> None:
        self._risk_matrix = {
            "critical": {"score": 10, "priority": 1},
            "high": {"score": 7, "priority": 2},
            "medium": {"score": 4, "priority": 3},
            "low": {"score": 1, "priority": 4},
        }
        self._environment_multipliers = {
            "production": 2.0,
            "staging": 1.5,
            "development": 1.0,
            "testing": 0.8,
        }

    def assess_risk(
        self, findings: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        綜合風險評估

        Args:
            findings: 漏洞發現列表
            context: 環境上下文資訊

        Returns:
            風險評估結果
        """
        if not findings:
            return {
                "overall_risk_score": 0,
                "risk_level": "none",
                "total_findings": 0,
                "priority_findings": [],
                "recommendations": ["無發現安全漏洞"],
            }

        # 計算總風險分數
        total_risk_score = 0.0
        priority_findings = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        environment = context.get("environment", "development").lower()
        env_multiplier = self._environment_multipliers.get(environment, 1.0)

        for finding in findings:
            severity = finding.get("severity", "low").lower()
            if severity not in self._risk_matrix:
                severity = "low"

            # 基礎風險分數
            base_score = self._risk_matrix[severity]["score"]

            # 環境調整
            adjusted_score = base_score * env_multiplier

            # 考慮可利用性
            exploitability = finding.get("exploitability", "medium").lower()
            if exploitability == "high":
                adjusted_score *= 1.5
            elif exploitability == "low":
                adjusted_score *= 0.7

            # 考慮資產重要性
            asset_importance = context.get("asset_importance", "medium").lower()
            if asset_importance == "critical":
                adjusted_score *= 1.8
            elif asset_importance == "high":
                adjusted_score *= 1.3

            finding["calculated_risk_score"] = adjusted_score
            total_risk_score += adjusted_score
            severity_counts[severity] += 1

            # 收集高優先級發現
            if severity in ["critical", "high"] or adjusted_score >= 8:
                priority_findings.append(finding)

        # 排序優先級發現
        priority_findings.sort(
            key=lambda x: x.get("calculated_risk_score", 0), reverse=True
        )

        # 確定整體風險等級
        avg_risk_score = total_risk_score / len(findings)
        overall_risk_level = self._determine_risk_level(avg_risk_score)

        # 生成建議
        recommendations = self._generate_recommendations(
            severity_counts, overall_risk_level, environment
        )

        result = {
            "overall_risk_score": round(total_risk_score, 2),
            "average_risk_score": round(avg_risk_score, 2),
            "risk_level": overall_risk_level,
            "environment": environment,
            "environment_multiplier": env_multiplier,
            "total_findings": len(findings),
            "severity_breakdown": severity_counts,
            "priority_findings": priority_findings[:10],  # 前10個高風險
            "recommendations": recommendations,
        }

        logger.info(
            f"Risk assessment completed: {len(findings)} findings, "
            f"overall risk level: {overall_risk_level}"
        )
        return result

    def _determine_risk_level(self, avg_score: float) -> str:
        """根據平均分數確定風險等級"""
        if avg_score >= 8:
            return "critical"
        elif avg_score >= 5:
            return "high"
        elif avg_score >= 2:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self, severity_counts: dict[str, int], risk_level: str, environment: str
    ) -> list[str]:
        """生成風險緩解建議"""
        recommendations = []

        if severity_counts["critical"] > 0:
            recommendations.append(
                f"立即處理 {severity_counts['critical']} 個關鍵級別漏洞"
            )

        if severity_counts["high"] > 0:
            recommendations.append(f"優先處理 {severity_counts['high']} 個高風險漏洞")

        if risk_level in ["critical", "high"] and environment == "production":
            recommendations.append("建議考慮暫時下線相關功能直至漏洞修復")

        if risk_level == "critical":
            recommendations.append("建議啟動緊急響應程序")
            recommendations.append("建議進行額外的安全加固措施")

        if not recommendations:
            recommendations.append("繼續保持良好的安全實踐")

        return recommendations

    def compare_risk_trends(
        self, current_assessment: dict[str, Any], previous_assessment: dict[str, Any]
    ) -> dict[str, Any]:
        """比較風險趨勢變化"""
        current_score = current_assessment.get("overall_risk_score", 0)
        previous_score = previous_assessment.get("overall_risk_score", 0)

        score_change = current_score - previous_score
        trend = "stable"
        if score_change > 1:
            trend = "increasing"
        elif score_change < -1:
            trend = "decreasing"

        return {
            "current_score": current_score,
            "previous_score": previous_score,
            "score_change": round(score_change, 2),
            "trend": trend,
            "improvement_percentage": (
                round(
                    (previous_score - current_score) / max(previous_score, 1) * 100, 1
                )
                if previous_score > 0
                else 0
            ),
        }
