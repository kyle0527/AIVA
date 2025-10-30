"""
增強版風險評估引擎

整合業務上下文、資產價值、合規要求等多維度因素進行風險評估
"""



from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class EnhancedRiskAssessmentEngine:
    """
    增強版風險評估引擎

    基於漏洞、環境、業務影響、資產價值等多維度因素
    進行綜合風險評估和業務驅動的優先級排序。

    新增功能：
    - 業務重要性深度整合
    - 資料敏感度評估
    - 網路暴露度考量
    - 合規風險評估
    - 財務影響估算
    """

    def __init__(self) -> None:
        # 基礎風險矩陣
        self._risk_matrix = {
            "critical": {"score": 10, "priority": 1},
            "high": {"score": 7, "priority": 2},
            "medium": {"score": 4, "priority": 3},
            "low": {"score": 1, "priority": 4},
        }

        # 環境乘數
        self._environment_multipliers = {
            "production": 2.0,
            "staging": 1.5,
            "development": 1.0,
            "testing": 0.8,
        }

        # 業務重要性乘數
        self._business_criticality_multipliers = {
            "critical": 3.0,  # 業務關鍵系統
            "high": 2.0,  # 重要業務系統
            "medium": 1.0,  # 一般業務系統
            "low": 0.5,  # 非關鍵系統
        }

        # 資料敏感度乘數
        self._data_sensitivity_multipliers = {
            "highly_sensitive": 2.5,  # 高度敏感（信用卡、健康資料）
            "sensitive": 1.8,  # 敏感（PII）
            "internal": 1.2,  # 內部資料
            "public": 0.8,  # 公開資料
        }

        # 可利用性評估乘數
        self._exploitability_multipliers = {
            "proven": 2.0,  # 已有公開 exploit
            "high": 1.5,  # 高度可利用
            "medium": 1.0,  # 中等可利用性
            "low": 0.7,  # 低可利用性
            "theoretical": 0.4,  # 理論上可利用
        }

        # 資產網路暴露度乘數
        self._exposure_multipliers = {
            "internet_facing": 2.0,  # 直接暴露於互聯網
            "dmz": 1.5,  # DMZ 區域
            "internal_network": 1.0,  # 內部網路
            "isolated": 0.6,  # 隔離網路
        }

        # 合規要求標籤權重
        self._compliance_weights = {
            "pci-dss": 1.5,
            "hipaa": 1.5,
            "gdpr": 1.4,
            "sox": 1.3,
            "iso27001": 1.2,
        }

    def assess_risk(
        self, findings: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        綜合風險評估（增強版）

        Args:
            findings: 漏洞發現列表
            context: 環境上下文資訊，支援以下欄位：
                - environment: 環境類型
                - business_criticality: 業務重要性
                - data_sensitivity: 資料敏感度
                - asset_exposure: 網路暴露度
                - compliance_tags: 合規標籤列表
                - asset_value: 資產估值
                - user_base: 使用者基數

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

        # 提取上下文資訊
        environment = context.get("environment", "development").lower()
        business_criticality = context.get("business_criticality", "medium").lower()
        data_sensitivity = context.get("data_sensitivity", "internal").lower()
        asset_exposure = context.get("asset_exposure", "internal_network").lower()
        compliance_tags = context.get("compliance_tags", [])
        asset_value = context.get("asset_value", 0)
        user_base = context.get("user_base", 0)

        # 計算上下文乘數
        env_multiplier = self._environment_multipliers.get(environment, 1.0)
        business_multiplier = self._business_criticality_multipliers.get(
            business_criticality, 1.0
        )
        data_multiplier = self._data_sensitivity_multipliers.get(
            data_sensitivity, 1.0
        )
        exposure_multiplier = self._exposure_multipliers.get(asset_exposure, 1.0)

        # 計算合規乘數
        compliance_multiplier = 1.0
        for tag in compliance_tags:
            tag_lower = tag.lower()
            if tag_lower in self._compliance_weights:
                compliance_multiplier = max(
                    compliance_multiplier, self._compliance_weights[tag_lower]
                )

        # 計算總體上下文乘數（加權平均）
        context_multiplier = (
            env_multiplier * 0.3
            + business_multiplier * 0.25
            + data_multiplier * 0.2
            + exposure_multiplier * 0.15
            + compliance_multiplier * 0.1
        )

        logger.info(
            f"Context multipliers - Env: {env_multiplier}, Business: {business_multiplier}, "
            f"Data: {data_multiplier}, Exposure: {exposure_multiplier}, "
            f"Compliance: {compliance_multiplier}, Total: {context_multiplier:.2f}"
        )

        # 計算風險分數
        total_risk_score = 0.0
        business_risk_score = 0.0
        technical_risk_score = 0.0
        priority_findings = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for finding in findings:
            severity = finding.get("severity", "low").lower()
            if severity not in self._risk_matrix:
                severity = "low"

            # 基礎技術風險分數
            base_score = self._risk_matrix[severity]["score"]
            technical_risk_score += base_score

            # 可利用性評估
            exploitability = finding.get("exploitability", "medium").lower()
            exploit_multiplier = self._exploitability_multipliers.get(
                exploitability, 1.0
            )

            # 計算調整後的技術分數
            adjusted_tech_score = base_score * exploit_multiplier

            # 計算業務風險分數
            adjusted_business_score = adjusted_tech_score * context_multiplier

            # 額外考慮：資產價值或使用者基數
            if asset_value > 0 or user_base > 0:
                impact_factor = self._calculate_business_impact_factor(
                    asset_value, user_base
                )
                adjusted_business_score *= impact_factor

            # 記錄計算後的分數
            finding["calculated_technical_risk_score"] = round(adjusted_tech_score, 2)
            finding["calculated_business_risk_score"] = round(
                adjusted_business_score, 2
            )
            finding["context_multiplier"] = round(context_multiplier, 2)

            total_risk_score += adjusted_business_score
            business_risk_score += adjusted_business_score
            severity_counts[severity] += 1

            # 收集高優先級發現
            if severity in ["critical", "high"] or adjusted_business_score >= 15:
                priority_findings.append(finding)

        # 按業務風險分數排序
        priority_findings.sort(
            key=lambda x: x.get("calculated_business_risk_score", 0), reverse=True
        )

        # 確定整體風險等級
        avg_business_risk = business_risk_score / len(findings)
        overall_risk_level = self._determine_risk_level_enhanced(
            avg_business_risk, business_criticality, environment
        )

        # 生成增強的建議
        recommendations = self._generate_enhanced_recommendations(
            severity_counts=severity_counts,
            overall_risk_level=overall_risk_level,
            business_criticality=business_criticality,
            environment=environment,
            compliance_tags=compliance_tags,
            priority_findings=priority_findings,
        )

        # 計算業務影響估算
        business_impact = self._estimate_business_impact(
            severity_counts=severity_counts,
            asset_value=asset_value,
            user_base=user_base,
            business_criticality=business_criticality,
        )

        result = {
            "overall_risk_score": round(total_risk_score, 2),
            "average_risk_score": round(total_risk_score / len(findings), 2),
            "technical_risk_score": round(technical_risk_score, 2),
            "business_risk_score": round(business_risk_score, 2),
            "risk_level": overall_risk_level,
            "context": {
                "environment": environment,
                "environment_multiplier": env_multiplier,
                "business_criticality": business_criticality,
                "business_multiplier": business_multiplier,
                "data_sensitivity": data_sensitivity,
                "data_multiplier": data_multiplier,
                "asset_exposure": asset_exposure,
                "exposure_multiplier": exposure_multiplier,
                "compliance_tags": compliance_tags,
                "compliance_multiplier": compliance_multiplier,
                "total_context_multiplier": round(context_multiplier, 2),
            },
            "total_findings": len(findings),
            "severity_breakdown": severity_counts,
            "priority_findings": priority_findings[:10],
            "recommendations": recommendations,
            "business_impact": business_impact,
            "risk_trend": self._calculate_risk_trend(findings),
        }

        logger.info(
            f"Risk assessment completed: {len(findings)} findings, "
            f"business risk level: {overall_risk_level}, "
            f"business risk score: {business_risk_score:.2f}"
        )
        return result

    def _calculate_business_impact_factor(
        self, asset_value: float, user_base: int
    ) -> float:
        """
        計算業務影響因子

        基於資產價值和使用者基數評估潛在的業務影響
        """
        impact_factor = 1.0

        # 資產價值影響
        if asset_value > 0:
            if asset_value >= 10_000_000:  # $10M+
                impact_factor += 0.5
            elif asset_value >= 1_000_000:  # $1M+
                impact_factor += 0.3
            elif asset_value >= 100_000:  # $100K+
                impact_factor += 0.2

        # 使用者基數影響
        if user_base > 0:
            if user_base >= 1_000_000:  # 1M+ users
                impact_factor += 0.4
            elif user_base >= 100_000:  # 100K+ users
                impact_factor += 0.3
            elif user_base >= 10_000:  # 10K+ users
                impact_factor += 0.2

        return min(impact_factor, 2.5)  # 上限 2.5x

    def _determine_risk_level_enhanced(
        self, avg_score: float, business_criticality: str, environment: str
    ) -> str:
        """
        增強版風險等級判定

        考慮業務重要性和環境因素
        """
        # 基礎判定
        if avg_score >= 15:
            base_level = "critical"
        elif avg_score >= 10:
            base_level = "high"
        elif avg_score >= 5:
            base_level = "medium"
        else:
            base_level = "low"

        # 根據業務重要性和環境提升風險等級
        if business_criticality == "critical" and environment == "production":
            # 關鍵業務的生產環境，提升一級
            level_order = ["low", "medium", "high", "critical"]
            current_index = level_order.index(base_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]

        return base_level

    def _generate_enhanced_recommendations(
        self,
        severity_counts: dict[str, int],
        overall_risk_level: str,
        business_criticality: str,
        environment: str,
        compliance_tags: list[str],
        priority_findings: list[dict[str, Any]],
    ) -> list[str]:
        """生成增強的風險緩解建議"""
        recommendations = []

        # 嚴重性建議
        if severity_counts["critical"] > 0:
            recommendations.append(
                f"🚨 立即處理 {severity_counts['critical']} 個關鍵級別漏洞"
            )

        if severity_counts["high"] > 0:
            recommendations.append(f"⚠️ 優先處理 {severity_counts['high']} 個高風險漏洞")

        # 業務與環境特定建議
        if business_criticality == "critical" and environment == "production":
            if overall_risk_level in ["critical", "high"]:
                recommendations.append(
                    "💼 業務關鍵系統發現高風險漏洞，建議啟動緊急響應程序"
                )
                recommendations.append("🔒 考慮暫時下線受影響功能直至漏洞修復")

        # 合規建議
        if compliance_tags:
            compliance_str = ", ".join(compliance_tags)
            recommendations.append(
                f"📋 此資產受 {compliance_str} 合規要求約束，務必在規定時間內修復"
            )

        # 資料敏感度建議
        if len(priority_findings) > 0:
            top_vuln = priority_findings[0]
            recommendations.append(
                f"🎯 最高優先級：{top_vuln.get('vulnerability_type', '未知')} "
                f"(業務風險分數: {top_vuln.get('calculated_business_risk_score', 0):.1f})"
            )

        # 一般建議
        if overall_risk_level == "critical":
            recommendations.append("📞 建議立即通知相關利益相關者和管理層")
            recommendations.append("🛡️ 考慮實施臨時緩解措施（WAF 規則、IP 限制等）")

        if not recommendations:
            recommendations.append("✅ 繼續保持良好的安全實踐")

        return recommendations

    def _estimate_business_impact(
        self,
        severity_counts: dict[str, int],
        asset_value: float,
        user_base: int,
        business_criticality: str,
    ) -> dict[str, Any]:
        """估算業務影響"""
        # 潛在財務影響（簡化估算）
        financial_impact = 0.0
        if asset_value > 0:
            # 假設漏洞被利用可能導致資產價值的一定比例損失
            risk_percentage = (
                severity_counts["critical"] * 0.3
                + severity_counts["high"] * 0.15
                + severity_counts["medium"] * 0.05
            )
            financial_impact = asset_value * min(risk_percentage, 1.0)

        # 影響範圍
        affected_users = 0
        if user_base > 0:
            # 假設漏洞可能影響的使用者比例
            exposure_rate = (
                severity_counts["critical"] * 0.5
                + severity_counts["high"] * 0.3
                + severity_counts["medium"] * 0.1
            )
            affected_users = int(user_base * min(exposure_rate, 1.0))

        # 業務中斷風險
        disruption_risk = "low"
        if business_criticality == "critical":
            if severity_counts["critical"] > 0:
                disruption_risk = "high"
            elif severity_counts["high"] > 0:
                disruption_risk = "medium"

        return {
            "estimated_financial_impact": round(financial_impact, 2),
            "potentially_affected_users": affected_users,
            "business_disruption_risk": disruption_risk,
            "reputation_risk": self._assess_reputation_risk(
                severity_counts, business_criticality
            ),
        }

    def _assess_reputation_risk(
        self, severity_counts: dict[str, int], business_criticality: str
    ) -> str:
        """評估名譽風險"""
        if business_criticality == "critical":
            if severity_counts["critical"] > 0:
                return "high"
            elif severity_counts["high"] > 0:
                return "medium"
        return "low"

    def _calculate_risk_trend(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """計算風險趨勢（需要歷史資料）"""
        # 簡化版本：檢查是否有新發現的漏洞
        new_findings = sum(
            1 for f in findings if f.get("status", "").lower() == "new"
        )
        total = len(findings)

        trend = "stable"
        if new_findings > total * 0.7:
            trend = "increasing"
        elif new_findings < total * 0.3:
            trend = "decreasing"

        return {
            "trend": trend,
            "new_findings": new_findings,
            "total_findings": total,
            "new_percentage": round((new_findings / total * 100) if total > 0 else 0, 1),
        }

    def compare_risk_trends(
        self, current_assessment: dict[str, Any], previous_assessment: dict[str, Any]
    ) -> dict[str, Any]:
        """比較風險趨勢變化"""
        current_score = current_assessment.get("business_risk_score", 0)
        previous_score = previous_assessment.get("business_risk_score", 0)

        score_change = current_score - previous_score
        trend = "stable"
        if score_change > 5:
            trend = "increasing"
        elif score_change < -5:
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
