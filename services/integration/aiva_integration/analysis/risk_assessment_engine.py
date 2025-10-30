

from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class RiskAssessmentEngine:
    """
    增強版風險評估引擎

    基於發現的漏洞、環境資訊、業務影響、資產價值等多維度因素
    進行綜合風險評估和業務驅動的優先級排序。

    新增功能：
    - 業務重要性深度整合
    - 資產價值評估
    - 威脅情報關聯
    - 可利用性動態評估
    - 合規風險評估
    """

    def __init__(self) -> None:
        # 基礎風險矩陣
        self._risk_matrix = {
            "critical": {"score": 10, "priority": 1},
            "high": {"score": 7, "priority": 2},
            "medium": {"score": 4, "priority": 3},
            "low": {"score": 1, "priority": 4},
        }

        # 環境乘數（已存在，保持不變）
        self._environment_multipliers = {
            "production": 2.0,
            "staging": 1.5,
            "development": 1.0,
            "testing": 0.8,
        }

        # 業務重要性乘數（新增）
        self._business_criticality_multipliers = {
            "critical": 3.0,  # 業務關鍵系統
            "high": 2.0,  # 重要業務系統
            "medium": 1.0,  # 一般業務系統
            "low": 0.5,  # 非關鍵系統
        }

        # 資料敏感度乘數（新增）
        self._data_sensitivity_multipliers = {
            "highly_sensitive": 2.5,  # 高度敏感（如信用卡、健康資料）
            "sensitive": 1.8,  # 敏感（如 PII）
            "internal": 1.2,  # 內部資料
            "public": 0.8,  # 公開資料
        }

        # 可利用性評估乘數（擴展）
        self._exploitability_multipliers = {
            "proven": 2.0,  # 已有公開 exploit
            "high": 1.5,  # 高度可利用
            "medium": 1.0,  # 中等可利用性
            "low": 0.7,  # 低可利用性
            "theoretical": 0.4,  # 理論上可利用
        }

        # 資產網路暴露度乘數（新增）
        self._exposure_multipliers = {
            "internet_facing": 2.0,  # 直接暴露於互聯網
            "dmz": 1.5,  # DMZ 區域
            "internal_network": 1.0,  # 內部網路
            "isolated": 0.6,  # 隔離網路
        }

        # 合規要求標籤權重（新增）
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
                - environment: 環境類型 (production/staging/development/testing)
                - business_criticality: 業務重要性 (critical/high/medium/low)
                - data_sensitivity: 資料敏感度 (highly_sensitive/sensitive/internal/public)
                - asset_exposure: 網路暴露度 (internet_facing/dmz/internal_network/isolated)
                - compliance_tags: 合規標籤列表 (如 ["pci-dss", "gdpr"])
                - asset_value: 資產估值（可選，用於財務影響計算）
                - user_base: 使用者基數（可選，用於影響範圍計算）

        Returns:
            風險評估結果，包含業務驅動的優先級排序
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
        data_multiplier = self._data_sensitivity_multipliers.get(data_sensitivity, 1.0)
        exposure_multiplier = self._exposure_multipliers.get(asset_exposure, 1.0)

        # 計算合規乘數
        compliance_multiplier = 1.0
        for tag in compliance_tags:
            tag_lower = tag.lower()
            if tag_lower in self._compliance_weights:
                compliance_multiplier = max(
                    compliance_multiplier, self._compliance_weights[tag_lower]
                )

        # 計算總體上下文乘數
        # 使用加權平均而非簡單相乘，避免乘數過大
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

        # 計算總風險分數
        total_risk_score = 0.0
        business_risk_score = 0.0  # 業務風險分數
        technical_risk_score = 0.0  # 技術風險分數
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

            # 計算業務風險分數（整合所有上下文因素）
            adjusted_business_score = adjusted_tech_score * context_multiplier

            # 額外考慮：如果有資產價值或使用者基數，進一步調整
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

            # 收集高優先級發現（基於業務風險分數）
            if severity in ["critical", "high"] or adjusted_business_score >= 15:
                priority_findings.append(finding)

        # 按業務風險分數排序優先級發現
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
            "priority_findings": priority_findings[:10],  # 前10個最高業務風險
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

    def _calculate_risk_trend(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """
        計算風險趨勢

        Args:
            findings: 漏洞發現列表

        Returns:
            風險趨勢資訊
        """
        if not findings:
            return {
                "trend": "stable",
                "risk_distribution": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "total_findings": 0,
                "average_age_days": 0,
            }

        # 計算風險分布
        risk_distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        total_age_days = 0

        for finding in findings:
            severity = finding.get("severity", "low").lower()
            if severity in risk_distribution:
                risk_distribution[severity] += 1

            # 計算漏洞存在時間（如果有的話）
            created_at = finding.get("created_at")
            if created_at:
                # 這裡應該計算天數，暫時設為 0
                total_age_days += 0

        return {
            "trend": "stable",  # 需要歷史資料才能計算真實趨勢
            "risk_distribution": risk_distribution,
            "total_findings": len(findings),
            "average_age_days": total_age_days // max(len(findings), 1),
        }

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

    def _calculate_business_impact_factor(
        self, asset_value: float, user_base: int
    ) -> float:
        """
        計算業務影響因子（基於資產價值和使用者基數）

        Args:
            asset_value: 資產估值（貨幣單位）
            user_base: 使用者基數

        Returns:
            業務影響乘數 (1.0 - 3.0)
        """
        impact_factor = 1.0

        # 資產價值影響 (最大 1.5 倍)
        if asset_value > 0:
            # 對數縮放，避免極端值
            # 假設: $10K = 1.1x, $100K = 1.2x, $1M = 1.3x, $10M+ = 1.5x
            value_factor = min(1.0 + (asset_value / 1_000_000) * 0.1, 1.5)
            impact_factor *= value_factor

        # 使用者基數影響 (最大 2.0 倍)
        if user_base > 0:
            # 對數縮放
            # 假設: 1K users = 1.1x, 10K = 1.3x, 100K = 1.5x, 1M+ = 2.0x
            import math

            user_factor = min(1.0 + math.log10(user_base) * 0.15, 2.0)
            impact_factor *= user_factor

        # 總體不超過 3.0 倍
        return min(impact_factor, 3.0)

    def _determine_risk_level_enhanced(
        self, avg_risk_score: float, business_criticality: str, environment: str
    ) -> str:
        """
        增強版風險等級判定（考慮業務重要性和環境）

        Args:
            avg_risk_score: 平均業務風險分數
            business_criticality: 業務重要性 (critical/high/medium/low)
            environment: 環境類型 (production/staging/development/testing)

        Returns:
            風險等級: critical, high, medium, low
        """
        # 基礎等級判定
        if avg_risk_score >= 15:
            base_level = "critical"
        elif avg_risk_score >= 8:
            base_level = "high"
        elif avg_risk_score >= 3:
            base_level = "medium"
        else:
            base_level = "low"

        # 業務重要性升級規則
        if business_criticality == "critical" and base_level in ["high", "medium"]:
            # 關鍵業務系統：高風險提升為嚴重，中風險提升為高風險
            if base_level == "high":
                base_level = "critical"
            elif base_level == "medium":
                base_level = "high"

        # 生產環境升級規則
        if (
            environment == "production"
            and base_level == "medium"
            and avg_risk_score >= 5
        ):
            # 生產環境中風險提升為高風險（中等偏高的分數）
            base_level = "high"

        logger.debug(
            f"Enhanced risk level: {base_level} "
            f"(score={avg_risk_score:.2f}, "
            f"criticality={business_criticality}, "
            f"env={environment})"
        )

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
        """
        生成增強的業務驅動建議

        Args:
            severity_counts: 嚴重性統計
            overall_risk_level: 總體風險等級
            business_criticality: 業務重要性
            environment: 環境類型
            compliance_tags: 合規標籤
            priority_findings: 優先級發現列表

        Returns:
            建議列表
        """
        recommendations = []

        # 1. 緊急處置建議
        if severity_counts["critical"] > 0:
            recommendations.append(
                f"【緊急】立即處理 {severity_counts['critical']} 個嚴重漏洞"
            )
            if business_criticality == "critical" or environment == "production":
                recommendations.append("【緊急】建議啟動應急響應程序並通知管理層")

        # 2. 高優先級建議
        if severity_counts["high"] > 0:
            recommendations.append(
                f"【高優先級】24-48小時內處理 {severity_counts['high']} 個高風險漏洞"
            )

        # 3. 業務連續性建議
        if overall_risk_level == "critical" and environment == "production":
            recommendations.append(
                "【業務影響】建議評估是否需要臨時下線受影響功能或服務"
            )
            recommendations.append("【業務影響】建議啟動業務連續性計畫（BCP）")

        # 4. 合規性建議
        if compliance_tags:
            compliance_str = ", ".join(compliance_tags)
            recommendations.append(
                f"【合規要求】此系統受 {compliance_str} 約束，需遵循相應合規時間表"
            )
            if "pci-dss" in [tag.lower() for tag in compliance_tags]:
                recommendations.append("【PCI-DSS】嚴重及高風險漏洞需在30天內修復")
            if "gdpr" in [tag.lower() for tag in compliance_tags]:
                recommendations.append("【GDPR】涉及個資的漏洞需在72小時內評估並通報")

        # 5. 具體技術建議（基於優先級發現）
        vuln_types_seen = set()
        for finding in priority_findings[:5]:  # 前5個最高風險
            vuln_type = finding.get("vulnerability_type", "unknown")
            if vuln_type not in vuln_types_seen:
                vuln_types_seen.add(vuln_type)
                specific_rec = self._get_specific_recommendation(vuln_type)
                if specific_rec:
                    recommendations.append(f"【技術建議】{specific_rec}")

        # 6. 長期改善建議
        if severity_counts["medium"] > 3:
            recommendations.append(
                f"【改善計畫】規劃修復 {severity_counts['medium']} 個中風險漏洞"
            )

        # 7. 正向回饋
        if overall_risk_level == "low" and severity_counts["critical"] == 0:
            recommendations.append("【良好】繼續保持現有安全實踐")

        # 8. 預防性建議
        if environment == "production":
            recommendations.append("【預防】建議實施持續安全監控和定期滲透測試")

        return recommendations if recommendations else ["無特殊建議"]

    def _get_specific_recommendation(self, vuln_type: str) -> str | None:
        """
        根據漏洞類型提供具體技術建議

        Args:
            vuln_type: 漏洞類型

        Returns:
            技術建議字串，如果無匹配則返回 None
        """
        recommendations_map = {
            "sql_injection": "實施參數化查詢和 ORM，禁止動態 SQL 拼接",
            "xss": "實施嚴格的輸出編碼和 CSP（內容安全政策）",
            "csrf": "啟用 CSRF token 驗證並檢查 Referer/Origin header",
            "command_injection": "避免執行外部命令，使用安全的 API 替代",
            "path_traversal": "實施嚴格的路徑驗證和白名單機制",
            "xxe": "禁用外部實體解析，使用安全的 XML 解析器配置",
            "ssrf": "實施 URL 白名單和網路隔離，禁止訪問內部網路",
            "deserialization": "避免反序列化不可信資料，使用簽名驗證",
            "authentication_bypass": "加強認證機制，實施多因素認證（MFA）",
            "authorization_bypass": "實施嚴格的權限檢查和最小權限原則",
        }

        return recommendations_map.get(vuln_type.lower())

    def _estimate_business_impact(
        self,
        severity_counts: dict[str, int],
        asset_value: float,
        user_base: int,
        business_criticality: str,
    ) -> dict[str, Any]:
        """
        估算業務影響

        Args:
            severity_counts: 嚴重性統計
            asset_value: 資產估值
            user_base: 使用者基數
            business_criticality: 業務重要性

        Returns:
            業務影響估算結果
        """
        # 1. 計算潛在財務損失
        base_loss = 0.0
        if severity_counts["critical"] > 0:
            # 嚴重漏洞：每個估算 10% 資產價值損失風險
            base_loss += asset_value * 0.1 * severity_counts["critical"]
        if severity_counts["high"] > 0:
            # 高風險：每個估算 3% 資產價值損失風險
            base_loss += asset_value * 0.03 * severity_counts["high"]

        # 2. 計算影響使用者數
        affected_users = 0
        if severity_counts["critical"] > 0 or severity_counts["high"] > 0:
            # 嚴重/高風險可能影響所有使用者
            affected_users = user_base
        elif severity_counts["medium"] > 0:
            # 中風險可能影響部分使用者
            affected_users = int(user_base * 0.3)

        # 3. 業務中斷風險
        disruption_risk = "none"
        if severity_counts["critical"] > 0 and business_criticality == "critical":
            disruption_risk = "high"
        elif severity_counts["critical"] > 0 or (
            severity_counts["high"] > 2 and business_criticality in ["critical", "high"]
        ):
            disruption_risk = "medium"
        elif severity_counts["high"] > 0:
            disruption_risk = "low"

        # 4. 聲譽影響
        reputation_impact = "none"
        if severity_counts["critical"] > 0:
            reputation_impact = "high"
        elif severity_counts["high"] > 2:
            reputation_impact = "medium"
        elif severity_counts["high"] > 0:
            reputation_impact = "low"

        # 5. 合規性風險
        compliance_risk = "unknown"
        if severity_counts["critical"] > 0 or severity_counts["high"] > 0:
            compliance_risk = "potential_violation"

        return {
            "estimated_financial_loss": round(base_loss, 2),
            "currency": "USD",
            "affected_users": affected_users,
            "user_base": user_base,
            "affected_percentage": (
                round((affected_users / user_base) * 100, 1) if user_base > 0 else 0
            ),
            "business_disruption_risk": disruption_risk,
            "reputation_impact": reputation_impact,
            "compliance_risk": compliance_risk,
            "impact_summary": self._generate_impact_summary(
                base_loss, affected_users, disruption_risk, reputation_impact
            ),
        }

    def _generate_impact_summary(
        self,
        financial_loss: float,
        affected_users: int,
        disruption_risk: str,
        reputation_impact: str,
    ) -> str:
        """
        生成業務影響摘要

        Args:
            financial_loss: 財務損失估算
            affected_users: 受影響使用者數
            disruption_risk: 業務中斷風險
            reputation_impact: 聲譽影響

        Returns:
            影響摘要字串
        """
        summary_parts = []

        if financial_loss > 0:
            summary_parts.append(f"潛在財務損失 ${financial_loss:,.2f}")

        if affected_users > 0:
            summary_parts.append(f"可能影響 {affected_users:,} 名使用者")

        if disruption_risk in ["high", "medium"]:
            summary_parts.append(f"業務中斷風險: {disruption_risk}")

        if reputation_impact in ["high", "medium"]:
            summary_parts.append(f"聲譽影響: {reputation_impact}")

        return "；".join(summary_parts) if summary_parts else "影響較小"
