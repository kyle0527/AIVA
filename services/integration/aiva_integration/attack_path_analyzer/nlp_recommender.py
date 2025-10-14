"""
Attack Path NLP Recommender - 攻擊路徑自然語言推薦系統

為攻擊路徑分析結果提供自然語言解釋和優先修復建議
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from services.integration.aiva_integration.attack_path_analyzer.engine import (
    AttackPath,
    NodeType,
)


class RiskLevel(str, Enum):
    """風險等級"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PathRecommendation:
    """路徑推薦"""

    path_id: str
    risk_level: RiskLevel
    priority_score: float
    executive_summary: str
    technical_explanation: str
    business_impact: str
    remediation_steps: list[str]
    quick_wins: list[str]
    affected_assets: list[str]
    estimated_effort: str
    estimated_risk_reduction: float


class AttackPathNLPRecommender:
    """攻擊路徑自然語言推薦器"""

    def __init__(self):
        """初始化推薦器"""
        # 漏洞類型描述模板
        self._vuln_descriptions = {
            "SQLI": {
                "name": "SQL 注入",
                "impact": "攻擊者可以直接存取、竄改或刪除資料庫資料",
                "common_exploits": "資料洩露、身份認證繞過、資料篡改",
            },
            "XSS": {
                "name": "跨站腳本攻擊",
                "impact": "攻擊者可以竊取使用者 Session、執行惡意腳本、釣魚攻擊",
                "common_exploits": "帳號劫持、惡意重定向、資料竊取",
            },
            "SSRF": {
                "name": "伺服器端請求偽造",
                "impact": "攻擊者可以探測內網、存取內部服務",
                "common_exploits": "內網掃描、雲端元資料存取、內部 API 呼叫",
            },
            "IDOR": {
                "name": "不安全的直接物件引用",
                "impact": "攻擊者可以存取未授權的資料或功能",
                "common_exploits": "未授權資料存取、越權操作",
            },
            "BOLA": {
                "name": "物件層級授權缺失",
                "impact": "攻擊者可以存取其他使用者的資源",
                "common_exploits": "資料洩露、隱私侵犯",
            },
            "AUTHENTICATION_BYPASS": {
                "name": "身份認證繞過",
                "impact": "攻擊者可以未經授權存取系統",
                "common_exploits": "帳號接管、系統控制",
            },
            "RCE": {
                "name": "遠端程式碼執行",
                "impact": "攻擊者可以在伺服器上執行任意程式碼",
                "common_exploits": "完全系統控制、資料竊取、後門植入",
            },
        }

        # 節點類型描述
        self._node_descriptions = {
            NodeType.ATTACKER: "外部攻擊者",
            NodeType.ASSET: "應用資產",
            NodeType.VULNERABILITY: "安全漏洞",
            NodeType.DATABASE: "資料庫系統",
            NodeType.CREDENTIAL: "身份憑證",
            NodeType.API_ENDPOINT: "API 端點",
            NodeType.INTERNAL_NETWORK: "內部網路",
        }

    def analyze_and_recommend(
        self, paths: list[AttackPath], top_n: int = 5
    ) -> list[PathRecommendation]:
        """
        分析攻擊路徑並生成推薦

        Args:
            paths: 攻擊路徑列表
            top_n: 回傳前 N 條推薦

        Returns:
            路徑推薦列表
        """
        recommendations = []

        for path in paths[:top_n]:
            # 計算優先級分數
            priority_score = self._calculate_priority_score(path)

            # 判斷風險等級
            risk_level = self._determine_risk_level(priority_score, path)

            # 生成執行摘要
            executive_summary = self._generate_executive_summary(path, risk_level)

            # 生成技術解釋
            technical_explanation = self._generate_technical_explanation(path)

            # 生成業務影響說明
            business_impact = self._generate_business_impact(path, risk_level)

            # 生成修復步驟
            remediation_steps = self._generate_remediation_steps(path)

            # 識別快速修復項目
            quick_wins = self._identify_quick_wins(path)

            # 提取受影響資產
            affected_assets = self._extract_affected_assets(path)

            # 估算修復工作量
            estimated_effort = self._estimate_effort(path)

            # 估算風險降低程度
            risk_reduction = self._estimate_risk_reduction(path)

            recommendations.append(
                PathRecommendation(
                    path_id=path.path_id,
                    risk_level=risk_level,
                    priority_score=priority_score,
                    executive_summary=executive_summary,
                    technical_explanation=technical_explanation,
                    business_impact=business_impact,
                    remediation_steps=remediation_steps,
                    quick_wins=quick_wins,
                    affected_assets=affected_assets,
                    estimated_effort=estimated_effort,
                    estimated_risk_reduction=risk_reduction,
                )
            )

        # 按優先級排序
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        return recommendations

    def _calculate_priority_score(self, path: AttackPath) -> float:
        """
        計算優先級分數

        考慮因素:
        - 路徑總風險分數
        - 路徑長度（越短越危險）
        - 漏洞嚴重程度
        - 目標節點類型的敏感度

        Returns:
            優先級分數 (0-100)
        """
        # 基礎風險分數 (0-40分)
        base_risk = min(path.total_risk_score * 4, 40)

        # 路徑長度評分 (0-25分，越短分數越高)
        # 長度 1-3: 25分, 4-6: 15分, 7+: 5分
        if path.length <= 3:
            length_score = 25
        elif path.length <= 6:
            length_score = 15
        else:
            length_score = 5

        # 目標節點敏感度 (0-20分)
        target_node = path.nodes[-1] if path.nodes else {}
        target_labels = target_node.get("labels", [])
        sensitivity_scores = {
            "Database": 20,
            "Credential": 18,
            "APIEndpoint": 15,
            "InternalNetwork": 12,
            "Asset": 10,
        }
        sensitivity_score = 0
        for label in target_labels:
            sensitivity_score = max(sensitivity_score, sensitivity_scores.get(label, 10))

        # 嚴重漏洞數量 (0-15分)
        critical_vuln_count = sum(
            1
            for node in path.nodes
            if "Vulnerability" in node.get("labels", [])
            and node.get("severity") in ["CRITICAL", "HIGH"]
        )
        vuln_score = min(critical_vuln_count * 5, 15)

        total_score = base_risk + length_score + sensitivity_score + vuln_score

        return min(total_score, 100.0)

    def _determine_risk_level(
        self, priority_score: float, path: AttackPath
    ) -> RiskLevel:
        """判斷風險等級"""
        if priority_score >= 80 or path.total_risk_score >= 25:
            return RiskLevel.CRITICAL
        elif priority_score >= 60 or path.total_risk_score >= 15:
            return RiskLevel.HIGH
        elif priority_score >= 40 or path.total_risk_score >= 8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_executive_summary(
        self, path: AttackPath, risk_level: RiskLevel
    ) -> str:
        """生成執行摘要（給管理層看的）"""
        # 識別關鍵元素
        vuln_count = sum(
            1 for node in path.nodes if "Vulnerability" in node.get("labels", [])
        )

        target_node = path.nodes[-1] if path.nodes else {}
        target_type = self._get_node_type_name(target_node)

        critical_vulns = [
            node.get("name", "未知漏洞")
            for node in path.nodes
            if "Vulnerability" in node.get("labels", [])
            and node.get("severity") in ["CRITICAL", "HIGH"]
        ]

        risk_emoji = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
        }

        summary = f"{risk_emoji[risk_level]} **{risk_level.value.upper()} 風險攻擊路徑**\n\n"

        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            summary += f"發現一條高風險攻擊路徑，外部攻擊者可透過 {vuln_count} 個安全漏洞"
            summary += f"最終到達 **{target_type}**。"
        else:
            summary += f"發現一條潛在攻擊路徑，涉及 {vuln_count} 個安全漏洞，"
            summary += f"可能影響 **{target_type}**。"

        if critical_vulns:
            summary += f"\n\n**關鍵漏洞**: {', '.join(critical_vulns[:3])}"
            if len(critical_vulns) > 3:
                summary += f" 及其他 {len(critical_vulns) - 3} 個"

        summary += f"\n\n**路徑長度**: {path.length} 步"
        summary += f" | **風險評分**: {path.total_risk_score:.1f}/10"

        return summary

    def _generate_technical_explanation(self, path: AttackPath) -> str:
        """生成技術解釋（給技術團隊看的）"""
        explanation = "## 攻擊路徑技術分析\n\n"

        # 詳細路徑步驟
        explanation += "### 攻擊步驟\n\n"

        for i, node in enumerate(path.nodes):
            node_labels = node.get("labels", [])
            node_type = node_labels[0] if node_labels else "Unknown"

            if i == 0:
                explanation += f"**步驟 {i + 1}: 起點 - {self._get_node_description(node)}**\n"
                explanation += "  - 攻擊者從外部網路發起攻擊\n\n"

            elif "Vulnerability" in node_labels:
                vuln_name = node.get("name", "Unknown")
                severity = node.get("severity", "UNKNOWN")
                cwe = node.get("cwe", "N/A")

                explanation += f"**步驟 {i + 1}: 利用漏洞 - {vuln_name}**\n"
                explanation += f"  - **嚴重程度**: {severity}\n"
                explanation += f"  - **CWE 編號**: {cwe}\n"

                # 添加漏洞詳細說明
                if vuln_name in self._vuln_descriptions:
                    vuln_info = self._vuln_descriptions[vuln_name]
                    explanation += f"  - **影響**: {vuln_info['impact']}\n"
                    explanation += (
                        f"  - **常見利用方式**: {vuln_info['common_exploits']}\n"
                    )

                # 添加對應的邊資訊
                if i < len(path.edges):
                    edge = path.edges[i]
                    edge_type = edge.get("type", "UNKNOWN")
                    edge_risk = edge.get("risk", 0)
                    explanation += f"  - **攻擊效果**: {self._translate_edge_type(edge_type)}\n"
                    explanation += f"  - **路徑風險**: {edge_risk:.1f}\n"

                explanation += "\n"

            elif i == len(path.nodes) - 1:
                explanation += f"**步驟 {i + 1}: 攻擊目標 - {self._get_node_description(node)}**\n"
                explanation += f"  - 攻擊者成功到達 {self._get_node_type_name(node)}\n"
                explanation += "  - 可能的後果:\n"

                # 根據目標類型添加後果說明
                if "Database" in node_labels:
                    explanation += "    - 竊取、篡改或刪除敏感資料\n"
                    explanation += "    - 獲取使用者憑證和個人資訊\n"
                    explanation += "    - 破壞資料完整性\n"
                elif "Credential" in node_labels:
                    explanation += "    - 劫持使用者帳號\n"
                    explanation += "    - 橫向移動到其他系統\n"
                    explanation += "    - 持久化存取\n"
                elif "InternalNetwork" in node_labels:
                    explanation += "    - 探測內部網路拓撲\n"
                    explanation += "    - 存取內部服務和 API\n"
                    explanation += "    - 建立跳板進行進一步攻擊\n"
                elif "APIEndpoint" in node_labels:
                    explanation += "    - 未授權存取敏感 API\n"
                    explanation += "    - 資料洩露或篡改\n"
                    explanation += "    - 業務邏輯繞過\n"

                explanation += "\n"

        return explanation

    def _generate_business_impact(self, path: AttackPath, risk_level: RiskLevel) -> str:
        """生成業務影響說明"""
        impact = "## 業務影響評估\n\n"

        # 根據風險等級給出不同的影響說明
        if risk_level == RiskLevel.CRITICAL:
            impact += "### 🚨 嚴重業務影響\n\n"
            impact += "此攻擊路徑若被利用，可能導致:\n\n"
            impact += "- **資料洩露風險**: 極高，敏感資料可能完全洩露\n"
            impact += "- **服務中斷風險**: 高，可能導致服務完全停擺\n"
            impact += "- **財務損失**: 可能超過數百萬元，包含罰款、補償、商譽損失\n"
            impact += "- **法規合規**: 可能違反 GDPR、PCI-DSS 等法規，面臨鉅額罰款\n"
            impact += "- **商譽損害**: 嚴重，可能導致客戶流失和媒體負面報導\n\n"
            impact += "**建議行動**: 立即召集緊急會議，24小時內完成修復\n"

        elif risk_level == RiskLevel.HIGH:
            impact += "### ⚠️ 高度業務影響\n\n"
            impact += "此攻擊路徑具有顯著風險:\n\n"
            impact += "- **資料洩露風險**: 高，部分敏感資料可能洩露\n"
            impact += "- **服務中斷風險**: 中等，可能影響部分服務\n"
            impact += "- **財務損失**: 可能達到數十萬元\n"
            impact += "- **法規合規**: 需要注意合規風險\n"
            impact += "- **商譽損害**: 中等，需要謹慎處理\n\n"
            impact += "**建議行動**: 優先處理，一週內完成修復\n"

        elif risk_level == RiskLevel.MEDIUM:
            impact += "### ⚡ 中度業務影響\n\n"
            impact += "此攻擊路徑需要關注:\n\n"
            impact += "- **資料洩露風險**: 中等\n"
            impact += "- **服務中斷風險**: 低\n"
            impact += "- **財務損失**: 可能數萬元\n"
            impact += "- **商譽損害**: 有限\n\n"
            impact += "**建議行動**: 納入修復計劃，一個月內完成\n"

        else:
            impact += "### ℹ️ 低度業務影響\n\n"
            impact += "此攻擊路徑風險較低，但仍需注意。\n\n"
            impact += "**建議行動**: 常規維護週期內處理\n"

        # 添加受影響的利益相關者
        target_node = path.nodes[-1] if path.nodes else {}
        target_labels = target_node.get("labels", [])

        impact += "\n### 受影響的利益相關者\n\n"
        if "Database" in target_labels:
            impact += "- **資料所有者**: 客戶、使用者\n"
            impact += "- **負責團隊**: 資安團隊、DBA 團隊、應用開發團隊\n"
            impact += "- **需通知對象**: CISO、法務、客服、公關\n"
        elif "InternalNetwork" in target_labels:
            impact += "- **負責團隊**: 網路安全團隊、基礎架構團隊\n"
            impact += "- **需通知對象**: CISO、IT 主管\n"
        else:
            impact += "- **負責團隊**: 資安團隊、應用開發團隊\n"
            impact += "- **需通知對象**: CISO、產品負責人\n"

        return impact

    def _generate_remediation_steps(self, path: AttackPath) -> list[str]:
        """生成修復步驟（按優先級排序）"""
        steps = []

        # 提取所有漏洞
        vulnerabilities = [
            node
            for node in path.nodes
            if "Vulnerability" in node.get("labels", [])
        ]

        # 按嚴重程度排序
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        vulnerabilities.sort(
            key=lambda v: severity_order.get(v.get("severity", "LOW"), 4)
        )

        # 為每個漏洞生成修復建議
        for i, vuln in enumerate(vulnerabilities, 1):
            vuln_name = vuln.get("name", "Unknown")
            severity = vuln.get("severity", "UNKNOWN")

            step = f"【優先級 {i}】修復 {vuln_name} 漏洞 ({severity}):\n"

            # 根據漏洞類型提供具體建議
            if vuln_name == "SQLI":
                step += "  - 使用參數化查詢 (Prepared Statements)\n"
                step += "  - 啟用 ORM 框架的參數綁定功能\n"
                step += "  - 實施輸入驗證和過濾\n"
                step += "  - 使用最小權限資料庫帳號\n"
                step += "  - 部署 WAF 規則攔截 SQL 注入攻擊"

            elif vuln_name == "XSS":
                step += "  - 實施 Content Security Policy (CSP)\n"
                step += "  - 對所有使用者輸入進行 HTML 編碼\n"
                step += "  - 使用安全的模板引擎\n"
                step += "  - 啟用 HttpOnly 和 Secure Cookie 標誌\n"
                step += "  - 部署 WAF XSS 防護規則"

            elif vuln_name == "SSRF":
                step += "  - 實施嚴格的 URL 白名單\n"
                step += "  - 禁止存取內部 IP 範圍 (RFC1918)\n"
                step += "  - 禁止存取雲端元資料端點 (169.254.169.254)\n"
                step += "  - 使用獨立的網路隔離環境\n"
                step += "  - 限制出站連線"

            elif vuln_name in ["IDOR", "BOLA"]:
                step += "  - 實施嚴格的授權檢查\n"
                step += "  - 使用 UUID 取代順序 ID\n"
                step += "  - 在每個端點驗證使用者權限\n"
                step += "  - 實施 RBAC 或 ABAC 存取控制\n"
                step += "  - 記錄所有授權失敗嘗試"

            elif vuln_name == "AUTHENTICATION_BYPASS":
                step += "  - 修復認證邏輯漏洞\n"
                step += "  - 實施多因素認證 (MFA)\n"
                step += "  - 加強 Session 管理\n"
                step += "  - 實施帳號鎖定機制\n"
                step += "  - 定期審查認證程式碼"

            elif vuln_name == "RCE":
                step += "  - 【緊急】立即修補已知 RCE 漏洞\n"
                step += "  - 禁止執行使用者提供的程式碼\n"
                step += "  - 實施應用程式沙箱\n"
                step += "  - 移除或禁用危險函數\n"
                step += "  - 部署 Runtime Application Self-Protection (RASP)"

            else:
                step += f"  - 參考 CWE-{vuln.get('cwe', 'N/A')} 修復指南\n"
                step += "  - 實施輸入驗證和輸出編碼\n"
                step += "  - 進行安全程式碼審查\n"
                step += "  - 部署相應的安全控制措施"

            steps.append(step)

        # 添加通用的防護措施
        steps.append(
            "【通用防護】加強整體安全態勢:\n"
            "  - 部署 Web Application Firewall (WAF)\n"
            "  - 啟用詳細的安全日誌記錄\n"
            "  - 實施入侵檢測系統 (IDS/IPS)\n"
            "  - 定期進行滲透測試\n"
            "  - 建立安全事件響應計劃"
        )

        return steps

    def _identify_quick_wins(self, path: AttackPath) -> list[str]:
        """識別快速修復項目（低工作量、高效益）"""
        quick_wins = []

        vulnerabilities = [
            node
            for node in path.nodes
            if "Vulnerability" in node.get("labels", [])
        ]

        for vuln in vulnerabilities:
            vuln_name = vuln.get("name", "Unknown")

            # 識別可以快速修復的項目
            if vuln_name == "XSS":
                quick_wins.append(
                    "🚀 部署 Content Security Policy (CSP) Header (1小時內可完成)"
                )
                quick_wins.append(
                    "🚀 啟用 HttpOnly 和 Secure Cookie 標誌 (30分鐘內可完成)"
                )

            elif vuln_name == "SSRF":
                quick_wins.append(
                    "🚀 在防火牆層面阻擋內部 IP 存取 (1小時內可完成)"
                )
                quick_wins.append(
                    "🚀 加入雲端元資料端點黑名單 (30分鐘內可完成)"
                )

            elif vuln_name in ["IDOR", "BOLA"]:
                quick_wins.append(
                    "🚀 在中介軟體層面加入統一授權檢查 (半天可完成)"
                )
                quick_wins.append("🚀 啟用詳細的存取日誌記錄 (1小時內可完成)")

        # 通用快速修復
        if len(vulnerabilities) > 0:
            quick_wins.append(
                "🚀 部署 WAF 規則阻擋已知攻擊模式 (當天可完成)"
            )
            quick_wins.append(
                "🚀 限制錯誤訊息中的敏感資訊洩露 (半天可完成)"
            )

        # 去重
        return list(set(quick_wins))

    def _extract_affected_assets(self, path: AttackPath) -> list[str]:
        """提取受影響的資產"""
        assets = []

        for node in path.nodes:
            if "Asset" in node.get("labels", []):
                asset_value = node.get("value", node.get("name", "Unknown"))
                assets.append(asset_value)

        return assets

    def _estimate_effort(self, path: AttackPath) -> str:
        """估算修復工作量"""
        vuln_count = sum(
            1 for node in path.nodes if "Vulnerability" in node.get("labels", [])
        )

        critical_count = sum(
            1
            for node in path.nodes
            if "Vulnerability" in node.get("labels", [])
            and node.get("severity") == "CRITICAL"
        )

        # 基於漏洞數量和嚴重程度估算
        if critical_count >= 2 or vuln_count >= 5:
            return "高 (預估 2-4 週，需要多個團隊協作)"
        elif critical_count >= 1 or vuln_count >= 3:
            return "中 (預估 1-2 週，需要開發和測試資源)"
        else:
            return "低 (預估 2-5 天，可由單一團隊完成)"

    def _estimate_risk_reduction(self, path: AttackPath) -> float:
        """
        估算修復後的風險降低程度

        Returns:
            風險降低百分比 (0-100)
        """
        # 如果修復了路徑中的關鍵漏洞，整條攻擊路徑就會被阻斷
        # 因此風險降低程度與路徑的總風險成正比

        risk_score = path.total_risk_score

        if risk_score >= 25:
            return 95.0  # 極高風險路徑，修復後大幅降低風險
        elif risk_score >= 15:
            return 85.0
        elif risk_score >= 8:
            return 70.0
        else:
            return 50.0

    def _get_node_description(self, node: dict[str, Any]) -> str:
        """獲取節點描述"""
        node_labels = node.get("labels", [])
        node_type = node_labels[0] if node_labels else "Unknown"

        name = node.get("name", node.get("value", node.get("id", "Unknown")))

        type_name = self._node_descriptions.get(
            NodeType(node_type) if node_type in [e.value for e in NodeType] else None,
            node_type,
        )

        return f"{type_name}: {name}"

    def _get_node_type_name(self, node: dict[str, Any]) -> str:
        """獲取節點類型名稱"""
        node_labels = node.get("labels", [])

        for label in node_labels:
            if label in self._node_descriptions:
                return self._node_descriptions[label]

        return "系統資源"

    def _translate_edge_type(self, edge_type: str) -> str:
        """翻譯邊類型"""
        translations = {
            "EXPLOITS": "利用漏洞",
            "LEADS_TO": "導致存取",
            "GRANTS_ACCESS": "授予存取權限",
            "EXPOSES": "暴露資訊",
            "CONTAINS": "包含",
            "CAN_ACCESS": "可存取",
            "HAS_VULNERABILITY": "存在漏洞",
        }

        return translations.get(edge_type, edge_type)

    def generate_report(self, recommendations: list[PathRecommendation]) -> str:
        """
        生成完整的推薦報告

        Args:
            recommendations: 推薦列表

        Returns:
            Markdown 格式的報告
        """
        report = "# 攻擊路徑分析與修復建議報告\n\n"
        report += f"**生成時間**: {self._get_current_time()}\n\n"
        report += "---\n\n"

        # 執行摘要
        report += "## 📊 執行摘要\n\n"
        report += f"本次分析發現 **{len(recommendations)}** 條需要關注的攻擊路徑。\n\n"

        # 風險等級統計
        risk_counts = {level: 0 for level in RiskLevel}
        for rec in recommendations:
            risk_counts[rec.risk_level] += 1

        report += "### 風險等級分布\n\n"
        report += f"- 🔴 **CRITICAL**: {risk_counts[RiskLevel.CRITICAL]} 條\n"
        report += f"- 🟠 **HIGH**: {risk_counts[RiskLevel.HIGH]} 條\n"
        report += f"- 🟡 **MEDIUM**: {risk_counts[RiskLevel.MEDIUM]} 條\n"
        report += f"- 🟢 **LOW**: {risk_counts[RiskLevel.LOW]} 條\n\n"

        # 總體建議
        if risk_counts[RiskLevel.CRITICAL] > 0:
            report += "### ⚠️ 緊急建議\n\n"
            report += (
                f"發現 {risk_counts[RiskLevel.CRITICAL]} 條 **CRITICAL** 風險攻擊路徑，"
            )
            report += "建議立即召開緊急會議，在 24 小時內開始修復工作。\n\n"

        report += "---\n\n"

        # 詳細推薦
        report += "## 🎯 詳細推薦\n\n"

        for i, rec in enumerate(recommendations, 1):
            report += f"### 路徑 {i}: {rec.risk_level.value.upper()}\n\n"

            # 執行摘要
            report += rec.executive_summary + "\n\n"

            # 技術解釋
            report += rec.technical_explanation + "\n\n"

            # 業務影響
            report += rec.business_impact + "\n\n"

            # 快速修復
            if rec.quick_wins:
                report += "### ⚡ 快速修復建議\n\n"
                for quick_win in rec.quick_wins:
                    report += f"- {quick_win}\n"
                report += "\n"

            # 修復步驟
            report += "### 🔧 詳細修復步驟\n\n"
            for j, step in enumerate(rec.remediation_steps, 1):
                report += f"{j}. {step}\n\n"

            # 受影響資產
            if rec.affected_assets:
                report += "### 📦 受影響資產\n\n"
                for asset in rec.affected_assets:
                    report += f"- `{asset}`\n"
                report += "\n"

            # 修復評估
            report += "### 📈 修復評估\n\n"
            report += f"- **預估工作量**: {rec.estimated_effort}\n"
            report += (
                f"- **預估風險降低**: {rec.estimated_risk_reduction:.0f}%\n"
            )
            report += f"- **優先級分數**: {rec.priority_score:.1f}/100\n\n"

            report += "---\n\n"

        # 總結
        report += "## 📝 總結與後續行動\n\n"
        report += "### 建議的行動優先順序\n\n"

        critical_recs = [r for r in recommendations if r.risk_level == RiskLevel.CRITICAL]
        high_recs = [r for r in recommendations if r.risk_level == RiskLevel.HIGH]

        if critical_recs:
            report += "**立即行動 (24小時內)**:\n"
            for rec in critical_recs[:3]:
                report += f"- 修復路徑: {rec.path_id}\n"
            report += "\n"

        if high_recs:
            report += "**短期行動 (1週內)**:\n"
            for rec in high_recs[:3]:
                report += f"- 修復路徑: {rec.path_id}\n"
            report += "\n"

        report += "**中長期行動**:\n"
        report += "- 建立持續的漏洞掃描和攻擊路徑分析機制\n"
        report += "- 加強安全開發培訓\n"
        report += "- 實施安全左移 (Shift-Left Security)\n"
        report += "- 定期進行紅隊演練\n\n"

        return report

    def _get_current_time(self) -> str:
        """獲取當前時間字串"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
