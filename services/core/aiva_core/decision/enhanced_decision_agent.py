#!/usr/bin/env python3
"""AIVA 決策代理增強模組
用途: 整合風險評估和經驗驅動決策，提升 AI 決策的智能化水平
基於: BioNeuron_模型_AI核心大腦.md 中的決策代理分析

Compliance Note:
- 修正日期: 2025-10-25
- 修正項目: 移除重複定義的 RiskLevel，改用 aiva_common.enums.RiskLevel
- 符合架構原則: 使用 aiva_common 統一枚舉定義
"""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sys
from typing import Any

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

# 使用 aiva_common 的統一枚舉定義
from services.aiva_common.enums import RiskLevel

# Compliance Note: 2025-10-26 - 移除重複定義，統一使用 bio_neuron_master.py 中的 OperationMode
from ..bio_neuron_master import OperationMode


class DecisionContext:
    """決策上下文"""

    def __init__(self):
        self.risk_level = RiskLevel.LOW
        self.discovered_vulns = []
        self.attempts_without_success = 0
        self.target_info = {}
        self.previous_results = []
        self.time_constraints = None
        self.available_tools = []
        self.mode_restrictions = []


class Decision:
    """決策結果"""

    def __init__(
        self, action: str, params: dict[str, Any] = None, confidence: float = 0.5
    ):
        self.action = action
        self.params = params or {}
        self.confidence = confidence
        self.reasoning = ""
        self.alternatives = []
        self.risk_assessment = None


class EnhancedDecisionAgent:
    """增強的決策代理"""

    def __init__(self, knowledge_base=None, experience_manager=None):
        self.knowledge_base = knowledge_base
        self.experience_manager = experience_manager
        self.decision_history = []
        self.risk_threshold = 0.7
        self.success_threshold = 3  # 失敗嘗試的閾值

        # 決策規則引擎
        self.decision_rules = self._initialize_decision_rules()

        # 工具選擇偏好
        self.tool_preferences = {
            "sql_injection": ["sqlmap", "havij", "manual_test"],
            "xss": ["xsser", "xsstrike", "manual_test"],
            "directory_traversal": ["dirb", "gobuster", "manual_enum"],
            "port_scan": ["nmap", "masscan", "unicornscan"],
            "web_scan": ["nikto", "dirb", "wpscan"],
            "brute_force": ["hydra", "medusa", "john"],
        }

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """設置日誌記錄器"""
        logger = logging.getLogger("EnhancedDecisionAgent")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_decision_rules(self) -> list[dict[str, Any]]:
        """初始化決策規則"""
        return [
            {
                "name": "high_risk_confirmation",
                "condition": lambda ctx: ctx.risk_level
                in [RiskLevel.HIGH, RiskLevel.CRITICAL],
                "action": "REQUIRE_CONFIRMATION",
                "priority": 100,
                "description": "高風險操作需要用戶確認",
            },
            {
                "name": "sql_injection_found",
                "condition": lambda ctx: "sql_injection" in ctx.discovered_vulns,
                "action": "EXPLOIT_SQL_INJECTION",
                "priority": 90,
                "description": "發現 SQL 注入，深入測試",
            },
            {
                "name": "multiple_failures",
                "condition": lambda ctx: ctx.attempts_without_success
                >= self.success_threshold,
                "action": "CHANGE_STRATEGY",
                "priority": 80,
                "description": "多次失敗後改變策略",
            },
            {
                "name": "web_service_detected",
                "condition": lambda ctx: any(
                    "http" in str(tool).lower() for tool in ctx.available_tools
                ),
                "action": "WEB_ATTACK",
                "priority": 70,
                "description": "檢測到 Web 服務，執行 Web 攻擊",
            },
            {
                "name": "ssh_service_available",
                "condition": lambda ctx: any(
                    "ssh" in str(tool).lower() for tool in ctx.available_tools
                ),
                "action": "SSH_BRUTE_FORCE",
                "priority": 60,
                "description": "SSH 服務可用，嘗試爆破",
            },
        ]

    def make_decision(self, context: DecisionContext) -> Decision:
        """基於上下文做出智能決策

        Args:
            context: 決策上下文

        Returns:
            決策結果
        """
        self.logger.info(f"🤔 開始決策分析 - 風險等級: {context.risk_level.value}")

        # 1. 風險評估決策
        risk_decision = self._assess_risk_decision(context)
        if risk_decision:
            return risk_decision

        # 2. 經驗驅動決策
        experience_decision = self._make_experience_driven_decision(context)
        if experience_decision and experience_decision.confidence > 0.7:
            return experience_decision

        # 3. 規則引擎決策
        rule_decision = self._apply_decision_rules(context)
        if rule_decision:
            return rule_decision

        # 4. 預設決策
        default_decision = self._make_default_decision(context)

        # 記錄決策
        self._record_decision(context, default_decision)

        return default_decision

    def _assess_risk_decision(self, context: DecisionContext) -> Decision | None:
        """基於風險評估的決策"""
        if context.risk_level == RiskLevel.CRITICAL:
            decision = Decision(
                action="STOP_OPERATION",
                params={"reason": "Critical risk level detected"},
                confidence=1.0,
            )
            decision.reasoning = "檢測到重大風險，停止操作以避免損害"
            return decision

        if context.risk_level == RiskLevel.HIGH:
            decision = Decision(
                action="SWITCH_MODE",
                params={"mode": OperationMode.UI.value},  # 現在使用統一的小寫值 "ui"
                confidence=0.9,
            )
            decision.reasoning = "高風險操作，切換至 UI 模式要求用戶確認"
            return decision

        return None

    def _make_experience_driven_decision(
        self, context: DecisionContext
    ) -> Decision | None:
        """基於經驗的決策"""
        if not self.experience_manager:
            return None

        try:
            # 搜尋相似的成功經驗
            similar_experiences = self._find_similar_experiences(context)

            if not similar_experiences:
                return None

            # 選擇最佳經驗
            best_experience = max(
                similar_experiences, key=lambda x: x.get("success_score", 0)
            )

            if best_experience["success_score"] > 0.8:
                decision = Decision(
                    action=best_experience["recommended_action"],
                    params=best_experience.get("parameters", {}),
                    confidence=best_experience["success_score"],
                )
                decision.reasoning = (
                    f"基於類似成功經驗 (成功率: {best_experience['success_score']:.1%})"
                )
                return decision

        except Exception as e:
            self.logger.error(f"經驗驅動決策異常: {e}")

        return None

    def _find_similar_experiences(
        self, context: DecisionContext
    ) -> list[dict[str, Any]]:
        """查找相似的成功經驗"""
        # 模擬經驗查找 (實際實作應該查詢經驗管理器)
        mock_experiences = [
            {
                "scenario": "web_application_test",
                "target_type": "http_service",
                "vulnerabilities": ["sql_injection"],
                "recommended_action": "EXPLOIT_SQL_INJECTION",
                "parameters": {"tool": "sqlmap", "payload_type": "boolean"},
                "success_score": 0.85,
                "attempts": 2,
            },
            {
                "scenario": "network_penetration",
                "target_type": "ssh_service",
                "vulnerabilities": ["weak_credentials"],
                "recommended_action": "SSH_BRUTE_FORCE",
                "parameters": {"tool": "hydra", "wordlist": "common_passwords"},
                "success_score": 0.75,
                "attempts": 4,
            },
        ]

        # 簡單的相似度匹配
        similar = []
        for exp in mock_experiences:
            similarity = self._calculate_similarity(context, exp)
            if similarity > 0.6:
                exp["similarity"] = similarity
                similar.append(exp)

        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def _calculate_similarity(
        self, context: DecisionContext, experience: dict[str, Any]
    ) -> float:
        """計算上下文與經驗的相似度"""
        similarity = 0.0

        # 漏洞類型相似度
        ctx_vulns = set(context.discovered_vulns)
        exp_vulns = set(experience.get("vulnerabilities", []))

        if ctx_vulns and exp_vulns:
            intersection = len(ctx_vulns.intersection(exp_vulns))
            union = len(ctx_vulns.union(exp_vulns))
            similarity += (intersection / union) * 0.6

        # 工具可用性相似度
        if context.available_tools:
            recommended_tool = experience.get("parameters", {}).get("tool")
            if recommended_tool in context.available_tools:
                similarity += 0.4

        return min(similarity, 1.0)

    def _apply_decision_rules(self, context: DecisionContext) -> Decision | None:
        """應用決策規則引擎"""
        # 按優先級排序規則
        sorted_rules = sorted(
            self.decision_rules, key=lambda x: x["priority"], reverse=True
        )

        for rule in sorted_rules:
            try:
                if rule["condition"](context):
                    decision = self._execute_rule_action(rule, context)
                    if decision:
                        decision.reasoning = rule["description"]
                        self.logger.info(
                            f"✅ 觸發規則: {rule['name']} -> {rule['action']}"
                        )
                        return decision

            except Exception as e:
                self.logger.error(f"規則 {rule['name']} 執行異常: {e}")
                continue

        return None

    def _execute_rule_action(
        self, rule: dict[str, Any], context: DecisionContext
    ) -> Decision | None:
        """執行規則動作"""
        action = rule["action"]

        if action == "REQUIRE_CONFIRMATION":
            return Decision(
                action="SWITCH_MODE",
                params={
                    "mode": OperationMode.UI.value,
                    "message": "需要用戶確認",
                },  # 統一使用小寫值
                confidence=0.95,
            )

        elif action == "EXPLOIT_SQL_INJECTION":
            best_tool = self._select_best_tool("sql_injection", context.available_tools)
            return Decision(
                action="RUN_TOOL",
                params={"tool": best_tool, "target_vuln": "sql_injection"},
                confidence=0.8,
            )

        elif action == "CHANGE_STRATEGY":
            new_strategy = self._suggest_alternative_strategy(context)
            return Decision(
                action="CHANGE_APPROACH",
                params={"new_strategy": new_strategy},
                confidence=0.7,
            )

        elif action == "WEB_ATTACK":
            return Decision(
                action="RUN_TOOL",
                params={"tool": "web_scanner", "scan_type": "comprehensive"},
                confidence=0.75,
            )

        elif action == "SSH_BRUTE_FORCE":
            return Decision(
                action="RUN_TOOL",
                params={"tool": "hydra", "service": "ssh", "method": "brute_force"},
                confidence=0.6,
            )

        return None

    def _select_best_tool(self, attack_type: str, available_tools: list[str]) -> str:
        """選擇最佳工具"""
        preferred_tools = self.tool_preferences.get(attack_type, [])

        # 選擇第一個可用的偏好工具
        for tool in preferred_tools:
            if tool in available_tools:
                return tool

        # 如果沒有偏好工具可用，返回第一個可用工具
        return available_tools[0] if available_tools else "manual_test"

    def _suggest_alternative_strategy(self, context: DecisionContext) -> str:
        """建議替代策略"""
        strategies = [
            "passive_reconnaissance",
            "social_engineering",
            "physical_assessment",
            "wireless_testing",
            "client_side_attack",
        ]

        # 根據失敗次數選擇策略
        strategy_index = min(
            context.attempts_without_success - self.success_threshold,
            len(strategies) - 1,
        )
        return strategies[strategy_index]

    def _make_default_decision(self, context: DecisionContext) -> Decision:
        """預設決策邏輯"""
        # 如果有可用工具，選擇一個執行
        if context.available_tools:
            tool = context.available_tools[0]
            decision = Decision(
                action="RUN_TOOL", params={"tool": tool}, confidence=0.5
            )
            decision.reasoning = "無特定規則匹配，執行預設工具"
            return decision

        # 否則建議進行偵察
        decision = Decision(
            action="RECONNAISSANCE", params={"type": "passive"}, confidence=0.4
        )
        decision.reasoning = "無可用工具，建議進行被動偵察"
        return decision

    def _record_decision(self, context: DecisionContext, decision: Decision):
        """記錄決策歷史"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "context": {
                "risk_level": context.risk_level.value,
                "discovered_vulns": context.discovered_vulns,
                "attempts_without_success": context.attempts_without_success,
                "available_tools": context.available_tools,
            },
            "decision": {
                "action": decision.action,
                "params": decision.params,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
        }

        self.decision_history.append(record)

        # 限制歷史記錄大小
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]

    def get_decision_stats(self) -> dict[str, Any]:
        """獲取決策統計"""
        if not self.decision_history:
            return {"total_decisions": 0}

        # 統計決策類型
        action_counts = {}
        confidence_sum = 0

        for record in self.decision_history:
            action = record["decision"]["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
            confidence_sum += record["decision"]["confidence"]

        avg_confidence = confidence_sum / len(self.decision_history)

        return {
            "total_decisions": len(self.decision_history),
            "decision_types": action_counts,
            "average_confidence": f"{avg_confidence:.2f}",
            "most_common_decision": (
                max(action_counts, key=action_counts.get) if action_counts else "無"
            ),
            "recent_decisions": len(
                [
                    r
                    for r in self.decision_history
                    if datetime.fromisoformat(r["timestamp"])
                    > datetime.now() - timedelta(hours=1)
                ]
            ),
        }

    def export_decision_analysis(self, output_path: str = None) -> str:
        """匯出決策分析報告"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"decision_analysis_{timestamp}.json"

        analysis = {
            "agent_info": {
                "name": "Enhanced Decision Agent",
                "version": "1.0",
                "risk_threshold": self.risk_threshold,
                "success_threshold": self.success_threshold,
            },
            "statistics": self.get_decision_stats(),
            "decision_rules": [
                {
                    "name": rule["name"],
                    "description": rule["description"],
                    "priority": rule["priority"],
                }
                for rule in self.decision_rules
            ],
            "tool_preferences": self.tool_preferences,
            "decision_history": self.decision_history[-100:],  # 最近 100 個決策
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📊 決策分析報告已輸出: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"報告輸出失敗: {e}")
            return ""


# 使用範例和測試
def demo_enhanced_decision_agent():
    """示範增強決策代理功能"""
    print("🧠 AIVA 增強決策代理示範")
    print("=" * 50)

    # 創建決策代理
    agent = EnhancedDecisionAgent()

    # 測試場景 1: 高風險操作
    print("\n🔴 場景 1: 高風險操作")
    context1 = DecisionContext()
    context1.risk_level = RiskLevel.HIGH
    context1.available_tools = ["sqlmap", "nikto", "hydra"]

    decision1 = agent.make_decision(context1)
    print(f"決策: {decision1.action}")
    print(f"參數: {decision1.params}")
    print(f"信心度: {decision1.confidence:.2f}")
    print(f"理由: {decision1.reasoning}")

    # 測試場景 2: 發現 SQL 注入
    print("\n🎯 場景 2: 發現 SQL 注入漏洞")
    context2 = DecisionContext()
    context2.risk_level = RiskLevel.MEDIUM
    context2.discovered_vulns = ["sql_injection", "xss"]
    context2.available_tools = ["sqlmap", "xsser", "nikto"]

    decision2 = agent.make_decision(context2)
    print(f"決策: {decision2.action}")
    print(f"參數: {decision2.params}")
    print(f"信心度: {decision2.confidence:.2f}")
    print(f"理由: {decision2.reasoning}")

    # 測試場景 3: 多次失敗
    print("\n⚠️  場景 3: 多次攻擊失敗")
    context3 = DecisionContext()
    context3.risk_level = RiskLevel.LOW
    context3.attempts_without_success = 5
    context3.available_tools = ["nmap", "dirb", "hydra"]

    decision3 = agent.make_decision(context3)
    print(f"決策: {decision3.action}")
    print(f"參數: {decision3.params}")
    print(f"信心度: {decision3.confidence:.2f}")
    print(f"理由: {decision3.reasoning}")

    # 顯示統計
    stats = agent.get_decision_stats()
    print("\n📈 決策統計:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 匯出分析報告
    report_path = agent.export_decision_analysis()
    if report_path:
        print(f"\n📄 決策分析報告: {report_path}")


if __name__ == "__main__":
    demo_enhanced_decision_agent()
