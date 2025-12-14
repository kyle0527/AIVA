#!/usr/bin/env python3
"""AIVA 決策代理增強模組
用途: 整合風險評估和經驗驅動決策，提升 AI 決策的智能化水平
基於: BioNeuron_模型_AI核心大腦.md 中的決策代理分析

Compliance Note:
- 修正日期: 2025-10-25
- 修正項目: 移除重複定義的 RiskLevel，改用 aiva_common.enums.RiskLevel
- 符合架構原則: 使用 aiva_common 統一枚舉定義

Architecture Fix Note:
- 修復日期: 2025-11-16
- 修復項目: 問題三「決策交接不明確」
- 新增: decide() 方法返回 HighLevelIntent (cognitive_core → task_planning 數據合約)
- 向後兼容: 保留 make_decision() 方法
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

# 使用 aiva_common 的決策數據合約 (問題三修復)
from services.aiva_common.schemas import (
    HighLevelIntent,
    IntentType,
    TargetInfo,
    DecisionConstraints,
)

# Operation mode as string literal (bio_neuron_master.py 已移除)
from typing import Literal
OperationMode = Literal["ui", "ai", "chat"]


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
        self, action: str, params: dict[str, Any] | None = None, confidence: float = 0.5
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

    def decide(self, context: DecisionContext) -> HighLevelIntent:
        """做出高階決策 - 返回 HighLevelIntent (問題三修復)
        
        這是 cognitive_core → task_planning 的標準接口
        
        職責劃分：
        - cognitive_core (此方法): 決定「做什麼」(What) 和「為什麼」(Why)
        - task_planning: 決定「怎麼做」(How) - 生成具體的 AST
        
        Args:
            context: 決策上下文
            
        Returns:
            HighLevelIntent: 高階意圖 (包含目標、參數、約束、推理等)
        """
        self.logger.info(f"🤔 開始高階決策分析 - 風險等級: {context.risk_level.value}")
        
        # 使用現有的決策邏輯
        legacy_decision = self.make_decision(context)
        
        # 將 Legacy Decision 轉換為 HighLevelIntent
        intent = self._convert_decision_to_intent(legacy_decision, context)
        
        self.logger.info(f"✅ 生成高階意圖: {intent.intent_type.value} (信心度: {intent.confidence:.2f})")
        
        return intent
    
    def _convert_decision_to_intent(
        self, decision: Decision, context: DecisionContext
    ) -> HighLevelIntent:
        """將 Legacy Decision 轉換為 HighLevelIntent
        
        這是過渡期的轉換方法，未來可以直接生成 HighLevelIntent
        """
        # 映射 action 到 IntentType
        action_to_intent_type = {
            "EXPLOIT_SQL_INJECTION": IntentType.TEST_VULNERABILITY,
            "WEB_ATTACK": IntentType.SCAN_SURFACE,
            "SSH_BRUTE_FORCE": IntentType.EXPLOIT_TARGET,
            "CHANGE_STRATEGY": IntentType.ANALYZE_RESULTS,
            "STOP_OPERATION": IntentType.ANALYZE_RESULTS,
        }
        
        intent_type = action_to_intent_type.get(
            decision.action, IntentType.SCAN_SURFACE
        )
        
        # 構建目標信息
        target = TargetInfo(
            target_id=str(context.target_info.get("id", "unknown")),
            target_type=context.target_info.get("type", "url"),
            target_value=context.target_info.get("value", "unknown"),
            context=context.target_info,
        )
        
        # 構建約束條件
        constraints = DecisionConstraints(
            time_limit=context.time_constraints,
            risk_level=context.risk_level.value.lower(),
            stealth_mode=False,
            resource_limits={},
            forbidden_actions=[],
        )
        
        # 構建高階意圖
        intent = HighLevelIntent(
            intent_type=intent_type,
            target=target,
            parameters=decision.params,
            constraints=constraints,
            confidence=decision.confidence,
            reasoning=decision.reasoning or f"決策行動: {decision.action}",
            alternatives=[
                {"action": alt.get("action"), "confidence": alt.get("confidence")}
                for alt in (decision.alternatives or [])
            ],
            context={
                "discovered_vulns": context.discovered_vulns,
                "attempts_without_success": context.attempts_without_success,
                "previous_results": context.previous_results[:5],  # 只保留最近 5 個
            },
        )
        
        return intent

    def make_decision(self, context: DecisionContext) -> Decision:
        """基於上下文做出智能決策 (Legacy 方法，保持向後兼容)
        
        注意: 新代碼應使用 decide() 方法返回 HighLevelIntent

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
        # 實際查詢 experience_manager (移除硬編碼的 mock_experiences)
        if not self.experience_manager:
            # 沒有經驗管理器時，返回空列表而非假數據
            self.logger.debug("Experience manager not available, no historical data")
            return []
        
        try:
            # 查詢條件構建
            query_params = {
                "target_type": context.target_info.get("type"),
                "vulnerabilities": context.discovered_vulns,
                "risk_level": context.risk_level.value,
                "min_success_score": 0.6,
            }
            
            # 實際查詢經驗管理器
            experiences = self.experience_manager.query_similar_experiences(query_params)
            
            # 計算相似度並排序
            similar = []
            for exp in experiences:
                similarity = self._calculate_similarity(context, exp)
                if similarity > 0.6:
                    exp["similarity"] = similarity
                    similar.append(exp)
            
            return sorted(similar, key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to query experience manager: {e}")
            return []

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

    async def execute_decision(self, decision: Decision, context: DecisionContext) -> dict[str, Any]:
        """執行 AI 決策（實際調用模組）
        
        這是 AI 決策 → 實際執行的橋梁
        
        Args:
            decision: AI 決策結果
            context: 決策上下文
            
        Returns:
            執行結果
        """
        self.logger.info(f"🚀 執行 AI 決策: {decision.action}")
        
        try:
            # 根據決策動作執行對應操作
            if decision.action == "RUN_TOOL":
                return await self._execute_tool_decision(decision, context)
            
            elif decision.action in ["EXPLOIT_SQL_INJECTION", "WEB_ATTACK"]:
                return await self._execute_vulnerability_test(decision, context)
            
            elif decision.action == "SWITCH_MODE":
                return self._execute_mode_switch(decision, context)
            
            elif decision.action == "CHANGE_APPROACH":
                return self._execute_strategy_change(decision, context)
            
            elif decision.action == "STOP_OPERATION":
                return self._execute_stop(decision, context)
            
            else:
                self.logger.warning(f"⚠️ 未知決策動作: {decision.action}")
                return {
                    "success": False,
                    "error": f"Unknown decision action: {decision.action}"
                }
                
        except Exception as e:
            self.logger.error(f"❌ 決策執行失敗: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "decision": decision.action
            }
    
    async def _execute_tool_decision(self, decision: Decision, context: DecisionContext) -> dict[str, Any]:
        """執行工具相關決策"""
        tool = decision.params.get("tool")
        target_vuln = decision.params.get("target_vuln")
        
        self.logger.info(f"   🔧 使用工具: {tool}, 目標漏洞: {target_vuln}")
        
        # 直接使用 AICommandCenter 下達命令
        try:
            from services.aiva_common.command_center import get_command_center
            from services.aiva_common.schemas import AICommand, CommandType
            import uuid
            
            command_center = get_command_center()
            target = context.target_info.get("value", "http://localhost:3000")
            
            # 生成唯一的 scan_id
            scan_id = f"scan_{uuid.uuid4().hex[:12]}"
            command_id = f"{scan_id}_phase0"
            
            # 構建符合 Phase0StartPayload schema 的命令
            command = AICommand(
                command_id=command_id,
                command_type=CommandType.SCAN_PHASE0,
                target_module="scan",
                payload={
                    "scan_id": scan_id,
                    "targets": [target],
                    "max_depth": 3,
                    "timeout": 600,
                    # 可選參數
                    "custom_headers": {},
                }
            )
            
            # 執行命令
            result = await command_center.execute(command)
            
            return {
                "success": result.success,
                "data": result.result,  # 修正：使用 result 而非 result_data
                "error": result.error
            }
            
        except Exception as e:
            self.logger.error(f"❌ Command execution failed: {e}")
            # 返回實際失敗狀態，不再偽裝成功
            return {
                "success": False,
                "error": str(e),
                "tool": tool,
                "message": "Command execution failed - CommandCenter not available or error occurred",
                "requires_user_action": True
            }
    
    async def _execute_vulnerability_test(self, decision: Decision, context: DecisionContext) -> dict[str, Any]:
        """執行漏洞測試"""
        target = context.target_info.get("value", "http://localhost:3000")
        
        self.logger.info(f"   🎯 對目標 {target} 執行漏洞測試")
        
        try:
            from services.aiva_common.command_center import get_command_center
            from services.aiva_common.schemas import AICommand, CommandType
            import uuid
            
            command_center = get_command_center()
            
            # 生成唯一的 scan_id
            scan_id = f"scan_{uuid.uuid4().hex[:12]}"
            command_id = f"{scan_id}_phase0"
            
            # 決定掃描深度（SQLi 需要更深入）
            max_depth = 5 if decision.action == "EXPLOIT_SQL_INJECTION" else 3
            
            # 構建符合 Phase0StartPayload schema 的命令
            command = AICommand(
                command_id=command_id,
                command_type=CommandType.SCAN_PHASE0,
                target_module="scan",
                payload={
                    "scan_id": scan_id,
                    "targets": [target],
                    "max_depth": max_depth,
                    "timeout": 600,
                    "custom_headers": {},
                }
            )
            
            # 執行命令
            result = await command_center.execute(command)
            
            return {
                "success": result.success,
                "data": result.result,  # 修正：使用 result 而非 result_data
                "error": result.error
            }
            
        except Exception as e:
            self.logger.error(f"❌ Test execution failed: {e}")
            # 返回實際失敗狀態，不再偽裝成功
            return {
                "success": False,
                "error": str(e),
                "message": "Vulnerability test execution failed",
                "requires_user_action": True
            }
    
    def _execute_mode_switch(self, decision: Decision, context: DecisionContext) -> dict[str, Any]:
        """執行模式切換"""
        new_mode = decision.params.get("mode")
        message = decision.params.get("message", "Mode switch")
        
        self.logger.info(f"   🔄 切換模式到: {new_mode}")
        
        return {
            "success": True,
            "action": "mode_switch",
            "new_mode": new_mode,
            "message": message,
            "requires_user_action": True,
        }
    
    def _execute_strategy_change(self, decision: Decision, context: DecisionContext) -> dict[str, Any]:
        """執行策略變更"""
        new_strategy = decision.params.get("new_strategy")
        
        self.logger.info(f"   🔄 變更策略到: {new_strategy}")
        
        return {
            "success": True,
            "action": "strategy_change",
            "new_strategy": new_strategy,
            "reasoning": decision.reasoning,
        }
    
    def _execute_stop(self, decision: Decision, context: DecisionContext) -> dict[str, Any]:
        """執行停止操作"""
        reason = decision.params.get("reason", "Safety measure")
        
        self.logger.warning(f"   ⛔ 停止操作: {reason}")
        
        return {
            "success": True,
            "action": "stop",
            "reason": reason,
            "requires_user_action": True,
        }

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
                max(action_counts, key=lambda k: action_counts[k]) if action_counts else "無"
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

    def export_decision_analysis(self, output_path: str | None = None) -> str:
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
