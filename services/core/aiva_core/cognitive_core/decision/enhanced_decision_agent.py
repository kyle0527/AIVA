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
from typing import Any, Optional
import asyncio

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

# [新增] 引入真實神經網路引擎
from ..neural.real_neural_core import RealDecisionEngine

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

        # 設定日誌
        self.logger = self._setup_logger()

        # [新增] 初始化真實 AI 引擎
        # 不使用降級方案，如果載入失敗就讓錯誤暴露
        # 指向權重檔案的正確路徑
        weights_path = Path(__file__).parent.parent / "neural" / "weights" / "aiva_real_weights.pth"
        self.neural_engine = RealDecisionEngine(
            use_5m_model=True,
            weights_path=str(weights_path)
        )
        self.use_neural_decision = True
        self.logger.info("🧠 Real Neural Core (5M) 整合成功")

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

        self.logger.info("🛡️ 規則引擎已就緒")

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
        
        # 使用現有的決策邏輯（支援 async）
        legacy_decision = self._sync_make_decision(context)
        
        # 將 Legacy Decision 轉換為 HighLevelIntent
        intent = self._convert_decision_to_intent(legacy_decision, context)
        
        self.logger.info(f"✅ 生成高階意圖: {intent.intent_type.value} (信心度: {intent.confidence:.2f})")
        
        return intent
    
    def _sync_make_decision(self, context: DecisionContext) -> Decision:
        """同步包裝的 make_decision 方法"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已經在 event loop 中，創建 task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.make_decision(context))
                    return future.result()
            else:
                # 沒有運行的 loop，直接運行
                return loop.run_until_complete(self.make_decision(context))
        except RuntimeError:
            # 沒有 event loop，創建新的
            return asyncio.run(self.make_decision(context))
    
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

    async def make_decision(self, context: DecisionContext) -> Decision:
        """基於多模態評估做出智能決策
        邏輯：神經網路(直覺) + 經驗庫(記憶) + 規則引擎(安全邊界)
        
        注意: 新代碼應使用 decide() 方法返回 HighLevelIntent

        Args:
            context: 決策上下文

        Returns:
            決策結果
        """
        self.logger.info(f"🤔 開始多維度決策分析 - 風險: {context.risk_level.value}")

        # 1. 安全煞車 (規則優先 - 最高優先級)
        # 如果觸發高風險規則，直接攔截，不經過 AI
        risk_decision = self._assess_risk_decision(context)
        if risk_decision and risk_decision.action == "STOP_OPERATION":
            return risk_decision

        # 2. 並行獲取決策建議
        neural_task = self._make_neural_decision(context)
        exp_task = self._async_wrapper(self._make_experience_driven_decision, context)
        rule_task = self._async_wrapper(self._apply_decision_rules, context)

        # 等待所有決策模組返回
        neural_result, exp_result, rule_result = await asyncio.gather(
            neural_task, exp_task, rule_task
        )

        # 3. 集成學習決策 (Ensemble Learning)
        # 使用加權算法融合三方意見
        final_decision = self._ensemble_decision(
            neural=neural_result,
            experience=exp_result,
            rule=rule_result,
            context=context
        )

        # 4. 記錄並返回
        self._record_decision(context, final_decision)
        return final_decision

    async def _make_neural_decision(self, context: DecisionContext) -> Optional[Decision]:
        """[新增] 基於 5M 神經網路的真實 AI 決策"""
        # neural_engine 必須存在，不使用降級方案
        try:
            # A. 狀態序列化：將 Context 轉為 AI 可讀的 Prompt
            state_description = (
                f"TargetType: {context.target_info.get('type', 'unknown')} | "
                f"VulnsFound: {','.join(context.discovered_vulns) or 'None'} | "
                f"RiskLevel: {context.risk_level.value} | "
                f"FailCount: {context.attempts_without_success} | "
                f"AvailableTools: {','.join(context.available_tools[:3])}"
            )

            # B. 神經網路推論 (Forward Pass)
            # 使用 real_neural_core 的 generate_decision 方法
            # 注意：這裡使用 run_in_executor 避免阻塞 Event Loop
            loop = asyncio.get_event_loop()
            ai_result = await loop.run_in_executor(
                None, 
                lambda: self.neural_engine.generate_decision(
                    task_description="determine_optimal_action",
                    context=state_description
                )
            )

            # C. 解析輸出張量
            confidence = ai_result.get("confidence", 0.0)
            attack_vector = ai_result.get("attack_vector")
            
            # 過濾低信心度結果
            if confidence < 0.55:
                return None

            # D. 動作映射 (Vector -> Action)
            # 將 AI 的抽象意圖映射為系統可執行的具體指令
            action_map: dict[str, str] = {
                "sql_injection": "EXPLOIT_SQL_INJECTION",
                "cross_site_scripting": "WEB_ATTACK",
                "server_side_request_forgery": "RUN_TOOL",
                "reconnaissance": "RUN_TOOL",
                "file_upload": "WEB_ATTACK"
            }
            
            mapped_action = action_map.get(attack_vector or "", "RUN_TOOL")
            
            # 參數綁定
            tools = ai_result.get("recommended_tools", [])
            selected_tool = tools[0] if tools else "manual"

            decision = Decision(
                action=mapped_action,
                params={
                    "tool": selected_tool, 
                    "target_vuln": attack_vector,
                    "source": "neural_network_5m"
                },
                confidence=confidence
            )
            decision.reasoning = f"NeuralNet Suggestion: {ai_result.get('reasoning')}"
            return decision

        except Exception as e:
            self.logger.error(f"❌ 神經網路推論錯誤: {e}")
            return None

    def _ensemble_decision(self, neural: Optional[Decision], experience: Optional[Decision], rule: Optional[Decision], context: DecisionContext) -> Decision:
        """[新增] 加權決策融合算法"""
        candidates: list[tuple[Decision, float]] = []

        # 權重配置
        W_NEURAL = 0.5      # AI 直覺佔 50%
        W_EXPERIENCE = 0.3  # 歷史經驗佔 30%
        W_RULE = 0.2        # 硬性規則佔 20%

        # 1. 評分計算（使用元組存儲決策和分數）
        if neural:
            score = neural.confidence * W_NEURAL
            # AI 的建議如果有經驗支持，給予額外加成
            if experience and neural.action == experience.action:
                score += 0.1
            candidates.append((neural, score))
            
        if experience:
            score = experience.confidence * W_EXPERIENCE
            candidates.append((experience, score))
            
        if rule:
            score = rule.confidence * W_RULE
            # 規則通常是兜底，分數較低，但如果是高優先級規則則例外
            if rule.action == "REQUIRE_CONFIRMATION":
                score += 0.5 
            candidates.append((rule, score))

        # 2. 決策選擇
        if not candidates:
            self.logger.info("無有效決策，使用預設策略")
            return self._make_default_decision(context)

        # 選出分數最高的決策
        best_decision = max(candidates, key=lambda x: x[1])[0]
        
        self.logger.info(
            f"✅ 最終決策: {best_decision.action} "
            f"(來源: {best_decision.params.get('source', 'rule/exp')}, "
            f"加權分數: {getattr(best_decision, 'score', 0):.2f})"
        )
        
        return best_decision

    async def _async_wrapper(self, func, *args, **kwargs):
        """輔助方法：將同步函數包裝為異步"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

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
                params={"mode": "ui"},  # OperationMode 是 Literal 類型，直接使用字串值
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
                    "mode": "ui",  # OperationMode 是 Literal 類型
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
                },
                # 添加必需參數
                trace_id=scan_id,
                session_id=scan_id,
                parent_command_id=None,
                callback_url=None
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
                },
                # 添加必需參數
                trace_id=scan_id,
                session_id=scan_id,
                parent_command_id=None,
                callback_url=None
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
    
    def decide_scan_strategy(self, scan_context) -> dict[str, Any]:
        """基於 ScanTaskContext 做掃描策略決策
        
        這是新的標準化決策接口，使用統一的參數包。
        
        Args:
            scan_context: ScanTaskContext 或包含相關信息的字典
        
        Returns:
            決策結果 {
                "selected_tool": str,  # nmap/masscan
                "confidence": float,
                "reasoning": str,
                "suggested_params": dict
            }
        """
        # 提取上下文信息
        if hasattr(scan_context, 'constraints'):
            # ScanTaskContext 對象
            stealth_level = scan_context.constraints.stealth_level
            rate_limit = scan_context.constraints.rate_limit
            target = scan_context.target
        else:
            # 字典格式（向後兼容）
            constraints = scan_context.get('constraints', {})
            stealth_level = constraints.get('stealth_level', 'medium')
            rate_limit = constraints.get('rate_limit', 1000)
            target = scan_context.get('target', '')
        
        # 決策邏輯
        selected_tool = "nmap"  # 默認
        confidence = 0.7
        reasoning = "標準掃描配置"
        
        # 規則 1: 隱匿模式優先使用 Nmap（更低調）
        if str(stealth_level).lower() in ['high', 'paranoid']:
            selected_tool = "nmap"
            confidence = 0.9
            reasoning = "檢測到高隱匿需求，推薦 Nmap -sS -T2（低調掃描）"
        
        # 規則 2: 高速掃描優先使用 Masscan
        elif rate_limit > 2000:
            selected_tool = "masscan"
            confidence = 0.85
            reasoning = f"檢測到高速需求（{rate_limit} pps），推薦 Masscan"
        
        # 規則 3: 神經網路增強決策（如果可用）
        if self.use_neural_decision:
            try:
                # 構建簡化的決策上下文
                neural_input = f"Target: {target}, Stealth: {stealth_level}, Speed: {rate_limit}"
                loop = asyncio.new_event_loop()
                ai_result = loop.run_until_complete(
                    loop.run_in_executor(
                        None,
                        lambda: self.neural_engine.generate_decision(
                            task_description="select_scan_tool",
                            context=neural_input
                        )
                    )
                )
                loop.close()
                
                # 如果神經網路信心度更高，採用其建議
                if ai_result.get("confidence", 0) > confidence:
                    selected_tool = ai_result.get("recommended_tool", selected_tool)
                    confidence = ai_result["confidence"]
                    reasoning = f"AI 推薦: {ai_result.get('reasoning', reasoning)}"
            
            except Exception as e:
                self.logger.warning(f"神經網路決策失敗，使用規則引擎: {e}")
        
        # 構建建議參數
        suggested_params = {}
        if selected_tool == "nmap":
            if str(stealth_level).lower() in ['high', 'paranoid']:
                suggested_params = {
                    "scan_type": "-sS",  # SYN 掃描
                    "timing": "-T2",     # 慢速
                    "flags": ["--disable-arp-ping", "-f"]  # 分片
                }
            else:
                suggested_params = {
                    "scan_type": "-sS",
                    "timing": "-T4",
                    "flags": ["-Pn"]
                }
        elif selected_tool == "masscan":
            suggested_params = {
                "rate": min(rate_limit, 10000),  # 限制最大速率
                "wait": 0 if rate_limit > 5000 else 1
            }
        
        self.logger.info(
            f"🎯 掃描策略決策: {selected_tool} (信心度: {confidence:.2f})"
        )
        
        return {
            "selected_tool": selected_tool,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggested_params": suggested_params
        }


# 使用範例和測試
def demo_enhanced_decision_agent():
    """示範增強決策代理功能"""
    print("🧠 AIVA 增強決策代理示範")
    print("=" * 50)

    # 測試代碼已移除 - 使用獨立的測試文件進行測試
    # 如需測試請執行: python -m pytest tests/test_enhanced_decision_agent.py
    print("請使用專用測試套件進行功能測試")


if __name__ == "__main__":
    demo_enhanced_decision_agent()
