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
from typing import Any, Optional
import asyncio

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
    """決策上下文
    
    v2.1 (2026-01-08): 支援去語意化反射引擎
    """

    def __init__(self):
        self.risk_level = RiskLevel.LOW
        self.discovered_vulns = []
        self.attempts_without_success = 0
        self.target_info = {}
        self.previous_results = []
        self.time_constraints = None
        self.available_tools = []
        self.mode_restrictions = []
        # v2.1: 環境特徵（用於去語意化檢索）
        self.environment_features: dict[str, float] | None = None


class Decision:
    """決策結果
    
    v2.1 (2026-01-08): 支援 RAG 檢索建議
    """

    def __init__(
        self, 
        action: str, 
        params: dict[str, Any] | None = None, 
        confidence: float = 0.5,
        rag_suggestions: list[dict[str, Any]] | None = None
    ):
        self.action = action
        self.params = params or {}
        self.confidence = confidence
        self.reasoning = ""
        self.alternatives = []
        self.risk_assessment = None
        # v2.1: RAG 檢索建議
        self.rag_suggestions = rag_suggestions or []


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
        
        # 初始化 RAG 引擎（去語意化反射引擎整合）
        if knowledge_base is not None:
            from ..rag.rag_engine import RAGEngine
            self.rag_engine = RAGEngine(knowledge_base)
            self.logger.info("🔍 RAG Engine 已整合（支援去語意化檢索）")
        else:
            self.rag_engine = None

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
        
        # Bug Bounty 特化配置
        self.waf_bypass_strategies = self._initialize_waf_strategies()
        self.rate_limit_profiles = self._initialize_rate_profiles()
        self.bounty_value_matrix = self._initialize_bounty_matrix()

        self.logger.info("🛡️ 規則引擎已就緒")
        self.logger.info("🎯 Bug Bounty 模組已載入")

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
        邏輯：神經網路(直覺) + 經驗庫(記憶) + 規則引擎(安全邊界) + RAG檢索(去語意化)
        
        v2.1 (2026-01-08): 整合 HackOne 去語意化反射引擎
        
        注意: 新代碼應使用 decide() 方法返回 HighLevelIntent

        Args:
            context: 決策上下文

        Returns:
            決策結果
        """
        self.logger.info(f"🤔 開始多維度決策分析 - 風險: {context.risk_level.value}")

        # 0. 去語意化 RAG 檢索（HackOne v2.0）
        # 如果提供了環境特徵，使用去語意化檢索匹配能力
        rag_suggestions = None
        if hasattr(context, 'environment_features') and context.environment_features and self.rag_engine:
            try:
                # 快速檢索（< 5ms）
                rag_suggestions = await self.rag_engine.search_capabilities_by_environment(
                    environment_features=context.environment_features,
                    top_k=3
                )
                self.logger.info(f"🔍 RAG 檢索建議: {len(rag_suggestions)} 個能力")
            except Exception as e:
                self.logger.warning(f"RAG 檢索失敗: {e}")
        
        # 1. 安全煞車 (規則優先 - 最高優先級)
        # 如果觸發高風險規則，直接攔截，不經過 AI
        risk_decision = self._assess_risk_decision(context)
        if risk_decision and risk_decision.action == "STOP_OPERATION":
            return risk_decision

        # 2. 並行獲取決策建議
        neural_task = self._make_neural_decision(context, rag_suggestions)
        exp_task = self._async_wrapper(self._make_experience_driven_decision, context)
        rule_task = self._async_wrapper(self._apply_decision_rules, context)

        # 等待所有決策模組返回
        neural_result, exp_result, rule_result = await asyncio.gather(
            neural_task, exp_task, rule_task
        )

        # 3. 集成學習決策 (Ensemble Learning)
        # 使用加權算法融合三方意見（包含 RAG 建議）
        final_decision = self._ensemble_decision(
            neural=neural_result,
            experience=exp_result,
            rule=rule_result,
            context=context,
            rag_suggestions=rag_suggestions
        )

        # 4. 記錄並返回
        self._record_decision(context, final_decision)
        return final_decision

    async def _make_neural_decision(
        self, 
        context: DecisionContext,
        rag_suggestions: list[dict[str, Any]] | None = None
    ) -> Optional[Decision]:
        """[新增] 基於 5M 神經網路的真實 AI 決策
        
        v2.1 (2026-01-08): 整合 RAG 建議到神經決策
        """
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
            
            # 整合 RAG 建議（去語意化特徵匹配結果）
            if rag_suggestions:
                top_capability = rag_suggestions[0]
                state_description += (
                    f" | RAG_Suggestion: {top_capability['capability_id']} "
                    f"(score: {top_capability['match_score']:.2f})"
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

    def _ensemble_decision(
        self, 
        neural: Optional[Decision], 
        experience: Optional[Decision], 
        rule: Optional[Decision], 
        context: DecisionContext,
        rag_suggestions: list[dict[str, Any]] | None = None
    ) -> Decision:
        """加權決策融合算法
        
        v2.1 (2026-01-08): 整合 RAG 建議到決策過程
        """
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
            # v2.1: RAG 建議加成
            if rag_suggestions and len(rag_suggestions) > 0:
                score += 0.05  # RAG 匹配給予小加成
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
        """智能掃描策略決策 (增強版)
        
        基於目標特徵、技術棧、安全設備等多因素智能選擇最佳掃描策略。
        整合了 Bug Bounty 經驗和神經網路決策。
        
        Args:
            scan_context: ScanTaskContext 或包含相關信息的字典
        
        Returns:
            決策結果 {
                "selected_tool": str,     # 主要掃描工具
                "confidence": float,      # 決策信心度
                "reasoning": str,         # 決策理由
                "suggested_params": dict, # 建議參數
                "scan_strategy": str,     # 掃描策略 
                "estimated_time": int     # 預計耗時(秒)
            }
        """
        # 提取上下文信息
        if hasattr(scan_context, 'constraints'):
            stealth_level = scan_context.constraints.stealth_level
            rate_limit = scan_context.constraints.rate_limit
            target = scan_context.target
            intent = getattr(scan_context, 'intent', 'reconnaissance')
        else:
            constraints = scan_context.get('constraints', {})
            stealth_level = constraints.get('stealth_level', 'medium')
            rate_limit = constraints.get('rate_limit', 1000)
            target = scan_context.get('target', '')
            intent = scan_context.get('intent', 'reconnaissance')
        
        # 目標分析
        target_analysis = self._analyze_scan_target(target)
        is_web_app = target_analysis['is_web_app']
        has_waf = target_analysis['has_waf']
        tech_stack = target_analysis['tech_stack']
        
        # 智能決策邏輯
        selected_tool = "nmap"  # 默認
        confidence = 0.7
        reasoning_parts = []
        scan_strategy = "standard"
        
        # === 規則 1: 基於目標類型選擇工具 ===
        if is_web_app:
            # Web 應用偏向使用更精細的掃描
            if 'api' in target.lower():
                selected_tool = "nmap"  # API 需要精確的端口識別
                scan_strategy = "api_focused"
                reasoning_parts.append("API 端點檢測")
            elif any(tech in tech_stack for tech in ['nodejs', 'react', 'angular']):
                selected_tool = "nmap"  # SPA 應用特殊處理
                scan_strategy = "spa_optimized"
                reasoning_parts.append("SPA 應用優化")
            else:
                selected_tool = "nmap"
                scan_strategy = "web_comprehensive"
                reasoning_parts.append("Web 應用全面掃描")
        
        # === 規則 2: 基於隱匿需求調整 ===
        stealth = str(stealth_level).lower()
        if stealth in ['high', 'paranoid']:
            selected_tool = "nmap"  # Nmap 更適合隱匿掃描
            scan_strategy = "stealth"
            confidence = min(0.95, confidence + 0.2)
            reasoning_parts.append(f"高隱匿需求({stealth})")
        elif stealth == 'low' and rate_limit > 3000:
            selected_tool = "masscan"  # 高速掃描
            scan_strategy = "fast_discovery"
            confidence = min(0.9, confidence + 0.15)
            reasoning_parts.append(f"高速發現模式({rate_limit} pps)")
        
        # === 規則 3: WAF 檢測調整策略 ===
        if has_waf:
            if selected_tool == "masscan":
                selected_tool = "nmap"  # WAF 環境下降級到 Nmap
                scan_strategy = "waf_evasion"
                reasoning_parts.append("WAF 檢測，切換規避模式")
            else:
                scan_strategy = "waf_evasion"
                reasoning_parts.append("WAF 環境，啟用規避技術")
        
        # === 規則 4: 神經網路增強決策 ===
        if self.use_neural_decision:
            try:
                neural_context = (
                    f"Target: {target[:50]} | "
                    f"Type: {'WebApp' if is_web_app else 'Infrastructure'} | "
                    f"WAF: {'Yes' if has_waf else 'No'} | "
                    f"Stealth: {stealth} | Rate: {rate_limit} | "
                    f"Tech: {','.join(tech_stack[:3]) if tech_stack else 'unknown'}"
                )
                
                ai_result = self.neural_engine.generate_decision(
                    task_description="intelligent_scan_strategy",
                    context=neural_context
                )
                
                ai_confidence = ai_result.get("confidence", 0)
                ai_tool = ai_result.get("recommended_tool", "")
                
                # AI 信心度高於當前規則時採用 AI 建議
                if ai_confidence > confidence + 0.1:
                    if ai_tool.lower() in ['nmap', 'masscan']:
                        selected_tool = ai_tool.lower()
                        confidence = ai_confidence
                        reasoning_parts.append(f"AI 推薦: {ai_result.get('reasoning', 'Neural decision')}")
                
            except Exception as e:
                self.logger.debug(f"神經網路決策失敗: {e}")
        
        # === 構建掃描參數 ===
        suggested_params = self._build_scan_params(
            selected_tool, scan_strategy, stealth_level, rate_limit, has_waf
        )
        
        # === 預估掃描時間 ===
        estimated_time = self._estimate_scan_time(
            selected_tool, scan_strategy, target_analysis
        )
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "標準掃描策略"
        
        self.logger.info(
            f"🎯 智能掃描策略: {selected_tool}({scan_strategy}) 信心度:{confidence:.2f} 預計:{estimated_time//60}分鐘"
        )
        
        return {
            "selected_tool": selected_tool,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggested_params": suggested_params,
            "scan_strategy": scan_strategy,
            "estimated_time": estimated_time
        }
    
    def _analyze_scan_target(self, target: str) -> dict[str, Any]:
        """分析掃描目標特徵"""
        target_lower = target.lower()
        
        # 判斷是否為 Web 應用
        web_indicators = ['http', 'www', 'api', 'app', 'web', 'portal']
        is_web_app = any(indicator in target_lower for indicator in web_indicators)
        
        # 簡單 WAF 檢測 (基於域名)
        waf_indicators = ['cloudflare', 'akamai', 'incapsula', 'aws']
        has_waf = any(indicator in target_lower for indicator in waf_indicators)
        
        # 技術棧推測
        tech_indicators = {
            'nodejs': ['node', 'npm', 'express'],
            'php': ['php', 'wordpress', 'laravel'],
            'python': ['django', 'flask', 'fastapi'],
            'java': ['spring', 'tomcat', 'struts'],
            'react': ['react', 'redux'],
            'angular': ['angular', 'ng']
        }
        
        tech_stack = []
        for tech, indicators in tech_indicators.items():
            if any(ind in target_lower for ind in indicators):
                tech_stack.append(tech)
        
        return {
            'is_web_app': is_web_app,
            'has_waf': has_waf,
            'tech_stack': tech_stack,
            'target_type': 'web' if is_web_app else 'infrastructure'
        }
    
    def _build_scan_params(self, tool: str, strategy: str, stealth: str, rate: int, has_waf: bool) -> dict:
        """構建掃描參數"""
        params = {}
        
        if tool == "nmap":
            if strategy == "stealth" or has_waf:
                params = {
                    "scan_type": "-sS",
                    "timing": "-T2",
                    "flags": ["--disable-arp-ping", "-f", "--source-port", "53"]
                }
            elif strategy == "api_focused":
                params = {
                    "scan_type": "-sV",
                    "timing": "-T4",
                    "flags": ["--script", "http-enum,http-headers"],
                    "ports": "80,443,8080,8443,3000,8000"
                }
            elif strategy == "web_comprehensive":
                params = {
                    "scan_type": "-sV -sC",
                    "timing": "-T4",
                    "flags": ["--script", "vuln"],
                    "ports": "1-65535"
                }
            else:  # standard
                params = {
                    "scan_type": "-sS",
                    "timing": "-T4",
                    "flags": ["-Pn"]
                }
        
        elif tool == "masscan":
            params = {
                "rate": min(rate, 10000 if not has_waf else 1000),
                "wait": 0 if rate > 5000 else 1,
                "ports": "1-65535" if strategy != "fast_discovery" else "80,443,22,21,25,53,110,143,993,995"
            }
        
        return params
    
    def _estimate_scan_time(self, tool: str, strategy: str, target_analysis: dict) -> int:
        """預估掃描時間 (秒)"""
        base_time = 300  # 5分鐘基礎時間
        
        if tool == "masscan":
            if strategy == "fast_discovery":
                return 60  # 1分鐘快速發現
            else:
                return 180  # 3分鐘全端口
        
        elif tool == "nmap":
            if strategy == "stealth":
                return 1800  # 30分鐘隱匿掃描
            elif strategy == "web_comprehensive":
                return 900   # 15分鐘全面掃描
            elif strategy == "api_focused":
                return 300   # 5分鐘 API 掃描
            else:
                return base_time
        
        return base_time
    
    # ═══════════════════════════════════════════════════════════════
    # Bug Bounty 特化決策方法 (對應 13 步驟外閉環)
    # ═══════════════════════════════════════════════════════════════
    
    def decide_phase1_strategy(
        self,
        phase0_result: dict[str, Any],
        target_value: float = 1000.0,
        program_scope: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """步驟 6: AI 決策是否需要 Phase1 深度掃描 (HackerOne/Bugcrowd 實戰優化版)
        
        實際 Bug Bounty 場景考量：
        1. Program Scope 限制（必須遵守範圍）
        2. Rate Limiting 和 WAF 檢測（避免被封禁）
        3. 高價值漏洞類型優先（SSRF > SQLi > XSS）
        4. 時間投資回報率（專業 Hunter 的核心指標）
        5. 歷史數據：同類型目標成功率
        
        OWASP WSTG 對應：
        - 4.1 Information Gathering → Phase0
        - 4.2 Configuration Testing → Phase1 起點
        - 4.7 Input Validation Testing → Phase1 核心
        
        Args:
            phase0_result: Phase0 偵察結果
                - summary: {urls_found, forms_found, apis_found, subdomains_found}
                - fingerprints: {technologies, waf_detected, waf_vendor, server_type}
                - endpoints: [{url, method, params, response_info}]
                - recommendations: {priority_targets, needs_deep_scan}
            target_value: 預期獎金價值 (美元)，基於 Program 歷史支付
            program_scope: Program 範圍限制
                - in_scope: [domain patterns]
                - out_of_scope: [excluded patterns]
                - vulnerability_types: [accepted vuln types]
                - testing_restrictions: {rate_limit, no_automated_tools, etc}
            
        Returns:
            決策結果字典
        """
        self.logger.info("🧠 [Step 6: Phase1 Strategy] 開始 AI 決策...")
        
        # === 1. 提取 Phase0 情報 ===
        summary = phase0_result.get("summary", {})
        fingerprints = phase0_result.get("fingerprints", {})
        recommendations = phase0_result.get("recommendations", {})
        endpoints = phase0_result.get("endpoints", [])
        
        urls_found = summary.get("urls_found", 0)
        forms_found = summary.get("forms_found", 0)
        apis_found = summary.get("apis_found", 0)
        subdomains_found = summary.get("subdomains_found", 0)
        waf_detected = fingerprints.get("waf_detected", False)
        waf_vendor = fingerprints.get("waf_vendor")
        technologies = fingerprints.get("technologies", [])
        
        # === 2. Program Scope 合規檢查 ===
        scope_violations = []
        testing_restrictions = {}
        if program_scope:
            testing_restrictions = program_scope.get("testing_restrictions", {})
            # 檢查是否有禁止自動化測試
            if testing_restrictions.get("no_automated_tools"):
                self.logger.warning("   ⚠️ Program 禁止自動化工具，切換為手動輔助模式")
        
        # === 3. 高價值目標識別 (OWASP Top 10 + Bug Bounty 熱門) ===
        high_value_indicators = {
            "api_endpoints": apis_found,  # API 通常有 IDOR/Auth 問題
            "file_upload_forms": sum(1 for e in endpoints if "upload" in e.get("url", "").lower()),
            "auth_endpoints": sum(1 for e in endpoints if any(k in e.get("url", "").lower() for k in ["login", "auth", "oauth", "token", "session"])),
            "admin_panels": sum(1 for e in endpoints if any(k in e.get("url", "").lower() for k in ["admin", "dashboard", "manage", "config"])),
            "payment_flows": sum(1 for e in endpoints if any(k in e.get("url", "").lower() for k in ["payment", "checkout", "billing", "subscription"])),
            "user_input_heavy": forms_found,
            "graphql_endpoints": sum(1 for e in endpoints if "graphql" in e.get("url", "").lower()),
        }
        
        # 計算高價值評分 (0-1)
        high_value_score = min(1.0, (
            high_value_indicators["api_endpoints"] * 0.15 +
            high_value_indicators["file_upload_forms"] * 0.2 +  # 文件上傳是高價值目標
            high_value_indicators["auth_endpoints"] * 0.15 +
            high_value_indicators["admin_panels"] * 0.2 +  # Admin 面板通常獎金高
            high_value_indicators["payment_flows"] * 0.25 +  # 支付相關最高價值
            high_value_indicators["graphql_endpoints"] * 0.1
        ) / 5.0)
        
        self.logger.info(f"   📊 高價值目標評分: {high_value_score:.2f}")
        self.logger.info(f"   📊 發現: {apis_found} APIs, {forms_found} Forms, {subdomains_found} Subdomains")
        
        # === 4. 技術棧風險評估 ===
        tech_risk_multiplier = 1.0
        tech_insights = []
        
        # 高風險技術棧 (基於歷史 CVE 和 Bug Bounty 統計)
        high_risk_techs = {
            "php": 1.3,  # 歷史上大量漏洞
            "wordpress": 1.4,  # 插件生態系統問題
            "struts": 1.5,  # Apache Struts 歷史漏洞
            "spring": 1.2,  # Spring4Shell 等
            "laravel": 1.1,
            "rails": 1.1,
            "node": 1.0,
            "java": 1.2,
            "asp.net": 1.1,
        }
        
        for tech in technologies:
            tech_lower = tech.lower()
            for risk_tech, multiplier in high_risk_techs.items():
                if risk_tech in tech_lower:
                    tech_risk_multiplier = max(tech_risk_multiplier, multiplier)
                    tech_insights.append(f"{tech} (風險係數 {multiplier})")
        
        # === 5. 神經網路決策 (5M 專用模型) ===
        ai_confidence = 0.5
        ai_attack_vector = "reconnaissance"
        ai_recommended_focus = []
        
        if self.use_neural_decision:
            try:
                # 構建 Bug Bounty 專用上下文
                neural_context = (
                    f"Program: {program_scope.get('name', 'unknown') if program_scope else 'unknown'} | "
                    f"Scope: {urls_found} URLs, {apis_found} APIs, {forms_found} Forms | "
                    f"Tech: {','.join(technologies[:5])} | "
                    f"WAF: {waf_vendor if waf_detected else 'None'} | "
                    f"HighValue: {high_value_score:.2f} | "
                    f"Upload: {high_value_indicators['file_upload_forms']}, Admin: {high_value_indicators['admin_panels']}"
                )
                
                ai_result = self.neural_engine.generate_decision(
                    task_description="decide_phase1_strategy",
                    context=neural_context
                )
                
                ai_confidence = ai_result.get("confidence", 0.5)
                ai_attack_vector = ai_result.get("attack_vector", "reconnaissance")
                ai_recommended_focus = ai_result.get("recommended_tools", [])
                
                self.logger.info(f"   🧠 AI 信心度: {ai_confidence:.2f}, 推薦向量: {ai_attack_vector}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ 神經網路決策回退: {e}")
        
        # === 6. ROI 計算 (實際 Bug Bounty 經濟學) ===
        estimated_time_hours = self._estimate_phase1_time(phase0_result) / 3600.0
        
        # 基於高價值目標和技術風險調整預期獎金
        adjusted_target_value = target_value * high_value_score * tech_risk_multiplier
        
        # WAF 會降低成功率約 30-50%
        if waf_detected:
            adjusted_target_value *= 0.6
        
        expected_value = adjusted_target_value * ai_confidence
        roi = expected_value / max(estimated_time_hours, 0.5)
        
        self.logger.info(f"   💰 預期價值: ${expected_value:.0f}, ROI: ${roi:.0f}/hr")
        
        # === 7. 決策邏輯 (多因素加權) ===
        need_phase1 = False
        reasoning_parts = []
        priority_targets = []
        
        # 規則 1: 高價值目標檢測
        if high_value_score > 0.5:
            need_phase1 = True
            reasoning_parts.append(f"高價值目標評分 {high_value_score:.2f}")
            
            # 識別優先測試目標
            if high_value_indicators["payment_flows"] > 0:
                priority_targets.append({"type": "payment", "priority": 1, "vuln_focus": ["idor", "logic_bypass", "race_condition"]})
            if high_value_indicators["admin_panels"] > 0:
                priority_targets.append({"type": "admin", "priority": 2, "vuln_focus": ["auth_bypass", "privilege_escalation"]})
            if high_value_indicators["file_upload_forms"] > 0:
                priority_targets.append({"type": "upload", "priority": 3, "vuln_focus": ["file_upload_rce", "path_traversal"]})
            if high_value_indicators["api_endpoints"] > 3:
                priority_targets.append({"type": "api", "priority": 4, "vuln_focus": ["idor", "mass_assignment", "rate_limit_bypass"]})
        
        # 規則 2: 攻擊面評估
        attack_surface_score = (urls_found * 0.01 + forms_found * 0.05 + apis_found * 0.1)
        if attack_surface_score > 1.0:
            need_phase1 = True
            reasoning_parts.append(f"廣大攻擊面 (surface={attack_surface_score:.2f})")
        
        # 規則 3: WAF 策略
        if waf_detected:
            reasoning_parts.append(f"WAF: {waf_vendor or 'unknown'}")
            if ai_confidence > 0.65:  # 提高門檻，因為 WAF 增加難度
                need_phase1 = True
                reasoning_parts.append("AI 建議嘗試 WAF 繞過")
            else:
                reasoning_parts.append("WAF 存在，謹慎評估")
        
        # 規則 4: ROI 門檻 (專業 Hunter 標準)
        if roi > 75:  # $75/hr 是專業 Hunter 的參考門檻
            need_phase1 = True
            reasoning_parts.append(f"高 ROI: ${roi:.0f}/hr")
        elif roi < 30:
            if not high_value_score > 0.7:  # 除非有超高價值目標
                need_phase1 = False
                reasoning_parts.append(f"低 ROI: ${roi:.0f}/hr")
        
        # 規則 5: 技術棧風險
        if tech_risk_multiplier > 1.2:
            need_phase1 = True
            reasoning_parts.append(f"高風險技術棧: {', '.join(tech_insights)}")
        
        # 規則 6: Program 推薦
        if recommendations.get("needs_deep_scan") or recommendations.get("priority_high"):
            need_phase1 = True
            reasoning_parts.append("Phase0 強烈建議深度掃描")
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "預設保守策略"
        
        # === 8. 構建返回結果 ===
        result = {
            "need_phase1": need_phase1,
            "reasoning": reasoning,
            "decision_source": "neural_network" if ai_confidence > 0.6 else "rule_engine",
            
            # 經濟指標
            "roi": roi,
            "estimated_time_hours": estimated_time_hours,
            "expected_value": expected_value,
            "adjusted_target_value": adjusted_target_value,
            
            # AI 分析
            "ai_confidence": ai_confidence,
            "ai_attack_vector": ai_attack_vector,
            "ai_recommended_focus": ai_recommended_focus,
            
            # 目標分析
            "high_value_score": high_value_score,
            "high_value_indicators": high_value_indicators,
            "priority_targets": priority_targets,
            
            # 技術分析
            "tech_risk_multiplier": tech_risk_multiplier,
            "tech_insights": tech_insights,
            "technologies_detected": technologies,
            
            # Phase1 執行配置
            "phase1_config": {
                "scan_depth": "intensive" if high_value_score > 0.7 else "standard",
                "focus_areas": [t["type"] for t in priority_targets[:3]],
                "time_budget_minutes": int(estimated_time_hours * 60),
                "parallel_workers": 3 if not waf_detected else 1,
            }
        }
        
        # WAF 繞過策略
        if waf_detected:
            result["waf_bypass_plan"] = self._decide_waf_bypass(waf_vendor, ai_confidence)
            result["phase1_config"]["stealth_mode"] = True
            result["phase1_config"]["delay_between_requests"] = result["waf_bypass_plan"].get("delay_multiplier", 2.0)
            self.logger.info(f"   🛡️ WAF 繞過策略已配置: delay×{result['waf_bypass_plan'].get('delay_multiplier', 1)}")
        
        self.logger.info(f"   ✅ 決策: {'執行 Phase1 (優先: {})'.format(','.join([t['type'] for t in priority_targets[:2]])) if need_phase1 else '跳過 Phase1'}")
        return result
    
    def decide_phase2_targets(
        self,
        phase1_result: dict[str, Any],
        max_targets: int = 10,
        program_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """步驟 9: AI 決策 Phase2 攻擊目標優先級排序 (HackerOne/Bugcrowd 實戰優化版)
        
        實際 Bug Bounty 目標優先級邏輯 (基於專業 Hunter 經驗):
        
        優先級 Tier 1 (Critical 獎金 $10k+):
        - Account Takeover (ATO) 鏈
        - RCE/SSRF 到內部服務
        - 支付/金融繞過
        - PII 大規模洩露
        
        優先級 Tier 2 (High 獎金 $3k-$10k):
        - SQL Injection (有數據影響)
        - IDOR 到敏感資源
        - Auth Bypass
        - Privilege Escalation
        
        優先級 Tier 3 (Medium 獎金 $500-$3k):
        - XSS (Stored > DOM > Reflected)
        - CSRF (敏感操作)
        - 信息洩露 (API keys, credentials)
        
        OWASP WSTG 對應：
        - 4.6 Session Management → Session Fixation, Cookie 問題
        - 4.7 Input Validation → SQLi, XSS, Command Injection
        - 4.10 Business Logic → 邏輯漏洞、競態條件
        
        Args:
            phase1_result: Phase1 掃描結果
            max_targets: 最大返回目標數量
            program_context: Program 上下文 (獎金表、重複情報等)
            
        Returns:
            排序後的目標列表
        """
        self.logger.info("🎯 [Step 9: Phase2 Targets] 開始智慧目標優先級排序...")
        
        assets = phase1_result.get("assets", [])
        if not assets:
            self.logger.warning("   ⚠️ Phase1 未發現任何資產")
            return []
        
        # Program 上下文
        program_context = program_context or {}
        bounty_table = program_context.get("bounty_table", self._get_default_bounty_table())
        duplicate_intel = set(program_context.get("duplicate_intel", []))
        waf_info = phase1_result.get("waf_info", {})
        
        # === HackerOne/Bugcrowd 漏洞優先級映射 ===
        vuln_tier_mapping = {
            # Tier 1: Critical ($10k+)
            "rce": 1, "remote_code_execution": 1, "command_injection": 1,
            "ssrf": 1, "server_side_request_forgery": 1,
            "account_takeover": 1, "ato": 1,
            "payment_bypass": 1, "financial_manipulation": 1,
            # Tier 2: High ($3k-$10k)
            "sql_injection": 2, "sqli": 2,
            "idor": 2, "insecure_direct_object_reference": 2,
            "auth_bypass": 2, "authentication_bypass": 2,
            "privilege_escalation": 2, "xxe": 2, "ssti": 2,
            "path_traversal": 2, "lfi": 2, "rfi": 2,
            # Tier 3: Medium ($500-$3k)
            "xss_stored": 3, "stored_xss": 3,
            "xss_dom": 3, "xss_reflected": 3,
            "csrf": 3, "api_key_disclosure": 3,
            "cors_misconfiguration": 3, "open_redirect": 3,
        }
        
        targets_with_scores = []
        
        for asset in assets[:100]:  # 限制分析前 100 個資產
            vuln_type = asset.get("vuln_type", "unknown").lower().replace(" ", "_")
            tier = vuln_tier_mapping.get(vuln_type, 4)
            if tier == 4:  # 跳過低價值漏洞
                continue
            
            # 計算多維度評分
            bounty_value_score = bounty_table.get(vuln_type, 500) / 10000
            waf_interference_score = self._calculate_waf_interference(asset, phase1_result)
            historical_success_score = self._query_historical_success(asset)
            duplicate_risk = 0.5 if vuln_type in duplicate_intel else 0.0
            
            # AI 神經網路評分
            ai_score = 0.5
            attack_vector = vuln_type
            recommended_tools = self._get_default_tools_for_vuln(vuln_type)
            
            if self.use_neural_decision:
                try:
                    ai_result = self.neural_engine.generate_decision(
                        task_description="target_prioritization",
                        context=f"Target: {asset.get('url', '')[:80]} | VulnType: {vuln_type} | Tier: {tier}"
                    )
                    ai_score = ai_result.get("confidence", 0.5)
                    attack_vector = ai_result.get("attack_vector", vuln_type)
                    recommended_tools = ai_result.get("recommended_tools", recommended_tools)
                except Exception as e:
                    self.logger.debug(f"   AI 評分失敗: {e}")
            
            # 綜合評分 (加權)
            final_score = (
                ai_score * 0.35 +
                bounty_value_score * 0.25 +
                (1.0 - waf_interference_score) * 0.15 +
                historical_success_score * 0.10 +
                (1.0 - duplicate_risk) * 0.15
            )
            
            # Tier 加權
            if tier == 1:
                final_score *= 1.3
            elif tier == 2:
                final_score *= 1.1
            
            targets_with_scores.append({
                "asset": asset,
                "score": min(1.0, final_score),
                "tier": tier,
                "cvss_estimate": {1: 9.0, 2: 7.5, 3: 5.5}.get(tier, 5.0),
                "attack_vector": attack_vector,
                "recommended_tools": recommended_tools,
                "estimated_bounty": bounty_table.get(vuln_type, 500),
                "duplicate_risk": duplicate_risk,
                "reasoning": f"Tier{tier}|AI:{ai_score:.2f}|獎金:{bounty_value_score:.2f}|WAF干擾:{waf_interference_score:.2f}"
            })
        
        # 排序: 先 Tier，再 score
        targets_with_scores.sort(key=lambda x: (x['tier'], -x['score']))
        top_targets = targets_with_scores[:max_targets]
        
        self.logger.info(f"   ✅ 已選出 {len(top_targets)} 個高優先級目標")
        for idx, t in enumerate(top_targets[:3], 1):
            tier_icon = {1: "🔴", 2: "🟠", 3: "🟡"}.get(t['tier'], "⚪")
            self.logger.info(f"   {tier_icon} #{idx} Tier{t['tier']} {t['asset'].get('url', 'N/A')[:40]}... (分數: {t['score']:.2f})")
        
        return top_targets
    
    def _get_default_tools_for_vuln(self, vuln_type: str) -> list[str]:
        """獲取漏洞類型對應的默認工具"""
        return {
            "sql_injection": ["sqlmap", "burp_intruder"], "sqli": ["sqlmap", "burp_intruder"],
            "xss_stored": ["xsstrike", "dalfox"], "xss_dom": ["domdig"], "xss_reflected": ["xsstrike"],
            "ssrf": ["ssrfmap", "burp_collaborator"], "rce": ["nuclei", "custom_payload"],
            "idor": ["autorize", "burp_match_replace"], "csrf": ["burp_csrf_poc"],
            "xxe": ["xxeinjector"], "ssti": ["tplmap"],
        }.get(vuln_type, ["burp_suite", "manual_analysis"])
    
    def _get_default_bounty_table(self) -> dict[str, int]:
        """默認獎金表 (基於 HackerOne 平均值)"""
        return {
            "rce": 15000, "ssrf": 8000, "account_takeover": 12000,
            "sql_injection": 5000, "sqli": 5000, "idor": 3000,
            "auth_bypass": 4000, "privilege_escalation": 3500,
            "xxe": 3000, "ssti": 4000, "path_traversal": 2000,
            "xss_stored": 1500, "xss_dom": 1000, "xss_reflected": 500,
            "csrf": 1000, "api_key_disclosure": 1500,
            "cors_misconfiguration": 500, "open_redirect": 300,
        }
    
    def evaluate_phase2_results(
        self,
        phase2_results: list[dict[str, Any]],
        time_budget_remaining: float,
        program_info: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """步驟 11: AI 評估 Phase2 結果並決定後續行動 (HackerOne 實戰優化版)
        
        決策選項 (基於 HackerOne 經驗):
        1. SUBMIT_REPORT - 提交報告
           - 高信心漏洞 (confidence > 0.85)
           - 可複現 POC 已準備
           - CVSS 評分已評估
        
        2. CONTINUE_DEEP_DIVE - 繼續深挖
           - 發現潛在攻擊鏈
           - 時間允許進一步探索
           - 可能提升漏洞嚴重性
        
        3. CHAIN_VULNERABILITIES - 串聯漏洞 (新增)
           - 發現多個低/中危漏洞
           - 可以組合成高危攻擊鏈
           - 例: XSS + CSRF = ATO
        
        4. SWITCH_STRATEGY - 切換策略
           - 當前攻擊向量無效
           - WAF 持續封鎖
           - 需要嘗試其他方法
        
        5. ABANDON_TARGET - 放棄目標
           - ROI 過低
           - 疑似 honeypot
           - 重複風險過高
        
        HackerOne 報告品質考量:
        - 清晰的重現步驟
        - 商業影響說明
        - 修復建議
        - CVSS 評分合理性
        
        Args:
            phase2_results: Phase2 測試結果列表
                - [{vuln_type, severity, confidence, poc_ready, reproducible, url, params}]
            time_budget_remaining: 剩餘時間預算（秒）
            program_info: Program 資訊
                - avg_response_time_days: 平均響應時間
                - duplicate_rate: 重複率
                - avg_bounty_by_severity: {CRITICAL: $, HIGH: $, ...}
            
        Returns:
            評估結果字典
        """
        self.logger.info("📊 [Step 11: Phase2 Evaluation] 評估攻擊結果...")
        
        program_info = program_info or {}
        duplicate_rate = program_info.get("duplicate_rate", 0.3)
        
        # === 1. 詳細統計分析 ===
        total_findings = len(phase2_results)
        
        # 按嚴重性分類
        severity_counts = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0
        }
        for r in phase2_results:
            sev = r.get("severity", "INFO").upper()
            if sev in severity_counts:
                severity_counts[sev] += 1
        
        critical_high_count = severity_counts["CRITICAL"] + severity_counts["HIGH"]
        
        # POC 準備度
        poc_ready_count = sum(1 for r in phase2_results if r.get("poc_ready", False))
        reproducible_count = sum(1 for r in phase2_results if r.get("reproducible", False))
        
        # 平均信心度
        avg_confidence = (
            sum(r.get("confidence", 0) for r in phase2_results) / max(total_findings, 1)
        )
        
        # 預估獎金
        bounty_table = program_info.get("avg_bounty_by_severity", {
            "CRITICAL": 10000, "HIGH": 3000, "MEDIUM": 500, "LOW": 100
        })
        estimated_total_bounty = sum(
            bounty_table.get(r.get("severity", "LOW").upper(), 100) * r.get("confidence", 0.5)
            for r in phase2_results
        )
        
        self.logger.info(f"   📈 發現: {total_findings} 個漏洞")
        self.logger.info(f"   🔴 Critical: {severity_counts['CRITICAL']}, High: {severity_counts['HIGH']}")
        self.logger.info(f"   🟠 Medium: {severity_counts['MEDIUM']}, Low: {severity_counts['LOW']}")
        self.logger.info(f"   📋 POC 已準備: {poc_ready_count}/{total_findings}")
        self.logger.info(f"   📈 平均信心度: {avg_confidence:.2f}")
        self.logger.info(f"   💰 預估獎金: ${estimated_total_bounty:.0f}")
        self.logger.info(f"   ⏱️ 剩餘時間: {time_budget_remaining / 60:.1f} 分鐘")
        
        # === 2. 攻擊鏈潛力分析 ===
        chain_potential = self._analyze_vulnerability_chains(phase2_results)
        
        # === 3. 神經網路決策輔助 ===
        ai_recommendation = None
        if self.use_neural_decision and phase2_results:
            try:
                context = (
                    f"Findings: {total_findings} | "
                    f"Critical/High: {critical_high_count} | "
                    f"Confidence: {avg_confidence:.2f} | "
                    f"POC Ready: {poc_ready_count} | "
                    f"ChainPotential: {chain_potential['score']:.2f} | "
                    f"TimeRemaining: {time_budget_remaining/3600:.1f}h | "
                    f"DuplicateRate: {duplicate_rate:.2f}"
                )
                ai_result = self.neural_engine.generate_decision(
                    task_description="evaluate_phase2_results",
                    context=context
                )
                ai_recommendation = ai_result.get("attack_vector", "continue")
            except Exception as e:
                self.logger.debug(f"   AI 決策失敗: {e}")
        
        # === 4. 決策邏輯 (多因素加權) ===
        action = "CONTINUE_DEEP_DIVE"  # 預設行動
        reasoning_parts = []
        priority = "NORMAL"
        
        # 規則 1: 發現高價值可報告漏洞
        if critical_high_count >= 1 and poc_ready_count >= 1 and avg_confidence > 0.85:
            action = "SUBMIT_REPORT"
            priority = "HIGH"
            reasoning_parts.append(f"發現 {critical_high_count} 個高危漏洞，POC 已準備")
        
        # 規則 2: 可串聯漏洞提升價值
        elif chain_potential["can_chain"] and chain_potential["score"] > 0.7:
            action = "CHAIN_VULNERABILITIES"
            priority = "HIGH"
            reasoning_parts.append(f"可串聯漏洞: {chain_potential['chain_description']}")
        
        # 規則 3: 多個中危漏洞，建議合併報告
        elif severity_counts["MEDIUM"] >= 3 and poc_ready_count >= 2:
            action = "SUBMIT_REPORT"
            priority = "MEDIUM"
            reasoning_parts.append("多個中危漏洞，建議整合報告")
        
        # 規則 4: 時間緊迫
        elif time_budget_remaining < 1800:  # 少於 30 分鐘
            if critical_high_count >= 1 or (total_findings > 0 and avg_confidence > 0.7):
                action = "SUBMIT_REPORT"
                priority = "URGENT"
                reasoning_parts.append("時間緊迫，提交現有發現")
            else:
                action = "ABANDON_TARGET"
                reasoning_parts.append("時間不足且收穫有限")
        
        # 規則 5: 效果不佳，切換策略
        elif avg_confidence < 0.4 and total_findings < 3:
            action = "SWITCH_STRATEGY"
            reasoning_parts.append("當前方法效果不佳")
            
            # AI 建議新策略
            if ai_recommendation:
                reasoning_parts.append(f"AI 建議: 嘗試 {ai_recommendation}")
        
        # 規則 6: 發現潛力，繼續深挖
        elif total_findings > 0 and time_budget_remaining > 3600:
            # 檢查是否值得繼續
            hourly_rate = estimated_total_bounty / max(1, (24 * 3600 - time_budget_remaining) / 3600)
            if hourly_rate > 50:  # $50/hr 門檻
                action = "CONTINUE_DEEP_DIVE"
                reasoning_parts.append(f"ROI 良好 (${hourly_rate:.0f}/hr)，繼續深挖")
            else:
                action = "SWITCH_STRATEGY"
                reasoning_parts.append(f"ROI 偏低 (${hourly_rate:.0f}/hr)，建議切換")
        
        # 規則 7: 重複風險評估
        elif duplicate_rate > 0.6 and severity_counts["MEDIUM"] + severity_counts["LOW"] > critical_high_count:
            action = "ABANDON_TARGET"
            reasoning_parts.append(f"高重複風險 ({duplicate_rate*100:.0f}%)")
        
        # 規則 8: 無有效發現
        else:
            action = "ABANDON_TARGET"
            reasoning_parts.append("未發現可利用漏洞")
        
        reasoning = " | ".join(reasoning_parts)
        
        # === 5. 生成下一步建議 ===
        next_steps = self._generate_next_steps(action, phase2_results)
        
        # === 6. 報告準備建議 (如果是 SUBMIT_REPORT) ===
        report_guidance = None
        if action == "SUBMIT_REPORT":
            report_guidance = self._generate_report_guidance(phase2_results, program_info)
        
        result = {
            "action": action,
            "priority": priority,
            "reasoning": reasoning,
            "findings_summary": {
                "total": total_findings,
                "by_severity": severity_counts,
                "critical_high": critical_high_count,
                "poc_ready": poc_ready_count,
                "reproducible": reproducible_count,
                "avg_confidence": avg_confidence,
                "estimated_bounty": estimated_total_bounty
            },
            "chain_analysis": chain_potential,
            "next_steps": next_steps,
            "report_guidance": report_guidance,
            "ai_recommendation": ai_recommendation,
            "time_metrics": {
                "remaining_minutes": time_budget_remaining / 60,
                "recommended_action_time": self._estimate_action_time(action)
            }
        }
        
        self.logger.info(f"   ✅ 建議行動: {action} (優先級: {priority})")
        return result
    
    def _analyze_vulnerability_chains(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """分析漏洞串聯潛力"""
        vuln_types = [r.get("vuln_type", "").lower() for r in results]
        
        # 已知攻擊鏈模式
        chain_patterns = [
            {
                "components": ["xss", "csrf"],
                "result": "Account Takeover",
                "severity_boost": "CRITICAL",
                "description": "XSS + CSRF → 帳戶劫持"
            },
            {
                "components": ["ssrf", "rce"],
                "result": "Remote Code Execution Chain",
                "severity_boost": "CRITICAL",
                "description": "SSRF → 內部服務 → RCE"
            },
            {
                "components": ["idor", "information_disclosure"],
                "result": "Mass Data Exposure",
                "severity_boost": "HIGH",
                "description": "IDOR + 信息洩露 → 批量數據提取"
            },
            {
                "components": ["sql_injection", "auth_bypass"],
                "result": "Full Database Access",
                "severity_boost": "CRITICAL",
                "description": "SQLi → 認證繞過 → 完整數據庫訪問"
            },
            {
                "components": ["open_redirect", "oauth"],
                "result": "OAuth Token Theft",
                "severity_boost": "HIGH",
                "description": "開放重定向 + OAuth → Token 竊取"
            }
        ]
        
        can_chain = False
        best_chain = None
        chain_score = 0.0
        
        for pattern in chain_patterns:
            components = pattern["components"]
            matches = sum(1 for c in components if any(c in vt for vt in vuln_types))
            if matches >= len(components):
                can_chain = True
                score = matches / len(components)
                if score > chain_score:
                    chain_score = score
                    best_chain = pattern
        
        return {
            "can_chain": can_chain,
            "score": chain_score,
            "chain_description": best_chain["description"] if best_chain else None,
            "severity_boost": best_chain["severity_boost"] if best_chain else None,
            "matched_pattern": best_chain
        }
    
    def _generate_report_guidance(
        self,
        results: list[dict[str, Any]],
        program_info: dict[str, Any]
    ) -> dict[str, Any]:
        """生成 HackerOne 報告撰寫指南"""
        top_finding = max(results, key=lambda x: x.get("confidence", 0)) if results else {}
        
        return {
            "title_template": f"[{top_finding.get('severity', 'Medium')}] {top_finding.get('vuln_type', 'Vulnerability')} in {top_finding.get('url', 'application')[:50]}",
            "sections": [
                "Summary (1-2 sentences)",
                "Steps to Reproduce (numbered list)",
                "Impact (business perspective)",
                "Proof of Concept (code/screenshots)",
                "Suggested Fix",
                "References (CVE, OWASP)"
            ],
            "cvss_estimate": self._calculate_cvss(top_finding),
            "tips": [
                "使用清晰的重現步驟",
                "強調商業影響",
                "提供修復建議",
                "附上視頻 POC 可提升報告品質"
            ]
        }
    
    def _calculate_cvss(self, finding: dict[str, Any]) -> dict[str, Any]:
        """計算 CVSS 3.1 評分估計"""
        severity = finding.get("severity", "MEDIUM").upper()
        base_scores = {"CRITICAL": 9.0, "HIGH": 7.5, "MEDIUM": 5.5, "LOW": 3.0, "INFO": 0.0}
        
        return {
            "base_score": base_scores.get(severity, 5.5),
            "vector_string": f"CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",  # 簡化範例
            "severity_rating": severity
        }
    
    def _estimate_action_time(self, action: str) -> int:
        """估計各行動所需時間（分鐘）"""
        time_estimates = {
            "SUBMIT_REPORT": 30,      # 撰寫報告
            "CONTINUE_DEEP_DIVE": 60, # 繼續測試
            "CHAIN_VULNERABILITIES": 45,  # 串聯攻擊
            "SWITCH_STRATEGY": 15,    # 切換策略
            "ABANDON_TARGET": 5       # 放棄
        }
        return time_estimates.get(action, 30)
    
    # ═══════════════════════════════════════════════════════════════
    # WAF/Rate Limit 自適應策略（輔助方法）
    # ═══════════════════════════════════════════════════════════════
    
    def _decide_waf_bypass(
        self,
        waf_vendor: Optional[str],
        base_confidence: float
    ) -> dict[str, Any]:
        """基於 WAF 類型和 AI 信心度決定繞過策略"""
        vendor = (waf_vendor or "unknown").lower()
        strategy = self.waf_bypass_strategies.get(vendor, self.waf_bypass_strategies["unknown"])
        
        # AI 調整策略激進度
        delay_multiplier = strategy['delay_multiplier']
        if base_confidence > 0.8:
            delay_multiplier *= 0.8  # 高信心 → 更快
        elif base_confidence < 0.5:
            delay_multiplier *= 1.5  # 低信心 → 更保守
        
        return {
            **strategy,
            'delay_multiplier': delay_multiplier,
            'confidence_adjusted': True
        }
    
    def adaptive_rate_limiting(
        self,
        target_url: str,
        phase: str,
        waf_detected: bool
    ) -> dict[str, float]:
        """自適應速率限制策略
        
        Args:
            target_url: 目標 URL
            phase: 階段 (phase0/phase1/phase2)
            waf_detected: 是否檢測到 WAF
            
        Returns:
            速率限制配置
        """
        profile = self.rate_limit_profiles.get(phase, self.rate_limit_profiles["phase1"])
        
        base_rate = profile['requests_per_second']
        
        # WAF 檢測 → 大幅降速
        if waf_detected:
            base_rate *= 0.2
        
        # 查詢歷史封禁率（如果有經驗管理器）
        if self.experience_manager:
            try:
                historical_ban_rate = self.experience_manager.get_ban_rate(target_url)
                if historical_ban_rate > 0.5:
                    base_rate *= 0.5
            except Exception:
                pass
        
        return {
            'requests_per_second': base_rate,
            'burst_size': int(base_rate * 2),
            'retry_after_429': 60,
            'backoff_multiplier': 2.0
        }
    
    # ═══════════════════════════════════════════════════════════════
    # 初始化和輔助方法
    # ═══════════════════════════════════════════════════════════════
    
    def _initialize_waf_strategies(self) -> dict[str, dict]:
        """初始化 WAF 繞過策略庫"""
        return {
            'cloudflare': {
                'delay_multiplier': 3.0,
                'payload_encoding': ['unicode', 'hex', 'double_url'],
                'user_agent_rotation': True,
                'header_randomization': True
            },
            'imperva': {
                'delay_multiplier': 2.5,
                'payload_encoding': ['case_swap', 'comment_injection'],
                'chunk_encoding': True
            },
            'aws_waf': {
                'delay_multiplier': 2.0,
                'payload_encoding': ['null_byte', 'newline'],
                'ip_rotation': True
            },
            'unknown': {
                'delay_multiplier': 2.0,
                'payload_encoding': ['basic'],
                'conservative_mode': True
            }
        }
    
    def _initialize_rate_profiles(self) -> dict[str, dict]:
        """初始化速率限制配置文件"""
        return {
            'phase0': {'requests_per_second': 100},
            'phase1': {'requests_per_second': 50},
            'phase2': {'requests_per_second': 20}
        }
    
    def _initialize_bounty_matrix(self) -> dict[str, float]:
        """初始化獎金價值矩陣（歸一化 0-1）"""
        return {
            'authentication_bypass': 0.9,
            'sql_injection': 0.8,
            'server_side_request_forgery': 0.85,
            'cross_site_scripting': 0.5,
            'idor': 0.7,
            'reconnaissance': 0.1
        }
    
    def _encode_phase0_features(self, phase0_result: dict) -> Any:
        """將 Phase0 結果編碼為神經網路輸入（簡化版）"""
        import torch
        features = torch.zeros(512)
        summary = phase0_result.get("summary", {})
        features[0] = summary.get("urls_found", 0) / 1000.0
        features[1] = summary.get("forms_found", 0) / 50.0
        features[2] = summary.get("apis_found", 0) / 20.0
        # ... 其他特徵
        return features
    
    def _encode_asset_features(self, asset: dict) -> Any:
        """將資產編碼為神經網路輸入（簡化版）"""
        import torch
        return torch.zeros(512)  # 簡化實現
    
    def _estimate_phase1_time(self, phase0_result: dict) -> float:
        """估算 Phase1 所需時間（秒）"""
        urls = phase0_result.get("summary", {}).get("urls_found", 0)
        return min(1800 + urls * 10, 7200)  # 30分鐘 + 每個URL 10秒，最多2小時
    
    def _calculate_bounty_value(self, asset: dict) -> float:
        """計算資產的獎金價值評分 (0-1)"""
        # 簡化實現：根據 URL 特徵評估
        url = asset.get("url", "").lower()
        if "admin" in url or "api" in url:
            return 0.8
        elif "login" in url or "auth" in url:
            return 0.7
        return 0.5
    
    def _calculate_waf_interference(self, asset: dict, phase1_result: dict) -> float:
        """計算 WAF 干擾評分 (0-1, 越高越糟)"""
        fingerprints = phase1_result.get("fingerprints", {})
        if fingerprints.get("waf_detected"):
            return 0.8
        return 0.2
    
    def _query_historical_success(self, asset: dict) -> float:
        """查詢歷史成功率 (0-1)"""
        # 需要經驗管理器支援
        if self.experience_manager:
            try:
                return self.experience_manager.get_success_rate(asset.get("url", ""))
            except Exception:
                pass
        return 0.5  # 預設中等成功率
    
    def _generate_next_steps(self, action: str, results: list) -> list[str]:
        """生成下一步建議"""
        if action == "SUBMIT_REPORT":
            return [
                "1. 驗證所有高危漏洞的可重現性",
                "2. 撰寫詳細的 PoC 和影響說明",
                "3. 提交到 HackerOne 平台"
            ]
        elif action == "CONTINUE_DEEP_DIVE":
            return [
                "1. 針對高優先級目標進行手動測試",
                "2. 嘗試 WAF 繞過技術",
                "3. 探索邊緣案例和業務邏輯漏洞"
            ]
        elif action == "SWITCH_STRATEGY":
            return [
                "1. 切換到更隱蔽的掃描模式",
                "2. 嘗試不同的 payload 編碼",
                "3. 調整攻擊向量"
            ]
        else:  # ABANDON_TARGET
            return [
                "1. 記錄失敗原因和學習經驗",
                "2. 切換到下一個目標",
                "3. 更新目標篩選策略"
            ]


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
