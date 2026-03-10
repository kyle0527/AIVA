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

Integration Note:
- 整合日期: 2026-01-19
- 整合項目: 雙CLI架構 + embedded_knowledge
- 新增: InternalLoopConnector 整合（內部閉環）
- 新增: ExternalLoopConnector 整合（外部閉環）
- 新增: embedded_knowledge 知識引擎整合
"""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import asyncio
import uuid

# [新增] 引入真實神經網路引擎
from ..neural.real_neural_core import RealDecisionEngine

# [新增] 引入內外部閉環連接器
from ..internal_loop_connector import InternalLoopConnector
from ..external_loop_connector import ExternalLoopConnector

# [新增] 引入 embedded_knowledge 知識引擎
from ..embedded_knowledge import (
    VulnerabilityDetector,
    CVEIdentifier,
    WAFBypassEngine,
    WebArchitectureAnalyzer,
    AttackContext,
)

# [Split] 拆分模組
from .bounty_strategy_agent import BountyStrategyAgent
from .knowledge_decision_mixin import KnowledgeDecisionMixin

# TYPE_CHECKING 前向引用
if TYPE_CHECKING:
    from aiva_common.schemas.commands import CLICommand

# 使用 aiva_common 的統一枚舉定義
from aiva_common.enums import RiskLevel

# 使用 aiva_common 的決策數據合約 (問題三修復)
from aiva_common.schemas import (
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


class EnhancedDecisionAgent(KnowledgeDecisionMixin):
    """增強的決策代理（繼承 KnowledgeDecisionMixin 取得知識決策方法）"""

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

        # Bug Bounty 策略代理（委派 Phase 1/2 決策）
        self.bounty_agent = BountyStrategyAgent(
            neural_engine=self.neural_engine,
            experience_manager=self.experience_manager,
            use_neural_decision=self.use_neural_decision,
        )

        self.logger.info("🛡️ 規則引擎已就緒")
        self.logger.info("🎯 Bug Bounty 策略代理已載入")

        # ========== [2026-01-19] 雙CLI架構整合 ==========
        # 內部閉環連接器：連接 internal_exploration 能力庫
        try:
            self.internal_connector = InternalLoopConnector(
                rag_knowledge_base=knowledge_base
            )
            self.logger.info("🔗 內部閉環連接器 (InternalLoopConnector) 已整合")
        except Exception as e:
            self.internal_connector = None
            self.logger.warning(f"⚠️ 內部閉環連接器初始化失敗: {e}")

        # 外部閉環連接器：連接執行結果 → 學習系統
        try:
            self.external_connector = ExternalLoopConnector()
            self.logger.info("🔗 外部閉環連接器 (ExternalLoopConnector) 已整合")
        except Exception as e:
            self.external_connector = None
            self.logger.warning(f"⚠️ 外部閉環連接器初始化失敗: {e}")

        # ========== [2026-01-19] embedded_knowledge 知識引擎整合 ==========
        # 漏洞檢測器：嵌入式漏洞判斷邏輯
        try:
            self.vuln_detector = VulnerabilityDetector()
            self.logger.info("🔍 漏洞檢測器 (VulnerabilityDetector) 已整合")
        except Exception as e:
            self.vuln_detector = None
            self.logger.warning(f"⚠️ 漏洞檢測器初始化失敗: {e}")

        # CVE 識別器：高危險 CVE 模組
        try:
            self.cve_identifier = CVEIdentifier()
            self.logger.info("🚨 CVE 識別器 (CVEIdentifier) 已整合")
        except Exception as e:
            self.cve_identifier = None
            self.logger.warning(f"⚠️ CVE 識別器初始化失敗: {e}")

        # WAF 繞過引擎：繞過技術字典
        try:
            self.waf_engine = WAFBypassEngine()
            self.logger.info("🛡️ WAF 繞過引擎 (WAFBypassEngine) 已整合")
        except Exception as e:
            self.waf_engine = None
            self.logger.warning(f"⚠️ WAF 繞過引擎初始化失敗: {e}")

        # Web 架構分析器：架構漏洞檢測
        try:
            self.web_analyzer = WebArchitectureAnalyzer()
            self.logger.info("🌐 Web 架構分析器 (WebArchitectureAnalyzer) 已整合")
        except Exception as e:
            self.web_analyzer = None
            self.logger.warning(f"⚠️ Web 架構分析器初始化失敗: {e}")

        self.logger.info("✅ 雙CLI架構 + embedded_knowledge 整合完成")

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

    def decide(self, context: DecisionContext, return_cli_command: bool = False) -> "HighLevelIntent | CLICommand":
        """做出高階決策 - 返回 HighLevelIntent 或 CLICommand

        架構升級（v2.0 - 2026-02-09）:
        - 支持返回 CLICommand（CLI 參數包驅動架構）
        - 保留 HighLevelIntent 向後兼容

        這是 cognitive_core → task_planning 的標準接口

        職責劃分：
        - cognitive_core (此方法): 決定「做什麼」(What) 和「為什麼」(Why)
        - task_planning: 決定「怎麼做」(How) - 生成具體的 AST 或執行 CLI

        Args:
            context: 決策上下文
            return_cli_command: 是否返回 CLICommand（新架構）

        Returns:
            HighLevelIntent（舊架構）或 CLICommand（新架構）
        """
        self.logger.info(f"🤔 開始高階決策分析 - 風險等級: {context.risk_level.value}")

        # 新架構：返回 CLICommand
        if return_cli_command:
            return self._decide_cli_command(context)

        # 舊架構：返回 HighLevelIntent（向後兼容）
        legacy_decision = self._sync_make_decision(context)
        intent = self._convert_decision_to_intent(legacy_decision, context)

        self.logger.info(f"✅ 生成高階意圖: {intent.intent_type.value} (信心度: {intent.confidence:.2f})")

        return intent

    def _decide_cli_command(self, context: DecisionContext) -> 'CLICommand':
        """生成 CLI 命令（新架構）

        從決策上下文中提取信息，產出標準化的 CLICommand。

        Args:
            context: 決策上下文

        Returns:
            CLICommand: CLI 參數包
        """
        from aiva_common.schemas.commands import CLICommand
        from ...task_planning.planner.tool_selector import ToolSelector

        # 1. 分析上下文提取意圖
        intent = self._extract_intent_from_context(context)
        target = context.target_info.get("url", "unknown")

        # 2. 使用 tool_selector 選擇 CLI 命令
        selector = ToolSelector()
        cli_cmd = selector.select_cli_command(
            intent=intent,
            target=target,
            context={
                "intensity": self._calculate_intensity(context),
                "mode": self._determine_mode(context),
                "risk_level": context.risk_level.value
            }
        )

        if cli_cmd is None:
            # 降級：生成默認命令
            self.logger.warning("tool_selector 未返回命令，使用降級策略")
            cli_cmd = CLICommand(
                flow_id="flow_1",  # 默認 flow
                target=target,
                flags={
                    "intensity": 0.5,
                    "mode": "normal"
                },
                command_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                metadata={
                    "intent": intent,
                    "fallback": True,
                    "risk_level": context.risk_level.value
                }
            )

        self.logger.info(f"✅ 生成 CLI 命令: {cli_cmd.flow_id} for {intent}")
        return cli_cmd

    def _extract_intent_from_context(self, context: DecisionContext) -> str:
        """從上下文提取意圖

        Args:
            context: 決策上下文

        Returns:
            意圖字符串（scan, sqli, xss, exploit 等）
        """
        # 檢查已發現的漏洞
        if context.discovered_vulns:
            for vuln in context.discovered_vulns:
                if "sql" in vuln.lower():
                    return "sqli_exploit"
                elif "xss" in vuln.lower():
                    return "xss_exploit"
                elif "ssrf" in vuln.lower():
                    return "ssrf_exploit"

        # 檢查可用工具
        available_tools_str = " ".join(str(t).lower() for t in context.available_tools)
        if "sql" in available_tools_str:
            return "sqli_detection"
        elif "xss" in available_tools_str:
            return "xss_detection"
        elif "scan" in available_tools_str:
            return "port_scan"

        # 默認：通用掃描
        return "general_scan"

    def _calculate_intensity(self, context: DecisionContext) -> float:
        """計算攻擊強度

        Args:
            context: 決策上下文

        Returns:
            強度值 0.0-1.0
        """
        base_intensity = 0.5

        # 根據風險等級調整
        if context.risk_level == RiskLevel.HIGH:
            base_intensity = 0.3  # 高風險降低強度
        elif context.risk_level == RiskLevel.LOW:
            base_intensity = 0.8  # 低風險提高強度

        # 根據失敗次數調整
        if context.attempts_without_success > 3:
            base_intensity = min(base_intensity + 0.2, 1.0)

        return base_intensity

    def _determine_mode(self, context: DecisionContext) -> str:
        """決定執行模式

        Args:
            context: 決策上下文

        Returns:
            模式字符串（stealth, normal, aggressive）
        """
        if context.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return "stealth"  # 高風險使用隱匿模式
        elif context.attempts_without_success > 5:
            return "aggressive"  # 多次失敗使用激進模式
        else:
            return "normal"

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

    # ========== 知識決策方法已移至 KnowledgeDecisionMixin ==========
    # query_internal_capabilities, record_execution_feedback,
    # analyze_target_vulnerabilities, identify_high_risk_cves,
    # generate_waf_bypass_payloads, analyze_web_architecture

    async def make_enhanced_decision(
        self,
        context: DecisionContext,
        use_embedded_knowledge: bool = True
    ) -> Decision:
        """增強版決策方法（整合所有知識源）

        這是整合了雙CLI架構和 embedded_knowledge 的完整決策方法。

        流程：
        1. 查詢內部能力庫 (InternalLoopConnector)
        2. 使用 embedded_knowledge 分析目標
        3. 神經網路 + 經驗 + 規則融合決策
        4. 記錄結果供學習 (ExternalLoopConnector)

        Args:
            context: 決策上下文
            use_embedded_knowledge: 是否使用嵌入式知識

        Returns:
            Decision: 增強的決策結果
        """
        self.logger.info("🧠 開始增強決策流程...")

        # 1. 查詢內部能力（同步方法）
        if self.internal_connector:
            target_type = context.target_info.get("type", "web")
            capabilities = self.query_internal_capabilities(
                query=f"{target_type} vulnerability scan",
                top_k=3
            )
            if capabilities:
                # capabilities 是 dict 列表，不是對象
                context.available_tools.extend([
                    c.get("metadata", {}).get("name", f"capability_{i}")
                    for i, c in enumerate(capabilities)
                ])

        # 2. 使用 embedded_knowledge 分析
        if use_embedded_knowledge:
            target_url = context.target_info.get("value")
            if target_url and isinstance(target_url, str):
                # 漏洞分析（使用正確的參數）
                vuln_result = self.analyze_target_vulnerabilities(
                    target_url=target_url,
                    response_body="",  # 空響應體，實際使用時會有數據
                    response_time=0.0
                )
                if "vulnerabilities" in vuln_result:
                    context.discovered_vulns.extend(
                        [v["type"] for v in vuln_result.get("vulnerabilities", [])]
                    )

                # 架構分析（使用正確的參數）
                arch_result = self.analyze_web_architecture(
                    response_headers={},  # 空 headers，實際使用時會有數據
                    response_body=""
                )
                if "architecture_type" in arch_result:
                    context.target_info["architecture"] = arch_result

        # 3. 執行標準決策流程
        decision = await self.make_decision(context)

        # 4. 增強決策參數
        decision.params["enhanced_mode"] = True
        decision.params["knowledge_sources"] = [
            "neural_network",
            "experience_db",
            "rule_engine",
            "internal_capabilities" if self.internal_connector else None,
            "embedded_knowledge" if use_embedded_knowledge else None
        ]
        decision.params["knowledge_sources"] = [
            s for s in decision.params["knowledge_sources"] if s
        ]

        self.logger.info(f"✅ 增強決策完成: {decision.action}")

        return decision

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
            from aiva_common.core.command_center import get_command_center
            from aiva_common.schemas import AICommand, CommandType
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
            from aiva_common.core.command_center import get_command_center
            from aiva_common.schemas import AICommand, CommandType
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
        else:
            constraints = scan_context.get('constraints', {})
            stealth_level = constraints.get('stealth_level', 'medium')
            rate_limit = constraints.get('rate_limit', 1000)
            target = scan_context.get('target', '')

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
            selected_tool, scan_strategy
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

    def _estimate_scan_time(self, tool: str, strategy: str) -> int:
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
    # Bug Bounty 特化決策方法（委派至 BountyStrategyAgent）
    # ═══════════════════════════════════════════════════════════════

    def decide_phase1_strategy(
        self,
        phase0_result: dict[str, Any],
        program_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """步驟 6: AI 決策 Phase1 策略 — 委派至 BountyStrategyAgent"""
        return self.bounty_agent.decide_phase1_strategy(
            phase0_result, program_scope
        )

    def decide_phase2_targets(
        self,
        phase1_result: dict[str, Any],
        max_targets: int = 10,
        program_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """步驟 9: AI 決策 Phase2 目標 — 委派至 BountyStrategyAgent"""
        return self.bounty_agent.decide_phase2_targets(
            phase1_result, max_targets, program_context
        )

    def evaluate_phase2_results(
        self,
        phase2_results: list[dict[str, Any]],
        time_budget_remaining: float,
        program_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """步驟 11: AI 評估 Phase2 結果 — 委派至 BountyStrategyAgent"""
        return self.bounty_agent.evaluate_phase2_results(
            phase2_results, time_budget_remaining, program_info
        )

    def adaptive_rate_limiting(
        self,
        target_url: str,
        phase: str,
        waf_detected: bool,
    ) -> dict[str, float]:
        """自適應速率限制策略 — 委派至 BountyStrategyAgent"""
        return self.bounty_agent.adaptive_rate_limiting(
            target_url, phase, waf_detected
        )


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

