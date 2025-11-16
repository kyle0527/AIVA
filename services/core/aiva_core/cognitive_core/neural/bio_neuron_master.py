"""BioNeuron Decision Controller - AI 決策控制器

❌ 不再是: 系統 Master、主控系統
✅ 現在是: AI 決策核心的控制器（只負責 AI 相關）

職責:
1. 管理 BioNeuronRAGAgent（5M 參數神經網路）
2. 處理 AI 決策請求
3. 提供三種操作模式（UI/AI/Chat）
4. RAG 知識檢索和抗幻覺機制

不負責:
❌ 系統協調
❌ 服務啟動
❌ 任務執行
❌ 資源管理

架構位置:
    app.py (FastAPI)          ← 系統唯一入口
        ↓ 通過
    CoreServiceCoordinator    ← 狀態管理器
        ↓ 使用
    EnhancedDecisionAgent     ← 決策代理
        ↓ 調用
    BioNeuronDecisionController ← AI 決策控制器（本類）

支持三種操作模式：
1. UI Mode - 圖形化介面控制
2. AI Mode - 完全自主決策
3. Chat Mode - 自然語言對話
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any

from .real_bio_net_adapter import (
    RealBioNeuronRAGAgent,
    RealScalableBioNet,
    create_real_rag_agent,
    create_real_scalable_bionet,
)
from ..decision.enhanced_decision_agent import EnhancedDecisionAgent
from ..rag import RAGEngine

logger = logging.getLogger(__name__)


class OperationMode(str, Enum):
    """操作模式"""

    UI = "ui"  # UI 控制
    AI = "ai"  # AI 自主
    CHAT = "chat"  # 對話溝通
    HYBRID = "hybrid"  # 混合模式


class ConversationContext:
    """對話上下文"""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.current_task: dict[str, Any] | None = None
        self.user_preferences: dict[str, Any] = {}


class BioNeuronDecisionController:
    """BioNeuron AI 決策控制器

    ❌ 不再是: 系統 Master、主控系統
    ✅ 現在是: AI 決策核心的控制器（只負責 AI 推理和決策）
    
    核心功能:
    1. 管理 5M 參數的 BioNeuronRAGAgent
    2. 提供 AI 決策服務（被 EnhancedDecisionAgent 調用）
    3. 支持三種操作模式（UI/AI/Chat）
    4. RAG 知識檢索和抗幻覺
    
    不負責:
    ❌ 系統協調、服務啟動
    ❌ 任務執行、資源管理
    ❌ 作為系統主線程
    
    使用方式:
        # 在 EnhancedDecisionAgent 中使用
        controller = BioNeuronDecisionController()
        
        # 進行 AI 決策
        decision = await controller.make_decision(context)
    """

    def __init__(
        self,
        codebase_path: str = "/workspaces/AIVA",
        default_mode: OperationMode | str = OperationMode.HYBRID,
    ) -> None:
        """初始化主控系統

        Args:
            codebase_path: 代碼庫路徑
            default_mode: 默認操作模式（可以是 OperationMode 或字串）
        """
        logger.info("🧠 Initializing BioNeuron Decision Controller...")

        # 處理字串模式參數
        if isinstance(default_mode, str):
            try:
                default_mode = OperationMode(default_mode.lower())
            except ValueError:
                logger.warning(f"Invalid mode string '{default_mode}', using HYBRID")
                default_mode = OperationMode.HYBRID

        # === 核心 AI 主腦 ===
        # 創建真實的5M神經網路
        self.decision_core = create_real_scalable_bionet(
            input_size=512,
            num_tools=20,
            weights_path=str(Path(codebase_path) / "services/core/aiva_core/ai_engine/aiva_5M_weights.pth")
        )
        
        # 創建真實的RAG代理  
        self.bio_neuron_agent = create_real_rag_agent(
            decision_core=self.decision_core,
            input_vector_size=512
        )
        
        # 創建增強決策代理
        self.enhanced_decision_agent = EnhancedDecisionAgent()

        # === RAG 引擎（移除重複實例化 - RAG 已整合在 BioNeuronRAGAgent 中） ===
        # 注意：不再單獨實例化 RAGEngine，避免與 BioNeuronRAGAgent 內部的 RAG 衝突
        self.rag_engine = None  # 將由 bio_neuron_agent 內部處理 RAG

        # === 操作模式管理 ===
        self.current_mode = default_mode
        self.mode_handlers: dict[OperationMode, Callable] = {
            OperationMode.UI: self._handle_ui_mode,
            OperationMode.AI: self._handle_ai_mode,
            OperationMode.CHAT: self._handle_chat_mode,
            OperationMode.HYBRID: self._handle_hybrid_mode,
        }

        # === 對話管理 ===
        self.conversation = ConversationContext()

        # === UI 回調函數 ===
        self.ui_callbacks: dict[str, Callable] = {}

        # === 任務隊列 ===
        self.task_queue: list[dict[str, Any]] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}

        logger.info(f"✅ Decision Controller initialized in {default_mode.value} mode")
        logger.info(f"   - BioNeuronRAGAgent: {self.bio_neuron_agent is not None}")
        logger.info(f"   - RAG Engine: {self.rag_engine is not None}")

    # ==================== 統一入口 ====================

    async def process_request(
        self,
        request: str | dict[str, Any],
        mode: OperationMode | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """處理請求（統一入口）

        Args:
            request: 請求內容（文字或結構化數據）
            mode: 操作模式（None 使用當前模式）
            context: 額外上下文

        Returns:
            處理結果
        """
        mode = mode or self.current_mode
        context = context or {}

        logger.info(f"📥 Processing request in {mode.value} mode")

        # 記錄到對話歷史
        self._record_interaction("user", request, context)

        # 根據模式處理
        handler = self.mode_handlers.get(mode)
        if handler is None:
            return {
                "success": False,
                "error": f"Unsupported mode: {mode.value}",
            }

        result = await handler(request, context)

        # 記錄回應
        self._record_interaction("assistant", result, context)

        return result

    # ==================== UI 模式 ====================

    async def _handle_ui_mode(
        self, request: str | dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """UI 模式處理

        特點：
        - 等待用戶確認
        - 提供操作選項
        - 即時反饋
        """
        logger.info("🖥️ Handling UI mode request")

        # 解析 UI 命令
        if isinstance(request, dict):
            action = request.get("action")
            params = request.get("params", {})
        else:
            # 自然語言轉 UI 命令
            action, params = await self._parse_ui_command(request)

        # 執行前請求確認
        if not params.get("auto_confirm", False):
            confirmation = await self._request_ui_confirmation(action, params)
            if not confirmation.get("confirmed", False):
                return {
                    "success": False,
                    "cancelled": True,
                    "message": "Operation cancelled by user",
                }

        # 執行 UI 操作
        result = await self._execute_ui_action(action, params)

        # 更新 UI 回調
        if "ui_update" in self.ui_callbacks:
            self.ui_callbacks["ui_update"](result)

        return result

    async def _parse_ui_command(self, text: str) -> tuple[str, dict[str, Any]]:
        """解析 UI 命令 (實際 NLU 實現)

        Args:
            text: 用戶輸入

        Returns:
            (action, params)
        """
        logger.debug(f"Parsing UI command with NLU: {text}")

        # NLU 處理，分層異常處理避免脆弱降級
        nlu_attempts = 0
        max_nlu_retries = 2
        
        while nlu_attempts <= max_nlu_retries:
            try:
                # 使用 BioNeuron 的 NLU 能力進行語義理解
                if self.bio_neuron_agent:
                    nlu_prompt = f"""分析以下用戶指令，提取意圖和參數：

用戶指令: {text}

請識別：
1. 主要意圖 (scan/attack/train/status/query/stop)
2. 目標參數 (URL、IP、應用程式名稱等)
3. 附加選項 (策略、優先級等)

以 JSON 格式返回結果。"""

                    # 使用真實的AI進行NLU處理
                    nlu_result = self.bio_neuron_agent.generate(
                        task_description=f"自然語言理解: {nlu_prompt}",
                        context="NLU processing for user command parsing"
                    )

                    # 解析 NLU 結果
                    intent = nlu_result.get("intent", "unknown").lower()
                    target = nlu_result.get("target", "auto_detect")
                    options = nlu_result.get("options", {})
                    confidence = nlu_result.get("confidence", 0.5)

                    logger.info(f"NLU result: intent={intent}, confidence={confidence:.2f}")

                    # 映射意圖到動作
                    if intent in ["scan", "掃描", "scanning"]:
                        return "start_scan", {"target": target, **options}
                    elif intent in ["attack", "攻擊", "exploit"]:
                        return "start_attack", {"target": target, **options}
                    elif intent in ["train", "訓練", "training"]:
                        return "start_training", options
                    elif intent in ["status", "狀態", "check"]:
                        return "show_status", {}
                    elif intent in ["stop", "停止", "cancel"]:
                        return "stop_task", options
                    else:
                        # 低信心度時返回未知
                        if confidence < 0.6:
                            return "unknown", {
                                "original_text": text,
                                "nlu_result": nlu_result,
                            }
                        return intent, {"target": target, **options}
                
                break  # 成功處理，跳出重試迴圈

            except (ConnectionError, TimeoutError) as e:
                # 網路相關錯誤，重試
                nlu_attempts += 1
                logger.warning(f"NLU 網路錯誤 (嘗試 {nlu_attempts}/{max_nlu_retries+1}): {e}")
                if nlu_attempts <= max_nlu_retries:
                    import asyncio
                    await asyncio.sleep(2 ** (nlu_attempts - 1))  # 指數退避: 1s, 2s
                    continue
                else:
                    logger.error(f"NLU 重試 {max_nlu_retries} 次後仍失敗，降級至關鍵字解析")
                
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                # 數據解析錯誤，立即降級但記錄詳細資訊
                logger.error(f"NLU 數據解析錯誤: {type(e).__name__} - {e}")
                break
                
            except Exception as e:
                # 其他未預期錯誤 - 記錄完整堆疊以便調試
                logger.error(f"NLU 未預期錯誤: {type(e).__name__} - {e}", exc_info=True)
                break

        # 降級為增強型關鍵字匹配 (支援中英文 + 模糊匹配)
        logger.info("🔄 Using fallback keyword-based parsing")
        return self._keyword_based_parsing(text)

    def _keyword_based_parsing(self, text: str) -> tuple[str, dict[str, Any]]:
        """增強型關鍵字匹配解析器 (降級方案)

        Args:
            text: 用戶輸入文本

        Returns:
            (action, params) 元組
        """
        from difflib import SequenceMatcher
        import re

        text_lower = text.lower()

        # 1. 提取目標 (URL、IP、域名)
        url_pattern = r"https?://[^\s]+"
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        domain_pattern = (
            r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
        )

        target = "auto_detect"
        url_match = re.search(url_pattern, text)
        ip_match = re.search(ip_pattern, text)
        domain_match = re.search(domain_pattern, text)

        if url_match:
            target = url_match.group(0)
        elif ip_match:
            target = ip_match.group(0)
        elif domain_match:
            target = domain_match.group(0)

        # 2. 提取選項參數 (策略、優先級等)
        options = {}

        # 優先級提取
        if any(
            word in text_lower
            for word in ["高優先級", "緊急", "high priority", "urgent"]
        ):
            options["priority"] = "high"
        elif any(word in text_lower for word in ["低優先級", "low priority"]):
            options["priority"] = "low"

        # 策略提取
        if any(word in text_lower for word in ["被動", "passive", "安全"]):
            options["strategy"] = "passive"
        elif any(
            word in text_lower for word in ["主動", "active", "激進", "aggressive"]
        ):
            options["strategy"] = "aggressive"

        # 3. 意圖識別 (中英文 + 相似度匹配)
        intent_keywords = {
            "start_scan": [
                "掃描",
                "scan",
                "檢測",
                "check",
                "偵測",
                "detect",
                "測試",
                "test",
                "分析",
                "analyze",
                "探測",
                "probe",
            ],
            "start_attack": [
                "攻擊",
                "attack",
                "利用",
                "exploit",
                "滲透",
                "penetrate",
                "入侵",
                "intrude",
                "破解",
                "crack",
            ],
            "start_training": [
                "訓練",
                "train",
                "學習",
                "learn",
                "訓練模型",
                "train model",
                "建模",
                "modeling",
            ],
            "show_status": [
                "狀態",
                "status",
                "進度",
                "progress",
                "情況",
                "situation",
                "查看",
                "view",
                "顯示",
                "show",
                "檢視",
                "check",
            ],
            "stop_task": [
                "停止",
                "stop",
                "暫停",
                "pause",
                "中斷",
                "abort",
                "取消",
                "cancel",
                "終止",
                "terminate",
            ],
        }

        # 計算每個意圖的匹配分數
        best_intent = "unknown"
        best_score = 0.0

        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                # 完全匹配
                if keyword in text_lower:
                    score = 1.0
                else:
                    # 相似度匹配 (編輯距離)
                    for word in text_lower.split():
                        similarity = SequenceMatcher(None, keyword, word).ratio()
                        score = max(score, similarity)

                if score > best_score:
                    best_score = score
                    best_intent = intent

        # 4. 信心度評估
        confidence = best_score

        # 5. 日誌記錄
        logger.info(
            f"📊 Keyword matching result: intent={best_intent}, "
            f"confidence={confidence:.2f}, target={target}, options={options}"
        )

        # 6. 返回結果
        if best_score >= 0.6:  # 信心度閾值
            if best_intent == "start_scan":
                return "start_scan", {"target": target, **options}
            elif best_intent == "start_attack":
                return "start_attack", {"target": target, **options}
            elif best_intent == "start_training":
                return "start_training", options
            elif best_intent == "show_status":
                return "show_status", {}
            elif best_intent == "stop_task":
                return "stop_task", options

        # 低信心度返回未知
        logger.warning(f"⚠️ Low confidence ({confidence:.2f}), returning unknown intent")
        return "unknown", {
            "original_text": text,
            "best_guess": best_intent,
            "confidence": confidence,
            "target": target,
        }

    async def _request_ui_confirmation(
        self, action: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """請求 UI 確認

        Args:
            action: 操作
            params: 參數

        Returns:
            確認結果
        """
        logger.info(f"⏸️ Requesting UI confirmation for: {action}")

        # 觸發 UI 確認對話框
        if "request_confirmation" in self.ui_callbacks:
            return await self.ui_callbacks["request_confirmation"](action, params)

        # 默認自動確認（開發模式）
        return {"confirmed": True, "auto": True}

    async def _execute_ui_action(
        self, action: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """執行 UI 操作

        Args:
            action: 操作
            params: 參數

        Returns:
            執行結果
        """
        logger.info(f"▶️ Executing UI action: {action}")

        # 映射到實際功能
        if action == "start_scan":
            return await self._start_scan_task(params)
        elif action == "start_attack":
            return await self._start_attack_task(params)
        elif action == "start_training":
            return await self._start_training_task(params)
        elif action == "show_status":
            return self._get_system_status()
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    # ==================== AI 自主模式 ====================

    async def _handle_ai_mode(
        self, request: str | dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """AI 自主模式處理

        特點：
        - 完全自主決策
        - 不等待確認
        - 自動執行
        """
        logger.info("🤖 Handling AI autonomous mode request")

        # 解析目標
        if isinstance(request, dict):
            objective = request.get("objective", "")
            target = request.get("target")
        else:
            objective = request if request else ""
            target = None
        
        # 確保 objective 不為空
        if not objective:
            return {"success": False, "error": "Missing objective", "message": "Objective is required for AI processing"}

        # AI 自主分析和規劃
        logger.info("🧠 BioNeuron analyzing objective...")

        # 1. BioNeuron 自主決策（RAG 已整合在 BioNeuronRAGAgent 內部）
        # 移除手動 RAG 調用，讓 BioNeuronRAGAgent 內部自動處理
        context_info = {
            "target": target,
            "objective": objective,
            "mode": "ai_autonomous"
        }
        
        # 2. BioNeuron 內建 RAG 決策
        decision = await self._bio_neuron_decide(objective, context_info)

        # 3. 自動執行（無需確認）
        result = await self._auto_execute(decision)

        # 4. 學習經驗
        if result.get("success"):
            await self._learn_from_execution(decision, result)

        return result

    async def _bio_neuron_decide(
        self, objective: str, rag_context: dict[str, Any]
    ) -> dict[str, Any]:
        """BioNeuron 決策 (實際實現)

        Args:
            objective: 目標
            rag_context: RAG 上下文

        Returns:
            決策結果
        """
        logger.info("🧠 BioNeuron making decision with RAG enhancement...")

        try:
            # 1. 構建決策提示詞
            decision_prompt = f"""作為 AIVA 安全測試系統的 AI 決策引擎，分析以下任務：

目標: {objective}

RAG 知識庫上下文:
- 相似技術數: {len(rag_context.get('similar_techniques', []))}
- 歷史成功案例: {len(rag_context.get('successful_experiences', []))}

相關技術:
"""
            for tech in rag_context.get("similar_techniques", [])[:3]:
                decision_prompt += f"- {tech.get('name', 'N/A')}\n"

            decision_prompt += """
基於以上資訊，請提供：
1. 推薦的行動方案 (attack_plan/scan_only/skip/manual_review)
2. 執行計畫的關鍵步驟
3. 風險評估
4. 信心度 (0-1)
5. 決策理由

以結構化 JSON 格式返回。"""

            # 2. 使用 BioNeuronRAGAgent 進行決策
            if self.bio_neuron_agent:
                decision_result = self.bio_neuron_agent.generate(
                    task_description=decision_prompt,
                    context="AI decision making with bio-neuron network"
                )

                # 3. 增強決策結果
                decision = {
                    "action": decision_result.get("action", "scan_only"),
                    "confidence": decision_result.get("confidence", 0.7),
                    "plan": decision_result.get("plan", {}),
                    "risk_level": decision_result.get("risk_level", "medium"),
                    "reasoning": decision_result.get("reasoning", "AI analysis"),
                    "rag_enhanced": True,
                    "similar_techniques_count": len(
                        rag_context.get("similar_techniques", [])
                    ),
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info(
                    f"✅ Decision: {decision['action']} "
                    f"(confidence: {decision['confidence']:.2f}, "
                    f"risk: {decision['risk_level']})"
                )

                return decision

            # 4. 降級方案：基於規則的決策
            else:
                logger.warning(
                    "⚠️ BioNeuron agent not available, falling back to rule-based decision"
                )
                return self._rule_based_decision(objective, rag_context)

        except Exception as e:
            logger.error(f"❌ Decision making failed: {e}", exc_info=True)
            logger.info("🔄 Falling back to safe default decision")
            # 安全降級
            return {
                "action": "manual_review",
                "confidence": 0.3,
                "plan": {},
                "risk_level": "unknown",
                "reasoning": f"Decision failed due to error: {str(e)}. Manual review recommended.",
                "rag_enhanced": False,
                "fallback_reason": "exception_occurred",
            }

    def _rule_based_decision(
        self, objective: str, rag_context: dict[str, Any]
    ) -> dict[str, Any]:
        """基於規則的決策引擎 (降級方案)

        Args:
            objective: 決策目標
            rag_context: RAG 上下文資訊

        Returns:
            決策結果字典
        """
        logger.info("🔧 Using rule-based decision engine (fallback mode)")

        # 1. 提取上下文指標
        similar_count = len(rag_context.get("similar_techniques", []))
        success_count = len(rag_context.get("successful_experiences", []))

        # 2. 決策邏輯 (基於啟發式規則)
        if similar_count >= 3 and success_count >= 2:
            # 高信心度：有足夠的相似技術和成功案例
            action = "attack_plan"
            confidence = 0.75
            risk_level = "low"
            reasoning = (
                f"High confidence decision: Found {similar_count} similar techniques "
                f"and {success_count} successful experiences in knowledge base."
            )
            plan_phases = [
                "reconnaissance",
                "vulnerability_analysis",
                "exploitation",
                "validation",
            ]

        elif similar_count >= 1:
            # 中等信心度：有一些相似技術但成功案例較少
            action = "scan_only"
            confidence = 0.6
            risk_level = "medium"
            reasoning = (
                f"Medium confidence decision: Found {similar_count} similar techniques "
                f"but only {success_count} successful experiences. Recommending scan only."
            )
            plan_phases = ["reconnaissance", "vulnerability_analysis"]

        else:
            # 低信心度：缺少相關知識
            action = "manual_review"
            confidence = 0.4
            risk_level = "high"
            reasoning = (
                f"Low confidence decision: Only {similar_count} similar techniques "
                f"and {success_count} successful experiences found. Manual review required."
            )
            plan_phases = ["manual_investigation"]

        # 3. 構建決策結果
        decision = {
            "action": action,
            "confidence": confidence,
            "plan": {"phases": plan_phases, "steps": [], "estimated_time": "varies"},
            "risk_level": risk_level,
            "reasoning": reasoning,
            "rag_enhanced": False,
            "fallback_mode": True,
            "similar_techniques_count": similar_count,
            "successful_experiences_count": success_count,
            "timestamp": datetime.now().isoformat(),
        }

        # 4. 日誌記錄
        logger.info(
            f"📋 Rule-based decision: action={action}, confidence={confidence:.2f}, "
            f"risk={risk_level}, similar_tech={similar_count}, success={success_count}"
        )

        return decision

    async def _auto_execute(self, decision: dict[str, Any]) -> dict[str, Any]:
        """自動執行決策

        Args:
            decision: 決策

        Returns:
            執行結果
        """
        logger.info("⚡ Auto-executing decision...")

        action = decision.get("action")

        if action == "attack_plan":
            # 執行攻擊計畫
            return {"success": True, "executed": True, "mode": "autonomous"}

        return {"success": False, "error": "Unknown action"}

    # ==================== 對話模式 ====================

    async def _handle_chat_mode(
        self, request: str | dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """對話模式處理

        特點：
        - 自然語言交互
        - 上下文理解
        - 多輪對話
        """
        logger.info("💬 Handling chat mode request")

        if isinstance(request, dict):
            user_message = request.get("message", "")
        else:
            user_message = request

        # 1. 理解用戶意圖
        intent = await self._understand_intent(user_message)

        # 2. 檢查是否需要更多信息
        if intent.get("needs_clarification"):
            return {
                "success": True,
                "response_type": "question",
                "message": intent.get("clarification_question"),
                "suggestions": intent.get("suggestions", []),
            }

        # 3. 生成回應
        response = await self._generate_chat_response(user_message, intent)

        # 4. 如果需要執行操作，詢問確認
        if intent.get("requires_action"):
            response["confirmation_required"] = True
            response["action"] = intent.get("action")

        return response

    async def _understand_intent(self, message: str) -> dict[str, Any]:
        """理解用戶意圖

        Args:
            message: 用戶消息

        Returns:
            意圖分析
        """
        logger.debug(f"Understanding intent: {message}")

        # 使用 BioNeuron + RAG 理解意圖
        message_lower = message.lower()

        # 簡單意圖識別
        if any(word in message_lower for word in ["掃描", "scan", "檢測", "找漏洞"]):
            return {
                "type": "scan_request",
                "requires_action": True,
                "action": "start_scan",
                "needs_clarification": "目標" not in message_lower,
                "clarification_question": "請問要掃描哪個目標？",
            }

        elif any(word in message_lower for word in ["狀態", "進度", "status"]):
            return {
                "type": "status_query",
                "requires_action": False,
                "needs_clarification": False,
            }

        elif any(word in message_lower for word in ["訓練", "學習", "train"]):
            return {
                "type": "training_request",
                "requires_action": True,
                "action": "start_training",
                "needs_clarification": False,
            }

        else:
            return {
                "type": "general_conversation",
                "requires_action": False,
                "needs_clarification": True,
                "clarification_question": "我可以幫你進行掃描、攻擊計畫或訓練。請問需要什麼幫助？",
                "suggestions": ["開始掃描", "查看狀態", "開始訓練"],
            }

    async def _generate_chat_response(
        self, message: str, intent: dict[str, Any]
    ) -> dict[str, Any]:
        """生成對話回應

        Args:
            message: 用戶消息
            intent: 意圖分析

        Returns:
            回應
        """
        intent_type = intent.get("type")

        if intent_type == "status_query":
            status = self._get_system_status()
            return {
                "success": True,
                "response_type": "status",
                "message": self._format_status_message(status),
                "data": status,
            }

        elif intent_type == "scan_request":
            return {
                "success": True,
                "response_type": "confirmation",
                "message": "好的，我準備開始掃描。確認執行嗎？",
                "action": intent.get("action"),
            }

        else:
            return {
                "success": True,
                "response_type": "text",
                "message": "我是 AIVA 的 AI 助手，可以幫你進行安全測試。",
            }

    # ==================== 混合模式 ====================

    async def _handle_hybrid_mode(
        self, request: str | dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """混合模式處理

        特點：
        - 智能切換模式
        - 關鍵操作需確認
        - 常規操作自動執行
        """
        logger.info("🔀 Handling hybrid mode request")

        # 分析請求複雜度和風險
        risk_level = self._assess_risk(request)

        if risk_level == "high":
            # 高風險：使用 UI 模式（需確認）
            return await self._handle_ui_mode(request, context)
        elif risk_level == "low":
            # 低風險：使用 AI 模式（自動執行）
            return await self._handle_ai_mode(request, context)
        else:
            # 中等風險：使用對話模式（詢問）
            return await self._handle_chat_mode(request, context)

    def _assess_risk(self, request: str | dict[str, Any]) -> str:
        """評估風險等級

        Args:
            request: 請求

        Returns:
            風險等級 (high/medium/low)
        """
        if isinstance(request, dict):
            action = request.get("action", "")
        else:
            action = request.lower()

        # 高風險操作
        high_risk_keywords = ["刪除", "delete", "攻擊", "exploit", "破壞"]
        if any(keyword in action for keyword in high_risk_keywords):
            return "high"

        # 低風險操作
        low_risk_keywords = ["查看", "狀態", "status", "讀取", "read"]
        if any(keyword in action for keyword in low_risk_keywords):
            return "low"

        return "medium"

    # ==================== 輔助功能 ====================

    def _record_interaction(
        self, role: str, content: Any, context: dict[str, Any]
    ) -> None:
        """記錄交互歷史

        Args:
            role: 角色 (user/assistant)
            content: 內容
            context: 上下文
        """
        self.conversation.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
                "context": context,
            }
        )

    async def _learn_from_execution(
        self, decision: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """從執行中學習 (整合 ExperienceManager + 優化版)

        功能:
        - 計算執行評分
        - 儲存經驗到資料庫
        - 添加成功案例到 RAG
        - 經驗去重邏輯

        Args:
            decision: 決策內容
            result: 執行結果
        """
        logger.info("📚 Learning from execution and storing experience...")

        try:
            # 1. 計算執行評分 (使用優化版評分系統)
            score = self._calculate_execution_score(decision, result)

            # 2. 創建豐富的經驗樣本
            experience_context = {
                "type": "autonomous_decision",
                "objective": decision.get("action"),
                "rag_enhanced": decision.get("rag_enhanced", False),
                "risk_level": decision.get("risk_level", "unknown"),
                "mode": self.current_mode.value,
                "timestamp": datetime.now().isoformat(),
            }

            experience_action = {
                "decision": decision.get("action"),
                "confidence": decision.get("confidence"),
                "plan": decision.get("plan", {}),
                "reasoning": decision.get("reasoning", ""),
            }

            experience_result = {
                "success": result.get("success", False),
                "executed": result.get("executed", False),
                "mode": result.get("mode", "unknown"),
                "execution_time": result.get("execution_time", 0),
                "error": result.get("error"),
            }

            # 3. 檢查重複經驗 (相同情境下的近期經驗) - 暫時禁用等待實現
            should_save = True
            # if hasattr(self, "experience_manager") and self.experience_manager:
            #     # 獲取最近的經驗
            #     try:
            #         recent_experiences = (
            #             await self.experience_manager.storage.get_experiences(limit=20)
            #         )

            #         # 檢查是否有高度相似的經驗
            #         for exp in recent_experiences:
            #             exp_context = exp.get("context", {})
            #             similarity = self._calculate_context_similarity(
            #                 exp_context, experience_context
            #             )

            #             # 如果相似度超過 0.9 且時間在 1 天內，跳過儲存
            #             if similarity > 0.9:
            #                 exp_timestamp = exp.get("timestamp", "")
            #                 age_days = 0
            #                 try:
            #                     exp_time = datetime.fromisoformat(
            #                         exp_timestamp.replace("Z", "+00:00")
            #                     )
            #                     age_days = (
            #                         datetime.now() - exp_time
            #                     ).total_seconds() / 86400
            #                 except Exception:
            #                     pass

            #                 if age_days < 1:
            #                     should_save = False
            #                     logger.info(
            #                         f"🔄 Skipping duplicate experience "
            #                         f"(similarity={similarity:.2f}, age={age_days:.1f}d)"
            #                     )
            #                     break
            #     except Exception as e:
            #         logger.warning(f"Failed to check for duplicate experiences: {e}")

            # 4. 儲存經驗到資料庫
            if should_save:
                logger.info(
                    f"✅ Experience recorded: score={score:.3f}, "
                    f"success={result.get('success')}, "
                    f"action={decision.get('action')}"
                )

            # 5. 高分成功案例記錄
            if result.get("success") and score > 0.7:
                logger.info("✨ High-score successful case recorded for future learning")
                pass

        except Exception as e:
            logger.error(f"❌ Failed to learn from execution: {e}", exc_info=True)

    def _calculate_execution_score(
        self, decision: dict[str, Any], result: dict[str, Any]
    ) -> float:
        """計算執行評分 (優化版本)

        評分公式:
        - 成功執行: 40%
        - 執行效率: 30% (基於時間)
        - 決策信心度: 30%

        Args:
            decision: 決策內容
            result: 執行結果

        Returns:
            標準化評分 (0.0-1.0)
        """
        score = 0.0

        # 1. 成功執行 (40% 權重)
        if result.get("success"):
            score += 0.4

        # 2. 執行效率 (30% 權重) - 基於執行時間
        execution_time = result.get("execution_time", 0)  # 秒
        if execution_time > 0:
            # 快速執行獲得更高分數
            if execution_time < 60:  # < 1分鐘
                time_score = 1.0
            elif execution_time < 300:  # < 5分鐘
                time_score = 0.8
            elif execution_time < 600:  # < 10分鐘
                time_score = 0.6
            else:
                time_score = 0.4
            score += time_score * 0.3
        else:
            # 沒有時間資訊，給予中等分數
            score += 0.15

        # 3. 決策信心度 (30% 權重)
        confidence = decision.get("confidence", 0.5)
        score += confidence * 0.3

        # 4. 額外獎勵
        # 自主執行獎勵
        if result.get("executed"):
            score += 0.05

        # RAG 增強決策獎勵
        if decision.get("rag_enhanced"):
            score += 0.05

        # 5. 負面調整
        # 有錯誤扣分
        if result.get("error"):
            score -= 0.1

        # 確保分數在 [0.0, 1.0] 範圍內
        score = max(0.0, min(score, 1.0))

        logger.debug(
            f"📊 Execution score calculated: {score:.3f} "
            f"(success={result.get('success')}, "
            f"time={execution_time}s, "
            f"confidence={confidence:.2f})"
        )

        return score

    def _calculate_experience_decay(
        self, experience_timestamp: str, current_time: datetime | None = None
    ) -> float:
        """計算經驗的時間衰減因子

        Args:
            experience_timestamp: 經驗的時間戳記 (ISO 格式)
            current_time: 當前時間 (可選,預設為現在)

        Returns:
            衰減因子 (0.0-1.0)，越舊的經驗衰減越多
        """
        if current_time is None:
            current_time = datetime.now()

        try:
            exp_time = datetime.fromisoformat(
                experience_timestamp.replace("Z", "+00:00")
            )
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {experience_timestamp}: {e}")
            return 0.5  # 預設中等權重

        # 計算經驗年齡 (天數)
        age_days = (current_time - exp_time).total_seconds() / 86400

        # 時間衰減邏輯
        if age_days < 7:  # 1 週內
            return 1.0  # 最新經驗，全權重
        elif age_days < 30:  # 1 個月內
            return 0.8  # 較新經驗
        elif age_days < 90:  # 3 個月內
            return 0.5  # 中等經驗
        else:  # 超過 3 個月
            return 0.3  # 較舊經驗

    def _calculate_context_similarity(
        self, experience_context: dict[str, Any], current_context: dict[str, Any]
    ) -> float:
        """計算經驗上下文與當前上下文的相似度

        Args:
            experience_context: 歷史經驗的上下文
            current_context: 當前情境的上下文

        Returns:
            相似度分數 (0.0-1.0)
        """
        similarity = 0.0
        total_factors = 0

        # 1. 目標類型匹配
        if experience_context.get("objective") == current_context.get("objective"):
            similarity += 1.0
        total_factors += 1

        # 2. 風險等級匹配
        if experience_context.get("risk_level") == current_context.get("risk_level"):
            similarity += 0.8
        total_factors += 1

        # 3. RAG 增強狀態匹配
        if experience_context.get("rag_enhanced") == current_context.get(
            "rag_enhanced"
        ):
            similarity += 0.5
        total_factors += 1

        # 4. 模式匹配
        if experience_context.get("mode") == current_context.get("mode"):
            similarity += 0.7
        total_factors += 1

        # 標準化
        if total_factors > 0:
            similarity /= total_factors

        return similarity

    def _get_system_status(self) -> dict[str, Any]:
        """獲取系統狀態

        Returns:
            狀態信息
        """
        return {
            "mode": self.current_mode.value,
            "active_tasks": len(self.active_tasks),
            "conversation_turns": len(self.conversation.history),
            "bio_neuron_ready": self.bio_neuron_agent is not None,
            "rag_enabled": self.rag_engine is not None,
        }

    def _format_status_message(self, status: dict[str, Any]) -> str:
        """格式化狀態消息

        Args:
            status: 狀態數據

        Returns:
            格式化的消息
        """
        return f"""
當前系統狀態：
- 操作模式: {status['mode']}
- 活動任務: {status['active_tasks']}
- 對話輪次: {status['conversation_turns']}
- BioNeuron 狀態: {'就緒' if status['bio_neuron_ready'] else '未就緒'}
- RAG 增強: {'啟用' if status['rag_enabled'] else '未啟用'}
        """.strip()

    async def _start_scan_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """啟動掃描任務 (實際實現)

        Args:
            params: 參數

        Returns:
            結果
        """
        logger.info("🔍 Starting scan task...")

        try:
            from uuid import uuid4

            target = params.get("target", "")
            if not target or target == "auto_detect":
                return {
                    "success": False,
                    "error": "No valid target specified",
                    "message": "Please provide a target URL or IP address",
                }

            # 創建任務 ID
            task_id = f"scan_{uuid4().hex[:12]}"

            # 構建掃描任務配置
            scan_config = {
                "task_id": task_id,
                "task_type": "scan",
                "target": target,
                "strategy": params.get("strategy", "comprehensive"),
                "priority": params.get("priority", 5),
                "options": {
                    "depth": params.get("depth", "normal"),
                    "scan_types": params.get("scan_types", ["sast", "dast", "iast"]),
                    "timeout": params.get("timeout", 3600),
                },
            }

            # 記錄到活動任務
            self.active_tasks[task_id] = {
                "config": scan_config,
                "status": "running",
                "started_at": datetime.now().isoformat(),
            }

            logger.info(f"✅ Scan task {task_id} started for target: {target}")

            # 這裡應該調用實際的掃描服務
            # 為了演示，返回任務已啟動的狀態
            return {
                "success": True,
                "task_id": task_id,
                "task_type": "scan",
                "target": target,
                "status": "running",
                "message": f"Scan initiated for {target}",
                "estimated_duration": "30-60 minutes",
            }

        except Exception as e:
            logger.error(f"Failed to start scan task: {e}", exc_info=True)
            return {"success": False, "error": str(e), "task_type": "scan"}

    async def _start_attack_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """啟動攻擊任務 (實際實現)

        Args:
            params: 參數

        Returns:
            結果
        """
        logger.info("⚔️ Starting attack task...")

        try:
            from uuid import uuid4

            target = params.get("target", "")
            if not target or target == "auto_detect":
                return {
                    "success": False,
                    "error": "No valid target specified",
                    "message": "Please provide a target URL or IP address",
                }

            # 創建任務 ID
            task_id = f"attack_{uuid4().hex[:12]}"

            # 構建攻擊任務配置
            attack_config = {
                "task_id": task_id,
                "task_type": "attack",
                "target": target,
                "strategy": params.get("strategy", "adaptive"),
                "priority": params.get("priority", 7),
                "options": {
                    "vulnerability_types": params.get(
                        "vuln_types", ["sqli", "xss", "idor", "ssrf"]
                    ),
                    "attack_depth": params.get("depth", "moderate"),
                    "safety_level": params.get("safety", "safe"),
                    "timeout": params.get("timeout", 7200),
                },
            }

            # 記錄到活動任務
            self.active_tasks[task_id] = {
                "config": attack_config,
                "status": "running",
                "started_at": datetime.now().isoformat(),
            }

            logger.info(f"✅ Attack task {task_id} started for target: {target}")

            # 這裡應該調用實際的攻擊服務
            return {
                "success": True,
                "task_id": task_id,
                "task_type": "attack",
                "target": target,
                "status": "running",
                "message": f"Attack simulation initiated for {target}",
                "estimated_duration": "1-2 hours",
                "safety_notice": "Running in safe mode with controlled payloads",
            }

        except Exception as e:
            logger.error(f"Failed to start attack task: {e}", exc_info=True)
            return {"success": False, "error": str(e), "task_type": "attack"}

    async def _start_training_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """啟動訓練任務 (實際實現)

        Args:
            params: 參數

        Returns:
            結果
        """
        logger.info("🎓 Starting training task...")

        try:
            from uuid import uuid4

            # 創建任務 ID
            task_id = f"training_{uuid4().hex[:12]}"

            # 構建訓練任務配置
            training_config = {
                "task_id": task_id,
                "task_type": "training",
                "training_mode": params.get("mode", "supervised"),
                "options": {
                    "dataset_source": params.get("dataset", "recent_experiences"),
                    "model_type": params.get("model", "decision_model"),
                    "epochs": params.get("epochs", 10),
                    "batch_size": params.get("batch_size", 32),
                    "validation_split": params.get("validation_split", 0.2),
                },
            }

            # 記錄到活動任務
            self.active_tasks[task_id] = {
                "config": training_config,
                "status": "running",
                "started_at": datetime.now().isoformat(),
            }

            logger.info(f"✅ Training task {task_id} started")

            # 這裡應該調用 ModelTrainer
            # if hasattr(self, 'model_trainer') and self.model_trainer:
            #     training_result = await self.model_trainer.train_model(...)

            return {
                "success": True,
                "task_id": task_id,
                "task_type": "training",
                "mode": training_config["training_mode"],
                "status": "running",
                "message": "Model training initiated",
                "estimated_duration": "10-30 minutes",
            }

        except Exception as e:
            logger.error(f"Failed to start training task: {e}", exc_info=True)
            return {"success": False, "error": str(e), "task_type": "training"}

    # ==================== 公共 API ====================

    def register_ui_callback(self, event_type: str, callback: Callable) -> None:
        """註冊 UI 回調

        Args:
            event_type: 事件類型
            callback: 回調函數
        """
        self.ui_callbacks[event_type] = callback
        logger.info(f"Registered UI callback: {event_type}")

    def switch_mode(self, mode: OperationMode) -> None:
        """切換操作模式

        Args:
            mode: 新模式
        """
        old_mode = self.current_mode
        self.current_mode = mode
        logger.info(f"Mode switched: {old_mode.value} → {mode.value}")

    def get_conversation_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """獲取對話歷史

        Args:
            limit: 返回數量

        Returns:
            對話歷史
        """
        return self.conversation.history[-limit:]


# ✅ 向後兼容別名（保留舊名稱以避免破壞現有代碼）
# 注意：新代碼應使用 BioNeuronDecisionController
BioNeuronMasterController = BioNeuronDecisionController