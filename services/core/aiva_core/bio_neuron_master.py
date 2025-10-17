"""
BioNeuron Master Controller - BioNeuronRAGAgent 主控系統

支持三種操作模式：
1. UI Mode - 圖形化介面控制
2. AI Mode - 完全自主決策
3. Chat Mode - 自然語言對話

架構：
┌─────────────────────────────────────────┐
│      BioNeuronRAGAgent (主腦)           │
│  - 決策核心 (500萬參數神經網路)          │
│  - RAG 知識檢索                          │
│  - 抗幻覺機制                            │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐   ┌───▼────┐
│UI Mode│   │AI Mode  │   │Chat Mode│
│ 介面  │   │ 自主    │   │ 對話   │
└───────┘   └─────────┘   └────────┘
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum
import logging
from typing import Any

from aiva_core.ai_engine import BioNeuronRAGAgent
from aiva_core.rag import RAGEngine

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


class BioNeuronMasterController:
    """BioNeuron 主控系統

    統一管理 BioNeuronRAGAgent 的三種操作模式
    """

    def __init__(
        self,
        codebase_path: str = "/workspaces/AIVA",
        default_mode: OperationMode = OperationMode.HYBRID,
    ) -> None:
        """初始化主控系統

        Args:
            codebase_path: 代碼庫路徑
            default_mode: 默認操作模式
        """
        logger.info("[BRAIN] Initializing BioNeuron Master Controller...")

        # === 核心 AI 主腦 ===
        self.bio_neuron_agent = BioNeuronRAGAgent(
            codebase_path=codebase_path,
            enable_planner=True,
            enable_tracer=True,
            enable_experience=True,
        )

        # === RAG 增強（整合到主腦） ===
        from aiva_core.rag import KnowledgeBase, VectorStore

        vector_store = VectorStore(backend="memory")
        knowledge_base = KnowledgeBase(vector_store=vector_store)
        self.rag_engine = RAGEngine(knowledge_base=knowledge_base)

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

        logger.info(f"[OK] Master Controller initialized in {default_mode.value} mode")
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

        logger.info(f"[U+1F4E5] Processing request in {mode.value} mode")

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
        logger.info("[UI] Handling UI mode request")

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
        """解析 UI 命令

        Args:
            text: 用戶輸入

        Returns:
            (action, params)
        """
        # 使用 BioNeuron 理解用戶意圖
        # TODO: 實際 NLU 實現
        logger.debug(f"Parsing UI command: {text}")

        # 簡單的關鍵字匹配
        text_lower = text.lower()

        if "掃描" in text_lower or "scan" in text_lower:
            return "start_scan", {"target": "auto_detect"}
        elif "攻擊" in text_lower or "attack" in text_lower:
            return "start_attack", {"target": "auto_detect"}
        elif "訓練" in text_lower or "train" in text_lower:
            return "start_training", {}
        elif "狀態" in text_lower or "status" in text_lower:
            return "show_status", {}
        else:
            return "unknown", {"original_text": text}

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
        logger.info(f"[U+23F8][U+FE0F] Requesting UI confirmation for: {action}")

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
        logger.info(f"[U+25B6][U+FE0F] Executing UI action: {action}")

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
        logger.info("[AI] Handling AI autonomous mode request")

        # 解析目標
        if isinstance(request, dict):
            objective = request.get("objective")
            target = request.get("target")
        else:
            objective = request
            target = None

        # AI 自主分析和規劃
        logger.info("[BRAIN] BioNeuron analyzing objective...")

        # 1. 使用 RAG 獲取相關知識
        if target:
            rag_context = self.rag_engine.enhance_attack_plan(
                target=target,
                objective=objective,
            )
        else:
            rag_context = {}

        # 2. BioNeuron 決策
        # TODO: 實際決策邏輯
        decision = await self._bio_neuron_decide(objective, rag_context)

        # 3. 自動執行（無需確認）
        result = await self._auto_execute(decision)

        # 4. 學習經驗
        if result.get("success"):
            await self._learn_from_execution(decision, result)

        return result

    async def _bio_neuron_decide(
        self, objective: str, rag_context: dict[str, Any]
    ) -> dict[str, Any]:
        """BioNeuron 決策

        Args:
            objective: 目標
            rag_context: RAG 上下文

        Returns:
            決策結果
        """
        logger.info("[BRAIN] BioNeuron making decision...")

        # 使用 BioNeuronRAGAgent 的決策核心
        # TODO: 整合實際決策
        decision = {
            "action": "attack_plan",
            "confidence": 0.85,
            "plan": None,  # TODO: 實際計畫
            "reasoning": "Based on RAG context and neural decision",
        }

        logger.info(
            f"Decision made: {decision['action']} "
            f"(confidence: {decision['confidence']:.2%})"
        )

        return decision

    async def _auto_execute(self, decision: dict[str, Any]) -> dict[str, Any]:
        """自動執行決策

        Args:
            decision: 決策

        Returns:
            執行結果
        """
        logger.info("[FAST] Auto-executing decision...")

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
        logger.info("[CHAT] Handling chat mode request")

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
        # TODO: 實際 NLU 實現

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
        logger.info("[MIX] Handling hybrid mode request")

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
        """從執行中學習

        Args:
            decision: 決策
            result: 結果
        """
        logger.info("[DOCS] Learning from execution...")

        # 創建經驗樣本
        # TODO: 整合 ExperienceManager

        # 添加到 RAG 知識庫
        # TODO: 添加知識

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
        """啟動掃描任務

        Args:
            params: 參數

        Returns:
            結果
        """
        logger.info("[SEARCH] Starting scan task...")
        # TODO: 實際掃描邏輯
        return {"success": True, "task_type": "scan", "status": "started"}

    async def _start_attack_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """啟動攻擊任務

        Args:
            params: 參數

        Returns:
            結果
        """
        logger.info("[U+2694][U+FE0F] Starting attack task...")
        # TODO: 實際攻擊邏輯
        return {"success": True, "task_type": "attack", "status": "started"}

    async def _start_training_task(self, params: dict[str, Any]) -> dict[str, Any]:
        """啟動訓練任務

        Args:
            params: 參數

        Returns:
            結果
        """
        logger.info("[U+1F393] Starting training task...")
        # TODO: 實際訓練邏輯
        return {"success": True, "task_type": "training", "status": "started"}

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
