"""
AIVA Common AI Dialog Assistant - 對話助手組件

此文件提供符合 aiva_common 規範的對話助手實現，
支援多輪對話、上下文管理和智能響應生成。

設計特點:
- 實現 IDialogAssistant 介面
- 整合現有 aiva_common 消息 Schema
- 支援多輪對話和上下文追蹤
- 異步消息處理機制
- 對話歷史管理
- 智能意圖識別

架構位置:
- 屬於 Common 層的共享組件
- 支援五大模組架構的對話需求
- 與 MessageHeader、AivaMessage 等現有 Schema 整合
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..enums import ModuleName
from ..schemas import AIVARequest, AIVAResponse
from .interfaces import IDialogAssistant

logger = logging.getLogger(__name__)


class DialogIntent(Enum):
    """對話意圖枚舉"""

    # 基本對話
    GREETING = "greeting"
    FAREWELL = "farewell"
    SMALL_TALK = "small_talk"

    # 查詢類
    INFORMATION_REQUEST = "information_request"
    STATUS_INQUIRY = "status_inquiry"
    HELP_REQUEST = "help_request"

    # 操作類
    COMMAND_EXECUTION = "command_execution"
    TASK_CREATION = "task_creation"
    CONFIGURATION_CHANGE = "configuration_change"

    # 安全相關
    SECURITY_QUERY = "security_query"
    VULNERABILITY_REPORT = "vulnerability_report"
    THREAT_ANALYSIS = "threat_analysis"

    # 系統相關
    SYSTEM_STATUS = "system_status"
    LOG_ANALYSIS = "log_analysis"
    PERFORMANCE_QUERY = "performance_query"

    # 特殊意圖
    UNCLEAR = "unclear"
    ERROR_HANDLING = "error_handling"
    ESCALATION = "escalation"


class DialogTurn(BaseModel):
    """對話輪次"""

    turn_id: str = Field(default_factory=lambda: f"turn_{uuid4().hex[:8]}")
    user_message: str
    assistant_response: str
    intent: DialogIntent
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context_used: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DialogSession(BaseModel):
    """對話會話"""

    session_id: str = Field(default_factory=lambda: f"session_{uuid4().hex[:12]}")
    user_id: str
    module_name: ModuleName
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))
    turns: list[DialogTurn] = Field(default_factory=list)  # type: ignore
    context: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    session_metadata: dict[str, Any] = Field(default_factory=dict)

    def add_turn(self, turn: DialogTurn) -> None:
        """添加對話輪次"""
        self.turns.append(turn)  # type: ignore
        self.last_activity = datetime.now(UTC)

    def get_recent_turns(self, count: int = 5) -> list[DialogTurn]:
        """獲取最近的對話輪次"""
        return self.turns[-count:] if self.turns else []

    def update_context(self, new_context: dict[str, Any]) -> None:
        """更新會話上下文"""
        self.context.update(new_context)
        self.last_activity = datetime.now(UTC)

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """檢查會話是否過期"""
        return (datetime.now(UTC) - self.last_activity).total_seconds() > (
            timeout_minutes * 60
        )


class DialogConfig(BaseModel):
    """對話助手配置"""

    # 會話管理
    session_timeout_minutes: int = Field(default=30, ge=1, le=1440)
    max_sessions_per_user: int = Field(default=5, ge=1, le=50)
    max_turns_per_session: int = Field(default=100, ge=10, le=1000)

    # 響應生成
    max_response_length: int = Field(default=2000, ge=100, le=8000)
    response_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    enable_context_memory: bool = True

    # 意圖識別
    intent_confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    enable_intent_logging: bool = True
    fallback_intent: DialogIntent = DialogIntent.UNCLEAR

    # 內容過濾
    enable_content_filter: bool = True
    blocked_words: set[str] = Field(default_factory=set)
    max_consecutive_unclear: int = Field(default=3, ge=1, le=10)

    # 系統整合
    enable_system_integration: bool = True
    allowed_modules: set[str] = Field(
        default_factory=lambda: {
            ModuleName.INTEGRATION,
            ModuleName.CORE,
            "scan_features",
            ModuleName.COMMON,
            "services",
        }
    )


class IntentClassifier:
    """意圖分類器"""

    def __init__(self):
        # 基於關鍵詞的簡單意圖識別規則
        self.intent_patterns = {
            DialogIntent.GREETING: [
                r"\b(hello|hi|hey|good\s+(morning|afternoon|evening)|greetings)\b",
                r"\b(你好|嗨|早安|午安|晚安|問好)\b",
            ],
            DialogIntent.FAREWELL: [
                r"\b(bye|goodbye|farewell|see\s+you|talk\s+later)\b",
                r"\b(再見|拜拜|告別|回頭見)\b",
            ],
            DialogIntent.INFORMATION_REQUEST: [
                r"\b(what|how|when|where|why|tell\s+me|explain|describe)\b",
                r"\b(什麼|如何|何時|哪裡|為什麼|告訴我|解釋|描述)\b",
            ],
            DialogIntent.STATUS_INQUIRY: [
                r"\b(status|state|condition|health|running|active)\b",
                r"\b(狀態|情況|條件|健康|運行|活躍)\b",
            ],
            DialogIntent.HELP_REQUEST: [
                r"\b(help|assist|support|guide|tutorial|how\s+to)\b",
                r"\b(幫助|協助|支援|指導|教程|如何)\b",
            ],
            DialogIntent.COMMAND_EXECUTION: [
                r"\b(run|execute|start|stop|restart|kill|launch)\b",
                r"\b(運行|執行|開始|停止|重啟|終止|啟動)\b",
            ],
            DialogIntent.SECURITY_QUERY: [
                r"\b(security|vulnerability|threat|attack|breach|exploit)\b",
                r"\b(安全|漏洞|威脅|攻擊|入侵|利用)\b",
            ],
            DialogIntent.SYSTEM_STATUS: [
                r"\b(system|server|service|process|memory|cpu|disk)\b",
                r"\b(系統|伺服器|服務|進程|記憶體|處理器|硬碟)\b",
            ],
        }

    def classify_intent(self, message: str) -> tuple[DialogIntent, float]:
        """分類對話意圖

        Args:
            message: 用戶消息

        Returns:
            意圖和信心度
        """
        message_lower = message.lower()
        best_intent = DialogIntent.UNCLEAR
        best_confidence = 0.0

        for intent, patterns in self.intent_patterns.items():
            confidence = 0.0

            for pattern in patterns:
                matches = re.findall(pattern, message_lower, re.IGNORECASE)
                if matches:
                    # 根據匹配數量和長度計算信心度
                    confidence += len(matches) * 0.3  # type: ignore
                    confidence += min(
                        sum(
                            len(match) if isinstance(match, str) else len(" ".join(match))  # type: ignore
                            for match in matches
                        )
                        / len(message),  # type: ignore
                        0.5,
                    )  # type: ignore

            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent

        # 標準化信心度
        best_confidence = min(best_confidence, 1.0)

        return best_intent, best_confidence


class ResponseGenerator:
    """響應生成器"""

    def __init__(self, config: DialogConfig):
        self.config = config

        # 預定義響應模板
        self.response_templates = {
            DialogIntent.GREETING: [
                "你好！我是 AIVA 助手，很高興為您服務。有什麼可以幫助您的嗎？",
                "Hello! I'm AIVA Assistant. How can I help you today?",
                "嗨！歡迎使用 AIVA 系統。請告訴我您需要什麼協助。",
            ],
            DialogIntent.FAREWELL: [
                "再見！如果還有其他問題，隨時找我。",
                "Goodbye! Feel free to reach out if you need any assistance.",
                "謝謝使用 AIVA！期待下次為您服務。",
            ],
            DialogIntent.HELP_REQUEST: [
                "我可以幫助您進行安全掃描、查詢系統狀態、分析威脅等。您想了解哪個方面？",
                "I can assist with security scanning, system status, threat analysis, and more. What would you like to know?",
                "AIVA 提供多種功能：漏洞掃描、威脅分析、系統監控等。請告訴我您的具體需求。",
            ],
            DialogIntent.UNCLEAR: [
                "抱歉，我不太理解您的意思。能否請您更詳細地說明？",
                "I'm not sure I understand. Could you please clarify what you're looking for?",
                "似乎您的問題不太清楚。請提供更多詳細信息，我會盡力幫助您。",
            ],
        }

    async def generate_response(
        self,
        message: str,
        intent: DialogIntent,
        confidence: float,
        context: dict[str, Any],
        session: DialogSession,
    ) -> str:
        """生成響應

        Args:
            message: 用戶消息
            intent: 識別的意圖
            confidence: 信心度
            context: 上下文信息
            session: 對話會話

        Returns:
            生成的響應
        """
        try:
            # 檢查信心度
            if confidence < self.config.intent_confidence_threshold:
                intent = self.config.fallback_intent

            # 獲取響應模板
            templates = self.response_templates.get(
                intent, self.response_templates[DialogIntent.UNCLEAR]
            )

            # 選擇響應模板 (簡單輪換)
            template_index = len(session.turns) % len(templates)  # type: ignore
            base_response = templates[template_index]

            # 根據上下文個性化響應
            personalized_response = await self._personalize_response(
                base_response, message, intent, context, session
            )

            # 應用長度限制
            if len(personalized_response) > self.config.max_response_length:  # type: ignore
                personalized_response = (
                    personalized_response[: self.config.max_response_length - 3] + "..."
                )

            return personalized_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "抱歉，處理您的請求時發生錯誤。請稍後再試。"

    async def _personalize_response(
        self,
        base_response: str,
        message: str,
        intent: DialogIntent,
        context: dict[str, Any],
        session: DialogSession,
    ) -> str:
        """個性化響應"""
        response = base_response

        # 添加用戶特定信息
        if "user_name" in context:
            response = response.replace("您", context["user_name"])

        # 添加上下文相關信息
        if intent == DialogIntent.STATUS_INQUIRY and "last_scan_time" in context:
            response += f" 上次掃描時間：{context['last_scan_time']}"

        # 添加會話歷史相關信息
        if len(session.turns) > 0:  # type: ignore
            last_intent = session.turns[-1].intent
            if last_intent == intent:
                response = "讓我繼續為您提供相關信息。" + response

        return response


class AIVADialogAssistant(IDialogAssistant):
    """AIVA 對話助手實現

    此類提供符合 aiva_common 規範的對話助手功能，
    支援多輪對話、意圖識別和智能響應生成。
    """

    def __init__(
        self,
        config: DialogConfig | None = None,
        module_name: ModuleName = ModuleName.COMMON,
    ):
        """初始化對話助手

        Args:
            config: 對話助手配置
            module_name: 所屬模組名稱
        """
        self.config = config or DialogConfig()
        self.module_name = module_name

        # 核心組件
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator(self.config)

        # 會話管理
        self.active_sessions: dict[str, DialogSession] = {}
        self.user_sessions: dict[str, list[str]] = {}  # user_id -> [session_ids]

        # 統計和監控
        self.total_turns = 0
        self.intent_stats: dict[DialogIntent, int] = {}
        self.start_time = datetime.now(UTC)

        # 清理任務
        self._cleanup_task: asyncio.Task[Any] | None = None
        self._start_cleanup_task()

        logger.info(f"AIVADialogAssistant initialized for module {module_name.value}")

    async def send_message(
        self, user_id: str, message: str, context: dict[str, Any] | None = None
    ) -> str:
        """發送消息並獲取響應

        Args:
            user_id: 用戶 ID
            message: 用戶消息
            context: 上下文信息

        Returns:
            助手響應
        """
        try:
            # 獲取或創建會話
            session = await self._get_or_create_session(user_id)

            # 更新會話上下文
            if context:
                session.update_context(context)

            # 內容過濾
            if self.config.enable_content_filter:
                filtered_message = self._filter_content(message)
                if filtered_message != message:
                    logger.warning(f"Content filtered for user {user_id}")
                    return "抱歉，您的消息包含不當內容。請重新表達。"

            # 意圖識別
            intent, confidence = self.intent_classifier.classify_intent(message)

            # 更新統計
            self.intent_stats[intent] = self.intent_stats.get(intent, 0) + 1

            # 生成響應
            response = await self.response_generator.generate_response(
                message, intent, confidence, session.context, session
            )

            # 創建對話輪次
            turn = DialogTurn(
                user_message=message,
                assistant_response=response,
                intent=intent,
                confidence=confidence,
                context_used=session.context.copy(),
                metadata={
                    "processing_time": datetime.now(UTC).isoformat(),
                    "session_id": session.session_id,
                },
            )

            # 添加到會話
            session.add_turn(turn)
            self.total_turns += 1

            # 檢查連續不清楚意圖
            await self._check_consecutive_unclear(session)

            logger.info(
                f"Dialog turn completed: user={user_id}, intent={intent.value}, "
                f"confidence={confidence:.2f}"
            )

            return response

        except Exception as e:
            logger.error(
                f"Error processing message from user {user_id}: {e}", exc_info=True
            )
            return "抱歉，處理您的消息時發生錯誤。請稍後再試。"

    async def get_session_history(
        self, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """獲取會話歷史

        Args:
            user_id: 用戶 ID
            session_id: 會話 ID (可選，默認為當前活躍會話)

        Returns:
            對話歷史列表
        """
        try:
            if session_id:
                session = self.active_sessions.get(session_id)
                if not session or session.user_id != user_id:
                    return []
            else:
                session = await self._get_current_session(user_id)
                if not session:
                    return []

            return [
                {
                    "turn_id": turn.turn_id,
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "intent": turn.intent.value,
                    "confidence": turn.confidence,
                    "timestamp": turn.timestamp.isoformat(),
                }
                for turn in session.turns
            ]

        except Exception as e:
            logger.error(f"Error getting session history for user {user_id}: {e}")
            return []

    async def clear_session(
        self, user_id: str, session_id: str | None = None
    ) -> bool:
        """清除會話

        Args:
            user_id: 用戶 ID
            session_id: 會話 ID (可選)

        Returns:
            是否成功清除
        """
        try:
            if session_id:
                session = self.active_sessions.get(session_id)
                if session and session.user_id == user_id:
                    del self.active_sessions[session_id]
                    if user_id in self.user_sessions:
                        self.user_sessions[user_id] = [
                            sid
                            for sid in self.user_sessions[user_id]
                            if sid != session_id
                        ]
                    logger.info(f"Session {session_id} cleared for user {user_id}")
                    return True
            else:
                # 清除用戶所有會話
                if user_id in self.user_sessions:
                    for sid in self.user_sessions[user_id]:
                        self.active_sessions.pop(sid, None)  # type: ignore
                    del self.user_sessions[user_id]
                    logger.info(f"All sessions cleared for user {user_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error clearing session for user {user_id}: {e}")
            return False

    def get_dialog_statistics(self) -> dict[str, Any]:
        """獲取對話統計信息

        Returns:
            統計信息字典
        """
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        return {
            "total_sessions": len(self.active_sessions),  # type: ignore
            "total_turns": self.total_turns,
            "uptime_seconds": uptime,
            "average_turns_per_session": (
                self.total_turns / len(self.active_sessions)  # type: ignore
                if self.active_sessions
                else 0
            ),
            "intent_distribution": {
                intent.value: count for intent, count in self.intent_stats.items()
            },
            "active_users": len(self.user_sessions),  # type: ignore
            "configuration": {
                "session_timeout_minutes": self.config.session_timeout_minutes,
                "max_sessions_per_user": self.config.max_sessions_per_user,
                "intent_confidence_threshold": self.config.intent_confidence_threshold,
            },
        }

    async def create_dialog_request(
        self,
        user_id: str,
        message: str,
        target_module: ModuleName,
        request_type: str = "dialog_message",
    ) -> AIVARequest:
        """創建對話請求 (用於模組間通信)

        Args:
            user_id: 用戶 ID
            message: 消息內容
            target_module: 目標模組
            request_type: 請求類型

        Returns:
            AIVA 請求對象
        """
        request_id = f"dialog_req_{uuid4().hex[:12]}"

        return AIVARequest(
            request_id=request_id,
            source_module=self.module_name.value,
            target_module=target_module.value,
            request_type=request_type,
            payload={
                "user_id": user_id,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            timestamp=datetime.now(UTC).isoformat(),
        )

    async def handle_dialog_response(self, response: AIVAResponse, user_id: str) -> str:
        """處理對話響應 (來自其他模組)

        Args:
            response: AIVA 響應對象
            user_id: 用戶 ID

        Returns:
            處理後的響應消息
        """
        try:
            if not response.success:
                error_msg = response.error_message or "未知錯誤"
                return f"抱歉，處理您的請求時發生錯誤：{error_msg}"

            if response.payload and "message" in response.payload:
                return response.payload["message"]

            return "請求已處理完成。"

        except Exception as e:
            logger.error(f"Error handling dialog response: {e}")
            return "抱歉，處理響應時發生錯誤。"

    async def _get_or_create_session(self, user_id: str) -> DialogSession:
        """獲取或創建用戶會話"""
        # 查找現有活躍會話
        current_session = await self._get_current_session(user_id)
        if current_session and not current_session.is_expired(
            self.config.session_timeout_minutes
        ):
            return current_session

        # 檢查用戶會話數量限制
        if user_id in self.user_sessions:
            user_session_ids = self.user_sessions[user_id]
            if len(user_session_ids) >= self.config.max_sessions_per_user:  # type: ignore
                # 移除最舊的會話
                oldest_session_id = user_session_ids[0]
                self.active_sessions.pop(oldest_session_id, None)  # type: ignore
                user_session_ids.remove(oldest_session_id)  # type: ignore

        # 創建新會話
        session = DialogSession(user_id=user_id, module_name=self.module_name)

        # 記錄會話
        self.active_sessions[session.session_id] = session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session.session_id)  # type: ignore

        logger.info(
            f"New dialog session created: {session.session_id} for user {user_id}"
        )
        return session

    async def _get_current_session(self, user_id: str) -> DialogSession | None:
        """獲取用戶當前會話"""
        if user_id not in self.user_sessions:
            return None

        user_session_ids = self.user_sessions[user_id]
        if not user_session_ids:
            return None

        # 返回最新的會話
        latest_session_id = user_session_ids[-1]
        return self.active_sessions.get(latest_session_id)

    def _filter_content(self, message: str) -> str:
        """內容過濾"""
        if not self.config.blocked_words:
            return message

        filtered_message = message
        for word in self.config.blocked_words:
            filtered_message = re.sub(
                re.escape(word),
                "*" * len(word),  # type: ignore
                filtered_message,
                flags=re.IGNORECASE,
            )

        return filtered_message

    async def _check_consecutive_unclear(self, session: DialogSession) -> None:
        """檢查連續不清楚意圖"""
        if len(session.turns) < self.config.max_consecutive_unclear:  # type: ignore
            return

        recent_turns = session.get_recent_turns(self.config.max_consecutive_unclear)
        unclear_count = sum(
            1 for turn in recent_turns if turn.intent == DialogIntent.UNCLEAR
        )

        if unclear_count >= self.config.max_consecutive_unclear:
            # 可以在這裡觸發升級機制
            logger.warning(
                f"User {session.user_id} has {unclear_count} consecutive unclear intents. "
                f"Consider escalation."
            )

    def _start_cleanup_task(self) -> None:
        """啟動清理任務"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self) -> None:
        """定期清理過期會話"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分鐘清理一次

                expired_sessions: list[str] = []
                for session_id, session in self.active_sessions.items():
                    if session.is_expired(self.config.session_timeout_minutes):
                        expired_sessions.append(session_id)  # type: ignore

                # 清理過期會話
                for session_id in expired_sessions:  # type: ignore
                    session = self.active_sessions.pop(session_id, None)  # type: ignore
                    if session:
                        user_session_ids = self.user_sessions.get(session.user_id, [])
                        if session_id in user_session_ids:
                            user_session_ids.remove(session_id)  # type: ignore

                        logger.info(f"Expired session cleaned up: {session_id}")

                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def cleanup(self) -> None:
        """清理資源"""
        try:
            # 取消清理任務
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # 清理會話
            self.active_sessions.clear()
            self.user_sessions.clear()

            logger.info("AIVADialogAssistant cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # ============================================================================
    # IDialogAssistant Interface Implementation (介面方法實現)
    # ============================================================================

    async def process_user_input(
        self,
        user_input: str,
        user_id: str = "default",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        處理使用者輸入並產生回應 (實現 IDialogAssistant 介面)

        Args:
            user_input: 使用者輸入文字
            user_id: 使用者識別碼
            context: 對話上下文

        Returns:
            包含意圖、回應和可執行動作的字典
        """
        try:
            # 使用現有的 send_message 方法處理
            response = await self.send_message(user_id, user_input, context)

            # 獲取當前會話以提取更多信息
            session_ids = self.user_sessions.get(user_id, [])
            current_session = None
            if session_ids:
                current_session = self.active_sessions.get(session_ids[-1])

            # 提取最後一輪對話的意圖信息
            intent_info = DialogIntent.UNCLEAR
            confidence = 0.0
            if current_session and current_session.turns:
                last_turn = current_session.turns[-1]
                intent_info = last_turn.intent
                confidence = last_turn.confidence

            # 構建返回結果
            result = {
                "intent": intent_info.value,
                "response": response,
                "confidence": confidence,
                "actions": [],  # 可執行動作列表 (預留接口)
                "context": current_session.context if current_session else {},
                "metadata": {
                    "user_id": user_id,
                    "session_id": (
                        current_session.session_id if current_session else None
                    ),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "module": self.module_name.value,
                },
            }

            # 根據意圖添加可執行動作
            if intent_info == DialogIntent.COMMAND_EXECUTION:
                result["actions"].append(  # type: ignore
                    {
                        "type": "command",
                        "description": "Execute system command",
                        "parameters": {"command": user_input},
                    }
                )
            elif intent_info == DialogIntent.STATUS_INQUIRY:
                result["actions"].append(  # type: ignore
                    {
                        "type": "status_check",
                        "description": "Check system status",
                        "parameters": {},
                    }
                )
            elif intent_info == DialogIntent.HELP_REQUEST:
                result["actions"].append(  # type: ignore
                    {
                        "type": "help",
                        "description": "Provide help information",
                        "parameters": {"topic": user_input},
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            return {
                "intent": DialogIntent.ERROR_HANDLING.value,
                "response": "處理您的請求時發生錯誤，請稍後再試。",
                "confidence": 0.0,
                "actions": [],
                "context": {},
                "metadata": {
                    "error": str(e),
                    "user_id": user_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            }

    async def identify_intent(self, user_input: str) -> tuple[str, dict[str, Any]]:
        """
        識別使用者意圖 (實現 IDialogAssistant 介面)

        Args:
            user_input: 使用者輸入

        Returns:
            (意圖類型, 提取的參數)
        """
        try:
            # 使用內部意圖分類器
            intent, confidence = self.intent_classifier.classify_intent(user_input)

            # 提取參數 (簡化實現，可根據需要擴展)
            parameters = {
                "confidence": confidence,
                "original_input": user_input,
                "intent_category": intent.value,
            }

            # 根據不同意圖類型提取特定參數
            if intent == DialogIntent.COMMAND_EXECUTION:
                # 提取命令參數
                command_match = re.search(r"執行|運行|啟動\s+(.*)", user_input)
                if command_match:
                    parameters["command"] = command_match.group(1).strip()

            elif intent == DialogIntent.STATUS_INQUIRY:
                # 提取狀態查詢對象
                status_match = re.search(
                    r"狀態|情況.*?(系統|服務|進程|網路)", user_input
                )
                if status_match:
                    parameters["status_target"] = status_match.group(1)

            elif intent == DialogIntent.INFORMATION_REQUEST:
                # 提取信息查詢主題
                info_keywords = re.findall(r"(什麼|如何|為什麼|何時|哪裡)", user_input)
                if info_keywords:
                    parameters["query_type"] = info_keywords[0]

            elif intent == DialogIntent.HELP_REQUEST:
                # 提取幫助主題
                help_match = re.search(r"幫助|協助.*?(.*)", user_input)
                if help_match:
                    parameters["help_topic"] = help_match.group(1).strip()

            return intent.value, parameters

        except Exception as e:
            logger.error(f"Error identifying intent: {e}", exc_info=True)
            return DialogIntent.ERROR_HANDLING.value, {
                "error": str(e),
                "confidence": 0.0,
                "original_input": user_input,
            }

    def get_conversation_history(
        self, limit: int = 10, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        獲取對話歷史 (實現 IDialogAssistant 介面)

        Args:
            limit: 返回記錄數限制
            user_id: 指定使用者ID (None 表示所有使用者)

        Returns:
            對話歷史記錄列表
        """
        try:
            history = []

            if user_id:
                # 獲取特定用戶的對話歷史
                session_ids = self.user_sessions.get(user_id, [])
                for session_id in session_ids:  # type: ignore
                    session = self.active_sessions.get(session_id)
                    if session:
                        for turn in session.turns:
                            history.append(  # type: ignore
                                {
                                    "turn_id": turn.turn_id,
                                    "user_id": user_id,
                                    "session_id": session_id,
                                    "user_message": turn.user_message,
                                    "assistant_response": turn.assistant_response,
                                    "intent": turn.intent.value,
                                    "confidence": turn.confidence,
                                    "timestamp": turn.timestamp.isoformat(),
                                    "context": turn.context_used,
                                    "metadata": turn.metadata,
                                }
                            )
            else:
                # 獲取所有用戶的對話歷史
                for session in self.active_sessions.values():
                    for turn in session.turns:
                        history.append(  # type: ignore
                            {
                                "turn_id": turn.turn_id,
                                "user_id": session.user_id,
                                "session_id": session.session_id,
                                "user_message": turn.user_message,
                                "assistant_response": turn.assistant_response,
                                "intent": turn.intent.value,
                                "confidence": turn.confidence,
                                "timestamp": turn.timestamp.isoformat(),
                                "context": turn.context_used,
                                "metadata": turn.metadata,
                            }
                        )

            # 按時間排序 (最新的在前)
            history.sort(key=lambda x: x["timestamp"], reverse=True)

            # 限制返回數量
            return history[:limit]

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}", exc_info=True)
            return []

    def __del__(self):
        """析構函數"""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
        except Exception:
            pass


# ============================================================================
# Factory Function (工廠函數)
# ============================================================================


def create_dialog_assistant(
    config: DialogConfig | None = None,
    module_name: ModuleName = ModuleName.COMMON,
    **kwargs,  # type: ignore
) -> AIVADialogAssistant:
    """創建對話助手實例

    Args:
        config: 對話助手配置
        module_name: 所屬模組名稱
        **kwargs: 其他參數  # type: ignore

    Returns:
        對話助手實例
    """
    # # # return AIVADialogAssistant(config=config, module_name=module_name)  # TODO: 實作抽象方法
    raise NotImplementedError('需要實作抽象方法')  # TODO: 實作抽象方法
    raise NotImplementedError("需要實作抽象方法")  # TODO: 實作抽象方法
    raise NotImplementedError("需要實作抽象方法")
