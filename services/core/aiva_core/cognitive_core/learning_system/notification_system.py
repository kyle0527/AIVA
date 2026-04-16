"""User Notification System - 用戶通知系統

當學習系統觸發 RAG 搜索時，即時通知用戶。

通知方式：
1. 日誌輸出（控制台）
2. API 響應（如果在 HTTP 請求中）
3. WebSocket 推送（實時通知）
4. 文件記錄（通知歷史）
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NotificationLevel(str, Enum):
    """通知級別"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationType(str, Enum):
    """通知類型"""
    UNKNOWN_SITUATION = "unknown_situation"
    RAG_TRIGGERED = "rag_triggered"
    RAG_COMPLETED = "rag_completed"
    RAG_FAILED = "rag_failed"
    LEARNING_STARTED = "learning_started"
    LEARNING_COMPLETED = "learning_completed"


class UserNotification:
    """用戶通知
    
    封裝通知信息
    """
    
    def __init__(
        self,
        notification_id: str,
        notification_type: NotificationType,
        level: NotificationLevel,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        self.notification_id = notification_id
        self.notification_type = notification_type
        self.level = level
        self.title = title
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "notification_id": self.notification_id,
            "type": self.notification_type,
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_log_message(self) -> str:
        """轉換為日誌訊息"""
        icon = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.CRITICAL: "🚨",
        }.get(self.level, "📢")
        
        return f"{icon} [{self.level.upper()}] {self.title} - {self.message}"


class NotificationSystem:
    """通知系統
    
    管理所有用戶通知
    """
    
    def __init__(
        self,
        log_to_console: bool = True,
        save_to_file: bool = True,
        notification_file: str = "services/core/aiva_core/cognitive_core/learning_system/data/notifications.jsonl",
    ):
        """初始化通知系統
        
        Args:
            log_to_console: 是否輸出到控制台
            save_to_file: 是否保存到文件
            notification_file: 通知文件路徑
        """
        self.log_to_console = log_to_console
        self.save_to_file = save_to_file
        self.notification_file = Path(notification_file)
        
        # 創建通知文件目錄
        if self.save_to_file:
            self.notification_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 通知回調列表（例如：WebSocket 推送）
        self._callbacks: list[Callable[[UserNotification], None]] = []
        
        # 通知歷史（內存中保留最近的通知）
        self._notification_history: list[UserNotification] = []
        self._max_history_size = 100
        
        logger.info(
            f"Notification system initialized "
            f"(console={log_to_console}, file={save_to_file})"
        )
    
    def register_callback(
        self,
        callback: Callable[[UserNotification], None]
    ) -> None:
        """註冊通知回調
        
        Args:
            callback: 回調函數，接收 UserNotification 對象
        """
        self._callbacks.append(callback)
        logger.info(f"Registered notification callback: {callback.__name__}")
    
    def notify(self, notification: UserNotification) -> None:
        """發送通知
        
        Args:
            notification: 通知對象
        """
        # 1. 輸出到控制台
        if self.log_to_console:
            log_level = {
                NotificationLevel.INFO: logging.INFO,
                NotificationLevel.WARNING: logging.WARNING,
                NotificationLevel.CRITICAL: logging.CRITICAL,
            }.get(notification.level, logging.INFO)
            
            logger.log(log_level, notification.to_log_message())
        
        # 2. 保存到文件
        if self.save_to_file:
            try:
                with open(self.notification_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(notification.to_dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to save notification to file: {e}")
        
        # 3. 調用所有回調
        for callback in self._callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")
        
        # 4. 保存到歷史
        self._notification_history.append(notification)
        if len(self._notification_history) > self._max_history_size:
            self._notification_history.pop(0)
    
    def notify_unknown_situation(
        self,
        alert_id: str,
        trigger_reason: str,
        data_snapshot: dict[str, Any],
        search_query: str,
    ) -> None:
        """通知未知情況
        
        當檢測到未知情況並觸發 RAG 時調用
        """
        notification = UserNotification(
            notification_id=f"notif_{alert_id}",
            notification_type=NotificationType.UNKNOWN_SITUATION,
            level=NotificationLevel.WARNING,
            title="檢測到未知情況",
            message=(
                f"系統遇到未知的安全情況，相似度低於閾值。"
                f"原因: {trigger_reason}"
            ),
            details={
                "alert_id": alert_id,
                "trigger_reason": trigger_reason,
                "data_snapshot": data_snapshot,
                "search_query": search_query,
                "action": "正在搜索外部知識資源...",
            },
        )
        
        self.notify(notification)
    
    def notify_rag_triggered(
        self,
        alert_id: str,
        search_query: str,
    ) -> None:
        """通知 RAG 已觸發"""
        notification = UserNotification(
            notification_id=f"notif_{alert_id}_rag_triggered",
            notification_type=NotificationType.RAG_TRIGGERED,
            level=NotificationLevel.INFO,
            title="RAG 搜索已啟動",
            message=f"正在搜索外部知識資源: {search_query}",
            details={
                "alert_id": alert_id,
                "search_query": search_query,
                "search_scope": [
                    "CVE 資料庫",
                    "技術文檔",
                    "安全研究報告",
                    "已有知識庫",
                ],
            },
        )
        
        self.notify(notification)
    
    def notify_rag_completed(
        self,
        alert_id: str,
        results_count: int,
        results_summary: list[dict[str, Any]],
    ) -> None:
        """通知 RAG 搜索完成"""
        level = NotificationLevel.INFO if results_count > 0 else NotificationLevel.WARNING
        
        notification = UserNotification(
            notification_id=f"notif_{alert_id}_rag_completed",
            notification_type=NotificationType.RAG_COMPLETED,
            level=level,
            title="RAG 搜索完成",
            message=(
                f"找到 {results_count} 個相關資源" if results_count > 0
                else "未找到相關資源"
            ),
            details={
                "alert_id": alert_id,
                "results_count": results_count,
                "results_summary": results_summary,
            },
        )
        
        self.notify(notification)
    
    def notify_rag_failed(
        self,
        alert_id: str,
        error_message: str,
    ) -> None:
        """通知 RAG 搜索失敗"""
        notification = UserNotification(
            notification_id=f"notif_{alert_id}_rag_failed",
            notification_type=NotificationType.RAG_FAILED,
            level=NotificationLevel.CRITICAL,
            title="RAG 搜索失敗",
            message=f"RAG 搜索過程中發生錯誤: {error_message}",
            details={
                "alert_id": alert_id,
                "error_message": error_message,
            },
        )
        
        self.notify(notification)
    
    def notify_learning_started(
        self,
        session_id: str,
        capability: str,
        data_sources: list[str],
    ) -> None:
        """通知學習開始"""
        notification = UserNotification(
            notification_id=f"notif_learning_{session_id}_started",
            notification_type=NotificationType.LEARNING_STARTED,
            level=NotificationLevel.INFO,
            title="學習會話開始",
            message=f"開始學習 {capability} 能力",
            details={
                "session_id": session_id,
                "capability": capability,
                "data_sources": data_sources,
            },
        )
        
        self.notify(notification)
    
    def notify_learning_completed(
        self,
        session_id: str,
        capability: str,
        improvements: dict[str, Any],
        validation_passed: bool,
    ) -> None:
        """通知學習完成"""
        level = NotificationLevel.INFO if validation_passed else NotificationLevel.WARNING
        
        notification = UserNotification(
            notification_id=f"notif_learning_{session_id}_completed",
            notification_type=NotificationType.LEARNING_COMPLETED,
            level=level,
            title="學習會話完成",
            message=(
                f"{capability} 學習完成，"
                f"{'驗證通過，已保存新權重' if validation_passed else '驗證未通過，已丟棄'}"
            ),
            details={
                "session_id": session_id,
                "capability": capability,
                "improvements": improvements,
                "validation_passed": validation_passed,
            },
        )
        
        self.notify(notification)
    
    def get_notification_history(
        self,
        limit: int = 20,
        notification_type: NotificationType | None = None,
    ) -> list[dict[str, Any]]:
        """獲取通知歷史
        
        Args:
            limit: 返回數量限制
            notification_type: 過濾通知類型
            
        Returns:
            通知列表
        """
        notifications = self._notification_history
        
        # 過濾類型
        if notification_type:
            notifications = [
                n for n in notifications
                if n.notification_type == notification_type
            ]
        
        # 返回最新的 N 條
        return [n.to_dict() for n in notifications[-limit:]]
    
    def clear_history(self) -> None:
        """清空通知歷史"""
        self._notification_history.clear()
        logger.info("Notification history cleared")


# 全局通知系統實例
_notification_system: NotificationSystem | None = None


def get_notification_system() -> NotificationSystem:
    """獲取全局通知系統實例"""
    global _notification_system
    if _notification_system is None:
        _notification_system = NotificationSystem()
    return _notification_system
