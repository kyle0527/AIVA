"""AIVA Context Manager - 上下文管理系統
從 aiva_core_v2 遷移到核心模組

分布式上下文和會話管理系統
"""

import asyncio
import logging
import time
from typing import Any

from ..task_planning.command_router import CommandContext


class ContextManager:
    """上下文管理器 - 處理分布式上下文和會話管理"""

    def __init__(self):
        self.logger = logging.getLogger("context_manager")
        self._contexts: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._context_locks: dict[str, asyncio.Lock] = {}

    async def create_context(self, context: CommandContext) -> str:
        """創建執行上下文"""
        context_id = f"{context.session_id}_{context.request_id}_{int(time.time())}"

        # 創建上下文鎖
        self._context_locks[context_id] = asyncio.Lock()

        async with self._context_locks[context_id]:
            self._contexts[context_id] = {
                "command_context": context,
                "created_at": time.time(),
                "state": "created",
                "variables": {},
                "history": [],
                "metadata": {
                    "command": context.command,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                },
            }

            # 更新會話信息
            if context.session_id:
                await self._update_session(
                    context.session_id, context_id, context.user_id
                )

        self.logger.debug(f"Created context: {context_id}")
        return context_id

    async def _update_session(
        self, session_id: str, context_id: str, user_id: str | None
    ):
        """更新會話信息"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "created_at": time.time(),
                "contexts": [],
                "user_id": user_id,
                "last_activity": time.time(),
            }

        self._sessions[session_id]["contexts"].append(context_id)
        self._sessions[session_id]["last_activity"] = time.time()

    async def get_context(self, context_id: str) -> dict[str, Any] | None:
        """獲取上下文"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                return self._contexts.get(context_id)
        return None

    async def update_context(self, context_id: str, updates: dict[str, Any]):
        """更新上下文"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                self._contexts[context_id].update(updates)
                self._contexts[context_id]["updated_at"] = time.time()

    async def set_variable(self, context_id: str, key: str, value: Any):
        """設置上下文變量"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                self._contexts[context_id]["variables"][key] = value

    async def get_variable(self, context_id: str, key: str) -> Any:
        """獲取上下文變量"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                return self._contexts[context_id]["variables"].get(key)
        return None

    async def add_history(self, context_id: str, entry: dict[str, Any]):
        """添加歷史記錄"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                entry["timestamp"] = time.time()
                self._contexts[context_id]["history"].append(entry)

                # 限制歷史記錄數量，避免內存洩漏
                if len(self._contexts[context_id]["history"]) > 1000:
                    self._contexts[context_id]["history"] = self._contexts[context_id][
                        "history"
                    ][-500:]

    async def get_context_history(
        self, context_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """獲取上下文歷史"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                history = self._contexts[context_id]["history"]
                if limit:
                    return history[-limit:]
                return history.copy()
        return []

    async def get_session_contexts(self, session_id: str) -> list[str]:
        """獲取會話的所有上下文ID"""
        if session_id in self._sessions:
            return self._sessions[session_id]["contexts"].copy()
        return []

    async def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """獲取會話信息"""
        return self._sessions.get(session_id)

    async def cleanup_context(self, context_id: str):
        """清理上下文"""
        if context_id in self._contexts:
            async with self._context_locks.get(context_id, asyncio.Lock()):
                del self._contexts[context_id]

            # 清理鎖
            if context_id in self._context_locks:
                del self._context_locks[context_id]

            self.logger.debug(f"Cleaned up context: {context_id}")

    async def cleanup_session(self, session_id: str):
        """清理會話及其相關上下文"""
        if session_id in self._sessions:
            # 清理會話中的所有上下文
            context_ids = self._sessions[session_id]["contexts"]
            for context_id in context_ids:
                await self.cleanup_context(context_id)

            # 清理會話
            del self._sessions[session_id]
            self.logger.debug(f"Cleaned up session: {session_id}")

    async def cleanup_expired_contexts(self, max_age_seconds: int = 3600):
        """清理過期的上下文（默認1小時）"""
        current_time = time.time()
        expired_contexts = []

        for context_id, context_data in self._contexts.items():
            if current_time - context_data["created_at"] > max_age_seconds:
                expired_contexts.append(context_id)

        for context_id in expired_contexts:
            await self.cleanup_context(context_id)

        if expired_contexts:
            self.logger.info(f"Cleaned up {len(expired_contexts)} expired contexts")

    async def cleanup_expired_sessions(self, max_idle_seconds: int = 7200):
        """清理過期的會話（默認2小時無活動）"""
        current_time = time.time()
        expired_sessions = []

        for session_id, session_data in self._sessions.items():
            if current_time - session_data["last_activity"] > max_idle_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.cleanup_session(session_id)

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def get_context_stats(self) -> dict[str, Any]:
        """獲取上下文統計信息"""
        current_time = time.time()
        active_contexts = 0
        active_sessions = 0

        # 統計活躍上下文（最近10分鐘內創建或更新）
        for context_data in self._contexts.values():
            last_activity = max(
                context_data["created_at"], context_data.get("updated_at", 0)
            )
            if current_time - last_activity < 600:  # 10分鐘
                active_contexts += 1

        # 統計活躍會話（最近30分鐘內活動）
        for session_data in self._sessions.values():
            if current_time - session_data["last_activity"] < 1800:  # 30分鐘
                active_sessions += 1

        return {
            "total_contexts": len(self._contexts),
            "total_sessions": len(self._sessions),
            "active_contexts": active_contexts,
            "active_sessions": active_sessions,
            "context_locks": len(self._context_locks),
        }


# 全局上下文管理器實例
_context_manager_instance = None


def get_context_manager() -> ContextManager:
    """獲取上下文管理器實例"""
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ContextManager()
    return _context_manager_instance
