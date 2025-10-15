"""
Trace Logger - 執行追蹤記錄器

負責記錄攻擊計畫執行過程中的所有操作，用於後續分析和強化學習
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any
from uuid import uuid4

from services.aiva_common.schemas import (
    SessionState,
    TraceRecord,
)

logger = logging.getLogger(__name__)


class TraceLogger:
    """執行追蹤記錄器

    訂閱 RabbitMQ 結果佇列，記錄每個任務執行的詳細資訊
    """

    def __init__(self, storage_backend: Any | None = None) -> None:
        """初始化追蹤記錄器

        Args:
            storage_backend: 儲存後端（資料庫連接等）
        """
        self.storage = storage_backend
        self.active_sessions: dict[str, SessionState] = {}
        logger.info("TraceLogger initialized")

    def create_trace_record(
        self,
        plan_id: str,
        step_id: str,
        session_id: str,
        tool_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        status: str,
        execution_time: float,
        error_message: str | None = None,
        environment_response: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """創建追蹤記錄

        Args:
            plan_id: 攻擊計畫 ID
            step_id: 步驟 ID
            session_id: 會話 ID
            tool_name: 工具名稱
            input_data: 輸入數據
            output_data: 輸出數據
            status: 執行狀態
            execution_time: 執行時間（秒）
            error_message: 錯誤訊息
            environment_response: 環境回應

        Returns:
            追蹤記錄
        """
        trace_id = f"trace_{uuid4().hex[:12]}"

        trace = TraceRecord(
            trace_id=trace_id,
            plan_id=plan_id,
            step_id=step_id,
            session_id=session_id,
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            status=status,
            error_message=error_message,
            execution_time_seconds=execution_time,
            timestamp=datetime.now(UTC),
            environment_response=environment_response or {},
        )

        logger.info(
            f"Created trace record {trace_id} for step {step_id} "
            f"(session={session_id}, status={status})"
        )

        return trace

    async def log_task_execution(
        self,
        session_id: str,
        plan_id: str,
        step_id: str,
        tool_name: str,
        input_params: dict[str, Any],
        result: dict[str, Any],
        status: str = "success",
        execution_time: float = 0.0,
        error: str | None = None,
    ) -> TraceRecord:
        """記錄任務執行

        Args:
            session_id: 會話 ID
            plan_id: 計畫 ID
            step_id: 步驟 ID
            tool_name: 工具名稱
            input_params: 輸入參數
            result: 執行結果
            status: 執行狀態
            execution_time: 執行時間
            error: 錯誤訊息

        Returns:
            追蹤記錄
        """
        # 創建追蹤記錄
        trace = self.create_trace_record(
            plan_id=plan_id,
            step_id=step_id,
            session_id=session_id,
            tool_name=tool_name,
            input_data=input_params,
            output_data=result,
            status=status,
            execution_time=execution_time,
            error_message=error,
            environment_response=result.get("environment_response", {}),
        )

        # 持久化到儲存後端
        if self.storage:
            await self._persist_trace(trace)

        # 更新會話狀態
        if session_id in self.active_sessions:
            self._update_session_state(session_id, step_id, status)

        return trace

    async def _persist_trace(self, trace: TraceRecord) -> None:
        """持久化追蹤記錄到儲存後端

        Args:
            trace: 追蹤記錄
        """
        try:
            if hasattr(self.storage, "save_trace"):
                await self.storage.save_trace(trace.model_dump())
                logger.debug(f"Persisted trace {trace.trace_id}")
            else:
                logger.warning("Storage backend does not support save_trace")
        except Exception as e:
            logger.error(f"Failed to persist trace {trace.trace_id}: {e}")

    def _update_session_state(self, session_id: str, step_id: str, status: str) -> None:
        """更新會話狀態

        Args:
            session_id: 會話 ID
            step_id: 步驟 ID
            status: 執行狀態
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found in active sessions")
            return

        session = self.active_sessions[session_id]

        # 更新完成步驟列表
        if status == "success" and step_id not in session.completed_steps:
            session.completed_steps.append(step_id)

        # 從待處理列表中移除
        if step_id in session.pending_steps:
            session.pending_steps.remove(step_id)

        # 更新當前步驟索引
        session.current_step_index += 1
        session.updated_at = datetime.now(UTC)

        logger.debug(f"Updated session {session_id}: step {step_id} {status}")

    async def get_session_traces(self, session_id: str) -> list[TraceRecord]:
        """獲取會話的所有追蹤記錄

        Args:
            session_id: 會話 ID

        Returns:
            追蹤記錄列表
        """
        if not self.storage:
            logger.warning("No storage backend configured")
            return []

        try:
            if hasattr(self.storage, "get_traces_by_session"):
                traces_data = await self.storage.get_traces_by_session(session_id)
                return [TraceRecord(**data) for data in traces_data]
            else:
                logger.warning("Storage backend does not support get_traces_by_session")
                return []
        except Exception as e:
            logger.error(f"Failed to get traces for session {session_id}: {e}")
            return []

    async def get_plan_traces(self, plan_id: str) -> list[TraceRecord]:
        """獲取攻擊計畫的所有追蹤記錄

        Args:
            plan_id: 計畫 ID

        Returns:
            追蹤記錄列表
        """
        if not self.storage:
            logger.warning("No storage backend configured")
            return []

        try:
            if hasattr(self.storage, "get_traces_by_plan"):
                traces_data = await self.storage.get_traces_by_plan(plan_id)
                return [TraceRecord(**data) for data in traces_data]
            else:
                logger.warning("Storage backend does not support get_traces_by_plan")
                return []
        except Exception as e:
            logger.error(f"Failed to get traces for plan {plan_id}: {e}")
            return []

    def create_session(
        self,
        plan_id: str,
        scan_id: str,
        steps: list[str],
        timeout_minutes: int = 30,
    ) -> SessionState:
        """創建新的會話

        Args:
            plan_id: 攻擊計畫 ID
            scan_id: 掃描 ID
            steps: 步驟 ID 列表
            timeout_minutes: 超時時間（分鐘）

        Returns:
            會話狀態
        """
        session_id = f"session_{uuid4().hex[:12]}"

        timeout_at = datetime.now(UTC)
        timeout_at = timeout_at.replace(minute=timeout_at.minute + timeout_minutes)

        session = SessionState(
            session_id=session_id,
            plan_id=plan_id,
            scan_id=scan_id,
            status="active",
            pending_steps=steps.copy(),
            timeout_at=timeout_at,
        )

        self.active_sessions[session_id] = session
        logger.info(
            f"Created session {session_id} for plan {plan_id} with {len(steps)} steps"
        )

        return session

    def get_session(self, session_id: str) -> SessionState | None:
        """獲取會話狀態

        Args:
            session_id: 會話 ID

        Returns:
            會話狀態，不存在則返回 None
        """
        return self.active_sessions.get(session_id)

    def update_session_status(self, session_id: str, status: str) -> None:
        """更新會話狀態

        Args:
            session_id: 會話 ID
            status: 新狀態
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].status = status
            self.active_sessions[session_id].updated_at = datetime.now(UTC)
            logger.info(f"Updated session {session_id} status to {status}")

    def complete_session(self, session_id: str) -> None:
        """完成會話

        Args:
            session_id: 會話 ID
        """
        self.update_session_status(session_id, "completed")

    def fail_session(self, session_id: str) -> None:
        """標記會話為失敗

        Args:
            session_id: 會話 ID
        """
        self.update_session_status(session_id, "failed")

    def abort_session(self, session_id: str) -> None:
        """中止會話

        Args:
            session_id: 會話 ID
        """
        self.update_session_status(session_id, "aborted")

    async def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """清理已完成的舊會話

        Args:
            max_age_hours: 最大保留時間（小時）

        Returns:
            清理的會話數量
        """
        cutoff_time = datetime.now(UTC)
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - max_age_hours)

        cleaned = 0
        sessions_to_remove = []

        for session_id, session in self.active_sessions.items():
            if (
                session.status in {"completed", "failed", "aborted"}
                and session.updated_at < cutoff_time
            ):
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old sessions")

        return cleaned
