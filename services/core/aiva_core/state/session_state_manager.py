

from datetime import UTC, datetime
from typing import Any

from services.aiva_common.schemas import ScanCompletedPayload, TaskUpdatePayload
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class SessionStateManager:
    """
    會話狀態管理器 - 測試會話狀態管理

    負責管理測試會話的狀態，包括進度追蹤、任務狀態、
    歷史記錄等。生產環境建議使用資料庫實作。
    """

    def __init__(self) -> None:
        self._scans: dict[str, ScanCompletedPayload] = {}
        self._tasks: dict[str, TaskUpdatePayload] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._session_history: dict[str, list[dict[str, Any]]] = {}

    async def record_scan_result(self, payload: ScanCompletedPayload) -> None:
        """記錄掃描結果"""
        self._scans[payload.scan_id] = payload

        # 同時更新會話狀態
        self.update_session_status(
            payload.scan_id, "scan_completed", {"scan_result": payload}
        )

    async def record_task_update(self, payload: TaskUpdatePayload) -> None:
        """記錄任務更新"""
        self._tasks[payload.task_id] = payload

    def get_session_status(self, scan_id: str) -> dict[str, str]:
        """獲取會話狀態摘要"""
        session = self._sessions.get(scan_id, {})
        return {
            "scan_id": scan_id,
            "status": session.get("status", "not_found"),
            "phase": session.get("current_phase", "unknown"),
            "progress": (
                f"{session.get('tasks_completed', 0)}/{session.get('tasks_total', 0)}"
            ),
        }

    def get_session_context(self, scan_id: str) -> dict[str, Any]:
        """獲取會話上下文資訊，用於策略調整"""
        session = self._sessions.get(scan_id, {})
        history = self._session_history.get(scan_id, [])

        return {
            "scan_id": scan_id,
            "current_status": session.get("status"),
            "completed_tasks": session.get("tasks_completed", 0),
            "total_tasks": session.get("tasks_total", 0),
            "findings_count": session.get("findings_count", 0),
            "waf_detected": session.get("waf_detected", False),
            "previous_results": history[-5:] if history else [],  # 最近5次結果
            "target_info": session.get("target_info", {}),
            "fingerprints": session.get("fingerprints", {}),
        }

    def update_context(self, scan_id: str, context_data: dict[str, Any]) -> None:
        """更新會話上下文資料"""
        if scan_id not in self._sessions:
            self._sessions[scan_id] = {
                "created_at": datetime.now(UTC).isoformat(),
                "status": "initialized",
                "tasks_completed": 0,
                "tasks_total": 0,
                "current_phase": "initial",
            }
            self._session_history[scan_id] = []

        # 更新上下文數據
        self._sessions[scan_id].update(context_data)
        self._sessions[scan_id]["updated_at"] = datetime.now(UTC).isoformat()

        logger.debug(f"Updated context for session {scan_id}")

    def update_session_status(
        self,
        session_id: str,
        status: str,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """更新會話狀態"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "created_at": datetime.now(UTC).isoformat(),
                "status": "initialized",
                "tasks_completed": 0,
                "tasks_total": 0,
                "current_phase": "initial",
            }
            self._session_history[session_id] = []

        self._sessions[session_id]["status"] = status
        self._sessions[session_id]["updated_at"] = datetime.now(UTC).isoformat()

        if additional_data:
            self._sessions[session_id].update(additional_data)

        # 記錄狀態變更歷史
        self._session_history[session_id].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "status": status,
                "data": additional_data or {},
            }
        )

        logger.info(f"Updated session {session_id} status to: {status}")
