"""
Trace Recorder - 軌跡記錄器

記錄任務執行過程中的所有詳細信息
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class TraceType(str, Enum):
    """軌跡類型"""

    TASK_START = "task_start"  # 任務開始
    TASK_END = "task_end"  # 任務結束
    HTTP_REQUEST = "http_request"  # HTTP 請求
    HTTP_RESPONSE = "http_response"  # HTTP 回應
    RPC_CALL = "rpc_call"  # RPC 調用
    RPC_RESPONSE = "rpc_response"  # RPC 回應
    LOG = "log"  # 日誌
    ERROR = "error"  # 錯誤
    TOOL_OUTPUT = "tool_output"  # 工具輸出
    DECISION = "decision"  # 決策點
    VALIDATION = "validation"  # 驗證結果


@dataclass
class TraceEntry:
    """軌跡條目

    記錄單一執行步驟的詳細信息
    """

    trace_id: str
    timestamp: datetime
    trace_type: TraceType
    task_id: str | None = None
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["trace_type"] = self.trace_type.value
        return data

    def to_json(self) -> str:
        """轉換為 JSON 字串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ExecutionTrace:
    """執行軌跡

    包含一個任務或攻擊流程的完整執行記錄
    """

    trace_session_id: str
    plan_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    entries: list[TraceEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_entry(self, entry: TraceEntry) -> None:
        """添加軌跡條目"""
        self.entries.append(entry)

    def get_entries_by_task(self, task_id: str) -> list[TraceEntry]:
        """獲取特定任務的所有軌跡"""
        return [e for e in self.entries if e.task_id == task_id]

    def get_entries_by_type(self, trace_type: TraceType) -> list[TraceEntry]:
        """獲取特定類型的所有軌跡"""
        return [e for e in self.entries if e.trace_type == trace_type]

    def finalize(self) -> None:
        """結束軌跡記錄"""
        self.end_time = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "trace_session_id": self.trace_session_id,
            "plan_id": self.plan_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_entries": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """轉換為 JSON 字串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class TraceRecorder:
    """軌跡記錄器

    負責記錄和管理執行軌跡
    """

    def __init__(self) -> None:
        """初始化記錄器"""
        self.active_traces: dict[str, ExecutionTrace] = {}
        logger.info("TraceRecorder initialized")

    def start_trace(
        self, plan_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> ExecutionTrace:
        """開始一個新的軌跡記錄

        Args:
            plan_id: 執行計畫 ID
            metadata: 元數據

        Returns:
            執行軌跡對象
        """
        trace_session_id = f"trace_{uuid4().hex[:8]}"
        trace = ExecutionTrace(
            trace_session_id=trace_session_id,
            plan_id=plan_id,
            metadata=metadata or {},
        )
        self.active_traces[trace_session_id] = trace

        logger.info(f"Started trace session {trace_session_id} for plan {plan_id}")
        return trace

    def record(
        self,
        trace_session_id: str,
        trace_type: TraceType,
        content: dict[str, Any],
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """記錄一條軌跡

        Args:
            trace_session_id: 軌跡會話 ID
            trace_type: 軌跡類型
            content: 內容
            task_id: 任務 ID
            metadata: 元數據

        Returns:
            軌跡條目
        """
        trace = self.active_traces.get(trace_session_id)
        if not trace:
            logger.warning(f"Trace session {trace_session_id} not found")
            return None  # type: ignore

        entry = TraceEntry(
            trace_id=f"{trace_session_id}_{len(trace.entries)}",
            timestamp=datetime.now(),
            trace_type=trace_type,
            task_id=task_id,
            content=content,
            metadata=metadata or {},
        )

        trace.add_entry(entry)
        logger.debug(
            f"Recorded {trace_type.value} trace for task {task_id} "
            f"in session {trace_session_id}"
        )
        return entry

    def record_task_start(
        self, trace_session_id: str, task_id: str, task_info: dict[str, Any]
    ) -> None:
        """記錄任務開始"""
        self.record(
            trace_session_id=trace_session_id,
            trace_type=TraceType.TASK_START,
            content=task_info,
            task_id=task_id,
        )

    def record_task_end(
        self,
        trace_session_id: str,
        task_id: str,
        result: dict[str, Any],
        success: bool = True,
    ) -> None:
        """記錄任務結束"""
        self.record(
            trace_session_id=trace_session_id,
            trace_type=TraceType.TASK_END,
            content={"result": result, "success": success},
            task_id=task_id,
        )

    def record_http_request(
        self,
        trace_session_id: str,
        task_id: str,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: str | None = None,
    ) -> None:
        """記錄 HTTP 請求"""
        self.record(
            trace_session_id=trace_session_id,
            trace_type=TraceType.HTTP_REQUEST,
            content={
                "method": method,
                "url": url,
                "headers": headers or {},
                "body": body,
            },
            task_id=task_id,
        )

    def record_http_response(
        self,
        trace_session_id: str,
        task_id: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: str | None = None,
    ) -> None:
        """記錄 HTTP 回應"""
        self.record(
            trace_session_id=trace_session_id,
            trace_type=TraceType.HTTP_RESPONSE,
            content={
                "status_code": status_code,
                "headers": headers or {},
                "body": body,
            },
            task_id=task_id,
        )

    def record_log(
        self, trace_session_id: str, task_id: str, level: str, message: str
    ) -> None:
        """記錄日誌"""
        self.record(
            trace_session_id=trace_session_id,
            trace_type=TraceType.LOG,
            content={"level": level, "message": message},
            task_id=task_id,
        )

    def record_error(
        self,
        trace_session_id: str,
        task_id: str,
        error: str,
        traceback: str | None = None,
    ) -> None:
        """記錄錯誤"""
        self.record(
            trace_session_id=trace_session_id,
            trace_type=TraceType.ERROR,
            content={"error": error, "traceback": traceback},
            task_id=task_id,
        )

    def finalize_trace(self, trace_session_id: str) -> ExecutionTrace | None:
        """結束軌跡記錄

        Args:
            trace_session_id: 軌跡會話 ID

        Returns:
            完成的執行軌跡
        """
        trace = self.active_traces.get(trace_session_id)
        if trace:
            trace.finalize()
            logger.info(
                f"Finalized trace session {trace_session_id} "
                f"with {len(trace.entries)} entries"
            )
            return trace
        return None

    def get_trace(self, trace_session_id: str) -> ExecutionTrace | None:
        """獲取軌跡"""
        return self.active_traces.get(trace_session_id)
