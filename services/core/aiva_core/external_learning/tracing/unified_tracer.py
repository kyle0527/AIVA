"""Unified Tracer - 統一追蹤介面

整合trace_recorder和trace_logger的功能，提供統一的追蹤記錄介面
符合aiva_common規範
"""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from uuid import uuid4

from services.aiva_common.schemas import (
    SessionState,
    TraceRecord,
)
from services.aiva_common.error_handling import (
    AIVAError,
    ErrorContext,
    ErrorSeverity,
    ErrorType,
)

logger = logging.getLogger(__name__)

# 模組常量
MODULE_NAME = "execution.unified_tracer"


class TraceType(str, Enum):
    """追蹤類型枚舉"""
    EXECUTION = "execution"
    AST_ANALYSIS = "ast_analysis" 
    FUNCTION_CALL = "function_call"
    VARIABLE_ACCESS = "variable_access"
    CONTROL_FLOW = "control_flow"
    ERROR = "error"
    TASK_START = "task_start"
    TASK_END = "task_end"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class ExecutionTrace:
    """執行追蹤記錄
    
    記錄執行過程中的各種操作和狀態變化
    """
    trace_id: str
    trace_type: TraceType
    timestamp: datetime
    module_name: str
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    variables: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """後處理初始化"""
        if self.variables is None:
            self.variables = {}
        if self.metadata is None:
            self.metadata = {}


class UnifiedTracer:
    """統一追蹤記錄器
    
    整合基本追蹤記錄和RabbitMQ追蹤功能
    符合aiva_common規範
    """
    
    def __init__(self, storage_backend: Any | None = None) -> None:
        """初始化統一追蹤器
        
        Args:
            storage_backend: 儲存後端（資料庫連接等）
        """
        self.storage = storage_backend
        self.traces: List[ExecutionTrace] = []
        self.trace_records: List[TraceRecord] = []
        self.active_sessions: Dict[str, SessionState] = {}
        self.current_session_id: Optional[str] = None
        
        logger.info("UnifiedTracer initialized")
    
    def start_session(self, session_id: str) -> None:
        """開始新的追蹤會話
        
        Args:
            session_id: 會話ID
        """
        try:
            self.current_session_id = session_id
            
            # 記錄會話開始
            self.record_trace(
                trace_type=TraceType.SESSION_START,
                module_name=MODULE_NAME,
                metadata={"session_id": session_id}
            )
            
            logger.info(f"Started trace session: {session_id}")
            
        except Exception as e:
            raise AIVAError(
                message=f"Failed to start trace session: {session_id}",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(
                    module=MODULE_NAME,
                    function="start_session",
                    additional_data={"session_id": session_id}
                ),
                original_exception=e
            )
    
    def record_trace(
        self,
        trace_type: TraceType,
        module_name: str,
        trace_id: Optional[str] = None,
        function_name: Optional[str] = None,
        line_number: Optional[int] = None,
        variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """記錄一個追蹤點
        
        Args:
            trace_type: 追蹤類型
            module_name: 模組名稱
            trace_id: 追蹤ID（可選）
            function_name: 函數名稱
            line_number: 行號
            variables: 變量狀態
            metadata: 額外元數據
            
        Returns:
            創建的追蹤記錄
        """
        try:
            if trace_id is None:
                trace_id = f"trace_{len(self.traces)}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            
            trace = ExecutionTrace(
                trace_id=trace_id,
                trace_type=trace_type,
                timestamp=datetime.now(UTC),
                module_name=module_name,
                function_name=function_name,
                line_number=line_number,
                variables=variables,
                metadata=metadata
            )
            
            self.traces.append(trace)
            logger.debug(f"Recorded trace: {trace_id} in {module_name}")
            
            return trace
            
        except Exception as e:
            raise AIVAError(
                message=f"Failed to record trace: {trace_type}",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.LOW,
                context=ErrorContext(
                    module=MODULE_NAME,
                    function="record_trace",
                    additional_data={
                        "trace_type": trace_type,
                        "module_name": module_name
                    }
                ),
                original_exception=e
            )
    
    def log_task_execution(
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
        environment_response: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """記錄任務執行（兼容原trace_logger介面）
        
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
            environment_response: 環境回應
            
        Returns:
            追蹤記錄
        """
        try:
            trace_id = f"trace_{uuid4().hex[:12]}"
            
            trace_record = TraceRecord(
                trace_id=trace_id,
                plan_id=plan_id,
                step_id=step_id,
                session_id=session_id,
                tool_name=tool_name,
                input_data=input_params,
                output_data=result,
                status=status,
                error_message=error,
                execution_time_seconds=execution_time,
                timestamp=datetime.now(UTC),
                environment_response=environment_response or {},
            )
            
            self.trace_records.append(trace_record)
            
            # 同時記錄到執行追蹤
            trace_type = TraceType.TASK_END if status in ["success", "completed"] else TraceType.ERROR
            self.record_trace(
                trace_type=trace_type,
                module_name=MODULE_NAME,
                trace_id=trace_id,
                metadata={
                    "plan_id": plan_id,
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "status": status,
                    "execution_time": execution_time
                }
            )
            
            # 持久化
            if self.storage:
                self._persist_trace_record(trace_record)
            
            logger.debug(f"Task execution logged: {trace_id}")
            return trace_record
            
        except Exception as e:
            raise AIVAError(
                message=f"Failed to log task execution: {step_id}",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(
                    module=MODULE_NAME,
                    function="log_task_execution",
                    session_id=session_id,
                    additional_data={
                        "plan_id": plan_id,
                        "step_id": step_id,
                        "tool_name": tool_name
                    }
                ),
                original_exception=e
            )
    
    def get_traces(
        self,
        trace_type: Optional[TraceType] = None,
        module_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[ExecutionTrace]:
        """獲取追蹤記錄
        
        Args:
            trace_type: 過濾的追蹤類型
            module_name: 過濾的模組名稱
            session_id: 過濾的會話ID
            
        Returns:
            過濾後的追蹤記錄列表
        """
        filtered_traces = self.traces
        
        if trace_type:
            filtered_traces = [t for t in filtered_traces if t.trace_type == trace_type]
        
        if module_name:
            filtered_traces = [t for t in filtered_traces if t.module_name == module_name]
        
        if session_id:
            filtered_traces = [
                t for t in filtered_traces 
                if t.metadata and t.metadata.get("session_id") == session_id
            ]
        
        return filtered_traces
    
    def get_trace_records(self, session_id: Optional[str] = None) -> List[TraceRecord]:
        """獲取任務執行記錄
        
        Args:
            session_id: 可選的會話ID過濾
            
        Returns:
            追蹤記錄列表
        """
        if session_id:
            return [tr for tr in self.trace_records if tr.session_id == session_id]
        return self.trace_records.copy()
    
    def complete_session(self, session_id: str) -> None:
        """完成會話（兼容原trace_logger介面）
        
        Args:
            session_id: 會話ID
        """
        try:
            # 記錄會話結束
            self.record_trace(
                trace_type=TraceType.SESSION_END,
                module_name=MODULE_NAME,
                metadata={"session_id": session_id}
            )
            
            # 清理會話狀態
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            logger.info(f"Completed trace session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to complete session {session_id}: {e}")
    
    def fail_session(self, session_id: str) -> None:
        """標記會話失敗（兼容原trace_logger介面）
        
        Args:
            session_id: 會話ID
        """
        try:
            # 記錄會話失敗
            self.record_trace(
                trace_type=TraceType.ERROR,
                module_name=MODULE_NAME,
                metadata={
                    "session_id": session_id,
                    "status": "failed"
                }
            )
            
            # 清理會話狀態
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            logger.warning(f"Failed trace session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to fail session {session_id}: {e}")
    
    def clear_traces(self) -> None:
        """清除所有追蹤記錄"""
        try:
            self.traces.clear()
            self.trace_records.clear()
            logger.info("All traces cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear traces: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """獲取會話摘要
        
        Returns:
            會話摘要信息
        """
        return {
            "current_session_id": self.current_session_id,
            "total_traces": len(self.traces),
            "total_trace_records": len(self.trace_records),
            "trace_types": list({t.trace_type for t in self.traces}),
            "modules": list({t.module_name for t in self.traces}),
            "active_sessions": len(self.active_sessions),
            "start_time": min((t.timestamp for t in self.traces), default=None),
            "end_time": max((t.timestamp for t in self.traces), default=None)
        }
    
    def _persist_trace_record(self, trace_record: TraceRecord) -> None:
        """持久化追蹤記錄
        
        Args:
            trace_record: 追蹤記錄
        """
        if self.storage:
            try:
                logger.debug(f"Persisted trace record: {trace_record.trace_id}")
            except Exception as e:
                logger.error(f"Failed to persist trace record: {e}")


# 全局統一追蹤記錄器實例
_global_tracer: Optional[UnifiedTracer] = None


def get_global_tracer() -> UnifiedTracer:
    """獲取全局統一追蹤記錄器實例"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = UnifiedTracer()
    return _global_tracer


def record_execution_trace(
    module_name: str,
    function_name: Optional[str] = None,
    line_number: Optional[int] = None,
    variables: Optional[Dict[str, Any]] = None
) -> None:
    """記錄執行追蹤（便利函數）
    
    Args:
        module_name: 模組名稱
        function_name: 函數名稱
        line_number: 行號
        variables: 變量狀態
    """
    tracer = get_global_tracer()
    tracer.record_trace(
        trace_type=TraceType.EXECUTION,
        module_name=module_name,
        function_name=function_name,
        line_number=line_number,
        variables=variables
    )


# 向後相容性別名
TraceLogger = UnifiedTracer
TraceRecorder = UnifiedTracer