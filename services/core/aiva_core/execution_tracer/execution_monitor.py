"""Execution Monitor - 執行監控器

監控任務執行狀態並協調 Trace 記錄
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..planner.orchestrator import ExecutionPlan
    from ..planner.task_converter import ExecutableTask
    from ..planner.tool_selector import ToolDecision

from .trace_recorder import ExecutionTrace, TraceRecorder, TraceType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """執行上下文

    包含任務執行時的所有相關信息
    """

    plan_id: str
    task_id: str
    tool_decision: ToolDecision
    trace_session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionMonitor:
    """執行監控器

    監控任務執行並記錄詳細軌跡
    """

    def __init__(self, trace_recorder: TraceRecorder | None = None) -> None:
        """初始化監控器

        Args:
            trace_recorder: 軌跡記錄器（如果未提供則創建新的）
        """
        self.trace_recorder = trace_recorder or TraceRecorder()
        self.active_contexts: dict[str, ExecutionContext] = {}
        logger.info("ExecutionMonitor initialized")

    def start_monitoring(
        self, plan: ExecutionPlan, metadata: dict[str, Any] | None = None
    ) -> ExecutionTrace:
        """開始監控執行計畫

        Args:
            plan: 執行計畫
            metadata: 元數據

        Returns:
            執行軌跡
        """
        trace_metadata = {
            "plan_id": plan.plan_id,
            "attack_type": plan.metadata.get("attack_type"),
            "total_tasks": plan.metadata.get("total_tasks"),
            **(metadata or {}),
        }

        trace = self.trace_recorder.start_trace(
            plan_id=plan.plan_id, metadata=trace_metadata
        )

        logger.info(
            f"Started monitoring plan {plan.plan_id} "
            f"with trace session {trace.trace_session_id}"
        )
        return trace

    def start_task_execution(
        self,
        trace_session_id: str,
        task: ExecutableTask,
        tool_decision: ToolDecision,
    ) -> ExecutionContext:
        """開始執行任務

        Args:
            trace_session_id: 軌跡會話 ID
            task: 任務
            tool_decision: 工具決策

        Returns:
            執行上下文
        """
        context = ExecutionContext(
            plan_id=trace_session_id,
            task_id=task.task_id,
            tool_decision=tool_decision,
            trace_session_id=trace_session_id,
            metadata={
                "task_type": task.task_type,
                "action": task.action,
                "service_type": tool_decision.service_type.value,
            },
        )

        self.active_contexts[task.task_id] = context

        # 記錄任務開始
        task_info = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "action": task.action,
            "parameters": task.parameters,
            "service": tool_decision.service_type.value,
            "service_function": tool_decision.service_function,
        }
        self.trace_recorder.record_task_start(trace_session_id, task.task_id, task_info)

        logger.info(f"Started task execution: {task.task_id} ({task.task_type})")
        return context

    def record_step(
        self,
        context: ExecutionContext,
        step_name: str,
        step_data: dict[str, Any],
        step_type: TraceType = TraceType.LOG,
    ) -> None:
        """記錄執行步驟

        Args:
            context: 執行上下文
            step_name: 步驟名稱
            step_data: 步驟數據
            step_type: 軌跡類型
        """
        content = {"step": step_name, "data": step_data}
        self.trace_recorder.record(
            trace_session_id=context.trace_session_id,
            trace_type=step_type,
            content=content,
            task_id=context.task_id,
        )

    def record_tool_invocation(
        self,
        context: ExecutionContext,
        tool_name: str,
        input_params: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        """記錄工具調用

        Args:
            context: 執行上下文
            tool_name: 工具名稱
            input_params: 輸入參數
            output: 輸出結果
        """
        content = {"tool": tool_name, "input": input_params, "output": output}
        self.trace_recorder.record(
            trace_session_id=context.trace_session_id,
            trace_type=TraceType.TOOL_OUTPUT,
            content=content,
            task_id=context.task_id,
        )

    def record_decision_point(
        self,
        context: ExecutionContext,
        decision_type: str,
        options: list[str],
        chosen_option: str,
        reason: str,
    ) -> None:
        """記錄決策點

        Args:
            context: 執行上下文
            decision_type: 決策類型
            options: 可選項
            chosen_option: 選擇的選項
            reason: 選擇原因
        """
        content = {
            "decision_type": decision_type,
            "options": options,
            "chosen": chosen_option,
            "reason": reason,
        }
        self.trace_recorder.record(
            trace_session_id=context.trace_session_id,
            trace_type=TraceType.DECISION,
            content=content,
            task_id=context.task_id,
        )

    def complete_task_execution(
        self,
        context: ExecutionContext,
        result: dict[str, Any],
        success: bool = True,
    ) -> None:
        """完成任務執行

        Args:
            context: 執行上下文
            result: 執行結果
            success: 是否成功
        """
        self.trace_recorder.record_task_end(
            trace_session_id=context.trace_session_id,
            task_id=context.task_id,
            result=result,
            success=success,
        )

        # 移除活躍上下文
        if context.task_id in self.active_contexts:
            del self.active_contexts[context.task_id]

        status = "successfully" if success else "with errors"
        logger.info(f"Task {context.task_id} completed {status}")

    def record_error(
        self, context: ExecutionContext, error: str, traceback: str | None = None
    ) -> None:
        """記錄錯誤

        Args:
            context: 執行上下文
            error: 錯誤訊息
            traceback: 錯誤追蹤
        """
        self.trace_recorder.record_error(
            trace_session_id=context.trace_session_id,
            task_id=context.task_id,
            error=error,
            traceback=traceback,
        )

    def finalize_monitoring(self, trace_session_id: str) -> ExecutionTrace | None:
        """結束監控

        Args:
            trace_session_id: 軌跡會話 ID

        Returns:
            完成的執行軌跡
        """
        trace = self.trace_recorder.finalize_trace(trace_session_id)

        if trace:
            logger.info(
                f"Monitoring completed for session {trace_session_id}: "
                f"{len(trace.entries)} trace entries recorded"
            )

        return trace

    def get_active_tasks(self) -> list[str]:
        """獲取正在執行的任務列表"""
        return list(self.active_contexts.keys())

    def get_trace_summary(self, trace_session_id: str) -> dict[str, Any] | None:
        """獲取軌跡摘要

        Args:
            trace_session_id: 軌跡會話 ID

        Returns:
            摘要字典
        """
        trace = self.trace_recorder.get_trace(trace_session_id)
        if not trace:
            return None

        # 統計各類型軌跡數量
        type_counts = {}
        for entry in trace.entries:
            trace_type = entry.trace_type.value
            type_counts[trace_type] = type_counts.get(trace_type, 0) + 1

        # 計算執行時間
        duration = None
        if trace.end_time:
            duration = (trace.end_time - trace.start_time).total_seconds()

        return {
            "trace_session_id": trace.trace_session_id,
            "plan_id": trace.plan_id,
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "duration_seconds": duration,
            "total_entries": len(trace.entries),
            "entry_type_counts": type_counts,
            "metadata": trace.metadata,
        }
