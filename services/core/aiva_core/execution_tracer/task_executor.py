"""Task Executor - 任務執行器

實際執行任務並與各種服務整合
"""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..planner.task_converter import ExecutableTask
    from ..planner.tool_selector import ToolDecision

from .execution_monitor import ExecutionContext, ExecutionMonitor

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """執行結果"""

    task_id: str
    success: bool
    output: dict[str, Any]
    error: str | None = None
    trace_session_id: str | None = None


class TaskExecutor:
    """任務執行器

    執行具體任務並記錄執行軌跡
    """

    def __init__(self, execution_monitor: ExecutionMonitor | None = None) -> None:
        """初始化執行器

        Args:
            execution_monitor: 執行監控器
        """
        self.monitor = execution_monitor or ExecutionMonitor()
        logger.info("TaskExecutor initialized")

    async def execute_task(
        self,
        task: ExecutableTask,
        tool_decision: ToolDecision,
        trace_session_id: str,
    ) -> ExecutionResult:
        """執行任務

        Args:
            task: 可執行任務
            tool_decision: 工具決策
            trace_session_id: 軌跡會話 ID

        Returns:
            執行結果
        """
        # 開始執行上下文
        context = self.monitor.start_task_execution(
            trace_session_id=trace_session_id, task=task, tool_decision=tool_decision
        )

        try:
            # 根據服務類型執行不同的邏輯
            output = await self._execute_by_service_type(context, task, tool_decision)

            # 記錄成功
            self.monitor.complete_task_execution(context, output, success=True)

            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                output=output,
                trace_session_id=trace_session_id,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.task_id} failed: {error_msg}")

            # 記錄錯誤
            self.monitor.record_error(context, error_msg)
            self.monitor.complete_task_execution(
                context, {"error": error_msg}, success=False
            )

            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                output={},
                error=error_msg,
                trace_session_id=trace_session_id,
            )

    async def _execute_by_service_type(
        self,
        context: ExecutionContext,
        task: ExecutableTask,
        tool_decision: ToolDecision,
    ) -> dict[str, Any]:
        """根據服務類型執行任務

        Args:
            context: 執行上下文
            task: 任務
            tool_decision: 工具決策

        Returns:
            執行輸出
        """
        service_type = tool_decision.service_type.value

        # 記錄決策
        self.monitor.record_decision_point(
            context=context,
            decision_type="service_selection",
            options=["scan", "function", "integration", "core"],
            chosen_option=service_type,
            reason=f"Based on task type: {task.task_type}",
        )

        # TODO: 實際與各服務整合
        # 目前使用 Mock 實現

        if "scan" in service_type:
            return await self._execute_scan_service(context, task, tool_decision)
        elif "function" in service_type:
            return await self._execute_function_service(context, task, tool_decision)
        elif "integration" in service_type:
            return await self._execute_integration_service(context, task, tool_decision)
        else:
            return await self._execute_core_service(context, task, tool_decision)

    async def _execute_scan_service(
        self,
        context: ExecutionContext,
        task: ExecutableTask,
        tool_decision: ToolDecision,
    ) -> dict[str, Any]:
        """執行掃描服務

        Args:
            context: 執行上下文
            task: 任務
            tool_decision: 工具決策

        Returns:
            掃描結果
        """
        self.monitor.record_step(
            context, "scan_target", {"url": task.parameters.get("url")}
        )

        # Mock 實現
        result = {
            "scanned_urls": 10,
            "discovered_parameters": 5,
            "scan_duration": 2.5,
        }

        self.monitor.record_tool_invocation(
            context,
            tool_name="scan_service",
            input_params=task.parameters,
            output=result,
        )

        return result

    async def _execute_function_service(
        self,
        context: ExecutionContext,
        task: ExecutableTask,
        tool_decision: ToolDecision,
    ) -> dict[str, Any]:
        """執行功能服務（漏洞測試）

        Args:
            context: 執行上下文
            task: 任務
            tool_decision: 工具決策

        Returns:
            測試結果
        """
        self.monitor.record_step(
            context,
            "exploit_vulnerability",
            {
                "target": task.parameters.get("url"),
                "payload": task.parameters.get("payload"),
            },
        )

        # Mock 實現
        result = {
            "vulnerability_found": True,
            "severity": "high",
            "confidence": 0.85,
            "evidence": "SQL error message detected",
        }

        self.monitor.record_tool_invocation(
            context,
            tool_name=tool_decision.service_type.value,
            input_params=task.parameters,
            output=result,
        )

        return result

    async def _execute_integration_service(
        self,
        context: ExecutionContext,
        task: ExecutableTask,
        tool_decision: ToolDecision,
    ) -> dict[str, Any]:
        """執行整合服務

        Args:
            context: 執行上下文
            task: 任務
            tool_decision: 工具決策

        Returns:
            執行結果
        """
        self.monitor.record_step(context, "integrate_task", task.parameters)

        # Mock 實現
        result = {"status": "completed", "message": "Integration task executed"}

        self.monitor.record_tool_invocation(
            context,
            tool_name="integration_service",
            input_params=task.parameters,
            output=result,
        )

        return result

    async def _execute_core_service(
        self,
        context: ExecutionContext,
        task: ExecutableTask,
        tool_decision: ToolDecision,
    ) -> dict[str, Any]:
        """執行核心服務（分析）

        Args:
            context: 執行上下文
            task: 任務
            tool_decision: 工具決策

        Returns:
            分析結果
        """
        self.monitor.record_step(context, "analyze_data", task.parameters)

        # Mock 實現
        result = {
            "analysis_complete": True,
            "findings": 3,
            "recommendations": ["test parameter X", "check for SQLi", "validate CSRF"],
        }

        self.monitor.record_tool_invocation(
            context,
            tool_name="core_analyzer",
            input_params=task.parameters,
            output=result,
        )

        return result
