"""AIVA Execution Planner - 執行計劃器
從 aiva_core_v2 遷移到核心模組

異步執行計劃和步驟編排系統
"""

import asyncio
import logging
import time
from typing import Any

# aiva_common 統一錯誤處理
from aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context as create_error_ctx,
)
from aiva_common.cross_language import get_error_handler
from ..command_router import CommandContext, CommandType, ExecutionResult

MODULE_NAME = "execution_planner"


class ExecutionPlanner:
    """執行計劃器 - 負責異步執行計劃和步驟編排"""

    def __init__(self):
        self.logger = logging.getLogger("execution_planner")
        self._execution_queue: list[dict[str, Any]] = []
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._plan_history: dict[str, dict[str, Any]] = {}
        self._execution_lock = asyncio.Lock()

    def create_execution_plan(
        self, context: CommandContext, route_info: dict[str, Any]
    ) -> dict[str, Any]:
        """創建執行計劃"""
        plan = {
            "plan_id": f"plan_{int(time.time())}_{id(context)}",
            "context": context,
            "route_info": route_info,
            "steps": [],
            "estimated_time": 0.0,
            "resources_required": [],
            "dependencies": [],
            "created_at": time.time(),
            "status": "created",
        }

        # 根據命令類型制定執行步驟
        if route_info["type"] == CommandType.SIMPLE:
            plan["steps"] = [
                {
                    "type": "validate_input",
                    "handler": "input_validator",
                    "critical": True,
                },
                {
                    "type": "execute_command",
                    "handler": "simple_executor",
                    "critical": True,
                },
                {
                    "type": "format_output",
                    "handler": "output_formatter",
                    "critical": False,
                },
            ]
            plan["estimated_time"] = 1.0

        elif route_info["type"] == CommandType.SCAN:
            plan["steps"] = [
                {
                    "type": "validate_target",
                    "handler": "target_validator",
                    "critical": True,
                },
                {"type": "prepare_scan", "handler": "scan_preparer", "critical": True},
                {"type": "execute_scan", "handler": "rust_scanner", "critical": True},
                {
                    "type": "process_results",
                    "handler": "result_processor",
                    "critical": True,
                },
                {
                    "type": "generate_report",
                    "handler": "report_generator",
                    "critical": False,
                },
            ]
            plan["estimated_time"] = 30.0
            plan["resources_required"] = ["rust_adapter", "scan_engine"]

        elif route_info["type"] == CommandType.ANALYSIS:
            plan["steps"] = [
                {
                    "type": "validate_input",
                    "handler": "input_validator",
                    "critical": True,
                },
                {
                    "type": "gather_context",
                    "handler": "context_gatherer",
                    "critical": False,
                },
                {
                    "type": "analyze_data",
                    "handler": "analysis_engine",
                    "critical": True,
                },
                {
                    "type": "generate_insights",
                    "handler": "insight_generator",
                    "critical": False,
                },
                {
                    "type": "format_response",
                    "handler": "response_formatter",
                    "critical": False,
                },
            ]
            plan["estimated_time"] = 5.0
            plan["resources_required"] = ["analysis_engine"]

        elif route_info["requires_ai"]:
            plan["steps"] = [
                {
                    "type": "analyze_intent",
                    "handler": "intent_analyzer",
                    "critical": True,
                },
                {
                    "type": "gather_context",
                    "handler": "context_gatherer",
                    "critical": False,
                },
                {"type": "ai_reasoning", "handler": "ai_engine", "critical": True},
                {
                    "type": "execute_actions",
                    "handler": "action_executor",
                    "critical": True,
                },
                {
                    "type": "format_response",
                    "handler": "response_formatter",
                    "critical": False,
                },
            ]
            plan["estimated_time"] = 10.0
            plan["resources_required"] = ["ai_engine", "rag_system"]

        elif route_info["type"] == CommandType.REPORT:
            plan["steps"] = [
                {
                    "type": "validate_input",
                    "handler": "input_validator",
                    "critical": True,
                },
                {"type": "gather_data", "handler": "data_gatherer", "critical": True},
                {"type": "process_data", "handler": "data_processor", "critical": True},
                {
                    "type": "generate_report",
                    "handler": "report_generator",
                    "critical": True,
                },
                {
                    "type": "export_results",
                    "handler": "export_handler",
                    "critical": False,
                },
            ]
            plan["estimated_time"] = 15.0
            plan["resources_required"] = ["report_engine"]

        else:
            # 默認簡單執行計劃
            plan["steps"] = [
                {
                    "type": "validate_input",
                    "handler": "input_validator",
                    "critical": True,
                },
                {
                    "type": "execute_command",
                    "handler": "generic_executor",
                    "critical": True,
                },
                {
                    "type": "format_output",
                    "handler": "output_formatter",
                    "critical": False,
                },
            ]
            plan["estimated_time"] = 2.0

        # 保存計劃歷史
        self._plan_history[plan["plan_id"]] = plan

        self.logger.info(
            f"Created execution plan: {plan['plan_id']} with {len(plan['steps'])} steps"
        )
        return plan

    async def execute_plan(self, plan: dict[str, Any]) -> ExecutionResult:
        """執行計劃"""
        plan_id = plan["plan_id"]
        start_time = time.time()

        async with self._execution_lock:
            try:
                plan["status"] = "executing"
                plan["execution_start"] = start_time

                self.logger.info(f"Executing plan: {plan_id}")

                # 檢查資源可用性
                if not await self._check_resources(plan["resources_required"]):
                    raise AIVAError(
                        "Required resources not available",
                        error_type=ErrorType.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                        context=create_error_ctx(module=MODULE_NAME, function="execute_plan")
                    )

                # 執行各個步驟
                step_results = []
                context_data = {}  # 步驟間共享的上下文數據

                for i, step in enumerate(plan["steps"]):
                    step_start = time.time()

                    try:
                        self.logger.debug(f"Executing step {i}: {step['type']}")

                        # 準備步驟執行上下文
                        step_context = {
                            "plan": plan,
                            "step_index": i,
                            "previous_results": step_results,
                            "shared_context": context_data,
                        }

                        result = await self._execute_step(step, step_context)
                        step_duration = time.time() - step_start

                        step_result = {
                            "step_index": i,
                            "step_type": step["type"],
                            "result": result,
                            "duration": step_duration,
                            "status": "success",
                        }

                        step_results.append(step_result)

                        # 更新共享上下文
                        if isinstance(result, dict) and "context_updates" in result:
                            context_data.update(result["context_updates"])

                        self.logger.debug(
                            f"Step {i} completed: {step['type']} ({step_duration:.2f}s)"
                        )

                    except Exception as e:
                        step_duration = time.time() - step_start
                        step_result = {
                            "step_index": i,
                            "step_type": step["type"],
                            "error": str(e),
                            "duration": step_duration,
                            "status": "failed",
                        }

                        step_results.append(step_result)

                        # 如果是關鍵步驟失敗，停止執行
                        if step.get("critical", True):
                            self.logger.error(f"Critical step {i} failed: {e}")
                            raise
                        else:
                            self.logger.warning(f"Non-critical step {i} failed: {e}")

                execution_time = time.time() - start_time
                plan["status"] = "completed"
                plan["execution_end"] = time.time()

                # 匯總結果
                final_result = self._aggregate_results(step_results, context_data)

                return ExecutionResult(
                    success=True,
                    result=final_result,
                    execution_time=execution_time,
                    metadata={
                        "plan_id": plan_id,
                        "step_results": step_results,
                        "steps_completed": len(
                            [r for r in step_results if r["status"] == "success"]
                        ),
                        "total_steps": len(plan["steps"]),
                        "estimated_time": plan["estimated_time"],
                        "actual_time": execution_time,
                    },
                )

            except Exception as e:
                execution_time = time.time() - start_time
                plan["status"] = "failed"
                plan["execution_end"] = time.time()
                plan["error"] = str(e)

                # 創建錯誤上下文
                error_context = create_error_context(
                    service_name="execution_planner",
                    function_name="execute_plan",
                    additional_context={
                        "plan_id": plan_id,
                        "steps_completed": len(
                            [r for r in step_results if r.get("status") == "success"]
                        ),
                        "total_steps": len(plan["steps"]),
                    },
                )

                error_handler_instance = get_error_handler()
                aiva_error = error_handler_instance.handle_error(e, error_context)

                return ExecutionResult(
                    success=False,
                    error=aiva_error,
                    execution_time=execution_time,
                    metadata={
                        "plan_id": plan_id,
                        "failed_at_step": len(step_results),
                        "total_steps": len(plan["steps"]),
                    },
                )

    def _check_resources(self, required_resources: list[str]) -> bool:
        """檢查所需資源是否可用"""
        # 實現資源檢查邏輯
        available = {"ai_engine", "rust_adapter", "scan_engine", "database"}
        return all(resource in available for resource in required_resources)

    async def _execute_step(
        self, step: dict[str, Any], step_context: dict[str, Any]
    ) -> Any:
        """執行單個步驟"""
        handler_name = step["handler"]

        # 模擬不同類型的步驟執行
        if handler_name == "input_validator":
            return await self._validate_input(step_context)
        elif handler_name == "simple_executor":
            return await self._execute_simple_command(step_context)
        elif handler_name == "output_formatter":
            return await self._format_output(step_context)
        elif handler_name == "ai_engine":
            return await self._execute_ai_task(step_context)
        elif handler_name == "rust_scanner":
            return await self._execute_rust_scan(step_context)
        elif handler_name == "report_generator":
            return await self._generate_report(step_context)
        else:
            # 默認處理器
            return await self._execute_generic_step(step_context)

    def _validate_input(self, context: dict[str, Any]) -> dict[str, Any]:
        """輸入驗證步驟"""
        command_context = context["plan"]["context"]

        # 基本驗證邏輯
        if not command_context.command:
            raise AIVAError(
                "Command is required",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=create_error_ctx(module=MODULE_NAME, function="_validate_input")
            )

        return {
            "validation_result": "passed",
            "validated_command": command_context.command,
            "validated_args": command_context.args,
        }

    async def _execute_simple_command(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行簡單命令"""
        command_context = context["plan"]["context"]

        # 模擬命令執行
        await asyncio.sleep(0.1)  # 模擬執行時間

        return {
            "command": command_context.command,
            "status": "completed",
            "message": f"Command '{command_context.command}' executed successfully",
        }

    def _format_output(self, context: dict[str, Any]) -> dict[str, Any]:
        """格式化輸出"""
        previous_results = context["previous_results"]

        # 收集所有執行結果
        results = []
        for result in previous_results:
            if result["status"] == "success":
                results.append(result["result"])

        return {
            "formatted_output": {
                "status": "success",
                "results": results,
                "timestamp": time.time(),
            }
        }

    async def _execute_ai_task(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行AI任務"""
        # 模擬AI處理
        await asyncio.sleep(1.0)  # 모拟AI推理時間

        return {
            "ai_result": "AI processing completed",
            "confidence": 0.95,
            "reasoning": "Based on the input analysis...",
        }

    async def _execute_rust_scan(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行Rust掃描"""
        # 模擬Rust掃描
        await asyncio.sleep(2.0)  # 模擬掃描時間

        return {
            "scan_result": "Scan completed",
            "vulnerabilities_found": 0,
            "scan_duration": 2.0,
        }

    async def _generate_report(self, context: dict[str, Any]) -> dict[str, Any]:
        """生成報告"""
        # 模擬報告生成
        await asyncio.sleep(0.5)

        return {"report_generated": True, "report_type": "standard", "pages": 5}

    async def _execute_generic_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """通用步驟執行器"""
        await asyncio.sleep(0.1)
        return {"status": "completed", "message": "Generic step completed"}

    def _aggregate_results(
        self, step_results: list[dict[str, Any]], context_data: dict[str, Any]
    ) -> dict[str, Any]:
        """匯總執行結果"""
        successful_steps = [r for r in step_results if r["status"] == "success"]
        failed_steps = [r for r in step_results if r["status"] == "failed"]

        # 收集最終結果
        final_output = None
        for result in reversed(step_results):
            if result["status"] == "success" and "formatted_output" in result.get(
                "result", {}
            ):
                final_output = result["result"]["formatted_output"]
                break

        if not final_output:
            # 如果沒有格式化輸出，創建默認輸出
            final_output = {
                "status": "completed" if not failed_steps else "partial",
                "results": [r["result"] for r in successful_steps],
                "timestamp": time.time(),
            }

        return {
            "output": final_output,
            "execution_summary": {
                "total_steps": len(step_results),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "total_duration": sum(r["duration"] for r in step_results),
            },
            "context_data": context_data,
        }

    def get_plan_status(self, plan_id: str) -> dict[str, Any] | None:
        """獲取計劃狀態"""
        if plan_id in self._plan_history:
            plan = self._plan_history[plan_id]
            return {
                "plan_id": plan_id,
                "status": plan.get("status", "unknown"),
                "created_at": plan.get("created_at"),
                "execution_start": plan.get("execution_start"),
                "execution_end": plan.get("execution_end"),
                "estimated_time": plan.get("estimated_time"),
                "steps_count": len(plan.get("steps", [])),
            }
        return None

    def cancel_plan(self, plan_id: str) -> bool:
        """取消計劃執行"""
        if plan_id in self._running_tasks:
            task = self._running_tasks[plan_id]
            task.cancel()
            del self._running_tasks[plan_id]

            if plan_id in self._plan_history:
                self._plan_history[plan_id]["status"] = "cancelled"

            self.logger.info(f"Cancelled plan: {plan_id}")
            return True
        return False

    def get_execution_stats(self) -> dict[str, Any]:
        """獲取執行統計資訊"""
        total_plans = len(self._plan_history)
        completed_plans = len(
            [p for p in self._plan_history.values() if p.get("status") == "completed"]
        )
        failed_plans = len(
            [p for p in self._plan_history.values() if p.get("status") == "failed"]
        )
        running_plans = len(self._running_tasks)

        avg_execution_time = 0
        if completed_plans > 0:
            execution_times = []
            for plan in self._plan_history.values():
                if (
                    plan.get("status") == "completed"
                    and "execution_start" in plan
                    and "execution_end" in plan
                ):
                    execution_times.append(
                        plan["execution_end"] - plan["execution_start"]
                    )

            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)

        return {
            "total_plans": total_plans,
            "completed_plans": completed_plans,
            "failed_plans": failed_plans,
            "running_plans": running_plans,
            "success_rate": (completed_plans / max(total_plans, 1)) * 100,
            "average_execution_time": avg_execution_time,
        }


# 全局執行計劃器實例
_execution_planner_instance = None


def get_execution_planner() -> ExecutionPlanner:
    """獲取執行計劃器實例"""
    global _execution_planner_instance
    if _execution_planner_instance is None:
        _execution_planner_instance = ExecutionPlanner()
    return _execution_planner_instance
