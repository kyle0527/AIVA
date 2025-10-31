"""
AIVA Common AI Plan Executor - 可插拔計劃執行器

此文件提供符合 aiva_common 規範的計劃執行器基礎實現，
支援可插拔架構設計，與五大模組架構相容。

設計特點:
- 實現 IPlanExecutor 介面
- 支援 _wait_for_result() 和結果訂閱機制
- 異步執行和追蹤支援
- 可替換的執行策略
- 與現有 services.core.aiva_core.execution.plan_executor 相容

架構位置:
- 屬於 Common 層的共享組件
- 可被 Integration 層和 Core 層使用
- 支援跨模組的計劃執行需求
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..schemas import (
    AttackPlan,
    AttackStep,
    PlanExecutionMetrics,
    PlanExecutionResult,
    TraceRecord,
)
from .interfaces import IPlanExecutor

logger = logging.getLogger(__name__)


class ExecutionConfig:
    """執行配置類"""

    def __init__(
        self,
        default_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_sandbox: bool = True,
        enable_tracing: bool = True,
        result_subscription_timeout: float = 60.0,
    ):
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_sandbox = enable_sandbox
        self.enable_tracing = enable_tracing
        self.result_subscription_timeout = result_subscription_timeout


class AIVAPlanExecutor(IPlanExecutor):
    """AIVA 可插拔計劃執行器實現

    此類提供符合 aiva_common 規範的計劃執行功能，
    支援可插拔架構並與五大模組架構相容。
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        message_queue_client: Any | None = None,
        storage_backend: Any | None = None,
        trace_logger: Any | None = None,
    ):
        """初始化計劃執行器

        Args:
            config: 執行配置
            message_queue_client: 訊息佇列客戶端 (可插拔)
            storage_backend: 儲存後端 (可插拔)
            trace_logger: 追蹤記錄器 (可插拔)
        """
        self.config = config or ExecutionConfig()
        self.mq_client = message_queue_client
        self.storage = storage_backend
        self.trace_logger = trace_logger

        # 執行狀態管理
        self.active_executions: dict[str, dict[str, Any]] = {}
        self.pending_results: dict[str, asyncio.Future[Any]] = {}
        self.execution_metrics: dict[str, dict[str, Any]] = {}

        # 結果訂閱機制
        self._result_subscribers: dict[str, list[asyncio.Queue[Any]]] = {}
        self._subscription_lock = asyncio.Lock()

        logger.info("AIVAPlanExecutor initialized with pluggable architecture")

    async def execute_plan(
        self,
        plan: AttackPlan,
        sandbox_mode: bool = True,
        timeout_minutes: int = 30,
    ) -> PlanExecutionResult:
        """執行攻擊計劃

        Args:
            plan: 攻擊計劃
            sandbox_mode: 沙箱模式
            timeout_minutes: 超時分鐘數

        Returns:
            執行結果
        """
        execution_id = f"exec_{uuid4().hex[:12]}"
        start_time = datetime.now(UTC)

        logger.info(
            f"Starting plan execution {execution_id} for plan {plan.plan_id} "
            f"(sandbox: {sandbox_mode}, timeout: {timeout_minutes}min)"
        )

        # 初始化執行狀態
        self.active_executions[execution_id] = {
            "plan_id": plan.plan_id,
            "start_time": start_time,
            "status": "running",
            "current_step": 0,
            "total_steps": len(plan.steps),
            "sandbox_mode": sandbox_mode,
        }

        trace_records: list[TraceRecord] = []
        findings: list[dict[str, Any]] = []
        anomalies: list[str] = []

        try:
            # 執行計劃步驟
            for step_index, step in enumerate(plan.steps):
                # 更新執行狀態
                self.active_executions[execution_id]["current_step"] = step_index + 1

                logger.info(
                    f"Executing step {step_index + 1}/{len(plan.steps)}: "
                    f"{step.step_id} ({step.action})"
                )

                # 檢查依賴關係
                if not await self._check_step_dependencies(step, plan, trace_records):
                    logger.warning(
                        f"Step {step.step_id} dependencies not met, skipping"
                    )
                    continue

                # 執行步驟
                trace = await self._execute_single_step(
                    execution_id, plan, step, sandbox_mode
                )
                trace_records.append(trace)

                # 收集結果
                if trace.output_data.get("findings"):
                    findings.extend(trace.output_data["findings"])

                if trace.status == "failed":
                    anomalies.append(
                        f"Step {step.step_id} failed: {trace.error_message}"
                    )

                # 檢查是否應該繼續
                if not await self._should_continue_execution(step, trace, plan):
                    logger.info(f"Stopping execution after step {step.step_id}")
                    break

            # 計算執行指標
            metrics = self._calculate_execution_metrics(
                execution_id, plan, trace_records, start_time
            )

            # 生成執行結果
            result = PlanExecutionResult(
                result_id=f"result_{execution_id}",
                plan_id=plan.plan_id,
                session_id=execution_id,
                plan=plan,
                trace=trace_records,
                metrics=metrics,
                findings=findings,
                anomalies=anomalies,
                recommendations=self._generate_recommendations(metrics, trace_records),
                status="completed" if metrics.completion_rate >= 0.8 else "partial",
                completed_at=datetime.now(UTC),
            )

            # 持久化結果
            if self.storage:
                await self._persist_execution_result(result)

            # 更新執行狀態
            self.active_executions[execution_id]["status"] = "completed"
            self.execution_metrics[execution_id] = metrics.model_dump()

            logger.info(
                f"Plan execution {execution_id} completed: "
                f"{metrics.completed_steps}/{metrics.expected_steps} steps, "
                f"success_rate={metrics.success_rate:.2%}"
            )

            return result

        except Exception as e:
            logger.error(f"Plan execution {execution_id} failed: {e}", exc_info=True)

            # 更新執行狀態
            self.active_executions[execution_id]["status"] = "failed"
            self.active_executions[execution_id]["error"] = str(e)

            anomalies.append(f"Execution error: {str(e)}")

            # 返回失敗結果
            metrics = self._calculate_execution_metrics(
                execution_id, plan, trace_records, start_time
            )

            return PlanExecutionResult(
                result_id=f"result_{execution_id}",
                plan_id=plan.plan_id,
                session_id=execution_id,
                plan=plan,
                trace=trace_records,
                metrics=metrics,
                findings=findings,
                anomalies=anomalies,
                recommendations=["執行失敗，建議檢查系統狀態和計劃配置"],
                status="failed",
                completed_at=datetime.now(UTC),
            )

    async def wait_for_result(
        self, task_id: str, timeout: float = 30.0
    ) -> dict[str, Any]:
        """等待任務結果 (實現結果訂閱機制)

        此方法實現了報告中要求的 _wait_for_result() 功能，
        支援異步結果訂閱和超時處理。

        Args:
            task_id: 任務ID
            timeout: 超時秒數

        Returns:
            任務執行結果
        """
        logger.info(f"Waiting for result of task {task_id} (timeout: {timeout}s)")

        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout

        # 初始化result_queue為None
        result_queue: asyncio.Queue[Any] | None = None

        try:
            # 1. 檢查是否已有結果
            if task_id in self.pending_results:
                future = self.pending_results[task_id]
                try:
                    remaining_timeout = max(
                        0, end_time - asyncio.get_event_loop().time()
                    )
                    result = await asyncio.wait_for(future, timeout=remaining_timeout)
                    logger.info(f"Retrieved cached result for task {task_id}")
                    return result
                except TimeoutError:
                    logger.warning(
                        f"Timeout waiting for cached result of task {task_id}"
                    )
                    return self._create_timeout_result(task_id, timeout)

            # 2. 創建結果訂閱
            result_queue = asyncio.Queue(maxsize=1)
            async with self._subscription_lock:
                if task_id not in self._result_subscribers:
                    self._result_subscribers[task_id] = []
                self._result_subscribers[task_id].append(result_queue)

            # 3. 發送任務 (如果有 MQ 客戶端)
            if self.mq_client:
                task_sent = await self._send_task_to_queue(task_id, timeout)
                if not task_sent:
                    logger.warning(f"Failed to send task {task_id} to message queue")

            # 4. 等待結果
            try:
                remaining_timeout = max(0, end_time - asyncio.get_event_loop().time())
                result = await asyncio.wait_for(
                    result_queue.get(), timeout=remaining_timeout
                )
                logger.info(f"Received subscribed result for task {task_id}")
                return result

            except TimeoutError:
                logger.warning(
                    f"Timeout waiting for subscribed result of task {task_id}"
                )
                return self._create_timeout_result(task_id, timeout)

        except Exception as e:
            logger.error(f"Error waiting for result of task {task_id}: {e}")
            return self._create_error_result(task_id, str(e))

        finally:
            # 清理訂閱
            if result_queue is not None:
                async with self._subscription_lock:
                    if task_id in self._result_subscribers:
                        try:
                            self._result_subscribers[task_id].remove(result_queue)
                            if not self._result_subscribers[task_id]:
                                del self._result_subscribers[task_id]
                        except ValueError:
                            pass  # 隊列已被移除

    async def get_execution_status(self, plan_id: str) -> dict[str, Any]:
        """獲取執行狀態

        Args:
            plan_id: 計劃ID

        Returns:
            執行狀態信息
        """
        # 查找匹配的執行
        for exec_id, exec_info in self.active_executions.items():
            if exec_info["plan_id"] == plan_id:
                status = {
                    "execution_id": exec_id,
                    "plan_id": plan_id,
                    "status": exec_info["status"],
                    "current_step": exec_info["current_step"],
                    "total_steps": exec_info["total_steps"],
                    "start_time": exec_info["start_time"].isoformat(),
                    "sandbox_mode": exec_info["sandbox_mode"],
                    "elapsed_time": (
                        datetime.now(UTC) - exec_info["start_time"]
                    ).total_seconds(),
                }

                if "error" in exec_info:
                    status["error"] = exec_info["error"]

                if exec_id in self.execution_metrics:
                    status["metrics"] = self.execution_metrics[exec_id]

                return status

        return {
            "plan_id": plan_id,
            "status": "not_found",
            "message": f"No active execution found for plan {plan_id}",
        }

    async def notify_result(self, task_id: str, result: dict[str, Any]) -> None:
        """通知任務結果 (供外部系統調用)

        Args:
            task_id: 任務ID
            result: 任務結果
        """
        logger.info(f"Received result notification for task {task_id}")

        # 通知所有訂閱者
        async with self._subscription_lock:
            if task_id in self._result_subscribers:
                subscribers = self._result_subscribers[task_id].copy()
                for queue in subscribers:
                    try:
                        await queue.put(result)
                    except Exception as e:
                        logger.warning(
                            f"Failed to notify subscriber for task {task_id}: {e}"
                        )

                # 清理訂閱者
                del self._result_subscribers[task_id]

        # 設置 future 結果
        if task_id in self.pending_results:
            future = self.pending_results[task_id]
            if not future.done():
                future.set_result(result)

    async def _execute_single_step(
        self, execution_id: str, plan: AttackPlan, step: AttackStep, sandbox_mode: bool
    ) -> TraceRecord:
        """執行單個步驟

        Args:
            execution_id: 執行ID
            plan: 攻擊計劃
            step: 攻擊步驟
            sandbox_mode: 沙箱模式

        Returns:
            追蹤記錄
        """
        step_start_time = datetime.now(UTC)
        trace_id = f"trace_{uuid4().hex[:8]}"

        try:
            # 準備任務參數
            task_params = {
                "step_id": step.step_id,
                "action": step.action,
                "tool_type": step.tool_type,
                "target": step.target,
                "parameters": step.parameters,
                "sandbox_mode": sandbox_mode,
            }

            # 生成任務ID
            task_id = f"task_{step.step_id}_{uuid4().hex[:8]}"

            # 等待步驟結果
            result = await self.wait_for_result(task_id, timeout=step.timeout_seconds)

            execution_time = (datetime.now(UTC) - step_start_time).total_seconds()

            # 創建追蹤記錄
            trace = TraceRecord(
                trace_id=trace_id,
                plan_id=plan.plan_id,
                step_id=step.step_id,
                session_id=execution_id,
                tool_name=step.tool_type,
                input_data=task_params,
                output_data=result,
                status="success" if result.get("success", False) else "failed",
                error_message=result.get("error"),
                execution_time_seconds=execution_time,
                timestamp=step_start_time,
                environment_response=result.get("environment_response", {}),
                metadata={
                    "execution_id": execution_id,
                    "sandbox_mode": sandbox_mode,
                    "task_id": task_id,
                },
            )

            # 記錄追蹤 (如果有追蹤記錄器)
            if self.trace_logger:
                await self.trace_logger.log_trace(trace)

            return trace

        except Exception as e:
            execution_time = (datetime.now(UTC) - step_start_time).total_seconds()
            error_msg = f"Step execution error: {str(e)}"

            logger.error(f"Step {step.step_id} execution failed: {e}", exc_info=True)

            return TraceRecord(
                trace_id=trace_id,
                plan_id=plan.plan_id,
                step_id=step.step_id,
                session_id=execution_id,
                tool_name=step.tool_type,
                input_data={"step_id": step.step_id, "error": "execution_failed"},
                output_data={},
                status="error",
                error_message=error_msg,
                execution_time_seconds=execution_time,
                timestamp=step_start_time,
                metadata={"execution_id": execution_id, "exception": str(e)},
            )

    async def _check_step_dependencies(
        self, step: AttackStep, plan: AttackPlan, completed_traces: list[TraceRecord]
    ) -> bool:
        """檢查步驟依賴關係

        Args:
            step: 攻擊步驟
            plan: 攻擊計劃
            completed_traces: 已完成的追蹤記錄

        Returns:
            依賴是否滿足
        """
        if step.step_id not in plan.dependencies:
            return True

        required_steps = plan.dependencies[step.step_id]
        completed_steps = {
            trace.step_id for trace in completed_traces if trace.status == "success"
        }

        return all(dep_step in completed_steps for dep_step in required_steps)

    async def _should_continue_execution(
        self, step: AttackStep, trace: TraceRecord, plan: AttackPlan
    ) -> bool:
        """判斷是否應該繼續執行

        Args:
            step: 當前步驟
            trace: 追蹤記錄
            plan: 攻擊計劃

        Returns:
            是否繼續執行
        """
        # 如果步驟失敗且沒有重試次數，停止執行
        if trace.status in ["failed", "error"] and step.retry_count == 0:
            return False

        # 如果達到目標，可以選擇提前停止
        if trace.output_data.get("goal_achieved", False):
            return False

        return True

    def _calculate_execution_metrics(
        self,
        execution_id: str,
        plan: AttackPlan,
        trace_records: list[TraceRecord],
        start_time: datetime,
    ) -> PlanExecutionMetrics:
        """計算執行指標

        Args:
            execution_id: 執行ID
            plan: 攻擊計劃
            trace_records: 追蹤記錄
            start_time: 開始時間

        Returns:
            執行指標
        """
        expected_steps = len(plan.steps)
        executed_steps = len(trace_records)
        completed_steps = sum(1 for t in trace_records if t.status == "success")
        failed_steps = sum(1 for t in trace_records if t.status in ["failed", "error"])
        skipped_steps = expected_steps - executed_steps

        completion_rate = (
            completed_steps / expected_steps if expected_steps > 0 else 0.0
        )
        success_rate = completed_steps / executed_steps if executed_steps > 0 else 0.0
        sequence_accuracy = self._calculate_sequence_accuracy(plan, trace_records)

        goal_achieved = completion_rate >= 0.8 and success_rate >= 0.9
        reward_score = (
            completion_rate * 0.4 + success_rate * 0.4 + sequence_accuracy * 0.2
        )

        total_execution_time = (datetime.now(UTC) - start_time).total_seconds()

        return PlanExecutionMetrics(
            plan_id=plan.plan_id,
            session_id=execution_id,
            expected_steps=expected_steps,
            executed_steps=executed_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            extra_actions=0,
            completion_rate=completion_rate,
            success_rate=success_rate,
            sequence_accuracy=sequence_accuracy,
            goal_achieved=goal_achieved,
            reward_score=reward_score,
            total_execution_time=total_execution_time,
        )

    def _calculate_sequence_accuracy(
        self, plan: AttackPlan, trace_records: list[TraceRecord]
    ) -> float:
        """計算順序準確度"""
        if not trace_records:
            return 0.0

        expected_order = [step.step_id for step in plan.steps]
        actual_order = [trace.step_id for trace in trace_records]

        matches = sum(
            1
            for i, step_id in enumerate(actual_order)
            if i < len(expected_order) and step_id == expected_order[i]
        )

        return matches / len(expected_order) if expected_order else 0.0

    def _generate_recommendations(
        self, metrics: PlanExecutionMetrics, trace_records: list[TraceRecord]
    ) -> list[str]:
        """生成改進建議"""
        recommendations: list[str] = []

        if metrics.success_rate < 0.5:
            recommendations.append("成功率過低，建議檢查工具配置和目標可達性")

        if metrics.completion_rate < 0.7:
            recommendations.append("完成率不足，建議優化步驟依賴和錯誤處理")

        if metrics.sequence_accuracy < 0.8:
            recommendations.append("執行順序偏差，建議檢查依賴關係定義")

        failed_traces = [t for t in trace_records if t.status in ["failed", "error"]]
        if failed_traces:
            recommendations.append(
                f"有 {len(failed_traces)} 個步驟失敗，建議增加重試機制"
            )

        return recommendations

    async def _send_task_to_queue(self, task_id: str, timeout: float) -> bool:
        """發送任務到訊息佇列"""
        if not self.mq_client:
            return False

        try:
            message = {
                "task_id": task_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "timeout": timeout,
            }
            await self.mq_client.publish("tasks.execution", message)
            return True
        except Exception as e:
            logger.error(f"Failed to send task {task_id} to queue: {e}")
            return False

    async def _persist_execution_result(self, result: PlanExecutionResult) -> None:
        """持久化執行結果"""
        if not self.storage:
            return

        try:
            await self.storage.save_execution_result(result.model_dump())
            logger.debug(f"Persisted execution result {result.result_id}")
        except Exception as e:
            logger.error(f"Failed to persist execution result: {e}")

    def _create_timeout_result(self, task_id: str, timeout: float) -> dict[str, Any]:
        """創建超時結果"""
        return {
            "task_id": task_id,
            "success": False,
            "status": "timeout",
            "error": f"Task execution timeout after {timeout}s",
            "findings": [],
            "metadata": {"timeout": timeout},
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _create_error_result(self, task_id: str, error: str) -> dict[str, Any]:
        """創建錯誤結果"""
        return {
            "task_id": task_id,
            "success": False,
            "status": "error",
            "error": error,
            "findings": [],
            "metadata": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def cleanup(self) -> None:
        """清理資源"""
        try:
            # 清理活躍執行
            self.active_executions.clear()

            # 清理待處理結果
            for future in self.pending_results.values():
                if not future.done():
                    future.cancel()
            self.pending_results.clear()

            # 清理訂閱者
            async with self._subscription_lock:
                self._result_subscribers.clear()

            # 清理指標
            self.execution_metrics.clear()

            logger.info("AIVAPlanExecutor cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# ============================================================================
# Factory Function (工廠函數)
# ============================================================================


def create_plan_executor(
    config: ExecutionConfig | None = None, **kwargs: Any
) -> AIVAPlanExecutor:
    """創建計劃執行器實例

    Args:
        config: 執行配置
        **kwargs: 其他參數 (mq_client, storage, trace_logger)

    Returns:
        計劃執行器實例
    """
    return AIVAPlanExecutor(config=config, **kwargs)
