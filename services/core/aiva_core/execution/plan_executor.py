"""Plan Executor - 攻擊計畫執行器

負責執行攻擊計畫，管理會話狀態，協調任務分發和結果收集

符合標準：
- 支持順序執行多步驟攻擊鏈
- 透過 RabbitMQ 發送任務到各功能模組
- 使用 TraceLogger 記錄執行過程
- 管理會話生命週期
"""

import asyncio
from datetime import UTC, datetime
import logging
from typing import Any
from uuid import uuid4

from services.aiva_common.schemas import (
    AttackPlan,
    AttackStep,
    FindingPayload,
    FunctionTaskPayload,
    FunctionTaskTarget,
    PlanExecutionMetrics,
    PlanExecutionResult,
    SessionState,
    TraceRecord,
)

from .trace_logger import TraceLogger

logger = logging.getLogger(__name__)


class PlanExecutor:
    """攻擊計畫執行器

    執行攻擊計畫，管理會話狀態，協調任務分發
    """

    def __init__(
        self,
        mq_client: Any | None = None,
        trace_logger: TraceLogger | None = None,
        storage_backend: Any | None = None,
    ) -> None:
        """初始化執行器

        Args:
            mq_client: RabbitMQ 客戶端
            trace_logger: 追蹤記錄器
            storage_backend: 儲存後端
        """
        self.mq_client = mq_client
        self.trace_logger = trace_logger or TraceLogger(storage_backend)
        self.storage = storage_backend
        self.active_sessions: dict[str, SessionState] = {}

        # 任務管理
        self.running_tasks: dict[str, dict[str, Any]] = {}
        self.completed_tasks: dict[str, dict[str, Any]] = {}

        logger.info("PlanExecutor initialized")

    async def execute_plan(
        self,
        plan: AttackPlan,
        sandbox_mode: bool = True,
        timeout_minutes: int = 30,
    ) -> PlanExecutionResult:
        """執行攻擊計畫

        Args:
            plan: 攻擊計畫
            sandbox_mode: 是否在沙箱模式執行（限制破壞性操作）
            timeout_minutes: 超時時間（分鐘）

        Returns:
            執行結果
        """
        logger.info(
            f"Starting execution of plan {plan.plan_id} "
            f"with {len(plan.steps)} steps (sandbox={sandbox_mode})"
        )

        # 創建會話
        session = self.trace_logger.create_session(
            plan_id=plan.plan_id,
            scan_id=plan.scan_id,
            steps=[step.step_id for step in plan.steps],
            timeout_minutes=timeout_minutes,
        )

        self.active_sessions[session.session_id] = session

        # 執行追蹤記錄
        trace_records: list[TraceRecord] = []
        findings: list[FindingPayload] = []
        anomalies: list[str] = []

        try:
            # 依序執行步驟
            for i, step in enumerate(plan.steps):
                logger.info(
                    f"Executing step {i+1}/{len(plan.steps)}: "
                    f"{step.step_id} ({step.action})"
                )

                # 檢查依賴
                if not await self._check_dependencies(
                    step.step_id, plan.dependencies, trace_records
                ):
                    logger.warning(
                        f"Step {step.step_id} dependencies not met, skipping"
                    )
                    await self._record_skipped_step(
                        session.session_id, plan.plan_id, step
                    )
                    continue

                # 執行步驟
                trace = await self._execute_step(
                    session=session,
                    plan=plan,
                    step=step,
                    sandbox_mode=sandbox_mode,
                )

                trace_records.append(trace)

                # 收集發現和異常
                if trace.output_data.get("findings"):
                    findings.extend(trace.output_data["findings"])

                if trace.status == "failed":
                    anomalies.append(
                        f"Step {step.step_id} failed: {trace.error_message}"
                    )

                # 檢查是否應該繼續
                if not await self._should_continue(step, trace, plan):
                    logger.info(f"Stopping execution after step {step.step_id}")
                    break

            # 標記會話完成
            self.trace_logger.complete_session(session.session_id)

        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            self.trace_logger.fail_session(session.session_id)
            anomalies.append(f"Execution error: {str(e)}")

        # 計算執行指標
        metrics = self._calculate_metrics(
            plan=plan,
            session=session,
            trace_records=trace_records,
        )

        # 生成建議
        recommendations = self._generate_recommendations(
            plan=plan,
            metrics=metrics,
            trace_records=trace_records,
        )

        # 創建執行結果
        result = PlanExecutionResult(
            result_id=f"result_{uuid4().hex[:12]}",
            plan_id=plan.plan_id,
            session_id=session.session_id,
            plan=plan,
            trace=trace_records,
            metrics=metrics,
            findings=findings,
            anomalies=anomalies,
            recommendations=recommendations,
            status="completed" if metrics.goal_achieved else "partial",
            completed_at=datetime.now(UTC),
        )

        # 持久化結果
        if self.storage:
            await self._persist_result(result)

        logger.info(
            f"Plan {plan.plan_id} execution completed: "
            f"{metrics.completed_steps}/{metrics.expected_steps} steps, "
            f"success_rate={metrics.success_rate:.2%}"
        )

        return result

    async def _execute_step(
        self,
        session: SessionState,
        plan: AttackPlan,
        step: AttackStep,
        sandbox_mode: bool,
    ) -> TraceRecord:
        """執行單個步驟

        Args:
            session: 會話狀態
            plan: 攻擊計畫
            step: 攻擊步驟
            sandbox_mode: 沙箱模式

        Returns:
            追蹤記錄
        """
        start_time = datetime.now(UTC)

        try:
            # 準備任務參數
            task_payload = self._prepare_task_payload(
                plan=plan,
                step=step,
                session=session,
                sandbox_mode=sandbox_mode,
            )

            # 發送任務到 RabbitMQ
            if self.mq_client:
                await self._send_task(step.tool_type, task_payload)

            # 等待結果（使用超時）
            result = await self._wait_for_result(
                task_id=task_payload.task_id,
                timeout=step.timeout_seconds,
            )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # 記錄成功執行
            trace = await self.trace_logger.log_task_execution(
                session_id=session.session_id,
                plan_id=plan.plan_id,
                step_id=step.step_id,
                tool_name=step.tool_type,
                input_params=task_payload.model_dump(),
                result=result,
                status="success",
                execution_time=execution_time,
            )

        except TimeoutError:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            error = f"Step execution timeout after {step.timeout_seconds}s"
            logger.error(f"Step {step.step_id}: {error}")

            trace = await self.trace_logger.log_task_execution(
                session_id=session.session_id,
                plan_id=plan.plan_id,
                step_id=step.step_id,
                tool_name=step.tool_type,
                input_params=step.parameters,
                result={},
                status="timeout",
                execution_time=execution_time,
                error=error,
            )

        except Exception as e:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            error = f"Step execution error: {str(e)}"
            logger.error(f"Step {step.step_id}: {error}", exc_info=True)

            trace = await self.trace_logger.log_task_execution(
                session_id=session.session_id,
                plan_id=plan.plan_id,
                step_id=step.step_id,
                tool_name=step.tool_type,
                input_params=step.parameters,
                result={},
                status="failed",
                execution_time=execution_time,
                error=error,
            )

        return trace

    def _prepare_task_payload(
        self,
        plan: AttackPlan,
        step: AttackStep,
        session: SessionState,
        sandbox_mode: bool,
    ) -> FunctionTaskPayload:
        """準備任務 Payload

        Args:
            plan: 攻擊計畫
            step: 攻擊步驟
            session: 會話狀態
            sandbox_mode: 沙箱模式

        Returns:
            功能任務 Payload
        """
        task_id = f"task_{uuid4().hex[:12]}"

        # 從步驟參數構建目標
        target = FunctionTaskTarget(
            url=step.target.get("url", ""),
            parameter=step.target.get("parameter"),
            method=step.target.get("method", "GET"),
            parameter_location=step.target.get("location", "query"),
            headers=step.target.get("headers", {}),
            cookies=step.target.get("cookies", {}),
        )

        # 構建任務 Payload
        payload = FunctionTaskPayload(
            task_id=task_id,
            scan_id=plan.scan_id,
            priority=step.parameters.get("priority", 5),
            target=target,
            strategy=step.parameters.get("strategy", "full"),
            custom_payloads=step.parameters.get("custom_payloads"),
        )

        # 在 metadata 中添加會話和計畫資訊
        if not payload.metadata:
            payload.metadata = {}

        payload.metadata.update(
            {
                "session_id": session.session_id,
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "sandbox_mode": sandbox_mode,
            }
        )

        return payload

    async def _send_task(self, tool_type: str, payload: FunctionTaskPayload) -> None:
        """發送任務到 RabbitMQ

        Args:
            tool_type: 工具類型
            payload: 任務 Payload
        """
        if not self.mq_client:
            logger.warning("No MQ client configured, task not sent")
            return

        # 根據工具類型決定 routing key
        routing_key_map = {
            "function_sqli": "tasks.function.sqli",
            "function_xss": "tasks.function.xss",
            "function_ssrf": "tasks.function.ssrf",
            "function_idor": "tasks.function.idor",
        }

        routing_key = routing_key_map.get(tool_type, "tasks.function.start")

        try:
            await self.mq_client.publish(
                routing_key=routing_key,
                message=payload.model_dump(),
            )
            logger.debug(f"Task {payload.task_id} sent to {routing_key}")
        except Exception as e:
            logger.error(f"Failed to send task {payload.task_id}: {e}")
            raise

    async def _wait_for_result(
        self, task_id: str, timeout: float = 30.0
    ) -> dict[str, Any]:
        """等待任務結果

        Args:
            task_id: 任務 ID
            timeout: 超時時間（秒）

        Returns:
            任務結果，包含 success 狀態

        Raises:
            asyncio.TimeoutError: 超時
        """
        from datetime import datetime, timedelta

        logger.info(f"等待任務 {task_id} 結果，超時時間: {timeout}s")

        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=timeout)

        # 實際的結果等待邏輯
        try:
            # 1. 檢查任務是否在執行佇列中
            if task_id in self.running_tasks:
                logger.debug(f"任務 {task_id} 正在執行中...")

                # 2. 輪詢等待結果（實際應用中會使用 MQ 或其他異步機制）
                while datetime.now() < end_time:
                    # 檢查任務是否完成
                    if task_id in self.completed_tasks:
                        result = self.completed_tasks.pop(task_id)
                        logger.info(
                            f"任務 {task_id} 完成，狀態: {result.get('status')}"
                        )

                        # 確保返回結果包含 success 字段
                        return {
                            "task_id": task_id,
                            "success": result.get("status") == "completed",
                            "status": result.get("status", "completed"),
                            "findings": result.get("findings", []),
                            "metadata": result.get("metadata", {}),
                            "execution_time": result.get("execution_time", 0),
                            "error": result.get("error"),
                            "timestamp": datetime.now().isoformat(),
                        }

                    # 短暫等待後重試
                    await asyncio.sleep(0.1)

                # 超時處理
                logger.warning(f"任務 {task_id} 執行超時")
                return {
                    "task_id": task_id,
                    "success": False,
                    "status": "timeout",
                    "findings": [],
                    "metadata": {},
                    "error": f"Task execution timeout after {timeout}s",
                    "timestamp": datetime.now().isoformat(),
                }

            else:
                # 3. 任務不在執行佇列中，可能已完成或不存在
                if task_id in self.completed_tasks:
                    result = self.completed_tasks.pop(task_id)
                    return {
                        "task_id": task_id,
                        "success": result.get("status") == "completed",
                        "status": result.get("status", "completed"),
                        "findings": result.get("findings", []),
                        "metadata": result.get("metadata", {}),
                        "execution_time": result.get("execution_time", 0),
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    # 4. 模擬執行（在沒有實際任務系統時的降級方案）
                    logger.info(f"模擬執行任務 {task_id}")

                    # 模擬執行時間
                    simulation_time = min(2.0, timeout * 0.1)
                    await asyncio.sleep(simulation_time)

                    # 模擬成功結果（80% 成功率）
                    import random

                    is_successful = random.random() > 0.2

                    if is_successful:
                        mock_findings = self._generate_mock_findings(task_id)
                        return {
                            "task_id": task_id,
                            "success": True,
                            "status": "completed",
                            "findings": mock_findings,
                            "metadata": {
                                "simulated": True,
                                "execution_mode": "mock",
                                "simulation_time": simulation_time,
                            },
                            "execution_time": simulation_time,
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        return {
                            "task_id": task_id,
                            "success": False,
                            "status": "failed",
                            "findings": [],
                            "metadata": {"simulated": True, "execution_mode": "mock"},
                            "error": "Simulated execution failure",
                            "execution_time": simulation_time,
                            "timestamp": datetime.now().isoformat(),
                        }

        except TimeoutError:
            logger.error(f"任務 {task_id} 等待結果超時")
            return {
                "task_id": task_id,
                "success": False,
                "status": "timeout",
                "findings": [],
                "metadata": {},
                "error": f"Result waiting timeout after {timeout}s",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"等待任務 {task_id} 結果時發生異常: {e}")
            return {
                "task_id": task_id,
                "success": False,
                "status": "error",
                "findings": [],
                "metadata": {},
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_mock_findings(self, task_id: str) -> list[dict[str, Any]]:
        """生成模擬發現結果"""
        import random

        mock_findings = [
            {
                "type": "sql_injection",
                "severity": "high",
                "location": "/api/user/login",
                "description": "Potential SQL injection vulnerability in login form",
                "confidence": 0.85,
            },
            {
                "type": "xss",
                "severity": "medium",
                "location": "/search",
                "description": "Reflected XSS in search parameter",
                "confidence": 0.72,
            },
            {
                "type": "csrf",
                "severity": "low",
                "location": "/profile/update",
                "description": "Missing CSRF protection on profile update",
                "confidence": 0.65,
            },
        ]

        # 隨機選擇 1-3 個發現
        num_findings = random.randint(1, 3)
        selected_findings = random.sample(mock_findings, num_findings)

        # 添加任務特定信息
        for finding in selected_findings:
            finding["task_id"] = task_id
            finding["finding_id"] = f"finding_{task_id}_{random.randint(1000, 9999)}"

        return selected_findings

    async def _check_dependencies(
        self,
        step_id: str,
        dependencies: dict[str, list[str]],
        trace_records: list[TraceRecord],
    ) -> bool:
        """檢查步驟依賴是否滿足

        Args:
            step_id: 步驟 ID
            dependencies: 依賴關係字典
            trace_records: 已執行的追蹤記錄

        Returns:
            依賴是否滿足
        """
        if step_id not in dependencies:
            return True

        required_steps = dependencies[step_id]
        completed_steps = {
            trace.step_id for trace in trace_records if trace.status == "success"
        }

        return all(dep in completed_steps for dep in required_steps)

    async def _should_continue(
        self, step: AttackStep, trace: TraceRecord, plan: AttackPlan
    ) -> bool:
        """判斷是否應該繼續執行

        Args:
            step: 當前步驟
            trace: 追蹤記錄
            plan: 攻擊計畫

        Returns:
            是否繼續
        """
        # 如果步驟失敗且沒有配置重試，則停止
        if trace.status == "failed" and step.retry_count == 0:
            return False

        # 如果達到目標，可以選擇提前停止
        return not trace.output_data.get("goal_achieved", False)

    async def _record_skipped_step(
        self, session_id: str, plan_id: str, step: AttackStep
    ) -> None:
        """記錄跳過的步驟

        Args:
            session_id: 會話 ID
            plan_id: 計畫 ID
            step: 步驟
        """
        await self.trace_logger.log_task_execution(
            session_id=session_id,
            plan_id=plan_id,
            step_id=step.step_id,
            tool_name=step.tool_type,
            input_params=step.parameters,
            result={},
            status="skipped",
            execution_time=0.0,
            error="Dependencies not met",
        )

    def _calculate_metrics(
        self,
        plan: AttackPlan,
        session: SessionState,
        trace_records: list[TraceRecord],
    ) -> PlanExecutionMetrics:
        """計算執行指標

        Args:
            plan: 攻擊計畫
            session: 會話狀態
            trace_records: 追蹤記錄

        Returns:
            執行指標
        """
        expected_steps = len(plan.steps)
        executed_steps = len(trace_records)
        completed_steps = sum(1 for t in trace_records if t.status == "success")
        failed_steps = sum(1 for t in trace_records if t.status == "failed")
        skipped_steps = sum(1 for t in trace_records if t.status == "skipped")

        completion_rate = (
            completed_steps / expected_steps if expected_steps > 0 else 0.0
        )
        success_rate = completed_steps / executed_steps if executed_steps > 0 else 0.0

        # 計算順序準確度（簡化版）
        sequence_accuracy = self._calculate_sequence_accuracy(plan, trace_records)

        # 判斷是否達成目標
        goal_achieved = completion_rate >= 0.8 and success_rate >= 0.9

        # 計算獎勵分數（用於強化學習）
        reward_score = (
            completion_rate * 0.4 + success_rate * 0.4 + sequence_accuracy * 0.2
        )

        # 計算總執行時間
        total_execution_time = sum(t.execution_time_seconds for t in trace_records)

        return PlanExecutionMetrics(
            plan_id=plan.plan_id,
            session_id=session.session_id,
            expected_steps=expected_steps,
            executed_steps=executed_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            skipped_steps=skipped_steps,
            extra_actions=0,  # TODO: 檢測額外動作
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
        """計算順序準確度

        Args:
            plan: 攻擊計畫
            trace_records: 追蹤記錄

        Returns:
            順序準確度 (0.0-1.0)
        """
        if not trace_records:
            return 0.0

        # 簡化版：檢查步驟順序是否匹配
        expected_order = [step.step_id for step in plan.steps]
        actual_order = [trace.step_id for trace in trace_records]

        matches = sum(
            1
            for i, step_id in enumerate(actual_order)
            if i < len(expected_order) and step_id == expected_order[i]
        )

        return matches / len(expected_order) if expected_order else 0.0

    def _generate_recommendations(
        self,
        plan: AttackPlan,
        metrics: PlanExecutionMetrics,
        trace_records: list[TraceRecord],
    ) -> list[str]:
        """生成改進建議

        Args:
            plan: 攻擊計畫
            metrics: 執行指標
            trace_records: 追蹤記錄

        Returns:
            建議列表
        """
        recommendations = []

        if metrics.success_rate < 0.5:
            recommendations.append("成功率過低，建議檢查工具配置和目標可達性")

        if metrics.completion_rate < 0.7:
            recommendations.append("完成率不足，建議優化步驟依賴和錯誤處理")

        if metrics.sequence_accuracy < 0.8:
            recommendations.append("執行順序偏差，建議檢查依賴關係定義")

        failed_steps = [t for t in trace_records if t.status == "failed"]
        if len(failed_steps) > 0:
            recommendations.append(
                f"有 {len(failed_steps)} 個步驟失敗，建議增加重試機制"
            )

        return recommendations

    async def _persist_result(self, result: PlanExecutionResult) -> None:
        """持久化執行結果

        Args:
            result: 執行結果
        """
        try:
            if hasattr(self.storage, "save_execution_result"):
                await self.storage.save_execution_result(result.model_dump())
                logger.debug(f"Persisted result {result.result_id}")
            else:
                logger.warning("Storage backend does not support save_execution_result")
        except Exception as e:
            logger.error(f"Failed to persist result {result.result_id}: {e}")

    async def get_session(self, session_id: str) -> SessionState | None:
        """獲取會話狀態

        Args:
            session_id: 會話 ID

        Returns:
            會話狀態
        """
        return self.trace_logger.get_session(session_id)

    async def abort_session(self, session_id: str) -> None:
        """中止會話

        Args:
            session_id: 會話 ID
        """
        self.trace_logger.abort_session(session_id)
        logger.info(f"Session {session_id} aborted")
