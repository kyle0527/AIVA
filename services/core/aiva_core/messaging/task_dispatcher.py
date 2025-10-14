"""
Task Dispatcher - 任務派發器

負責將攻擊計畫轉換為任務並派發到各功能模組
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import logging
from typing import Any
from uuid import uuid4

from aiva_common.schemas import (
    AivaMessage,
    AttackPlan,
    AttackStep,
    FunctionTaskPayload,
    FunctionTaskTarget,
    MessageHeader,
    ModuleName,
    ScanStartPayload,
    Topic,
)

from .message_broker import MessageBroker

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """任務派發器

    將高層攻擊計畫轉換為具體任務，派發到對應的功能模組
    """

    def __init__(
        self,
        broker: MessageBroker,
        module_name: ModuleName = ModuleName.CORE,
    ) -> None:
        """初始化任務派發器

        Args:
            broker: 消息代理
            module_name: 當前模組名稱
        """
        self.broker = broker
        self.module_name = module_name
        self.task_callbacks: dict[str, Callable] = {}

        # 工具類型到路由鍵的映射
        self.tool_routing_map = {
            "function_sqli": "tasks.function.sqli",
            "function_xss": "tasks.function.xss",
            "function_ssrf": "tasks.function.ssrf",
            "function_idor": "tasks.function.idor",
            "scan_service": "tasks.scan.start",
        }

        logger.info("TaskDispatcher initialized")

    def _get_topic_for_tool(self, tool_type: str) -> Topic:
        """根據工具類型獲取對應的 Topic

        Args:
            tool_type: 工具類型

        Returns:
            Topic 枚舉
        """
        topic_map = {
            "function_sqli": Topic.TASK_FUNCTION_SQLI,
            "function_xss": Topic.TASK_FUNCTION_XSS,
            "function_ssrf": Topic.TASK_FUNCTION_SSRF,
            "function_idor": Topic.FUNCTION_IDOR_TASK,
            "scan_service": Topic.TASK_SCAN_START,
        }
        return topic_map.get(tool_type, Topic.TASK_FUNCTION_START)

    async def dispatch_attack_plan(
        self,
        plan: AttackPlan,
        session_id: str,
    ) -> list[str]:
        """派發完整攻擊計畫

        將攻擊計畫的所有步驟轉換為任務並派發

        Args:
            plan: 攻擊計畫
            session_id: 會話 ID

        Returns:
            已派發的任務 ID 列表
        """
        logger.info(
            f"Dispatching attack plan {plan.plan_id} "
            f"with {len(plan.steps)} steps (session={session_id})"
        )

        task_ids = []

        for i, step in enumerate(plan.steps):
            try:
                task_id = await self.dispatch_step(
                    step=step,
                    plan_id=plan.plan_id,
                    session_id=session_id,
                    scan_id=plan.scan_id,
                    step_index=i,
                )
                task_ids.append(task_id)
                logger.debug(f"Dispatched step {i+1}/{len(plan.steps)}: {task_id}")

            except Exception as e:
                logger.error(f"Failed to dispatch step {step.step_id}: {e}")
                # 繼續派發其他步驟

        logger.info(
            f"Plan {plan.plan_id} dispatched: {len(task_ids)}/{len(plan.steps)} tasks"
        )

        return task_ids

    async def dispatch_step(
        self,
        step: AttackStep,
        plan_id: str,
        session_id: str,
        scan_id: str,
        step_index: int = 0,
    ) -> str:
        """派發單個攻擊步驟

        Args:
            step: 攻擊步驟
            plan_id: 計畫 ID
            session_id: 會話 ID
            scan_id: 掃描 ID
            step_index: 步驟索引

        Returns:
            任務 ID
        """
        # 生成任務 ID
        task_id = f"task_{uuid4().hex[:12]}"

        # 構建任務 Payload
        task_payload = self._build_task_payload(
            task_id=task_id,
            step=step,
            plan_id=plan_id,
            session_id=session_id,
            scan_id=scan_id,
        )

        # 確定路由鍵
        routing_key = self.tool_routing_map.get(
            step.tool_type,
            "tasks.function.unknown",
        )

        # 構建消息（使用現有的 Topic 枚舉）
        topic = self._get_topic_for_tool(step.tool_type)
        message = self._build_message(
            topic=topic,
            payload=task_payload.model_dump(),
            correlation_id=task_id,
        )

        # 發送任務
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key=routing_key,
            message=message,
            correlation_id=task_id,
        )

        logger.info(
            f"Dispatched task {task_id} to {routing_key} "
            f"(step={step.step_id}, tool={step.tool_type})"
        )

        return task_id

    def _build_task_payload(
        self,
        task_id: str,
        step: AttackStep,
        plan_id: str,
        session_id: str,
        scan_id: str,
    ) -> FunctionTaskPayload:
        """構建功能任務 Payload

        Args:
            task_id: 任務 ID
            step: 攻擊步驟
            plan_id: 計畫 ID
            session_id: 會話 ID
            scan_id: 掃描 ID

        Returns:
            功能任務 Payload
        """
        # 從步驟目標構建任務目標
        target = FunctionTaskTarget(
            url=step.target.get("url", ""),
            parameter=step.target.get("parameter"),
            method=step.target.get("method", "GET"),
            parameter_location=step.target.get("location", "query"),
            headers=step.target.get("headers", {}),
            cookies=step.target.get("cookies", {}),
            form_data=step.target.get("form_data", {}),
            json_data=step.target.get("json_data"),
            body=step.target.get("body"),
        )

        # 構建任務 Payload
        payload = FunctionTaskPayload(
            task_id=task_id,
            scan_id=scan_id,
            priority=step.parameters.get("priority", 5),
            target=target,
            strategy=step.parameters.get("strategy", "full"),
            custom_payloads=step.parameters.get("custom_payloads"),
            metadata={
                "plan_id": plan_id,
                "session_id": session_id,
                "step_id": step.step_id,
                "step_action": step.action,
                "mitre_technique_id": step.mitre_technique_id,
                "mitre_tactic": step.mitre_tactic,
            },
        )

        return payload

    def _build_message(
        self,
        topic: Topic,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> AivaMessage:
        """構建 AIVA 消息

        Args:
            topic: 消息主題
            payload: 消息負載
            correlation_id: 關聯 ID

        Returns:
            AIVA 消息
        """
        header = MessageHeader(
            message_id=f"msg_{uuid4().hex[:12]}",
            trace_id=f"trace_{uuid4().hex[:12]}",
            correlation_id=correlation_id,
            source_module=self.module_name,
            timestamp=datetime.now(UTC),
        )

        message = AivaMessage(
            header=header,
            topic=topic,
            payload=payload,
        )

        return message

    async def dispatch_scan_task(
        self,
        scan_payload: ScanStartPayload,
    ) -> str:
        """派發掃描任務

        Args:
            scan_payload: 掃描任務 Payload

        Returns:
            掃描 ID
        """
        # 構建消息
        message = self._build_message(
            topic=Topic.TASK_SCAN_START,
            payload=scan_payload.model_dump(),
            correlation_id=scan_payload.scan_id,
        )

        # 發送任務
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="tasks.scan.start",
            message=message,
            correlation_id=scan_payload.scan_id,
        )

        logger.info(f"Dispatched scan task {scan_payload.scan_id}")

        return scan_payload.scan_id

    async def dispatch_batch_tasks(
        self,
        tasks: list[dict[str, Any]],
        routing_key: str,
    ) -> list[str]:
        """批量派發任務

        Args:
            tasks: 任務列表
            routing_key: 路由鍵

        Returns:
            任務 ID 列表
        """
        task_ids = []

        for task_data in tasks:
            task_id = task_data.get("task_id") or f"task_{uuid4().hex[:12]}"

            message = self._build_message(
                topic=Topic.TASK_FUNCTION_START,
                payload=task_data,
                correlation_id=task_id,
            )

            try:
                await self.broker.publish_message(
                    exchange_name="aiva.tasks",
                    routing_key=routing_key,
                    message=message,
                    correlation_id=task_id,
                )

                task_ids.append(task_id)

            except Exception as e:
                logger.error(f"Failed to dispatch task {task_id}: {e}")

        logger.info(f"Batch dispatched {len(task_ids)}/{len(tasks)} tasks")

        return task_ids

    async def send_control_command(
        self,
        command: str,
        target_module: ModuleName,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """發送控制命令到特定模組

        Args:
            command: 命令名稱
            target_module: 目標模組
            parameters: 命令參數
        """
        payload = {
            "command": command,
            "target_module": target_module.value,
            "parameters": parameters or {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        message = self._build_message(
            topic=Topic.CONFIG_GLOBAL_UPDATE,
            payload=payload,
        )

        routing_key = f"control.{target_module.value}.{command}"

        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key=routing_key,
            message=message,
        )

        logger.info(f"Sent control command '{command}' to {target_module.value}")

    async def send_feedback(
        self,
        task_id: str,
        feedback_type: str,
        data: dict[str, Any],
    ) -> None:
        """發送反饋到執行模組

        Args:
            task_id: 任務 ID
            feedback_type: 反饋類型
            data: 反饋數據
        """
        payload = {
            "task_id": task_id,
            "feedback_type": feedback_type,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        message = self._build_message(
            topic=Topic.FEEDBACK_CORE_STRATEGY,
            payload=payload,
            correlation_id=task_id,
        )

        await self.broker.publish_message(
            exchange_name="aiva.feedback",
            routing_key=f"feedback.{feedback_type}",
            message=message,
            correlation_id=task_id,
        )

        logger.debug(f"Sent feedback for task {task_id}: {feedback_type}")

    async def request_status(
        self,
        target_module: ModuleName,
        timeout: float = 5.0,
    ) -> dict[str, Any] | None:
        """請求模組狀態（RPC 模式）

        Args:
            target_module: 目標模組
            timeout: 超時時間

        Returns:
            模組狀態，超時則返回 None
        """
        correlation_id = f"status_req_{uuid4().hex[:12]}"

        payload = {
            "request_type": "status",
            "target_module": target_module.value,
        }

        message = self._build_message(
            topic=Topic.MODULE_HEARTBEAT,
            payload=payload,
            correlation_id=correlation_id,
        )

        try:
            # 創建 RPC 客戶端
            rpc_client = await self.broker.create_rpc_client(
                exchange_name="aiva.events",
                timeout=timeout,
            )

            # 發送請求並等待響應
            response = await rpc_client.call(
                routing_key=f"status.{target_module.value}",
                message=message.model_dump(),
                correlation_id=correlation_id,
            )

            logger.debug(f"Received status from {target_module.value}")
            return response

        except TimeoutError:
            logger.warning(f"Status request to {target_module.value} timed out")
            return None

    def register_callback(
        self,
        task_id: str,
        callback: Callable[[dict[str, Any]], Any],
    ) -> None:
        """註冊任務完成回調

        Args:
            task_id: 任務 ID
            callback: 回調函數
        """
        self.task_callbacks[task_id] = callback
        logger.debug(f"Registered callback for task {task_id}")

    def unregister_callback(self, task_id: str) -> None:
        """取消註冊回調

        Args:
            task_id: 任務 ID
        """
        if task_id in self.task_callbacks:
            del self.task_callbacks[task_id]
            logger.debug(f"Unregistered callback for task {task_id}")

    async def trigger_callback(
        self,
        task_id: str,
        result: dict[str, Any],
    ) -> None:
        """觸發任務回調

        Args:
            task_id: 任務 ID
            result: 任務結果
        """
        callback = self.task_callbacks.get(task_id)

        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)

                logger.debug(f"Triggered callback for task {task_id}")

            except Exception as e:
                logger.error(f"Error in callback for task {task_id}: {e}")

            finally:
                self.unregister_callback(task_id)
