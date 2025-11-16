"""Result Collector - 結果收集器

接收並處理各功能模組傳回的執行結果
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import json
import logging
from typing import Any

from aio_pika.abc import AbstractIncomingMessage

from services.aiva_common.schemas import (
    FindingPayload,
    ScanCompletedPayload,
    TaskUpdatePayload,
)

from .message_broker import MessageBroker

logger = logging.getLogger(__name__)


class ResultCollector:
    """結果收集器

    訂閱各模組的結果隊列，收集和處理執行結果
    """

    def __init__(
        self,
        broker: MessageBroker,
        storage_backend: Any | None = None,
    ) -> None:
        """初始化結果收集器

        Args:
            broker: 消息代理
            storage_backend: 儲存後端
        """
        self.broker = broker
        self.storage = storage_backend
        self.result_handlers: dict[str, list[Callable]] = {}
        self.pending_results: dict[str, dict[str, Any]] = {}

        logger.info("ResultCollector initialized")

    async def start(self) -> None:
        """啟動結果收集"""
        logger.info("Starting result collection...")

        # 訂閱各類型結果
        await self._subscribe_scan_results()
        await self._subscribe_function_results()
        await self._subscribe_task_updates()
        await self._subscribe_findings()

        logger.info("Result collection started")

    async def _subscribe_scan_results(self) -> None:
        """訂閱掃描結果"""
        await self.broker.subscribe(
            queue_name="core.scan.results",
            routing_keys=["results.scan.completed", "results.scan.failed"],
            exchange_name="aiva.results",
            callback=self._handle_scan_result,
        )
        logger.debug("Subscribed to scan results")

    async def _subscribe_function_results(self) -> None:
        """訂閱功能模組結果"""
        await self.broker.subscribe(
            queue_name="core.function.results",
            routing_keys=[
                "results.function.sqli",
                "results.function.xss",
                "results.function.ssrf",
                "results.function.idor",
            ],
            exchange_name="aiva.results",
            callback=self._handle_function_result,
        )
        logger.debug("Subscribed to function results")

    async def _subscribe_task_updates(self) -> None:
        """訂閱任務狀態更新"""
        await self.broker.subscribe(
            queue_name="core.task.updates",
            routing_keys=["events.task.*"],
            exchange_name="aiva.events",
            callback=self._handle_task_update,
        )
        logger.debug("Subscribed to task updates")

    async def _subscribe_findings(self) -> None:
        """訂閱漏洞發現"""
        await self.broker.subscribe(
            queue_name="core.findings",
            routing_keys=["results.finding.*"],
            exchange_name="aiva.results",
            callback=self._handle_finding,
        )
        logger.debug("Subscribed to findings")

    async def _handle_scan_result(self, message: AbstractIncomingMessage) -> None:
        """處理掃描結果

        Args:
            message: 掃描結果消息
        """
        try:
            async with message.process():
                body = json.loads(message.body.decode())

                logger.info(
                    f"Received scan result: {body.get('payload', {}).get('scan_id')}"
                )

                # 解析掃描結果
                payload = ScanCompletedPayload(**body.get("payload", {}))

                # 存儲結果
                if self.storage:
                    await self._store_result("scan", payload.model_dump())

                # 觸發已註冊的處理器
                await self._trigger_handlers("scan_completed", payload)

                # 如果有等待此結果的請求，設置結果
                self._set_pending_result(payload.scan_id, payload.model_dump())

        except Exception as e:
            logger.error(f"Error handling scan result: {e}", exc_info=True)

    async def _handle_function_result(self, message: AbstractIncomingMessage) -> None:
        """處理功能模組結果

        Args:
            message: 功能模組結果消息
        """
        try:
            async with message.process():
                body = json.loads(message.body.decode())

                logger.info(
                    f"Received function result for task: {body.get('payload', {}).get('task_id')}"
                )

                payload = body.get("payload", {})
                task_id = payload.get("task_id")

                # 存儲結果
                if self.storage:
                    await self._store_result("function", payload)

                # 提取漏洞發現
                findings = payload.get("findings", [])
                if findings:
                    logger.info(
                        f"Extracted {len(findings)} findings from task {task_id}"
                    )
                    await self._trigger_handlers("findings_detected", findings)

                # 觸發處理器
                await self._trigger_handlers("function_completed", payload)

                # 設置等待結果
                if task_id:
                    self._set_pending_result(task_id, payload)

        except Exception as e:
            logger.error(f"Error handling function result: {e}", exc_info=True)

    async def _handle_task_update(self, message: AbstractIncomingMessage) -> None:
        """處理任務狀態更新

        Args:
            message: 任務更新消息
        """
        try:
            async with message.process():
                body = json.loads(message.body.decode())

                payload = TaskUpdatePayload(**body.get("payload", {}))

                logger.debug(f"Task {payload.task_id} status update: {payload.status}")

                # 存儲更新
                if self.storage:
                    await self._store_result("task_update", payload.model_dump())

                # 觸發處理器
                await self._trigger_handlers("task_updated", payload)

        except Exception as e:
            logger.error(f"Error handling task update: {e}", exc_info=True)

    async def _handle_finding(self, message: AbstractIncomingMessage) -> None:
        """處理漏洞發現

        Args:
            message: 漏洞發現消息
        """
        try:
            async with message.process():
                body = json.loads(message.body.decode())

                finding = FindingPayload(**body.get("payload", {}))

                logger.info(
                    f"Received finding: {finding.finding_id} "
                    f"({finding.vulnerability.name.value})"
                )

                # 存儲漏洞
                if self.storage:
                    await self._store_result("finding", finding.model_dump())

                # 觸發處理器
                await self._trigger_handlers("finding_received", finding)

        except Exception as e:
            logger.error(f"Error handling finding: {e}", exc_info=True)

    async def _store_result(self, result_type: str, data: dict[str, Any]) -> None:
        """存儲結果到後端

        Args:
            result_type: 結果類型
            data: 結果數據
        """
        try:
            if hasattr(self.storage, f"save_{result_type}_result"):
                method = getattr(self.storage, f"save_{result_type}_result")
                await method(data)
                logger.debug(f"Stored {result_type} result")
        except Exception as e:
            logger.error(f"Failed to store {result_type} result: {e}")

    async def _trigger_handlers(self, event_type: str, data: Any) -> None:
        """觸發已註冊的事件處理器

        Args:
            event_type: 事件類型
            data: 事件數據
        """
        handlers = self.result_handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in handler for {event_type}: {e}", exc_info=True)

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Any], Any],
    ) -> None:
        """註冊結果處理器

        Args:
            event_type: 事件類型
            handler: 處理函數
        """
        if event_type not in self.result_handlers:
            self.result_handlers[event_type] = []

        self.result_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    def unregister_handler(
        self,
        event_type: str,
        handler: Callable[[Any], Any],
    ) -> None:
        """取消註冊處理器

        Args:
            event_type: 事件類型
            handler: 處理函數
        """
        if event_type in self.result_handlers:
            try:
                self.result_handlers[event_type].remove(handler)
                logger.debug(f"Unregistered handler for {event_type}")
            except ValueError:
                pass

    def _set_pending_result(self, result_id: str, result: dict[str, Any]) -> None:
        """設置等待中的結果

        Args:
            result_id: 結果 ID（task_id, scan_id 等）
            result: 結果數據
        """
        self.pending_results[result_id] = {
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        logger.debug(f"Set pending result for {result_id}")

    async def wait_for_result(
        self,
        result_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> dict[str, Any] | None:
        """等待特定結果

        Args:
            result_id: 結果 ID
            timeout: 超時時間（秒）
            poll_interval: 輪詢間隔（秒）

        Returns:
            結果數據，超時則返回 None
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # 檢查是否已有結果
            if result_id in self.pending_results:
                result = self.pending_results.pop(result_id)
                return result["result"]

            # 檢查超時
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Timeout waiting for result {result_id}")
                return None

            # 等待一段時間後重試
            await asyncio.sleep(poll_interval)

    async def get_recent_results(
        self,
        result_type: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """獲取最近的結果

        Args:
            result_type: 結果類型
            limit: 數量限制

        Returns:
            結果列表
        """
        if not self.storage:
            return []

        try:
            if hasattr(self.storage, f"get_recent_{result_type}_results"):
                method = getattr(self.storage, f"get_recent_{result_type}_results")
                return await method(limit=limit)
        except Exception as e:
            logger.error(f"Failed to get recent {result_type} results: {e}")

        return []

    async def get_statistics(self) -> dict[str, Any]:
        """獲取收集統計信息

        Returns:
            統計信息
        """
        stats = {
            "registered_handlers": {
                event: len(handlers) for event, handlers in self.result_handlers.items()
            },
            "pending_results": len(self.pending_results),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # 從儲存後端獲取額外統計
        if self.storage and hasattr(self.storage, "get_result_statistics"):
            try:
                storage_stats = await self.storage.get_result_statistics()
                stats["storage"] = storage_stats
            except Exception as e:
                logger.error(f"Failed to get storage statistics: {e}")

        return stats
