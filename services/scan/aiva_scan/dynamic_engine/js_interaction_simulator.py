from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class InteractionType(Enum):
    """JavaScript 互動類型"""

    CLICK = "click"
    HOVER = "hover"
    INPUT = "input"
    SCROLL = "scroll"
    FOCUS = "focus"
    BLUR = "blur"
    SUBMIT = "submit"
    CHANGE = "change"


@dataclass
class JsEvent:
    """JavaScript 事件定義"""

    event_type: InteractionType
    selector: str
    value: str | None = None
    delay_ms: int = 100
    wait_for_response: bool = True


@dataclass
class InteractionResult:
    """互動結果"""

    success: bool
    event: JsEvent
    error: str | None = None
    dom_changes: list[str] = field(default_factory=list)
    network_requests: list[str] = field(default_factory=list)
    console_logs: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class JsInteractionSimulator:
    """
    模擬 JavaScript 互動

    用於在動態掃描過程中模擬用戶的 JavaScript 互動行為，
    包括點擊、輸入、懸停等操作，以觸發動態內容和行為。
    """

    def __init__(
        self,
        *,
        max_retry: int = 3,
        default_timeout_ms: int = 5000,
        enable_logging: bool = True,
    ) -> None:
        """
        初始化 JS 互動模擬器

        Args:
            max_retry: 最大重試次數
            default_timeout_ms: 默認超時時間（毫秒）
            enable_logging: 是否啟用日誌記錄
        """
        self.max_retry = max_retry
        self.default_timeout_ms = default_timeout_ms
        self.enable_logging = enable_logging
        self._event_queue: list[JsEvent] = []
        self._results: list[InteractionResult] = []

    def add_event(self, event: JsEvent) -> None:
        """添加事件到隊列"""
        self._event_queue.append(event)
        if self.enable_logging:
            logger.debug(
                f"Added event to queue: {event.event_type.value} on {event.selector}"
            )

    def add_click(
        self, selector: str, *, delay_ms: int = 100, wait: bool = True
    ) -> None:
        """添加點擊事件"""
        event = JsEvent(
            event_type=InteractionType.CLICK,
            selector=selector,
            delay_ms=delay_ms,
            wait_for_response=wait,
        )
        self.add_event(event)

    def add_input(
        self, selector: str, value: str, *, delay_ms: int = 100, wait: bool = True
    ) -> None:
        """添加輸入事件"""
        event = JsEvent(
            event_type=InteractionType.INPUT,
            selector=selector,
            value=value,
            delay_ms=delay_ms,
            wait_for_response=wait,
        )
        self.add_event(event)

    def add_hover(
        self, selector: str, *, delay_ms: int = 100, wait: bool = True
    ) -> None:
        """添加懸停事件"""
        event = JsEvent(
            event_type=InteractionType.HOVER,
            selector=selector,
            delay_ms=delay_ms,
            wait_for_response=wait,
        )
        self.add_event(event)

    def add_scroll(
        self, selector: str, *, delay_ms: int = 100, wait: bool = True
    ) -> None:
        """添加滾動事件"""
        event = JsEvent(
            event_type=InteractionType.SCROLL,
            selector=selector,
            delay_ms=delay_ms,
            wait_for_response=wait,
        )
        self.add_event(event)

    def add_submit(
        self, selector: str, *, delay_ms: int = 100, wait: bool = True
    ) -> None:
        """添加表單提交事件"""
        event = JsEvent(
            event_type=InteractionType.SUBMIT,
            selector=selector,
            delay_ms=delay_ms,
            wait_for_response=wait,
        )
        self.add_event(event)

    async def simulate_event(
        self, event: JsEvent, *, page: Any = None
    ) -> InteractionResult:
        """
        模擬單個事件

        Args:
            event: 要模擬的事件
            page: 瀏覽器頁面對象（如 Playwright 或 Selenium 的 page 對象）

        Returns:
            互動結果
        """
        import time

        start_time = time.time()
        result = InteractionResult(success=False, event=event)

        try:
            if self.enable_logging:
                logger.info(f"Simulating {event.event_type.value} on {event.selector}")

            # 如果沒有提供 page 對象，則僅記錄事件
            if page is None:
                if self.enable_logging:
                    logger.warning("No page object provided, skipping actual execution")
                result.success = True
                result.execution_time_ms = (time.time() - start_time) * 1000
                self._results.append(result)
                return result

            # 等待元素可見
            await self._wait_for_element(page, event.selector)

            # 執行互動
            if event.event_type == InteractionType.CLICK:
                await self._simulate_click(page, event)
            elif event.event_type == InteractionType.INPUT:
                await self._simulate_input(page, event)
            elif event.event_type == InteractionType.HOVER:
                await self._simulate_hover(page, event)
            elif event.event_type == InteractionType.SCROLL:
                await self._simulate_scroll(page, event)
            elif event.event_type == InteractionType.SUBMIT:
                await self._simulate_submit(page, event)
            else:
                raise ValueError(f"Unsupported event type: {event.event_type}")

            # 等待響應
            if event.wait_for_response and event.delay_ms > 0:
                await asyncio.sleep(event.delay_ms / 1000)

            result.success = True

        except Exception as e:
            result.error = str(e)
            if self.enable_logging:
                logger.error(
                    f"Failed to simulate {event.event_type.value}: {str(e)}",
                    exc_info=True,
                )

        result.execution_time_ms = (time.time() - start_time) * 1000
        self._results.append(result)
        return result

    async def execute_queue(self, *, page: Any = None) -> list[InteractionResult]:
        """
        執行事件隊列中的所有事件

        Args:
            page: 瀏覽器頁面對象

        Returns:
            所有互動結果列表
        """
        results = []
        for event in self._event_queue:
            result = await self.simulate_event(event, page=page)
            results.append(result)

            # 如果事件失敗且啟用重試
            if not result.success and self.max_retry > 0:
                for attempt in range(1, self.max_retry + 1):
                    if self.enable_logging:
                        logger.info(
                            f"Retrying event {event.event_type.value} "
                            f"(attempt {attempt}/{self.max_retry})"
                        )
                    await asyncio.sleep(0.5)  # 重試前等待
                    retry_result = await self.simulate_event(event, page=page)
                    if retry_result.success:
                        results[-1] = retry_result  # 替換失敗的結果
                        break

        return results

    async def _wait_for_element(self, page: Any, selector: str) -> None:
        """等待元素可見（抽象方法，需根據實際瀏覽器庫實現）"""
        # 這裡是佔位符實現，實際應根據使用的瀏覽器庫（如 Playwright）來實現
        if self.enable_logging:
            logger.debug(f"Waiting for element: {selector}")
        await asyncio.sleep(0.1)

    async def _simulate_click(self, page: Any, event: JsEvent) -> None:
        """模擬點擊事件"""
        # 佔位符實現
        # 實際實現範例（Playwright）: await page.click(event.selector)
        if self.enable_logging:
            logger.debug(f"Clicking element: {event.selector}")
        await asyncio.sleep(0.05)

    async def _simulate_input(self, page: Any, event: JsEvent) -> None:
        """模擬輸入事件"""
        # 佔位符實現
        # 實際實現範例（Playwright）: await page.fill(event.selector, event.value)
        if self.enable_logging:
            logger.debug(f"Inputting '{event.value}' to {event.selector}")
        await asyncio.sleep(0.05)

    async def _simulate_hover(self, page: Any, event: JsEvent) -> None:
        """模擬懸停事件"""
        # 佔位符實現
        # 實際實現範例（Playwright）: await page.hover(event.selector)
        if self.enable_logging:
            logger.debug(f"Hovering over element: {event.selector}")
        await asyncio.sleep(0.05)

    async def _simulate_scroll(self, page: Any, event: JsEvent) -> None:
        """模擬滾動事件"""
        # 佔位符實現
        # 實際實現範例（Playwright）:
        # await page.evaluate(
        #     f"document.querySelector('{event.selector}').scrollIntoView()"
        # )
        if self.enable_logging:
            logger.debug(f"Scrolling to element: {event.selector}")
        await asyncio.sleep(0.05)

    async def _simulate_submit(self, page: Any, event: JsEvent) -> None:
        """模擬表單提交事件"""
        # 佔位符實現
        # 實際實現範例（Playwright）:
        # await page.eval_on_selector(
        #     event.selector,
        #     "form => form.submit()"
        # )
        if self.enable_logging:
            logger.debug(f"Submitting form: {event.selector}")
        await asyncio.sleep(0.05)

    def clear_queue(self) -> None:
        """清空事件隊列"""
        self._event_queue.clear()
        if self.enable_logging:
            logger.debug("Event queue cleared")

    def get_results(self) -> list[InteractionResult]:
        """獲取所有互動結果"""
        return self._results.copy()

    def get_success_rate(self) -> float:
        """獲取成功率"""
        if not self._results:
            return 0.0
        successful = sum(1 for r in self._results if r.success)
        return successful / len(self._results)

    def get_stats(self) -> dict[str, Any]:
        """獲取統計信息"""
        if not self._results:
            return {
                "total_events": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
                "events_by_type": {},
            }

        successful = sum(1 for r in self._results if r.success)
        failed = len(self._results) - successful
        avg_time = sum(r.execution_time_ms for r in self._results) / len(self._results)

        return {
            "total_events": len(self._results),
            "successful": successful,
            "failed": failed,
            "success_rate": self.get_success_rate(),
            "avg_execution_time_ms": avg_time,
            "events_by_type": self._get_events_by_type(),
        }

    def _get_events_by_type(self) -> dict[str, int]:
        """按類型統計事件數量"""
        stats: dict[str, int] = {}
        for result in self._results:
            event_type = result.event.event_type.value
            stats[event_type] = stats.get(event_type, 0) + 1
        return stats
