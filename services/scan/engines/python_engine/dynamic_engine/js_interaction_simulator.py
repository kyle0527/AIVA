

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
        self._initialized = False
    
    def initialize(self, page: Any) -> None:
        """
        初始化並驗證瀏覽器頁面對象
        
        Args:
            page: 瀏覽器頁面對象 (Playwright/Selenium)
            
        Raises:
            RuntimeError: page 對象無效時
        """
        if page is None:
            raise RuntimeError(
                "瀏覽器頁面對象為 None。\n"
                "請確保已正確初始化瀏覽器引擎 (Playwright 或 Selenium)。\n"
                "安裝方法:\n"
                "- Playwright: pip install playwright && playwright install\n"
                "- Selenium: pip install selenium && 下載對應的 WebDriver"
            )
        
        self._initialized = True
        if self.enable_logging:
            logger.info("JS interaction simulator initialized successfully")

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
            
        Raises:
            RuntimeError: page 為 None 時
        """
        import time

        start_time = time.time()
        result = InteractionResult(success=False, event=event)

        # 強制要求 page 對象
        if page is None:
            raise RuntimeError(
                f"瀏覽器頁面對象為 None,無法執行 {event.event_type.value} 互動。\n"
                "請先調用 initialize(page) 方法初始化模擬器,並確保瀏覽器引擎正常運行。"
            )

        try:
            if self.enable_logging:
                logger.info(f"Simulating {event.event_type.value} on {event.selector}")

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
        """等待元素可見（支持 Playwright 和 Selenium）"""
        try:
            # 嘗試 Playwright API
            if hasattr(page, 'wait_for_selector'):
                await page.wait_for_selector(
                    selector, 
                    state='visible', 
                    timeout=self.default_timeout_ms
                )
            # 嘗試 Selenium API
            elif hasattr(page, 'find_element'):
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.common.by import By
                WebDriverWait(page, self.default_timeout_ms / 1000).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
                )
            else:
                # 回退到簡單等待
                if self.enable_logging:
                    logger.warning(f"Unknown page type, using simple wait for: {selector}")
                await asyncio.sleep(0.5)
                
            if self.enable_logging:
                logger.debug(f"Element found: {selector}")
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to find element {selector}: {e}")
            raise

    async def _simulate_click(self, page: Any, event: JsEvent) -> None:
        """模擬點擊事件（支持 Playwright 和 Selenium）"""
        try:
            if hasattr(page, 'click'):
                # Playwright API
                await page.click(event.selector, timeout=self.default_timeout_ms)
            elif hasattr(page, 'find_element'):
                # Selenium API
                from selenium.webdriver.common.by import By
                element = page.find_element(By.CSS_SELECTOR, event.selector)
                element.click()
            else:
                raise RuntimeError(f"Unsupported page type: {type(page)}")
                
            if self.enable_logging:
                logger.debug(f"Clicked element: {event.selector}")
            await asyncio.sleep(event.delay_ms / 1000.0)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to click {event.selector}: {e}")
            raise

    async def _simulate_input(self, page: Any, event: JsEvent) -> None:
        """模擬輸入事件（支持 Playwright 和 Selenium）"""
        if not event.value:
            if self.enable_logging:
                logger.warning(f"No value provided for input: {event.selector}")
            return
            
        try:
            if hasattr(page, 'fill'):
                # Playwright API
                await page.fill(event.selector, event.value, timeout=self.default_timeout_ms)
            elif hasattr(page, 'find_element'):
                # Selenium API
                from selenium.webdriver.common.by import By
                element = page.find_element(By.CSS_SELECTOR, event.selector)
                element.clear()
                element.send_keys(event.value)
            else:
                raise RuntimeError(f"Unsupported page type: {type(page)}")
                
            if self.enable_logging:
                logger.debug(f"Input '{event.value}' to {event.selector}")
            await asyncio.sleep(event.delay_ms / 1000.0)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to input to {event.selector}: {e}")
            raise

    async def _simulate_hover(self, page: Any, event: JsEvent) -> None:
        """模擬懸停事件（支持 Playwright 和 Selenium）"""
        try:
            if hasattr(page, 'hover'):
                # Playwright API
                await page.hover(event.selector, timeout=self.default_timeout_ms)
            elif hasattr(page, 'find_element'):
                # Selenium API
                from selenium.webdriver.common.by import By
                from selenium.webdriver.common.action_chains import ActionChains
                element = page.find_element(By.CSS_SELECTOR, event.selector)
                ActionChains(page).move_to_element(element).perform()
            else:
                raise RuntimeError(f"Unsupported page type: {type(page)}")
                
            if self.enable_logging:
                logger.debug(f"Hovered over element: {event.selector}")
            await asyncio.sleep(event.delay_ms / 1000.0)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to hover over {event.selector}: {e}")
            raise

    async def _simulate_scroll(self, page: Any, event: JsEvent) -> None:
        """模擬滾動事件（支持 Playwright 和 Selenium）"""
        try:
            if hasattr(page, 'evaluate'):
                # Playwright API
                await page.evaluate(
                    f"document.querySelector('{event.selector}').scrollIntoView({{behavior: 'smooth', block: 'center'}})"
                )
            elif hasattr(page, 'execute_script'):
                # Selenium API
                from selenium.webdriver.common.by import By
                element = page.find_element(By.CSS_SELECTOR, event.selector)
                page.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            else:
                raise RuntimeError(f"Unsupported page type: {type(page)}")
                
            if self.enable_logging:
                logger.debug(f"Scrolled to element: {event.selector}")
            await asyncio.sleep(event.delay_ms / 1000.0)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to scroll to {event.selector}: {e}")
            raise

    async def _simulate_submit(self, page: Any, event: JsEvent) -> None:
        """模擬表單提交事件（支持 Playwright 和 Selenium）"""
        try:
            if hasattr(page, 'eval_on_selector'):
                # Playwright API
                await page.eval_on_selector(
                    event.selector,
                    "form => form.submit()"
                )
            elif hasattr(page, 'execute_script'):
                # Selenium API
                from selenium.webdriver.common.by import By
                element = page.find_element(By.CSS_SELECTOR, event.selector)
                page.execute_script("arguments[0].submit();", element)
            else:
                raise RuntimeError(f"Unsupported page type: {type(page)}")
                
            if self.enable_logging:
                logger.debug(f"Submitted form: {event.selector}")
            await asyncio.sleep(event.delay_ms / 1000.0)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to submit form {event.selector}: {e}")
            raise

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
