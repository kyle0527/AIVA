

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class BrowserType(Enum):
    """支持的瀏覽器類型"""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class BrowserStatus(Enum):
    """瀏覽器實例狀態"""

    IDLE = "idle"  # 空閒
    BUSY = "busy"  # 使用中
    CRASHED = "crashed"  # 崩潰
    CLOSED = "closed"  # 已關閉


@dataclass
class BrowserInstance:
    """瀏覽器實例"""

    browser_id: str
    browser_type: BrowserType
    browser: Any  # Playwright Browser 對象
    status: BrowserStatus = BrowserStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    page_count: int = 0
    error_count: int = 0
    total_requests: int = 0


@dataclass
class PageInstance:
    """頁面實例"""

    page_id: str
    page: Any  # Playwright Page 對象
    browser_id: str
    created_at: datetime = field(default_factory=datetime.now)
    url: str | None = None
    is_active: bool = True


@dataclass
class PoolConfig:
    """瀏覽器池配置"""

    min_instances: int = 2
    max_instances: int = 10
    max_pages_per_browser: int = 5
    idle_timeout_seconds: int = 300
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    proxy: str | None = None
    user_agent: str | None = None
    viewport_width: int = 1920
    viewport_height: int = 1080
    enable_javascript: bool = True
    timeout_ms: int = 30000


class HeadlessBrowserPool:
    """
    無頭瀏覽器池管理器

    用於管理多個無頭瀏覽器實例，支持瀏覽器實例的創建、復用、
    回收等功能，提高動態掃描的效率和穩定性。

    特性：
    - 支持多種瀏覽器類型（Chromium、Firefox、WebKit）
    - 自動管理瀏覽器實例生命週期
    - 支持並發控制和資源限制
    - 提供詳細的統計和監控信息
    - 異常處理和自動恢復
    """

    def __init__(self, config: PoolConfig | None = None) -> None:
        """
        初始化瀏覽器池

        Args:
            config: 池配置，如果為 None 則使用默認配置
        """
        self.config = config or PoolConfig()
        self._browsers: dict[str, BrowserInstance] = {}
        self._pages: dict[str, PageInstance] = {}
        self._browser_semaphore = asyncio.Semaphore(self.config.max_instances)
        self._playwright: Any = None
        self._is_initialized = False
        self._next_browser_id = 0
        self._next_page_id = 0
        self._lock = asyncio.Lock()

        # 統計信息
        self._stats = {
            "browsers_created": 0,
            "browsers_closed": 0,
            "pages_created": 0,
            "pages_closed": 0,
            "total_requests": 0,
            "errors": 0,
        }

    async def initialize(self) -> None:
        """初始化瀏覽器池"""
        if self._is_initialized:
            logger.warning("Browser pool already initialized")
            return

        try:
            # 嘗試導入 Playwright
            from playwright.async_api import (  # type: ignore[import-not-found]
                async_playwright,
            )

            self._playwright = await async_playwright().start()
            logger.info("Playwright initialized successfully")

            # 標記為已初始化（在創建瀏覽器之前）
            self._is_initialized = True

            # 預創建最小數量的瀏覽器實例
            for _ in range(self.config.min_instances):
                await self._create_browser()

            logger.info(
                f"Browser pool initialized with {len(self._browsers)} instances"
            )

        except ImportError:
            logger.warning(
                "Playwright not installed. Browser pool will run in mock mode. "
                "Install with: pip install playwright && playwright install"
            )
            self._is_initialized = True  # 在模擬模式下仍然標記為已初始化

        except Exception as e:
            logger.error(f"Failed to initialize browser pool: {e}", exc_info=True)
            self._is_initialized = False  # 初始化失敗時重置狀態
            raise

    async def shutdown(self) -> None:
        """關閉瀏覽器池"""
        if not self._is_initialized:
            return

        logger.info(f"Shutting down browser pool with {len(self._browsers)} browsers")

        # 關閉所有頁面
        for page_instance in list(self._pages.values()):
            try:
                await self._close_page(page_instance.page_id)
            except Exception as e:
                logger.error(f"Error closing page {page_instance.page_id}: {e}")

        # 關閉所有瀏覽器
        for browser_instance in list(self._browsers.values()):
            try:
                await self._close_browser(browser_instance.browser_id)
            except Exception as e:
                logger.error(
                    f"Error closing browser {browser_instance.browser_id}: {e}"
                )

        # 關閉 Playwright
        if self._playwright:
            try:
                await self._playwright.stop()
                logger.info("Playwright stopped")
            except Exception as e:
                logger.error(f"Error stopping Playwright: {e}")

        self._is_initialized = False
        logger.info("Browser pool shut down successfully")

    async def _create_browser(self) -> BrowserInstance:
        """創建新的瀏覽器實例"""
        if not self._is_initialized:
            raise RuntimeError("Browser pool not initialized")

        async with self._lock:
            browser_id = f"browser_{self._next_browser_id}"
            self._next_browser_id += 1

        try:
            # 模擬模式（沒有 Playwright）
            if self._playwright is None:
                logger.debug(f"Creating mock browser: {browser_id}")
                browser_instance = BrowserInstance(
                    browser_id=browser_id,
                    browser_type=self.config.browser_type,
                    browser=None,  # Mock browser
                )
                self._browsers[browser_id] = browser_instance
                self._stats["browsers_created"] += 1
                return browser_instance

            # 獲取瀏覽器啟動器
            if self.config.browser_type == BrowserType.CHROMIUM:
                launcher = self._playwright.chromium
            elif self.config.browser_type == BrowserType.FIREFOX:
                launcher = self._playwright.firefox
            else:
                launcher = self._playwright.webkit

            # 配置瀏覽器選項
            launch_options: dict[str, Any] = {
                "headless": self.config.headless,
            }

            if self.config.proxy:
                launch_options["proxy"] = {"server": self.config.proxy}

            # 啟動瀏覽器
            browser = await launcher.launch(**launch_options)
            logger.info(
                f"Created {self.config.browser_type.value} browser: {browser_id}"
            )

            browser_instance = BrowserInstance(
                browser_id=browser_id,
                browser_type=self.config.browser_type,
                browser=browser,
            )

            self._browsers[browser_id] = browser_instance
            self._stats["browsers_created"] += 1

            return browser_instance

        except Exception as e:
            logger.error(f"Failed to create browser {browser_id}: {e}", exc_info=True)
            self._stats["errors"] += 1
            raise

    async def _close_browser(self, browser_id: str) -> None:
        """關閉瀏覽器實例"""
        browser_instance = self._browsers.get(browser_id)
        if not browser_instance:
            logger.warning(f"Browser {browser_id} not found")
            return

        try:
            # 關閉該瀏覽器的所有頁面
            pages_to_close = [
                p for p in self._pages.values() if p.browser_id == browser_id
            ]
            for page_instance in pages_to_close:
                await self._close_page(page_instance.page_id)

            # 關閉瀏覽器
            if browser_instance.browser:
                await browser_instance.browser.close()
                logger.debug(f"Closed browser: {browser_id}")

            browser_instance.status = BrowserStatus.CLOSED
            del self._browsers[browser_id]
            self._stats["browsers_closed"] += 1

        except Exception as e:
            logger.error(f"Error closing browser {browser_id}: {e}", exc_info=True)
            self._stats["errors"] += 1

    async def _create_page(self, browser_id: str | None = None) -> PageInstance:
        """
        創建新的頁面實例

        Args:
            browser_id: 指定的瀏覽器 ID，如果為 None 則自動選擇

        Returns:
            頁面實例
        """
        # 選擇瀏覽器
        if browser_id is None:
            browser_instance = await self._get_available_browser()
        else:
            browser_maybe = self._browsers.get(browser_id)
            if not browser_maybe:
                raise ValueError(f"Browser {browser_id} not found")
            browser_instance = browser_maybe

        async with self._lock:
            page_id = f"page_{self._next_page_id}"
            self._next_page_id += 1

        try:
            # 模擬模式
            if browser_instance.browser is None:
                logger.debug(f"Creating mock page: {page_id}")
                page_instance = PageInstance(
                    page_id=page_id,
                    page=None,
                    browser_id=browser_instance.browser_id,
                )
                self._pages[page_id] = page_instance
                browser_instance.page_count += 1
                self._stats["pages_created"] += 1
                return page_instance

            # 創建真實頁面
            context = await browser_instance.browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                user_agent=self.config.user_agent,
                java_script_enabled=self.config.enable_javascript,
            )

            page = await context.new_page()
            page.set_default_timeout(self.config.timeout_ms)

            logger.debug(
                f"Created page {page_id} in browser {browser_instance.browser_id}"
            )

            page_instance = PageInstance(
                page_id=page_id,
                page=page,
                browser_id=browser_instance.browser_id,
            )

            self._pages[page_id] = page_instance
            browser_instance.page_count += 1
            browser_instance.last_used = datetime.now()
            self._stats["pages_created"] += 1

            return page_instance

        except Exception as e:
            logger.error(f"Failed to create page {page_id}: {e}", exc_info=True)
            self._stats["errors"] += 1
            raise

    async def _close_page(self, page_id: str) -> None:
        """關閉頁面實例"""
        page_instance = self._pages.get(page_id)
        if not page_instance:
            logger.warning(f"Page {page_id} not found")
            return

        try:
            if page_instance.page:
                await page_instance.page.close()
                logger.debug(f"Closed page: {page_id}")

            # 更新瀏覽器頁面計數
            browser_instance = self._browsers.get(page_instance.browser_id)
            if browser_instance:
                browser_instance.page_count = max(0, browser_instance.page_count - 1)

            page_instance.is_active = False
            del self._pages[page_id]
            self._stats["pages_closed"] += 1

        except Exception as e:
            logger.error(f"Error closing page {page_id}: {e}", exc_info=True)
            self._stats["errors"] += 1

    async def _get_available_browser(self) -> BrowserInstance:
        """獲取可用的瀏覽器實例"""
        # 尋找空閒且頁面數未達上限的瀏覽器
        for browser_instance in self._browsers.values():
            if (
                browser_instance.status == BrowserStatus.IDLE
                and browser_instance.page_count < self.config.max_pages_per_browser
            ):
                return browser_instance

        # 如果沒有可用的，且未達到最大實例數，創建新的
        if len(self._browsers) < self.config.max_instances:
            return await self._create_browser()

        # 否則等待或使用頁面數最少的瀏覽器
        return min(self._browsers.values(), key=lambda b: b.page_count)

    @asynccontextmanager
    async def get_page(self, browser_id: str | None = None) -> AsyncIterator[Any]:
        """
        獲取一個頁面實例（上下文管理器）

        Args:
            browser_id: 指定的瀏覽器 ID

        Yields:
            頁面對象

        Example:
            async with pool.get_page() as page:
                await page.goto("https://example.com")
                content = await page.content()
        """
        if not self._is_initialized:
            raise RuntimeError("Browser pool not initialized. Call initialize() first.")

        page_instance = await self._create_page(browser_id)

        try:
            yield page_instance.page
        finally:
            await self._close_page(page_instance.page_id)

    async def execute_on_page(
        self, url: str, callback: Any, *, browser_id: str | None = None
    ) -> Any:
        """
        在頁面上執行回調函數

        Args:
            url: 要訪問的 URL
            callback: 異步回調函數，接收 page 對象作為參數
            browser_id: 指定的瀏覽器 ID

        Returns:
            回調函數的返回值
        """
        async with self.get_page(browser_id) as page:
            if page:  # 非模擬模式
                await page.goto(url)
                self._stats["total_requests"] += 1
            return await callback(page)

    def get_stats(self) -> dict[str, Any]:
        """獲取統計信息"""
        active_browsers = sum(
            1 for b in self._browsers.values() if b.status != BrowserStatus.CLOSED
        )
        active_pages = sum(1 for p in self._pages.values() if p.is_active)

        browser_types: dict[str, int] = defaultdict(int)
        for browser in self._browsers.values():
            browser_types[browser.browser_type.value] += 1

        return {
            "initialized": self._is_initialized,
            "active_browsers": active_browsers,
            "active_pages": active_pages,
            "total_browsers_created": self._stats["browsers_created"],
            "total_browsers_closed": self._stats["browsers_closed"],
            "total_pages_created": self._stats["pages_created"],
            "total_pages_closed": self._stats["pages_closed"],
            "total_requests": self._stats["total_requests"],
            "total_errors": self._stats["errors"],
            "browser_types": dict(browser_types),
            "config": {
                "min_instances": self.config.min_instances,
                "max_instances": self.config.max_instances,
                "max_pages_per_browser": self.config.max_pages_per_browser,
                "browser_type": self.config.browser_type.value,
                "headless": self.config.headless,
            },
        }

    async def cleanup_idle_browsers(self, idle_seconds: int | None = None) -> int:
        """
        清理空閒的瀏覽器實例

        Args:
            idle_seconds: 空閒時間閾值（秒），如果為 None 則使用配置值

        Returns:
            關閉的瀏覽器數量
        """
        if idle_seconds is None:
            idle_seconds = self.config.idle_timeout_seconds

        now = datetime.now()
        closed_count = 0

        for browser_instance in list(self._browsers.values()):
            # 保持最小實例數
            if len(self._browsers) <= self.config.min_instances:
                break

            idle_time = (now - browser_instance.last_used).total_seconds()
            if (
                browser_instance.page_count == 0
                and idle_time > idle_seconds
                and browser_instance.status == BrowserStatus.IDLE
            ):
                logger.info(
                    f"Closing idle browser {browser_instance.browser_id} "
                    f"(idle for {idle_time:.0f}s)"
                )
                await self._close_browser(browser_instance.browser_id)
                closed_count += 1

        return closed_count

    def get_browser_info(self, browser_id: str) -> dict[str, Any] | None:
        """獲取指定瀏覽器的詳細信息"""
        browser_instance = self._browsers.get(browser_id)
        if not browser_instance:
            return None

        return {
            "browser_id": browser_instance.browser_id,
            "browser_type": browser_instance.browser_type.value,
            "status": browser_instance.status.value,
            "page_count": browser_instance.page_count,
            "error_count": browser_instance.error_count,
            "total_requests": browser_instance.total_requests,
            "created_at": browser_instance.created_at.isoformat(),
            "last_used": browser_instance.last_used.isoformat(),
            "age_seconds": (
                datetime.now() - browser_instance.created_at
            ).total_seconds(),
        }

    def list_browsers(self) -> list[dict[str, Any]]:
        """列出所有瀏覽器實例信息"""
        result = []
        for bid in self._browsers:
            info = self.get_browser_info(bid)
            if info:
                result.append(info)
        return result

    def list_pages(self) -> list[dict[str, Any]]:
        """列出所有頁面實例信息"""
        pages_info = []
        for page_instance in self._pages.values():
            pages_info.append(
                {
                    "page_id": page_instance.page_id,
                    "browser_id": page_instance.browser_id,
                    "url": page_instance.url,
                    "is_active": page_instance.is_active,
                    "created_at": page_instance.created_at.isoformat(),
                    "age_seconds": (
                        datetime.now() - page_instance.created_at
                    ).total_seconds(),
                }
            )
        return pages_info
