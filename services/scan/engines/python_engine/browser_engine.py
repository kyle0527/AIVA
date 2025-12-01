"""Browser Engine - Playwright 真實化實作 (P1)

移除所有 sleep 模擬，接入真實 Playwright API
實現：
1. 真實 DOM 事件觸發（click, fill, hover）
2. 上下文感知輸入生成（Context-Aware Input）
3. DOM 變更監聽與 Hydration 檢測
4. 網絡閒置檢測（networkidle）
"""

import asyncio
import re
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime, UTC
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PlaywrightTimeout = Exception
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None  # type: ignore
    PlaywrightTimeout = Exception  # type: ignore

from aiva_common.utils.logging import get_logger
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

logger = get_logger(__name__)


class BrowserEngine:
    """真實瀏覽器引擎（Playwright）
    
    職責：
    1. 管理 Headless Browser 生命週期
    2. 執行真實 DOM 交互（click, fill, hover）
    3. 監聽 DOM 變更與網絡活動
    4. 生成上下文感知的測試輸入
    
    資源管理：
    - 每個 Worker 獨立 BrowserContext（Cookie 隔離）
    - 自動清理 Page 與 Context
    - 支持並發控制（最大 5 個 Context）
    """
    
    def __init__(self, headless: bool = True, max_contexts: int = 5):
        if not PLAYWRIGHT_AVAILABLE:
            raise AIVAError(
                message="Playwright not installed. Install: pip install playwright && playwright install chromium",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH
            )
        
        self.headless = headless
        self.max_contexts = max_contexts
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.contexts: List[BrowserContext] = []
        self._initialized = False
        
        logger.info(f"BrowserEngine initialized (headless={headless}, max_contexts={max_contexts})")
    
    async def initialize(self) -> None:
        """啟動 Playwright 和 Browser"""
        if self._initialized:
            return
        
        try:
            if async_playwright is None:
                raise AIVAError(
                    error_type=ErrorType.SYSTEM,
                    message="Playwright not available"
                )
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
            self._initialized = True
            logger.info("✅ Playwright Browser launched successfully")
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}", exc_info=True)
            raise AIVAError(
                message=f"Browser launch failed: {str(e)}",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH
            )
    
    async def create_context(self, user_agent: Optional[str] = None) -> BrowserContext:
        """創建隔離的 BrowserContext
        
        Args:
            user_agent: 自定義 User-Agent（模擬不同設備）
        
        Returns:
            BrowserContext 實例
        """
        if not self._initialized or not self.browser:
            await self.initialize()
        
        # 並發控制
        if len(self.contexts) >= self.max_contexts:
            logger.warning(f"Reached max contexts ({self.max_contexts}), closing oldest")
            await self.contexts[0].close()
            self.contexts.pop(0)
        
        context_options: Dict[str, Any] = {
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,  # 繞過自簽證書
        }
        
        if user_agent:
            context_options["user_agent"] = user_agent
        
        if self.browser is None:
            raise AIVAError(
                error_type=ErrorType.SYSTEM,
                message="Browser not initialized"
            )
        
        context = await self.browser.new_context(**context_options)
        self.contexts.append(context)
        
        logger.info(f"Created BrowserContext (total: {len(self.contexts)})")
        return context
    
    async def interact_with_page(
        self, 
        url: str, 
        actions: List[Dict[str, Any]],
        context: Optional[BrowserContext] = None
    ) -> Dict[str, Any]:
        """執行真實頁面交互
        
        Args:
            url: 目標 URL
            actions: 交互動作列表
            context: 可選的 BrowserContext（None 則創建新的）
        
        Returns:
            交互結果字典
        """
        if context is None:
            context = await self.create_context()
        
        if context is None:
            raise AIVAError(
                error_type=ErrorType.SYSTEM,
                message="Failed to create browser context"
            )
        
        page = await context.new_page()
        
        try:
            # 導航到目標頁面
            logger.info(f"Navigating to {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 設置監聽器
            await self._setup_dom_listeners(page)
            
            # 執行交互動作
            interaction_results = await self._execute_actions(page, actions)
            
            # 收集結果
            return await self._collect_interaction_results(page, url, interaction_results)
        
        except Exception as e:
            logger.error(f"Page interaction failed: {e}", exc_info=True)
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }
        
        finally:
            await page.close()
    
    async def _setup_dom_listeners(self, page: Page) -> None:
        """設置 DOM 變更監聽器"""
        await page.evaluate("""
            () => {
                window.__domChanges = [];
                const observer = new MutationObserver((mutations) => {
                    window.__domChanges.push({
                        timestamp: Date.now(),
                        type: mutations[0].type,
                        addedNodes: mutations[0].addedNodes.length,
                        removedNodes: mutations[0].removedNodes.length
                    });
                });
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true
                });
            }
        """)
    
    async def _execute_actions(self, page: Page, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """執行交互動作列表"""
        interaction_results = []
        
        for action in actions:
            result = await self._execute_single_action(page, action)
            interaction_results.append(result)
            # 等待 DOM 變更完成
            await asyncio.sleep(0.5)
        
        return interaction_results
    
    async def _execute_single_action(self, page: Page, action: Dict[str, Any]) -> Dict[str, Any]:
        """執行單一交互動作"""
        action_type = action.get("type")
        selector = action.get("selector")
        
        # 檢查 selector 是否為有效字符串
        if not isinstance(selector, str) or not selector.strip():
            logger.warning(f"Invalid selector for action {action_type}: {selector}")
            return {"action": action_type, "selector": selector, "success": False, "error": "Invalid selector"}
        
        try:
            if action_type == "click":
                await page.click(selector, force=True, timeout=5000)
                logger.info(f"✅ Clicked: {selector}")
                return {"action": "click", "selector": selector, "success": True}
            
            elif action_type == "fill":
                value = action.get("value")
                if not value:
                    value = await self._generate_context_aware_input(page, selector)
                
                await page.fill(selector, value, timeout=5000)
                logger.info(f"✅ Filled: {selector} = {value}")
                return {"action": "fill", "selector": selector, "value": value, "success": True}
            
            elif action_type == "hover":
                await page.hover(selector, timeout=5000)
                logger.info(f"✅ Hovered: {selector}")
                return {"action": "hover", "selector": selector, "success": True}
            
            else:
                return {"action": action_type, "selector": selector, "success": False, "error": "Unknown action type"}
                
        except PlaywrightTimeout:
            logger.warning(f"⚠️ Timeout on {action_type}: {selector}")
            return {"action": action_type, "selector": selector, "success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"❌ Error on {action_type} {selector}: {e}")
            return {"action": action_type, "selector": selector, "success": False, "error": str(e)}
    
    async def _collect_interaction_results(self, page: Page, url: str, interaction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """收集交互結果"""
        # 收集 DOM 變更記錄
        dom_changes = await page.evaluate("() => window.__domChanges || []")
        
        # 截圖（用於調試）
        screenshot = await page.screenshot(type="png", full_page=False)
        
        return {
            "url": url,
            "success": True,
            "interactions": interaction_results,
            "dom_changes": len(dom_changes),
            "dom_change_details": dom_changes[-5:] if dom_changes else [],
            "screenshot_size": len(screenshot),
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    async def _generate_context_aware_input(self, page: Page, selector: str) -> str:
        """生成上下文感知的測試輸入
        
        分析 Input 的 name, id, placeholder, aria-label 屬性
        
        Args:
            page: Playwright Page 實例
            selector: Input 選擇器
        
        Returns:
            生成的測試值
        """
        try:
            # 獲取元素屬性
            attributes = await page.evaluate("""
                (selector) => {
                    const el = document.querySelector(selector);
                    if (!el) return null;
                    return {{
                        name: el.name || '',
                        id: el.id || '',
                        type: el.type || '',
                        placeholder: el.placeholder || '',
                        ariaLabel: el.getAttribute('aria-label') || ''
                    }};
                }}
            """, selector)
            
            if not attributes:
                return "test_value"
            
            # 語義分析
            combined_text = (
                f"{attributes['name']} {attributes['id']} "
                f"{attributes['placeholder']} {attributes['ariaLabel']}"
            ).lower()
            
            # Email 檢測
            if re.search(r'email|e-mail|mail', combined_text):
                timestamp = int(datetime.now(UTC).timestamp())
                domain = urlparse(page.url).netloc.replace('www.', '')
                return f"test+{timestamp}@{domain}"
            
            # Search/Query 檢測（注入 XSS 探針）
            if re.search(r'search|query|q\b', combined_text):
                return "javascript://%250Aalert(1)//"
            
            # 數字檢測（邊界值）
            if re.search(r'age|year|quantity|amount|number', combined_text):
                return "-1"
            
            # 用戶名檢測
            if re.search(r'user|username|account|login', combined_text):
                return f"testuser_{int(datetime.now().timestamp())}"
            
            # 密碼檢測
            if re.search(r'pass|password|pwd', combined_text):
                return "TestP@ssw0rd123!"
            
            # 默認值
            return "test_input"
        
        except Exception as e:
            logger.error(f"Failed to generate context-aware input: {e}")
            return "default_value"
    
    async def extract_interactive_elements(self, url: str) -> List[Dict[str, Any]]:
        """提取頁面中所有可交互元素
        
        Args:
            url: 目標 URL
        
        Returns:
            可交互元素列表
        """
        context = await self.create_context()
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            elements = await page.evaluate("""
                () => {
                    const interactives = [];
                    
                    // Buttons
                    document.querySelectorAll('button, input[type="button"], input[type="submit"]').forEach(el => {
                        interactives.push({
                            type: 'button',
                            selector: el.id ? `#${el.id}` : `button:nth-child(${Array.from(el.parentNode.children).indexOf(el) + 1})`,
                            text: el.textContent.trim()
                        });
                    });
                    
                    // Input fields
                    document.querySelectorAll('input:not([type="hidden"]), textarea').forEach(el => {
                        interactives.push({
                            type: 'input',
                            selector: el.id ? `#${el.id}` : (el.name ? `[name="${el.name}"]` : null),
                            inputType: el.type,
                            name: el.name,
                            placeholder: el.placeholder
                        });
                    });
                    
                    // Links
                    document.querySelectorAll('a[href]').forEach((el, idx) => {
                        if (idx < 20) {  // 限制數量
                            interactives.push({
                                type: 'link',
                                href: el.href,
                                text: el.textContent.trim().substring(0, 50)
                            });
                        }
                    });
                    
                    return interactives.filter(el => el.selector || el.href);
                }
            """)
            
            logger.info(f"Extracted {len(elements)} interactive elements from {url}")
            return elements
        
        finally:
            await page.close()
    
    async def shutdown(self) -> None:
        """關閉所有 Context 和 Browser"""
        logger.info("Shutting down BrowserEngine...")
        
        for context in self.contexts:
            try:
                await context.close()
            except Exception as e:
                logger.error(f"Error closing context: {e}")
        
        self.contexts.clear()
        
        if self.browser:
            await self.browser.close()
        
        if self.playwright:
            await self.playwright.stop()
        
        self._initialized = False
        logger.info("✅ BrowserEngine shutdown complete")
