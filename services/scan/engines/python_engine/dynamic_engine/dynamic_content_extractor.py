

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from services.aiva_common.schemas import Asset
from services.aiva_common.utils import get_logger
from services.aiva_common.utils.ids import new_id

logger = get_logger(__name__)


class ContentType(Enum):
    """動態內容類型"""

    FORM = "form"
    LINK = "link"
    AJAX_ENDPOINT = "ajax_endpoint"
    WEBSOCKET = "websocket"
    API_CALL = "api_call"
    JAVASCRIPT_VARIABLE = "js_variable"
    DOM_ELEMENT = "dom_element"
    EVENT_LISTENER = "event_listener"


class ExtractionStrategy(Enum):
    """提取策略"""

    IMMEDIATE = "immediate"  # 立即提取（頁面加載後）
    AFTER_INTERACTION = "after_interaction"  # 互動後提取
    PERIODIC = "periodic"  # 定期提取
    ON_MUTATION = "on_mutation"  # DOM 變更時提取


@dataclass
class DynamicContent:
    """動態內容數據結構"""

    content_id: str
    content_type: ContentType
    url: str
    source_url: str
    extraction_time: datetime = field(default_factory=datetime.now)
    html_content: str | None = None
    text_content: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkRequest:
    """網絡請求數據"""

    request_id: str
    url: str
    method: str
    resource_type: str
    headers: dict[str, str] = field(default_factory=dict)
    post_data: str | None = None
    response_status: int | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractionConfig:
    """提取配置"""

    extract_forms: bool = True
    extract_links: bool = True
    extract_ajax: bool = True
    extract_websockets: bool = True
    extract_api_calls: bool = True
    extract_js_variables: bool = False
    extract_event_listeners: bool = False
    wait_for_network_idle: bool = True
    network_idle_timeout_ms: int = 2000
    max_wait_time_ms: int = 30000
    capture_screenshots: bool = False
    capture_network_requests: bool = True
    extract_hidden_elements: bool = True
    min_content_length: int = 10


class DynamicContentExtractor:
    """
    動態內容提取器

    從渲染後的頁面中提取動態生成的內容，包括：
    - 表單和輸入欄位
    - 動態生成的鏈接
    - AJAX 端點和 API 調用
    - WebSocket 連接
    - JavaScript 變量
    - DOM 元素和事件監聽器

    特性：
    - 支持多種提取策略
    - 網絡請求監控
    - DOM 變更觀察
    - 智能等待機制
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """
        初始化動態內容提取器

        Args:
            config: 提取配置，如果為 None 則使用默認配置
        """
        self.config = config or ExtractionConfig()
        self._network_requests: list[NetworkRequest] = []
        self._extracted_contents: list[DynamicContent] = []
        self._mutation_observer_script: str | None = None

    async def extract_from_url(
        self, url: str, *, page: Any = None, wait_time_ms: int | None = None
    ) -> list[DynamicContent]:
        """
        從 URL 提取動態內容

        Args:
            url: 要提取的 URL
            page: 瀏覽器頁面對象（Playwright Page）
            wait_time_ms: 等待時間（毫秒），如果為 None 則使用配置值

        Returns:
            提取的動態內容列表
        """
        if page is None:
            logger.warning("No page object provided, using static extraction")
            return await self._extract_static(url)

        try:
            # 設置網絡請求監聽
            if self.config.capture_network_requests:
                await self._setup_network_listener(page)

            # 訪問頁面
            await page.goto(url, wait_until="domcontentloaded")
            logger.info(f"Loaded page: {url}")

            # 等待網絡空閒
            if self.config.wait_for_network_idle:
                wait_time = wait_time_ms or self.config.network_idle_timeout_ms
                try:
                    await page.wait_for_load_state("networkidle", timeout=wait_time)
                except Exception as e:
                    logger.debug(f"Network idle timeout: {e}")

            # 提取內容
            contents = await self._extract_from_page(page, url)

            return contents

        except Exception as e:
            logger.error(f"Failed to extract from {url}: {e}", exc_info=True)
            return []

    async def extract_after_interaction(
        self, page: Any, url: str, *, wait_time_ms: int = 1000
    ) -> list[DynamicContent]:
        """
        在互動後提取內容

        Args:
            page: 瀏覽器頁面對象
            url: 當前 URL
            wait_time_ms: 互動後等待時間（毫秒）

        Returns:
            提取的動態內容列表
        """
        if page is None:
            return []

        try:
            # 等待一段時間讓動態內容加載
            await asyncio.sleep(wait_time_ms / 1000)

            # 提取內容
            return await self._extract_from_page(page, url)

        except Exception as e:
            logger.error(f"Failed to extract after interaction: {e}", exc_info=True)
            return []

    async def _extract_from_page(
        self, page: Any, source_url: str
    ) -> list[DynamicContent]:
        """從頁面提取所有類型的動態內容"""
        contents: list[DynamicContent] = []

        try:
            # 獲取頁面 HTML
            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            # 提取表單
            if self.config.extract_forms:
                forms = await self._extract_forms(soup, source_url)
                contents.extend(forms)

            # 提取鏈接
            if self.config.extract_links:
                links = await self._extract_links(soup, source_url)
                contents.extend(links)

            # 提取 AJAX 端點
            if self.config.extract_ajax:
                ajax_endpoints = await self._extract_ajax_endpoints(page, source_url)
                contents.extend(ajax_endpoints)

            # 提取 API 調用
            if self.config.extract_api_calls:
                api_calls = await self._extract_api_calls(source_url)
                contents.extend(api_calls)

            # 提取 JavaScript 變量
            if self.config.extract_js_variables:
                js_vars = await self._extract_js_variables(page, source_url)
                contents.extend(js_vars)

            # 提取事件監聽器
            if self.config.extract_event_listeners:
                listeners = await self._extract_event_listeners(page, source_url)
                contents.extend(listeners)

            self._extracted_contents.extend(contents)
            logger.info(f"Extracted {len(contents)} dynamic contents from {source_url}")

            return contents

        except Exception as e:
            logger.error(f"Failed to extract from page: {e}", exc_info=True)
            return []

    async def _extract_forms(
        self, soup: BeautifulSoup, source_url: str
    ) -> list[DynamicContent]:
        """提取表單"""
        contents: list[DynamicContent] = []

        forms = soup.find_all("form")
        for form in forms:
            try:
                # 確保 form 是 Tag 對象
                if not hasattr(form, "get"):
                    continue

                # 獲取表單屬性
                action = form.get("action", "")
                method_val = form.get("method", "GET")
                method = method_val.upper() if isinstance(method_val, str) else "GET"
                form_id = form.get("id", "")
                form_name = form.get("name", "")

                # 解析 action URL
                action_url = urljoin(source_url, action) if action else source_url

                # 提取輸入欄位
                inputs = []
                if hasattr(form, "find_all"):
                    input_elements = form.find_all(["input", "select", "textarea"])
                    if not isinstance(input_elements, list):
                        input_elements = []
                else:
                    input_elements = []

                for input_elem in input_elements:
                    if not hasattr(input_elem, "get"):
                        continue

                    input_data = {
                        "type": input_elem.get("type", "text"),
                        "name": input_elem.get("name", ""),
                        "id": input_elem.get("id", ""),
                        "value": input_elem.get("value", ""),
                        "required": (input_elem.get("required") is not None),
                    }
                    inputs.append(input_data)

                content = DynamicContent(
                    content_id=new_id("content"),
                    content_type=ContentType.FORM,
                    url=action_url,
                    source_url=source_url,
                    html_content=str(form),
                    attributes={
                        "method": method,
                        "id": form_id,
                        "name": form_name,
                        "inputs": inputs,
                    },
                )
                contents.append(content)

            except Exception as e:
                logger.debug(f"Failed to extract form: {e}")

        return contents

    async def _extract_links(
        self, soup: BeautifulSoup, source_url: str
    ) -> list[DynamicContent]:
        """提取鏈接"""
        contents: list[DynamicContent] = []
        seen_urls: set[str] = set()

        # 提取 <a> 標籤
        links = soup.find_all("a", href=True)
        for link in links:
            try:
                if not hasattr(link, "get"):
                    continue

                href = link.get("href", "")
                if not href or (
                    isinstance(href, str)
                    and href.startswith(("#", "javascript:", "mailto:"))
                ):
                    continue

                # 解析完整 URL
                full_url = urljoin(source_url, href)

                # 去重
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                # 獲取鏈接文本
                link_text = link.get_text(strip=True)

                # 獲取 rel 屬性並確保是正確類型
                rel_value = link.get("rel", [])
                if not isinstance(rel_value, list):
                    rel_value = []

                content = DynamicContent(
                    content_id=new_id("link"),
                    content_type=ContentType.LINK,
                    url=full_url,
                    source_url=source_url,
                    text_content=link_text,
                    attributes={
                        "href": href,
                        "text": link_text,
                        "rel": rel_value,
                        "target": link.get("target", ""),
                    },
                )
                contents.append(content)

            except Exception as e:
                logger.debug(f"Failed to extract link: {e}")

        return contents

    async def _extract_ajax_endpoints(
        self, page: Any, source_url: str
    ) -> list[DynamicContent]:
        """提取 AJAX 端點"""
        contents: list[DynamicContent] = []

        try:
            # 從網絡請求中提取 AJAX 端點
            ajax_requests = [
                req
                for req in self._network_requests
                if req.resource_type in ["xhr", "fetch"]
            ]

            for req in ajax_requests:
                content = DynamicContent(
                    content_id=new_id("ajax"),
                    content_type=ContentType.AJAX_ENDPOINT,
                    url=req.url,
                    source_url=source_url,
                    attributes={
                        "method": req.method,
                        "resource_type": req.resource_type,
                        "status": req.response_status,
                    },
                    metadata={
                        "headers": req.headers,
                        "response_headers": req.response_headers,
                    },
                )
                contents.append(content)

        except Exception as e:
            logger.debug(f"Failed to extract AJAX endpoints: {e}")

        return contents

    async def _extract_api_calls(self, source_url: str) -> list[DynamicContent]:
        """從網絡請求中提取 API 調用"""
        contents: list[DynamicContent] = []

        try:
            # 識別 API 調用（通常包含 /api/、/v1/、.json 等）
            api_patterns = [
                r"/api/",
                r"/v\d+/",
                r"\.json",
                r"/graphql",
                r"/rest/",
            ]

            for req in self._network_requests:
                url_lower = req.url.lower()
                if any(re.search(pattern, url_lower) for pattern in api_patterns):
                    content = DynamicContent(
                        content_id=new_id("api"),
                        content_type=ContentType.API_CALL,
                        url=req.url,
                        source_url=source_url,
                        attributes={
                            "method": req.method,
                            "status": req.response_status,
                            "post_data": req.post_data,
                        },
                        metadata={
                            "headers": req.headers,
                            "response_headers": req.response_headers,
                        },
                    )
                    contents.append(content)

        except Exception as e:
            logger.debug(f"Failed to extract API calls: {e}")

        return contents

    async def _extract_js_variables(
        self, page: Any, source_url: str
    ) -> list[DynamicContent]:
        """提取 JavaScript 變量"""
        contents: list[DynamicContent] = []

        try:
            # 執行 JavaScript 提取全局變量
            js_code = """
            () => {
                const vars = {};
                for (let key in window) {
                    try {
                        const value = window[key];
                        const type = typeof value;
                        if (
                            type === 'string' ||
                            type === 'number' ||
                            type === 'boolean'
                        ) {
                            vars[key] = { type, value };
                        } else if (type === 'object' && value !== null) {
                            vars[key] = {
                                type: 'object',
                                value: Object.keys(value).length + ' keys'
                            };
                        }
                    } catch (e) {
                        // Skip inaccessible properties
                    }
                }
                return vars;
            }
            """

            js_vars = await page.evaluate(js_code)

            for var_name, var_info in js_vars.items():
                content = DynamicContent(
                    content_id=new_id("jsvar"),
                    content_type=ContentType.JAVASCRIPT_VARIABLE,
                    url=source_url,
                    source_url=source_url,
                    text_content=str(var_info.get("value", "")),
                    attributes={
                        "variable_name": var_name,
                        "variable_type": var_info.get("type", ""),
                    },
                )
                contents.append(content)

        except Exception as e:
            logger.debug(f"Failed to extract JS variables: {e}")

        return contents

    async def _extract_event_listeners(
        self, page: Any, source_url: str
    ) -> list[DynamicContent]:
        """提取事件監聽器"""
        contents: list[DynamicContent] = []

        try:
            # 執行 JavaScript 查找事件監聽器
            js_code = """
            () => {
                const listeners = [];
                const elements = document.querySelectorAll('*');

                elements.forEach((elem, index) => {
                    if (index > 1000) return; // 限制數量

                    const events = [];
                    [
                        'click', 'submit', 'change',
                        'input', 'focus', 'blur'
                    ].forEach(eventType => {
                        if (elem['on' + eventType]) {
                            events.push(eventType);
                        }
                    });

                    if (events.length > 0) {
                        listeners.push({
                            tag: elem.tagName,
                            id: elem.id || '',
                            classes: elem.className || '',
                            events: events
                        });
                    }
                });

                return listeners;
            }
            """

            listeners_data = await page.evaluate(js_code)

            for listener in listeners_data:
                content = DynamicContent(
                    content_id=new_id("listener"),
                    content_type=ContentType.EVENT_LISTENER,
                    url=source_url,
                    source_url=source_url,
                    attributes={
                        "tag": listener.get("tag", ""),
                        "id": listener.get("id", ""),
                        "classes": listener.get("classes", ""),
                        "events": listener.get("events", []),
                    },
                )
                contents.append(content)

        except Exception as e:
            logger.debug(f"Failed to extract event listeners: {e}")

        return contents

    async def _setup_network_listener(self, page: Any) -> None:
        """設置網絡請求監聽器"""
        try:

            async def handle_request(request):
                """處理請求"""
                try:
                    req_data = NetworkRequest(
                        request_id=new_id("req"),
                        url=request.url,
                        method=request.method,
                        resource_type=request.resource_type,
                        headers=dict(request.headers),
                        post_data=request.post_data,
                    )
                    self._network_requests.append(req_data)
                except Exception as e:
                    logger.debug(f"Failed to capture request: {e}")

            async def handle_response(response):
                """處理響應"""
                try:
                    # 查找對應的請求
                    for req in self._network_requests:
                        if req.url == response.url:
                            req.response_status = response.status
                            req.response_headers = dict(response.headers)
                            break
                except Exception as e:
                    logger.debug(f"Failed to capture response: {e}")

            # 綁定事件處理器
            page.on("request", handle_request)
            page.on("response", handle_response)

        except Exception as e:
            logger.warning(f"Failed to setup network listener: {e}")

    async def _extract_static(self, url: str) -> list[DynamicContent]:
        """靜態提取（無瀏覽器）"""
        logger.info(f"Using static extraction for {url}")
        # 這裡可以使用 httpx + BeautifulSoup 進行基本的靜態提取
        # 暫時返回空列表
        return []

    def get_extracted_contents(self) -> list[DynamicContent]:
        """獲取所有已提取的內容"""
        return self._extracted_contents.copy()

    def get_network_requests(self) -> list[NetworkRequest]:
        """獲取所有網絡請求"""
        return self._network_requests.copy()

    def get_contents_by_type(self, content_type: ContentType) -> list[DynamicContent]:
        """按類型獲取內容"""
        return [c for c in self._extracted_contents if c.content_type == content_type]

    def convert_to_assets(
        self, contents: list[DynamicContent] | None = None
    ) -> list[Asset]:
        """
        將動態內容轉換為 Asset 對象

        Args:
            contents: 要轉換的內容列表，如果為 None 則使用所有已提取的內容

        Returns:
            Asset 對象列表
        """
        if contents is None:
            contents = self._extracted_contents

        assets: list[Asset] = []

        for content in contents:
            try:
                # 只轉換表單和鏈接
                if content.content_type == ContentType.FORM:
                    # 轉換表單為 Asset
                    params = []
                    for input_data in content.attributes.get("inputs", []):
                        if input_data.get("name"):
                            params.append(input_data["name"])

                    asset = Asset(
                        asset_id=content.content_id,
                        type="form",
                        value=content.url,
                        parameters=params,
                        has_form=True,
                    )
                    assets.append(asset)

                elif content.content_type == ContentType.LINK:
                    # 轉換鏈接為 Asset
                    asset = Asset(
                        asset_id=content.content_id,
                        type="url",
                        value=content.url,
                        parameters=None,
                        has_form=False,
                    )
                    assets.append(asset)

            except Exception as e:
                logger.debug(f"Failed to convert content to asset: {e}")

        return assets

    def get_stats(self) -> dict[str, Any]:
        """獲取統計信息"""
        type_counts: dict[str, int] = {}
        for content in self._extracted_contents:
            type_name = content.content_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_contents": len(self._extracted_contents),
            "total_network_requests": len(self._network_requests),
            "contents_by_type": type_counts,
            "config": {
                "extract_forms": self.config.extract_forms,
                "extract_links": self.config.extract_links,
                "extract_ajax": self.config.extract_ajax,
                "extract_api_calls": self.config.extract_api_calls,
                "capture_network_requests": self.config.capture_network_requests,
            },
        }

    def clear(self) -> None:
        """清空已提取的內容和網絡請求"""
        self._extracted_contents.clear()
        self._network_requests.clear()
        logger.debug("Cleared extracted contents and network requests")
