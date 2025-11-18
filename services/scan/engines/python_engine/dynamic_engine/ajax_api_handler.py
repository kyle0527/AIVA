"""
AJAX 和 API 端點處理器
擴展動態引擎以識別和處理 AJAX 請求和 API 端點
"""


import re
from typing import Any
from urllib.parse import urljoin, urlparse

from services.aiva_common.schemas import Asset
from services.aiva_common.utils import get_logger, new_id

logger = get_logger(__name__)


class AjaxApiHandler:
    """
    處理 AJAX 請求和 API 端點的識別與分析
    """

    # API 端點常見模式
    API_PATTERNS = [
        r"/api/v\d+/",
        r"/rest/",
        r"/graphql",
        r"/v\d+/",
        r"\.json$",
        r"\.xml$",
    ]

    # AJAX 請求特徵
    AJAX_HEADERS = {
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json",
    }

    def __init__(self):
        """初始化 AJAX/API 處理器"""
        self.discovered_endpoints: list[dict[str, Any]] = []
        self.api_pattern_regex = re.compile("|".join(self.API_PATTERNS), re.IGNORECASE)
        logger.debug("AjaxApiHandler initialized")

    async def analyze_javascript_for_ajax(
        self, js_content: str, base_url: str
    ) -> list[Asset]:
        """
        分析 JavaScript 代碼以提取 AJAX 請求

        Args:
            js_content: JavaScript 代碼內容
            base_url: 基礎 URL

        Returns:
            發現的 API 端點資產列表
        """
        assets: list[Asset] = []

        # 查找 fetch() 調用
        fetch_patterns = [
            r'fetch\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'fetch\s*\(\s*`([^`]+)`',
        ]

        for pattern in fetch_patterns:
            matches = re.finditer(pattern, js_content, re.IGNORECASE)
            for match in matches:
                endpoint = match.group(1)
                full_url = self._normalize_url(endpoint, base_url)
                if full_url and self._is_valid_endpoint(full_url):
                    asset = self._create_api_asset(full_url, "fetch")
                    assets.append(asset)
                    logger.debug(f"Found fetch() endpoint: {full_url}")

        # 查找 XMLHttpRequest 調用
        xhr_patterns = [
            r'\.open\s*\(\s*[\'"](\w+)[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]',
        ]

        for pattern in xhr_patterns:
            matches = re.finditer(pattern, js_content, re.IGNORECASE)
            for match in matches:
                method = match.group(1)
                endpoint = match.group(2)
                full_url = self._normalize_url(endpoint, base_url)
                if full_url and self._is_valid_endpoint(full_url):
                    asset = self._create_api_asset(full_url, "xhr", method)
                    assets.append(asset)
                    logger.debug(f"Found XHR endpoint: {method} {full_url}")

        # 查找 jQuery AJAX 調用
        jquery_patterns = [
            r'\$\.ajax\s*\(\s*\{[^}]*url\s*:\s*[\'"]([^\'"]+)[\'"]',
            r'\$\.get\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'\$\.post\s*\(\s*[\'"]([^\'"]+)[\'"]',
        ]

        for pattern in jquery_patterns:
            matches = re.finditer(pattern, js_content, re.IGNORECASE)
            for match in matches:
                endpoint = match.group(1)
                full_url = self._normalize_url(endpoint, base_url)
                if full_url and self._is_valid_endpoint(full_url):
                    asset = self._create_api_asset(full_url, "jquery")
                    assets.append(asset)
                    logger.debug(f"Found jQuery AJAX endpoint: {full_url}")

        # 查找 Axios 調用
        axios_patterns = [
            r'axios\.\w+\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'axios\s*\(\s*\{[^}]*url\s*:\s*[\'"]([^\'"]+)[\'"]',
        ]

        for pattern in axios_patterns:
            matches = re.finditer(pattern, js_content, re.IGNORECASE)
            for match in matches:
                endpoint = match.group(1)
                full_url = self._normalize_url(endpoint, base_url)
                if full_url and self._is_valid_endpoint(full_url):
                    asset = self._create_api_asset(full_url, "axios")
                    assets.append(asset)
                    logger.debug(f"Found Axios endpoint: {full_url}")

        logger.info(f"Extracted {len(assets)} API endpoints from JavaScript")
        return assets

    def _normalize_url(self, endpoint: str, base_url: str) -> str | None:
        """
        規範化 URL

        Args:
            endpoint: 端點 URL
            base_url: 基礎 URL

        Returns:
            規範化的完整 URL
        """
        try:
            # 移除模板變量
            endpoint = re.sub(r'\$\{[^}]+\}', '', endpoint)
            endpoint = re.sub(r'\{[^}]+\}', '', endpoint)

            # 如果是相對路徑，與 base_url 組合
            if not endpoint.startswith(('http://', 'https://', '//')):
                endpoint = urljoin(base_url, endpoint)

            # 驗證 URL 格式
            parsed = urlparse(endpoint)
            if parsed.scheme and parsed.netloc:
                return endpoint
            return None
        except Exception as e:
            logger.warning(f"Failed to normalize URL {endpoint}: {e}")
            return None

    def _is_valid_endpoint(self, url: str) -> bool:
        """
        判斷是否為有效的 API 端點

        Args:
            url: URL 字符串

        Returns:
            是否為有效端點
        """
        # 檢查是否匹配 API 模式
        if self.api_pattern_regex.search(url):
            return True

        # 檢查是否為靜態資源（排除）
        static_extensions = ['.css', '.js', '.jpg', '.png', '.gif', '.svg', '.woff', '.ttf']
        parsed = urlparse(url)
        return not any(parsed.path.lower().endswith(ext) for ext in static_extensions)

    def _create_api_asset(
        self, url: str, source: str, method: str = "GET"
    ) -> Asset:
        """
        創建 API 端點資產

        Args:
            url: API URL
            source: 來源（fetch, xhr, jquery, axios）
            method: HTTP 方法

        Returns:
            API 資產對象
        """
        asset = Asset(
            asset_id=new_id("asset"),
            type="API_ENDPOINT",
            value=url,
            metadata={
                "source": source,
                "method": method.upper(),
                "is_ajax": True,
            },
        )
        self.discovered_endpoints.append({
            "url": url,
            "source": source,
            "method": method,
        })
        return asset

    async def test_ajax_endpoint(
        self, url: str, method: str = "GET"
    ) -> dict[str, Any]:
        """
        測試 AJAX 端點

        Args:
            url: 端點 URL
            method: HTTP 方法

        Returns:
            測試結果
        """
        import httpx

        headers = self.AJAX_HEADERS.copy()
        result = {
            "url": url,
            "method": method,
            "accessible": False,
            "response_type": None,
            "status_code": None,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers)
                else:
                    response = await client.request(method, url, headers=headers)

                result["accessible"] = True
                result["status_code"] = response.status_code
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type:
                    result["response_type"] = "json"
                elif "xml" in content_type:
                    result["response_type"] = "xml"
                else:
                    result["response_type"] = "other"

                logger.debug(
                    f"AJAX endpoint test: {method} {url} -> {response.status_code}"
                )
        except Exception as e:
            logger.warning(f"Failed to test AJAX endpoint {url}: {e}")
            result["error"] = str(e)

        return result

    def get_discovered_endpoints(self) -> list[dict[str, Any]]:
        """
        獲取已發現的端點列表

        Returns:
            端點信息列表
        """
        return self.discovered_endpoints.copy()

    def get_statistics(self) -> dict[str, Any]:
        """
        獲取統計信息

        Returns:
            統計數據
        """
        stats = {
            "total_endpoints": len(self.discovered_endpoints),
            "by_source": {},
            "by_method": {},
        }

        for endpoint in self.discovered_endpoints:
            source = endpoint["source"]
            method = endpoint["method"]

            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            stats["by_method"][method] = stats["by_method"].get(method, 0) + 1

        return stats
