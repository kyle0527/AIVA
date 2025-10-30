

from typing import Any
from urllib.parse import urlparse

import httpx

from services.aiva_common.utils import get_logger
from services.aiva_common.utils.network import RateLimiter, RetryingAsyncClient

from ..authentication_manager import AuthenticationManager
from ..header_configuration import HeaderConfiguration

logger = get_logger(__name__)


class HiHttpClient:
    """
    高性能 HTTP 客戶端

    改進點:
    - 整合 RetryingAsyncClient 提供智能重試
    - 整合 RateLimiter 實現自適應速率限制
    - 支援並發連接池管理
    - 根據服務器響應自動調整請求速率

    特性:
    - 自動重試暫時性錯誤
    - 遵守 Retry-After 標頭
    - 全局和每主機雙層速率限制
    - 連接池復用
    """

    def __init__(
        self,
        auth: AuthenticationManager,
        headers: HeaderConfiguration,
        *,
        requests_per_second: float = 2.0,
        per_host_rps: float = 1.0,
        retries: int = 3,
        timeout: float = 20.0,
        pool_size: int = 10,
    ) -> None:
        """
        初始化 HTTP 客戶端

        Args:
            auth: 認證管理器
            headers: 標頭配置
            requests_per_second: 全局請求速率（每秒）
            per_host_rps: 每主機請求速率（每秒）
            retries: 重試次數
            timeout: 請求超時（秒）
            pool_size: 連接池大小
        """
        self._auth = auth
        self._headers = headers
        self._timeout = timeout
        self._pool_size = pool_size

        # 初始化速率限制器
        self._rate_limiter = RateLimiter(
            global_rps=requests_per_second,
            per_host_rps=per_host_rps,
        )

        # 初始化重試客戶端
        self._client = RetryingAsyncClient(
            retries=retries,
            timeout=timeout,
            follow_redirects=True,
            headers=self._headers.user_headers,
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 2,
            ),
        )

        logger.info(
            f"HTTP client initialized: {requests_per_second} global RPS, "
            f"{per_host_rps} per-host RPS, {retries} retries, {timeout}s timeout"
        )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response | None:
        """
        發送 GET 請求

        Args:
            url: 目標 URL
            **kwargs: 傳遞給 httpx 的額外參數

        Returns:
            響應對象，如果請求失敗則返回 None
        """
        # 提取主機名
        host = urlparse(url).netloc

        try:
            # 速率限制
            await self._rate_limiter.acquire(host)

            # 發送請求
            response = await self._client.get(url, **kwargs)

            # 更新速率限制器（基於響應狀態）
            self._rate_limiter.update_from_response(
                host,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

            logger.debug(f"GET {url} -> {response.status_code}")
            return response

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error for {url}: {e.response.status_code}")
            # 仍然更新速率限制器
            if hasattr(e, "response"):
                self._rate_limiter.update_from_response(
                    host,
                    status_code=e.response.status_code,
                    headers=dict(e.response.headers),
                )
            return None

        except httpx.RequestError as e:
            logger.warning(f"Request error for {url}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return None

    async def post(
        self, url: str, *, data: Any = None, json: Any = None, **kwargs: Any
    ) -> httpx.Response | None:
        """
        發送 POST 請求

        Args:
            url: 目標 URL
            data: 表單數據
            json: JSON 數據
            **kwargs: 傳遞給 httpx 的額外參數

        Returns:
            響應對象，如果請求失敗則返回 None
        """
        try:
            host = urlparse(url).netloc
            await self._rate_limiter.acquire(host)

            response = await self._client.post(url, data=data, json=json, **kwargs)

            self._rate_limiter.update_from_response(
                host,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

            logger.debug(f"POST {url} -> {response.status_code}")
            return response

        except Exception as e:
            logger.warning(f"POST request failed for {url}: {e}")
            return None

    async def close(self) -> None:
        """關閉客戶端並釋放資源"""
        await self._client.aclose()
        logger.info("HTTP client closed")

    async def __aenter__(self) -> HiHttpClient:
        """異步上下文管理器進入"""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """異步上下文管理器退出"""
        await self.close()

