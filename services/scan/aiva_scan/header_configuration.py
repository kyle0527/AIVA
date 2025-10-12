from __future__ import annotations

import random
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class HeaderConfiguration:
    """
    組裝和管理每個請求的 HTTP 頭。

    主要功能:
    - 管理默認和自定義 HTTP 頭
    - 提供常見的用戶代理字符串
    - 支持用戶代理輪換
    - 合併系統默認頭和用戶自定義頭
    - 動態調整請求頭

    使用範例:
        # 基本使用
        config = HeaderConfiguration({"X-API-Key": "secret"})
        headers = config.get_headers()

        # 使用自定義 User-Agent
        config = HeaderConfiguration({}, user_agent="Custom Bot 1.0")
        headers = config.get_headers()

        # 啟用 User-Agent 輪換
        config = HeaderConfiguration({}, rotate_user_agent=True)
        headers1 = config.get_headers()  # 隨機 UA
        headers2 = config.get_headers()  # 可能不同的 UA
    """

    # 默認的 User-Agent 池
    _DEFAULT_USER_AGENTS = [
        # Chrome on Windows
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        ),
        # Chrome on macOS
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        # Firefox on Windows
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
            "Gecko/20100101 Firefox/121.0"
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) "
            "Gecko/20100101 Firefox/120.0"
        ),
        # Firefox on macOS
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) "
            "Gecko/20100101 Firefox/121.0"
        ),
        # Safari on macOS
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.2 Safari/605.1.15"
        ),
        # Edge on Windows
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ),
        # Mobile Chrome
        (
            "Mozilla/5.0 (Linux; Android 13) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.6099.144 Mobile Safari/537.36"
        ),
        # Mobile Safari
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.2 Mobile/15E148 Safari/604.1"
        ),
    ]

    # 默認的系統頭
    _DEFAULT_HEADERS = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    def __init__(
        self,
        user_headers: dict[str, str] | None = None,
        *,
        user_agent: str | None = None,
        rotate_user_agent: bool = False,
        include_default_headers: bool = True,
    ) -> None:
        """
        初始化頭配置管理器

        Args:
            user_headers: 用戶自定義的 HTTP 頭
            user_agent: 自定義 User-Agent（如果提供，會覆蓋默認值）
            rotate_user_agent: 是否每次請求輪換 User-Agent
            include_default_headers: 是否包含默認的系統頭
        """
        self.user_headers = user_headers or {}
        self._custom_user_agent = user_agent
        self._rotate_user_agent = rotate_user_agent
        self._include_default_headers = include_default_headers
        self._current_user_agent = user_agent or self._get_random_user_agent()

        logger.debug(
            f"HeaderConfiguration initialized: "
            f"{len(self.user_headers)} custom headers, "
            f"rotate_ua={rotate_user_agent}"
        )

    def get_headers(self, *, for_json: bool = False) -> dict[str, str]:
        """
        獲取組裝後的 HTTP 頭

        Args:
            for_json: 是否為 JSON 請求（會添加 Content-Type）

        Returns:
            dict: 完整的 HTTP 頭字典
        """
        headers = {}

        # 1. 添加默認頭（如果啟用）
        if self._include_default_headers:
            headers.update(self._DEFAULT_HEADERS)

        # 2. 添加 User-Agent
        if self._rotate_user_agent:
            self._current_user_agent = self._get_random_user_agent()

        headers["User-Agent"] = self._current_user_agent

        # 3. 如果是 JSON 請求，調整 Content-Type 和 Accept
        if for_json:
            headers["Content-Type"] = "application/json"
            headers["Accept"] = "application/json, */*"

        # 4. 用戶自定義頭覆蓋默認值
        headers.update(self.user_headers)

        return headers

    def get_headers_for_api(self) -> dict[str, str]:
        """
        獲取適合 API 請求的頭

        Returns:
            dict: API 請求頭
        """
        headers = {
            "Accept": "application/json, */*",
            "Content-Type": "application/json",
            "User-Agent": self._current_user_agent,
        }

        # 添加用戶自定義頭
        headers.update(self.user_headers)

        return headers

    def add_header(self, key: str, value: str) -> None:
        """
        添加或更新一個自定義頭

        Args:
            key: 頭名稱
            value: 頭值
        """
        self.user_headers[key] = value
        logger.debug(f"Added/updated header: {key}={value}")

    def remove_header(self, key: str) -> None:
        """
        移除一個自定義頭

        Args:
            key: 頭名稱
        """
        if key in self.user_headers:
            del self.user_headers[key]
            logger.debug(f"Removed header: {key}")

    def set_user_agent(self, user_agent: str) -> None:
        """
        設置自定義 User-Agent

        Args:
            user_agent: User-Agent 字符串
        """
        self._custom_user_agent = user_agent
        self._current_user_agent = user_agent
        logger.debug(f"Set custom User-Agent: {user_agent[:50]}...")

    def enable_user_agent_rotation(self, enable: bool = True) -> None:
        """
        啟用或禁用 User-Agent 輪換

        Args:
            enable: 是否啟用
        """
        self._rotate_user_agent = enable
        logger.debug(f"User-Agent rotation: {'enabled' if enable else 'disabled'}")

    def set_referer(self, referer: str) -> None:
        """
        設置 Referer 頭

        Args:
            referer: Referer URL
        """
        self.add_header("Referer", referer)

    def set_authorization(self, token: str, auth_type: str = "Bearer") -> None:
        """
        設置 Authorization 頭

        Args:
            token: 認證令牌
            auth_type: 認證類型（Bearer, Basic 等）
        """
        self.add_header("Authorization", f"{auth_type} {token}")

    def set_cookie(self, cookie: str) -> None:
        """
        設置 Cookie 頭

        Args:
            cookie: Cookie 字符串
        """
        self.add_header("Cookie", cookie)

    def get_user_agent(self) -> str:
        """獲取當前的 User-Agent"""
        return self._current_user_agent

    def get_custom_headers(self) -> dict[str, str]:
        """獲取用戶自定義的頭"""
        return self.user_headers.copy()

    def clear_custom_headers(self) -> None:
        """清除所有自定義頭"""
        self.user_headers.clear()
        logger.debug("Cleared all custom headers")

    def merge_headers(self, additional_headers: dict[str, str]) -> None:
        """
        合併額外的頭

        Args:
            additional_headers: 要合併的頭字典
        """
        self.user_headers.update(additional_headers)
        logger.debug(f"Merged {len(additional_headers)} additional headers")

    def get_headers_summary(self) -> str:
        """
        獲取頭配置摘要

        Returns:
            str: 摘要文本
        """
        lines = [
            "Header Configuration:",
            f"  User-Agent: {self._current_user_agent[:60]}...",
            f"  Rotate UA: {self._rotate_user_agent}",
            f"  Custom Headers: {len(self.user_headers)}",
        ]

        if self.user_headers:
            lines.append("  Custom Header Keys:")
            for key in list(self.user_headers.keys())[:10]:
                lines.append(f"    - {key}")
            if len(self.user_headers) > 10:
                lines.append(f"    ... and {len(self.user_headers) - 10} more")

        return "\n".join(lines)

    def clone(self) -> HeaderConfiguration:
        """
        克隆當前配置

        Returns:
            HeaderConfiguration: 新的配置實例
        """
        new_config = HeaderConfiguration(
            user_headers=self.user_headers.copy(),
            user_agent=self._custom_user_agent,
            rotate_user_agent=self._rotate_user_agent,
            include_default_headers=self._include_default_headers,
        )
        return new_config

    def to_dict(self) -> dict[str, Any]:
        """
        將配置轉換為字典

        Returns:
            dict: 配置字典
        """
        return {
            "user_headers": self.user_headers,
            "user_agent": self._custom_user_agent,
            "rotate_user_agent": self._rotate_user_agent,
            "include_default_headers": self._include_default_headers,
            "current_user_agent": self._current_user_agent,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> HeaderConfiguration:
        """
        從字典創建配置

        Args:
            config_dict: 配置字典

        Returns:
            HeaderConfiguration: 新的配置實例
        """
        return cls(
            user_headers=config_dict.get("user_headers", {}),
            user_agent=config_dict.get("user_agent"),
            rotate_user_agent=config_dict.get("rotate_user_agent", False),
            include_default_headers=config_dict.get("include_default_headers", True),
        )

    def _get_random_user_agent(self) -> str:
        """
        獲取隨機 User-Agent

        Returns:
            str: 隨機的 User-Agent 字符串
        """
        return random.choice(self._DEFAULT_USER_AGENTS)

    @classmethod
    def get_default_user_agents(cls) -> list[str]:
        """獲取所有默認的 User-Agent 列表"""
        return cls._DEFAULT_USER_AGENTS.copy()

    @classmethod
    def get_default_headers(cls) -> dict[str, str]:
        """獲取默認的系統頭"""
        return cls._DEFAULT_HEADERS.copy()

    @classmethod
    def create_stealth_config(cls) -> HeaderConfiguration:
        """
        創建隱秘模式配置（模擬真實瀏覽器）

        Returns:
            HeaderConfiguration: 隱秘模式配置
        """
        config = cls(
            user_headers={
                "Accept-Language": ("en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7"),
                "Sec-Ch-Ua": (
                    '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"'
                ),
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            },
            rotate_user_agent=True,
        )
        return config

    @classmethod
    def create_api_config(cls, api_key: str | None = None) -> HeaderConfiguration:
        """
        創建 API 請求配置

        Args:
            api_key: API 密鑰（可選）

        Returns:
            HeaderConfiguration: API 配置
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if api_key:
            headers["X-API-Key"] = api_key

        config = cls(
            user_headers=headers,
            user_agent="AIVA-Scanner/1.0",
            include_default_headers=False,
        )
        return config

    @classmethod
    def create_minimal_config(cls) -> HeaderConfiguration:
        """
        創建最小配置（只有必要的頭）

        Returns:
            HeaderConfiguration: 最小配置
        """
        config = cls(
            user_headers={},
            user_agent="AIVA-Scanner/1.0",
            include_default_headers=False,
        )
        return config
