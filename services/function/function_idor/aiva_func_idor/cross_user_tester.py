"""
Cross User Tester for IDOR Detection

Implements cross-user testing according to OWASP WSTG-ATHZ-04 standards.
Tests for horizontal privilege escalation by attempting to access resources
belonging to other users.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import httpx
from pydantic import HttpUrl

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class CrossUserTestResult:
    """跨用戶測試結果 - 符合 OWASP 標準"""

    vulnerable: bool = False
    test_status: int = 0
    similarity_score: float = 0.0
    evidence: str = ""
    error_message: str | None = None


class CrossUserTester:
    """
    跨用戶測試器 - 實現 OWASP WSTG-ATHZ-04 標準

    根據 OWASP Web Security Testing Guide 4.5.4 -
    Testing for Insecure Direct Object References
    實現水平權限提升測試，檢測攻擊者是否能通過修改參數值訪問其他用戶的資源。
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def test_horizontal_idor(
        self,
        url: str | HttpUrl,
        resource_id: str,
        user_a_auth: dict[str, str] | None = None,
        user_b_auth: dict[str, str] | None = None,
        method: str = "GET",
    ) -> CrossUserTestResult:
        """
        執行水平 IDOR 測試

        根據 OWASP 標準，測試是否能夠通過修改資源 ID 訪問其他用戶的資源。

        Args:
            url: 測試目標 URL
            resource_id: 要測試的資源 ID
            user_a_auth: 用戶 A 的認證信息
            user_b_auth: 用戶 B 的認證信息
            method: HTTP 方法

        Returns:
            CrossUserTestResult: 測試結果
        """
        try:
            # 執行實際的 HTTP 請求測試
            headers = {}
            if user_b_auth:
                headers.update(user_b_auth)

            response = await self.client.request(
                method=method,
                url=str(url),
                headers=headers,
                timeout=30.0,
                follow_redirects=True
            )

            # 分析響應判斷是否存在 IDOR 漏洞
            # 成功響應碼且包含數據表示可能存在漏洞
            vulnerable = (
                response.status_code == 200
                and len(response.content) > 100
                and b"error" not in response.content.lower()
                and b"unauthorized" not in response.content.lower()
                and b"forbidden" not in response.content.lower()
            )

            return CrossUserTestResult(
                vulnerable=vulnerable,
                test_status=response.status_code,
                similarity_score=0.8 if vulnerable else 0.2,
                evidence=(
                    f"HTTP {response.status_code}, "
                    f"Content-Length: {len(response.content)}"
                ),
            )

        except TimeoutError:
            return CrossUserTestResult(
                vulnerable=False,
                test_status=0,
                similarity_score=0.0,
                evidence="Request timeout",
                error_message="Request timeout during IDOR test"
            )
        except Exception as e:
            return CrossUserTestResult(
                vulnerable=False,
                test_status=0,
                similarity_score=0.0,
                evidence=f"Error: {str(e)}",
                error_message=str(e)
            )

    async def test_cross_user_access(
        self,
        task: FunctionTaskPayload,
        resource_id: str,
        test_method: str = "direct"
    ) -> CrossUserTestResult:
        """
        測試跨用戶訪問

        Args:
            task: 功能任務負載
            resource_id: 測試的資源 ID
            test_method: 測試方法 ("direct", "modified", "bruteforce")

        Returns:
            CrossUserTestResult: 測試結果
        """
        # 構建測試 URL（替換原始 URL 中的資源 ID）
        url_str = str(task.target.url)

        # 簡單的 ID 替換邏輯
        if "?" in url_str:
            # URL 參數中的 ID
            parts = url_str.split("?")
            params = parts[1].split("&")
            for i, param in enumerate(params):
                if "=" in param:
                    key, value = param.split("=", 1)
                    if value.isdigit() or len(value) > 3:  # 可能是 ID
                        params[i] = f"{key}={resource_id}"
                        break
            test_url = f"{parts[0]}?{'&'.join(params)}"
        else:
            # URL 路[U+5F84]中的 ID
            test_url = re.sub(r'/\d+', f'/{resource_id}', url_str)
            if test_url == url_str:
                test_url = f"{url_str.rstrip('/')}/{resource_id}"

        return await self.test_horizontal_idor(
            url=test_url,
            resource_id=resource_id,
            method=task.target.method or "GET"
        )
