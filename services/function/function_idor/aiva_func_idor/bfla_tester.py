"""
BFLA (Broken Function Level Authorization) Tester
測試破碎的函式級授權漏洞

檢測普通使用者是否能執行管理員專用的 HTTP 方法或端點
"""

import asyncio
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import (
    Authentication,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    Vulnerability,
)
from services.aiva_common.utils import new_id

logger = logging.getLogger(__name__)


class BFLATestResult(BaseModel):
    """BFLA 測試結果"""

    endpoint: str
    method: str
    admin_status: int
    user_status: int
    is_vulnerable: bool
    response_diff: dict[str, Any] = Field(default_factory=dict)


class BFLATester:
    """破碎的函式級授權測試器"""

    # 高風險 HTTP 方法（通常需要管理員權限）
    PRIVILEGED_METHODS = ["DELETE", "PUT", "PATCH", "POST"]

    # 管理員端點模式
    ADMIN_PATTERNS = [
        "/admin/",
        "/api/admin/",
        "/api/v1/admin/",
        "/api/v2/admin/",
        "/management/",
        "/console/",
        "/dashboard/admin/",
    ]

    def __init__(
        self,
        admin_auth: Authentication,
        user_auth: Authentication,
        timeout: int = 10,
        max_concurrent: int = 5,
    ):
        """
        初始化 BFLA 測試器

        Args:
            admin_auth: 管理員認證
            user_auth: 普通使用者認證
            timeout: 請求超時時間（秒）
            max_concurrent: 最大並發數
        """
        self.admin_auth = admin_auth
        self.user_auth = user_auth
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _send_request(
        self,
        method: str,
        url: str,
        auth: Authentication,
    ) -> httpx.Response | None:
        """發送 HTTP 請求"""
        try:
            headers = self._build_headers(auth)
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                )
                return response
        except Exception as e:
            logger.warning(f"Request failed: {method} {url}, error: {e}")
            return None

    def _build_headers(self, auth: Authentication) -> dict[str, str]:
        """建立請求標頭"""
        headers = {}

        if auth.credentials:
            # Bearer token
            if "bearer_token" in auth.credentials:
                headers["Authorization"] = f"Bearer {auth.credentials['bearer_token']}"
            # Basic auth
            elif "username" in auth.credentials and "password" in auth.credentials:
                import base64

                credentials_str = (
                    f"{auth.credentials['username']}:{auth.credentials['password']}"
                )
                encoded = base64.b64encode(credentials_str.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

            # Cookies
            if "cookies" in auth.credentials:
                cookies = auth.credentials["cookies"]
                if isinstance(cookies, dict):
                    headers["Cookie"] = "; ".join(
                        [f"{k}={v}" for k, v in cookies.items()]
                    )

            # Custom headers
            if "custom_headers" in auth.credentials:
                custom_headers = auth.credentials["custom_headers"]
                if isinstance(custom_headers, dict):
                    headers.update(custom_headers)

        return headers

    async def test_endpoint(
        self,
        endpoint: str,
        base_methods: list[str] | None = None,
    ) -> list[BFLATestResult]:
        """
        測試單一端點的 BFLA 漏洞

        Args:
            endpoint: 目標端點 URL
            base_methods: 要測試的 HTTP 方法列表（預設為 PRIVILEGED_METHODS）

        Returns:
            測試結果列表
        """
        methods = base_methods or self.PRIVILEGED_METHODS
        results = []

        for method in methods:
            async with self.semaphore:
                result = await self._test_method(endpoint, method)
                if result:
                    results.append(result)
                    if result.is_vulnerable:
                        logger.warning(
                            f"BFLA detected: {method} {endpoint} "
                            f"(admin={result.admin_status}, user={result.user_status})"
                        )

        return results

    async def _test_method(
        self,
        endpoint: str,
        method: str,
    ) -> BFLATestResult | None:
        """測試特定 HTTP 方法"""
        # 1. 使用管理員帳號發送請求
        admin_response = await self._send_request(method, endpoint, self.admin_auth)
        if not admin_response:
            return None

        # 2. 使用普通使用者帳號發送請求
        user_response = await self._send_request(method, endpoint, self.user_auth)
        if not user_response:
            return None

        # 3. 分析回應差異
        is_vulnerable = self._is_bfla_vulnerable(admin_response, user_response)

        return BFLATestResult(
            endpoint=endpoint,
            method=method,
            admin_status=admin_response.status_code,
            user_status=user_response.status_code,
            is_vulnerable=is_vulnerable,
            response_diff={
                "admin_content_length": len(admin_response.content),
                "user_content_length": len(user_response.content),
                "admin_headers": dict(admin_response.headers),
                "user_headers": dict(user_response.headers),
            },
        )

    def _is_bfla_vulnerable(
        self,
        admin_response: httpx.Response,
        user_response: httpx.Response,
    ) -> bool:
        """
        判斷是否存在 BFLA 漏洞

        漏洞條件：
        1. 管理員請求成功（200-299）
        2. 普通使用者請求也成功（200-299）
        3. 但該操作應該受到限制
        """
        admin_success = 200 <= admin_response.status_code < 300
        user_success = 200 <= user_response.status_code < 300

        # 如果普通使用者能成功執行管理員操作，即為漏洞
        if admin_success and user_success:
            return True

        # 特殊情況：普通使用者收到 301/302 而非 403/401，可能是繞過
        return admin_success and user_response.status_code in [301, 302]

    async def batch_test_endpoints(
        self,
        endpoints: list[str],
    ) -> dict[str, list[BFLATestResult]]:
        """批次測試多個端點"""
        results = {}
        tasks = []

        for endpoint in endpoints:
            task = self.test_endpoint(endpoint)
            tasks.append((endpoint, task))

        for endpoint, task in tasks:
            try:
                endpoint_results = await task
                if endpoint_results:
                    results[endpoint] = endpoint_results
            except Exception as e:
                logger.error(f"Failed to test {endpoint}: {e}")

        return results

    def create_finding(
        self,
        test_result: BFLATestResult,
        task_id: str,
    ) -> FindingPayload:
        """根據測試結果建立 Finding"""
        finding_id = new_id("finding")

        # 判斷嚴重性
        severity = self._determine_severity(test_result)

        # 建立漏洞物件
        vulnerability = Vulnerability(
            name=VulnerabilityType.BOLA,  # BFLA 是 BOLA 的一種形式
            cwe="CWE-285",  # Improper Authorization
            severity=severity,
            confidence=Confidence.FIRM,
        )

        # 建立目標
        target = FindingTarget(
            url=test_result.endpoint,
            method=test_result.method,
            parameter=None,
        )

        # 建立證據
        evidence = FindingEvidence(
            request=(
                f"{test_result.method} {test_result.endpoint}\n"
                f"Authorization: [User Credentials]\n"
                f"Response Status: {test_result.user_status}"
            ),
            response=f"HTTP {test_result.user_status}\n[Response content omitted]",
            payload=None,
            proof=(
                f"1. 使用管理員帳號執行 {test_result.method} {test_result.endpoint}\n"
                f"   結果: HTTP {test_result.admin_status}\n"
                f"2. 使用普通使用者帳號執行相同請求\n"
                f"   結果: HTTP {test_result.user_status} (應為 403 Forbidden)\n"
                f"3. 普通使用者能成功執行管理員操作"
            ),
        )

        # 建立影響
        impact = FindingImpact(
            description=(
                f"普通使用者能夠執行應限制於管理員的 {test_result.method} 操作"
            ),
            business_impact=(
                "攻擊者可以使用普通使用者帳號執行管理員操作，"
                "可能導致未授權的資料修改、刪除或系統配置變更"
            ),
        )

        # 建立修復建議
        recommendation = FindingRecommendation(
            fix=(
                "1. 實施嚴格的函式級授權檢查\n"
                "2. 在控制器層驗證使用者角色與權限\n"
                "3. 使用 RBAC (Role-Based Access Control) 或 ABAC\n"
                "4. 預設拒絕策略（Deny by Default）"
            ),
            priority="HIGH",
        )

        return FindingPayload(
            finding_id=finding_id,
            task_id=task_id,
            scan_id=task_id.split("_")[0] + "_scan",
            status="detected",
            vulnerability=vulnerability,
            target=target,
            evidence=evidence,
            impact=impact,
            recommendation=recommendation,
        )

    def _determine_severity(self, test_result: BFLATestResult) -> Severity:
        """根據測試結果判斷嚴重性"""
        # DELETE 方法漏洞最嚴重
        if test_result.method == "DELETE":
            return Severity.CRITICAL

        # PUT/PATCH 次之
        if test_result.method in ["PUT", "PATCH"]:
            return Severity.HIGH

        # POST 為中等
        if test_result.method == "POST":
            return Severity.MEDIUM

        return Severity.MEDIUM


async def main():
    """測試範例"""
    # 模擬認證
    admin_auth = Authentication(
        method="bearer",
        credentials={
            "username": "admin",
            "password": "admin123",
            "bearer_token": "admin_token_12345",
        },
    )

    user_auth = Authentication(
        method="bearer",
        credentials={
            "username": "user",
            "password": "user123",
            "bearer_token": "user_token_67890",
        },
    )

    # 建立測試器
    tester = BFLATester(admin_auth=admin_auth, user_auth=user_auth)

    # 測試端點
    test_endpoints = [
        "https://example.com/api/v1/admin/users/123",
        "https://example.com/api/v1/admin/settings",
        "https://example.com/api/v1/users/123",
    ]

    results = await tester.batch_test_endpoints(test_endpoints)

    # 輸出結果
    for endpoint, endpoint_results in results.items():
        print(f"\n=== {endpoint} ===")
        for result in endpoint_results:
            print(f"  {result.method}: Vulnerable={result.is_vulnerable}")
            if result.is_vulnerable:
                finding = tester.create_finding(result, "test_task_123")
                print(f"    Finding ID: {finding.finding_id}")
                print(f"    Severity: {finding.vulnerability.severity.value}")


if __name__ == "__main__":
    asyncio.run(main())
