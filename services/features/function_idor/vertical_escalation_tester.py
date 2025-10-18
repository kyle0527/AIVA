"""
Vertical Escalation Tester for IDOR Detection

Implements vertical privilege escalation testing according to OWASP standards.
Tests for privilege escalation by attempting to access higher-privilege functions
with lower-privilege credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import httpx
from pydantic import HttpUrl

from services.aiva_common.schemas import FunctionTaskPayload


class PrivilegeLevel(Enum):
    """權限級別枚舉 - 符合 OWASP 標準"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    ANONYMOUS = "anonymous"


@dataclass
class VerticalTestResult:
    """垂直權限提升測試結果"""
    vulnerable: bool = False
    test_status: int = 0
    privilege_level: PrivilegeLevel = PrivilegeLevel.ANONYMOUS
    evidence: str = ""
    error_message: str | None = None


class VerticalEscalationTester:
    """
    垂直權限提升測試器 - 實現 OWASP WSTG-ATHZ-03 標準
    
    根據 OWASP Web Security Testing Guide 4.5.3 - Testing for Privilege Escalation
    實現垂直權限提升測試，檢測低權限用戶是否能訪問高權限功能。
    """
    
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
    
    def _infer_required_privilege(self, url: str) -> PrivilegeLevel:
        """
        根據 URL 推斷所需權限級別
        
        Args:
            url: 目標 URL
            
        Returns:
            PrivilegeLevel: 推斷的權限級別
        """
        url_lower = url.lower()
        
        # 管理員級別路徑
        admin_patterns = [
            "/admin/", "/administrator/", "/management/", "/dashboard/",
            "/settings/", "/config/", "/api/admin/", "/control/"
        ]
        
        for pattern in admin_patterns:
            if pattern in url_lower:
                return PrivilegeLevel.ADMIN
        
        # 用戶級別路徑（默認）
        return PrivilegeLevel.USER
    
    async def test_vertical_escalation(
        self,
        url: str | HttpUrl,
        test_privilege: PrivilegeLevel,
        required_privilege: PrivilegeLevel | None = None,
        auth_headers: dict[str, str] | None = None,
        method: str = "GET",
    ) -> VerticalTestResult:
        """
        執行垂直權限提升測試
        
        Args:
            url: 測試目標 URL
            test_privilege: 測試用戶的權限級別
            required_privilege: 所需的權限級別
            auth_headers: 認證標頭
            method: HTTP 方法
            
        Returns:
            VerticalTestResult: 測試結果
        """
        try:
            # 推斷所需權限級別
            if required_privilege is None:
                required_privilege = self._infer_required_privilege(str(url))
            
            # 構建請求頭
            headers = {}
            if auth_headers:
                headers.update(auth_headers)
            
            # 執行 HTTP 請求
            response = await self.client.request(
                method=method,
                url=str(url),
                headers=headers,
                timeout=30.0,
                follow_redirects=True
            )
            
            # 分析響應判斷是否存在垂直權限提升漏洞
            # 如果低權限用戶能成功訪問高權限功能，則存在漏洞
            privilege_levels = {
                PrivilegeLevel.ANONYMOUS: 0,
                PrivilegeLevel.GUEST: 1,
                PrivilegeLevel.USER: 2,
                PrivilegeLevel.ADMIN: 3,
            }
            
            test_level = privilege_levels.get(test_privilege, 0)
            required_level = privilege_levels.get(required_privilege, 3)
            
            # 低權限用戶成功訪問高權限功能表示存在漏洞
            vulnerable = (
                test_level < required_level
                and response.status_code in [200, 201, 202]
                and len(response.content) > 50
                and b"unauthorized" not in response.content.lower()
                and b"forbidden" not in response.content.lower()
                and b"access denied" not in response.content.lower()
            )
            
            return VerticalTestResult(
                vulnerable=vulnerable,
                test_status=response.status_code,
                privilege_level=test_privilege,
                evidence=f"HTTP {response.status_code}, Content-Length: {len(response.content)}, "
                         f"Test Level: {test_privilege.value}, Required: {required_privilege.value}",
            )
            
        except TimeoutError:
            return VerticalTestResult(
                vulnerable=False,
                test_status=0,
                privilege_level=test_privilege,
                evidence="Request timeout",
                error_message="Request timeout during vertical escalation test"
            )
        except Exception as e:
            return VerticalTestResult(
                vulnerable=False,
                test_status=0,
                privilege_level=test_privilege,
                evidence=f"Error: {str(e)}",
                error_message=str(e)
            )
    
    async def test_privilege_levels(
        self,
        task: FunctionTaskPayload,
        test_levels: list[PrivilegeLevel] | None = None,
    ) -> list[VerticalTestResult]:
        """
        測試多個權限級別
        
        Args:
            task: 功能任務負載
            test_levels: 要測試的權限級別列表
            
        Returns:
            list[VerticalTestResult]: 測試結果列表
        """
        if test_levels is None:
            test_levels = [PrivilegeLevel.GUEST, PrivilegeLevel.USER]
        
        results = []
        required_privilege = self._infer_required_privilege(str(task.target.url))
        
        for test_privilege in test_levels:
            result = await self.test_vertical_escalation(
                url=task.target.url,
                test_privilege=test_privilege,
                required_privilege=required_privilege,
                method=task.target.method or "GET"
            )
            results.append(result)
        
        return results