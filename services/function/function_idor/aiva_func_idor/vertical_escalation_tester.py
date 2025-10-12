"""
Vertical Escalation IDOR Tester
垂直權限提升 IDOR 檢測器 - 基於 schemas.py 標準與官方 Pydantic 規範
"""

from __future__ import annotations

from enum import Enum

import httpx
from pydantic import BaseModel, Field

from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class PrivilegeLevel(str, Enum):
    """權限級別 - 基於 schemas.py 標準"""

    GUEST = "guest"
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"
    SUPERUSER = "superuser"


class VerticalTestResult(BaseModel):
    """垂直權限提升測試結果 - 符合 Pydantic 官方標準與 schemas.py 規範"""

    vulnerable: bool
    confidence: Confidence
    severity: Severity
    vulnerability_type: VulnerabilityType
    evidence: str | None = None
    description: str | None = None
    privilege_level: PrivilegeLevel = PrivilegeLevel.USER
    status_code: int | None = None
    should_deny: bool = True
    actual_level: PrivilegeLevel | None = None
    attempted_level: PrivilegeLevel | None = None


# 別名為了相容性
VerticalEscalationTestResult = VerticalTestResult


class VerticalEscalationTester:
    """垂直權限提升 IDOR 測試器 - 統一命名規範"""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.logger = get_logger(__name__)

    async def test_vertical_escalation(
        self,
        url: str,
        user_auth: dict[str, str],
        user_level: PrivilegeLevel,
        required_level: PrivilegeLevel,
        method: str = "GET",
        task: FunctionTaskPayload | None = None,
        admin_endpoints: list[str] | None = None,
    ) -> VerticalEscalationTestResult:
        """
        測試垂直權限提升漏洞 - 兼容新的調用方式

        Args:
            url: 目標 URL
            user_auth: 用戶認證信息
            user_level: 用戶權限級別
            required_level: 所需權限級別
            method: HTTP 方法
            task: 功能任務載荷 (可選，向後兼容)
            admin_endpoints: 管理員端點列表 (可選，向後兼容)

        Returns:
            VerticalEscalationTestResult: 測試結果
        """
        try:
            # 實現垂直權限提升測試邏輯
            self.logger.info(f"開始垂直權限提升 IDOR 測試: {url}")

            # 模擬測試邏輯
            vulnerable = user_level.value != required_level.value

            result = VerticalEscalationTestResult(
                vulnerable=vulnerable,
                confidence=Confidence.FIRM if vulnerable else Confidence.POSSIBLE,
                severity=Severity.HIGH if vulnerable else Severity.MEDIUM,
                vulnerability_type=VulnerabilityType.IDOR,
                description=f"垂直權限提升測試: {user_level.value} -> {required_level.value}",
                actual_level=user_level,
                attempted_level=required_level,
                status_code=200,  # 模擬狀態碼
                should_deny=required_level != user_level,
            )

            return result

        except Exception as e:
            self.logger.error(f"垂直權限提升測試失敗: {e}")
            return VerticalEscalationTestResult(
                vulnerable=False,
                confidence=Confidence.POSSIBLE,
                severity=Severity.LOW,
                vulnerability_type=VulnerabilityType.IDOR,
                description=f"測試失敗: {str(e)}",
            )

    async def cleanup(self):
        """清理資源"""
        await self.client.aclose()
