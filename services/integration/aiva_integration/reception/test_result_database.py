"""
Test Result Database Interface

Abstract base class for test result database implementations.
"""



from abc import ABC, abstractmethod
from typing import Any

from services.aiva_common.schemas import FindingPayload


class TestResultDatabase(ABC):
    """測試結果資料庫介面"""

    @abstractmethod
    async def save_finding(self, finding: FindingPayload) -> None:
        """保存漏洞發現到資料庫"""
        ...

    @abstractmethod
    async def get_finding(self, finding_id: str) -> FindingPayload | None:
        """根據 ID 獲取漏洞發現"""
        ...

    @abstractmethod
    async def list_findings(
        self,
        scan_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FindingPayload]:
        """列出漏洞發現"""
        ...

    @abstractmethod
    async def count_findings(
        self,
        scan_id: str | None = None,
        severity: str | None = None,
    ) -> int:
        """統計漏洞發現數量"""
        ...

    @abstractmethod
    async def get_scan_summary(self, scan_id: str) -> dict[str, Any]:
        """獲取掃描摘要統計"""
        ...