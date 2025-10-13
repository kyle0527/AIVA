"""
Race Condition Tester - 競爭條件測試器

測試並發操作導致的業務邏輯缺陷:
- 庫存競爭
- 餘額競爭
- 限制繞過
- 重複消費
"""

from __future__ import annotations

import asyncio

import httpx

from services.aiva_common.enums import Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload
from services.aiva_common.utils import get_logger

from .finding_helper import create_bizlogic_finding

logger = get_logger(__name__)


class RaceConditionTester:
    """
    競爭條件測試器

    測試高並發場景下的業務邏輯缺陷
    """

    def __init__(self, client: httpx.AsyncClient | None = None):
        """初始化競爭條件測試器"""
        self.client = client or httpx.AsyncClient(timeout=10.0)

    async def test_inventory_race(
        self, purchase_api: str, product_id: str, task_id: str, scan_id: str, concurrent_purchases: int = 100
    ) -> list[FindingPayload]:
        """
        測試庫存競爭條件

        並發購買超過庫存的商品數量

        Args:
            purchase_api: 購買 API 端點
            product_id: 商品 ID
            task_id: 任務 ID
            scan_id: 掃描 ID
            concurrent_purchases: 並發購買次數

        Returns:
            list[FindingPayload]: 發現的漏洞
        """
        logger.info(
            f"Testing inventory race condition with {concurrent_purchases} concurrent purchases"
        )
        findings = []

        async def purchase():
            try:
                response = await self.client.post(
                    purchase_api, json={"product_id": product_id, "quantity": 1}
                )
                return response.json() if response.status_code == 200 else None
            except Exception as e:
                logger.debug(f"Purchase failed: {e}")
                return None

        # 執行並發購買
        tasks = [purchase() for _ in range(concurrent_purchases)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 計算成功購買次數
        successful_purchases = sum(
            1 for r in results
            if r and not isinstance(r, Exception) and isinstance(r, dict) and r.get("success")
        )

        logger.info(f"Successful purchases: {successful_purchases}/{concurrent_purchases}")

        # 如果成功購買次數超過庫存,表示存在競爭條件
        if successful_purchases > 10:  # 假設庫存為 10
            finding = create_bizlogic_finding(
                vuln_type=VulnerabilityType.RACE_CONDITION,
                severity=Severity.HIGH,
                target_url=purchase_api,
                method="POST",
                evidence_data={
                    "request": {"url": purchase_api, "method": "POST", "product_id": product_id},
                    "response": {"successful_purchases": successful_purchases},
                    "proof": f"並發購買 {concurrent_purchases} 次,成功 {successful_purchases} 次 (超過庫存)",
                },
                task_id=task_id,
                scan_id=scan_id,
            )
            findings.append(finding)
            logger.warning(
                f"Inventory race condition: {successful_purchases} purchases succeeded"
            )

        return findings

    async def test_balance_race(
        self, withdraw_api: str, amount: float, task_id: str, scan_id: str, concurrent_withdrawals: int = 50
    ) -> list[FindingPayload]:
        """
        測試餘額競爭條件

        並發提款超過帳戶餘額

        Args:
            withdraw_api: 提款 API 端點
            amount: 提款金額
            concurrent_withdrawals: 並發提款次數

        Returns:
            list[FindingPayload]: 發現的漏洞
        """
        logger.info(
            f"Testing balance race condition with {concurrent_withdrawals} concurrent withdrawals"
        )
        findings = []

        async def withdraw():
            try:
                response = await self.client.post(
                    withdraw_api, json={"amount": amount}
                )
                return response.json() if response.status_code == 200 else None
            except Exception as e:
                logger.debug(f"Withdrawal failed: {e}")
                return None

        tasks = [withdraw() for _ in range(concurrent_withdrawals)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_withdrawals = sum(
            1 for r in results
            if r and not isinstance(r, Exception) and isinstance(r, dict) and r.get("success")
        )

        logger.info(
            f"Successful withdrawals: {successful_withdrawals}/{concurrent_withdrawals}"
        )

        # 如果成功提款次數過多,可能存在餘額競爭
        if successful_withdrawals > 5:  # 假設帳戶只能提款 5 次
            finding = create_bizlogic_finding(
                vuln_type=VulnerabilityType.RACE_CONDITION,
                severity=Severity.CRITICAL,
                target_url=withdraw_api,
                method="POST",
                evidence_data={
                    "request": {"url": withdraw_api, "method": "POST", "amount": amount},
                    "response": {"successful_withdrawals": successful_withdrawals},
                    "proof": f"並發提款 {concurrent_withdrawals} 次,成功 {successful_withdrawals} 次 (超過預期)",
                },
                task_id=task_id,
                scan_id=scan_id,
            )
            findings.append(finding)
            logger.warning(
                f"Balance race condition: {successful_withdrawals} withdrawals succeeded"
            )

        return findings

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()
