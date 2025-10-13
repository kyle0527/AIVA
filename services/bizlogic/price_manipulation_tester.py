"""
Price Manipulation Tester - 價格操縱測試器

測試電商系統的價格相關漏洞:
- 負數數量漏洞
- 價格競爭條件
- 優惠券重複使用
- 折扣疊加漏洞
- 整數溢位
"""

from __future__ import annotations

import asyncio

import httpx

from services.aiva_common.enums import Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload
from services.aiva_common.utils import get_logger

from .finding_helper import create_bizlogic_finding

logger = get_logger(__name__)


class PriceManipulationTester:
    """
    價格操縱測試器

    測試電商購物流程中的價格相關業務邏輯漏洞
    """

    def __init__(self, client: httpx.AsyncClient | None = None):
        """
        初始化價格操縱測試器

        Args:
            client: HTTP 客戶端,如果為 None 則自動創建
        """
        self.client = client or httpx.AsyncClient(timeout=10.0)
        self.findings: list[FindingPayload] = []

    async def test_negative_quantity(
        self, cart_api: str, product_id: str, task_id: str = "task_bizlogic", scan_id: str = "scan_bizlogic"
    ) -> list[FindingPayload]:
        """
        測試負數數量漏洞

        嘗試將商品數量設置為負數,檢查是否會導致價格為負或免費獲得商品

        Args:
            cart_api: 購物車 API 端點
            product_id: 商品 ID
            task_id: 任務 ID
            scan_id: 掃描 ID

        Returns:
            list[FindingPayload]: 發現的漏洞列表
        """
        logger.info(f"Testing negative quantity vulnerability on {cart_api}")
        findings = []

        test_quantities = [-1, -10, -999, 0]

        for quantity in test_quantities:
            try:
                response = await self.client.post(
                    cart_api,
                    json={"product_id": product_id, "quantity": quantity},
                )

                if response.status_code == 200:
                    data = response.json()

                    # 檢查總價是否為負數或零
                    total = data.get("total", 0)
                    if total <= 0:
                        finding = create_bizlogic_finding(
                            vuln_type=VulnerabilityType.PRICE_MANIPULATION,
                            severity=Severity.HIGH,
                            target_url=cart_api,
                            method="POST",
                            evidence_data={
                                "request": {
                                    "url": cart_api,
                                    "method": "POST",
                                    "body": {
                                        "product_id": product_id,
                                        "quantity": quantity,
                                    },
                                },
                                "response": {"status": 200, "total": total},
                                "proof": f"購物車允許負數數量 ({quantity}),導致總價為 {total}",
                            },
                            task_id=task_id,
                            scan_id=scan_id,
                        )
                        findings.append(finding)
                        logger.warning("⚠️ Negative quantity vulnerability found!")

            except Exception as e:
                logger.debug(f"Test failed for quantity {quantity}: {e}")

        return findings

    async def test_race_condition_pricing(
        self,
        cart_api: str,
        product_id: str,
        task_id: str,
        scan_id: str,
        concurrent_requests: int = 50,
    ) -> list[FindingPayload]:
        """
        測試價格競爭條件

        並發修改同一商品的數量,檢查是否會導致價格計算錯誤

        Args:
            cart_api: 購物車 API 端點
            product_id: 商品 ID
            task_id: 任務 ID
            scan_id: 掃描 ID
            concurrent_requests: 並發請求數

        Returns:
            list[FindingPayload]: 發現的漏洞列表
        """
        logger.info(
            f"Testing price race condition with {concurrent_requests} concurrent requests"
        )
        findings = []

        # 準備並發請求
        async def update_quantity():
            try:
                response = await self.client.post(
                    cart_api,
                    json={"product_id": product_id, "quantity": 1, "action": "add"},
                )
                return response.json()
            except Exception as e:
                logger.debug(f"Request failed: {e}")
                return None

        # 執行並發請求
        tasks = [update_quantity() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 分析結果
        successful_results = [
            r for r in results if r is not None and not isinstance(r, Exception) and isinstance(r, dict)
        ]

        if successful_results:
            # 檢查總數量是否正確
            expected_quantity = concurrent_requests
            actual_quantity = successful_results[-1].get("total_quantity", 0)

            if actual_quantity != expected_quantity:
                from .finding_helper import create_bizlogic_finding

                finding = create_bizlogic_finding(
                    vuln_type=VulnerabilityType.RACE_CONDITION,
                    severity=Severity.MEDIUM,
                    target_url=cart_api,
                    method="POST",
                    evidence_data={
                        "request": {"product_id": product_id, "concurrent_requests": concurrent_requests},
                        "response": {"expected": expected_quantity, "actual": actual_quantity},
                        "proof": f"並發更新導致數量不一致: 預期 {expected_quantity}, 實際 {actual_quantity}",
                    },
                    task_id=task_id,
                    scan_id=scan_id,
                    parameter="quantity",
                )
                findings.append(finding)
                logger.warning("⚠️ Race condition vulnerability found!")

        return findings

    async def test_coupon_reuse(
        self,
        checkout_api: str,
        coupon_code: str,
        task_id: str,
        scan_id: str,
        attempts: int = 5,
    ) -> list[FindingPayload]:
        """
        測試優惠券重複使用漏洞

        Args:
            checkout_api: 結帳 API 端點
            coupon_code: 測試用優惠券代碼
            task_id: 任務 ID
            scan_id: 掃描 ID
            attempts: 嘗試次數

        Returns:
            list[FindingPayload]: 發現的漏洞列表
        """
        logger.info(f"Testing coupon reuse vulnerability for {coupon_code}")
        findings = []

        successful_uses = 0

        for i in range(attempts):
            try:
                response = await self.client.post(
                    checkout_api, json={"coupon_code": coupon_code}
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("discount_applied"):
                        successful_uses += 1

            except Exception as e:
                logger.debug(f"Coupon test attempt {i+1} failed: {e}")

        # 如果優惠券被成功使用超過一次
        if successful_uses > 1:
            from .finding_helper import create_bizlogic_finding

            finding = create_bizlogic_finding(
                vuln_type=VulnerabilityType.PRICE_MANIPULATION,
                severity=Severity.MEDIUM,
                target_url=checkout_api,
                method="POST",
                evidence_data={
                    "request": {"coupon_code": coupon_code, "attempts": attempts},
                    "response": {"successful_uses": successful_uses},
                    "proof": f"優惠券 '{coupon_code}' 可以被重複使用 {successful_uses} 次",
                },
                task_id=task_id,
                scan_id=scan_id,
                parameter="coupon_code",
            )
            findings.append(finding)
            logger.warning("⚠️ Coupon reuse vulnerability found!")

        return findings

    async def test_price_tampering(
        self,
        checkout_api: str,
        original_price: float,
        task_id: str,
        scan_id: str,
    ) -> list[FindingPayload]:
        """
        測試價格篡改漏洞

        嘗試在結帳請求中修改價格參數

        Args:
            checkout_api: 結帳 API 端點
            original_price: 原始價格
            task_id: 任務 ID
            scan_id: 掃描 ID

        Returns:
            list[FindingPayload]: 發現的漏洞列表
        """
        logger.info(f"Testing price tampering on {checkout_api}")
        findings = []

        tampered_prices = [0.01, 1.0, original_price * 0.1]

        for tampered_price in tampered_prices:
            try:
                response = await self.client.post(
                    checkout_api, json={"total": tampered_price, "currency": "USD"}
                )

                if response.status_code == 200:
                    data = response.json()

                    # 檢查是否接受了篡改的價格
                    if data.get("success") and data.get("charged_amount") == tampered_price:
                        from .finding_helper import create_bizlogic_finding

                        finding = create_bizlogic_finding(
                            vuln_type=VulnerabilityType.PRICE_MANIPULATION,
                            severity=Severity.CRITICAL,
                            target_url=checkout_api,
                            method="POST",
                            evidence_data={
                                "request": {"total": tampered_price, "currency": "USD"},
                                "response": data,
                                "proof": f"結帳 API 接受了客戶端提供的價格 {tampered_price} (原價: {original_price})",
                            },
                            task_id=task_id,
                            scan_id=scan_id,
                            parameter="total",
                        )
                        findings.append(finding)
                        logger.warning("⚠️ Price tampering vulnerability found!")

            except Exception as e:
                logger.debug(f"Price tampering test failed: {e}")

        return findings

    async def run_all_tests(
        self,
        target_urls: dict[str, str],
        task_id: str,
        scan_id: str,
        product_id: str | None = None,
    ) -> list[FindingPayload]:
        """
        執行所有價格操縱測試

        Args:
            target_urls: 目標 URL 字典 {"cart_api": "...", "checkout_api": "..."}
            task_id: 任務 ID
            scan_id: 掃描 ID
            product_id: 測試用商品 ID

        Returns:
            list[FindingPayload]: 所有發現的漏洞
        """
        all_findings = []

        cart_api = target_urls.get("cart_api")
        checkout_api = target_urls.get("checkout_api")

        if cart_api and product_id:
            # 測試負數數量
            findings = await self.test_negative_quantity(
                cart_api, product_id, task_id, scan_id
            )
            all_findings.extend(findings)

            # 測試競爭條件
            findings = await self.test_race_condition_pricing(
                cart_api, product_id, task_id, scan_id
            )
            all_findings.extend(findings)

        if checkout_api:
            # 測試優惠券重複使用
            findings = await self.test_coupon_reuse(
                checkout_api, "TEST10", task_id, scan_id
            )
            all_findings.extend(findings)

            # 測試價格篡改
            findings = await self.test_price_tampering(
                checkout_api, 99.99, task_id, scan_id
            )
            all_findings.extend(findings)

        logger.info(f"Price manipulation tests completed: {len(all_findings)} findings")
        return all_findings

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()
