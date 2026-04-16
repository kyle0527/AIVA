"""工作流程繞過掃描器 - 檢測多步驟業務流程的安全性

這是實際的安全掃描能力，用於驗證應用程式是否可以跳過關鍵業務步驟。
"""

import asyncio
import logging
from typing import Any

from aiva_common.enums import Confidence, Severity
import httpx

logger = logging.getLogger(__name__)


class WorkflowBypassScanner:
    """工作流程繞過掃描器 - 檢測業務流程繞過漏洞"""

    def __init__(self, target_url: str, authorization_token: str | None = None):
        """
        初始化工作流程繞過測試器
        
        Args:
            target_url: 目標 URL
            authorization_token: 授權令牌
        """
        self.target_url = target_url
        self.authorization_token = authorization_token
        
        if not authorization_token or len(authorization_token) < 32:
            logger.warning("No valid authorization token - running in limited mode")

    async def test_step_skipping(
        self,
        workflow_steps: list[str],
        skip_step_index: int
    ) -> list[dict[str, Any]]:
        """
        測試跳過工作流程步驟
        
        測試場景：嘗試跳過某個必要步驟，直接訪問後續步驟
        
        Args:
            workflow_steps: 工作流程步驟列表 (URL paths)
            skip_step_index: 要跳過的步驟索引
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            try:
                # 嘗試直接訪問跳過步驟後的端點
                for i in range(skip_step_index + 1, len(workflow_steps)):
                    target_step = workflow_steps[i]
                    
                    response = await client.get(f"{self.target_url}{target_step}")
                    
                    # 如果能成功訪問（200/302），可能存在步驟跳過漏洞
                    if response.status_code in [200, 302]:
                        findings.append({
                            "type": "workflow_step_skipping",
                            "severity": Severity.HIGH,
                            "confidence": Confidence.CERTAIN,
                            "skipped_step": workflow_steps[skip_step_index],
                            "accessed_step": target_step,
                            "response_code": response.status_code,
                            "evidence": f"Successfully accessed step {i} without completing step {skip_step_index}"
                        })
                        logger.warning(f"⚠️ Workflow bypass: skipped step {skip_step_index}, accessed step {i}")
            
            except Exception as e:
                logger.error(f"Step skipping test error: {e}")
        
        return findings

    async def test_direct_checkout(self, checkout_endpoint: str = "/checkout") -> list[dict[str, Any]]:
        """
        測試直接結帳繞過
        
        測試場景：跳過購物車，直接訪問結帳頁面
        
        Args:
            checkout_endpoint: 結帳端點
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # 1. 不添加商品到購物車，直接訪問結帳頁面
                response = await client.get(f"{self.target_url}{checkout_endpoint}")
                
                if response.status_code == 200:
                    # 檢查是否允許空購物車結帳
                    if "checkout" in response.text.lower() or "payment" in response.text.lower():
                        findings.append({
                            "type": "direct_checkout_bypass",
                            "severity": Severity.MEDIUM,
                            "confidence": Confidence.POSSIBLE,
                            "endpoint": checkout_endpoint,
                            "response_code": response.status_code,
                            "evidence": "Checkout page accessible without adding items to cart"
                        })
                        logger.warning("⚠️ Direct checkout bypass: accessed checkout without cart")
                
                # 2. 嘗試直接 POST 結帳請求
                post_response = await client.post(
                    f"{self.target_url}{checkout_endpoint}",
                    json={"items": [], "total": 0}
                )
                
                if post_response.status_code in [200, 201]:
                    findings.append({
                        "type": "empty_cart_checkout",
                        "severity": Severity.MEDIUM,
                        "confidence": Confidence.CERTAIN,
                        "endpoint": checkout_endpoint,
                        "response_code": post_response.status_code,
                        "evidence": "Empty cart checkout succeeded"
                    })
                    logger.warning("⚠️ Empty cart checkout succeeded")
            
            except Exception as e:
                logger.error(f"Direct checkout test error: {e}")
        
        return findings

    async def test_payment_bypass(
        self,
        order_endpoint: str = "/api/order",
        payment_endpoint: str = "/api/payment"
    ) -> list[dict[str, Any]]:
        """
        測試支付繞過
        
        測試場景：跳過支付步驟，直接創建訂單
        
        Args:
            order_endpoint: 訂單創建端點
            payment_endpoint: 支付端點
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # 嘗試不經過支付，直接創建訂單
                order_data = {
                    "items": [{"id": "test_product", "quantity": 1}],
                    "total": 100.0,
                    "payment_status": "paid"  # 嘗試直接設置為已支付
                }
                
                response = await client.post(
                    f"{self.target_url}{order_endpoint}",
                    json=order_data
                )
                
                if response.status_code in [200, 201]:
                    response_data = response.json()
                    
                    # 檢查訂單是否被創建且標記為已支付
                    if response_data.get("payment_status") == "paid":
                        findings.append({
                            "type": "payment_bypass",
                            "severity": Severity.CRITICAL,
                            "confidence": Confidence.CERTAIN,
                            "order_id": response_data.get("order_id"),
                            "evidence": "Order created with 'paid' status without actual payment"
                        })
                        logger.warning("⚠️ Payment bypass: order marked as paid without payment")
            
            except Exception as e:
                logger.error(f"Payment bypass test error: {e}")
        
        return findings

    async def test_verification_bypass(
        self,
        register_endpoint: str = "/api/register",
        verify_endpoint: str = "/api/verify"
    ) -> list[dict[str, Any]]:
        """
        測試驗證繞過
        
        測試場景：跳過郵箱/手機驗證，直接使用帳號
        
        Args:
            register_endpoint: 註冊端點
            verify_endpoint: 驗證端點
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # 1. 註冊帳號
                register_data = {
                    "username": f"test_user_{asyncio.get_event_loop().time()}",
                    "email": "test@example.com",
                    "password": "Test123!@#"
                }
                
                register_response = await client.post(
                    f"{self.target_url}{register_endpoint}",
                    json=register_data
                )
                
                if register_response.status_code in [200, 201]:
                    response_data = register_response.json()
                    
                    # 2. 不經過驗證，嘗試直接登入或使用功能
                    # 檢查是否需要驗證
                    if response_data.get("verified") is True:
                        findings.append({
                            "type": "verification_bypass",
                            "severity": Severity.MEDIUM,
                            "confidence": Confidence.CERTAIN,
                            "username": register_data["username"],
                            "evidence": "Account created with verified=True without verification step"
                        })
                        logger.warning("⚠️ Verification bypass: account pre-verified")
            
            except Exception as e:
                logger.error(f"Verification bypass test error: {e}")
        
        return findings

    async def test_admin_access_bypass(
        self,
        admin_endpoints: list[str] = []
    ) -> list[dict[str, Any]]:
        """
        測試管理員訪問繞過
        
        測試場景：未授權訪問管理員功能
        
        Args:
            admin_endpoints: 管理員端點列表
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        if not admin_endpoints:
            admin_endpoints = [
                "/admin",
                "/admin/users",
                "/admin/settings",
                "/api/admin/users",
                "/administrator"
            ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint in admin_endpoints:
                try:
                    response = await client.get(f"{self.target_url}{endpoint}")
                    
                    # 檢查是否能訪問管理員頁面
                    if response.status_code == 200:
                        if any(word in response.text.lower() for word in ["admin", "dashboard", "users", "settings"]):
                            findings.append({
                                "type": "admin_access_bypass",
                                "severity": Severity.CRITICAL,
                                "confidence": Confidence.CERTAIN,
                                "endpoint": endpoint,
                                "response_code": response.status_code,
                                "evidence": f"Admin page accessible without authentication: {endpoint}"
                            })
                            logger.warning(f"⚠️ Admin access bypass: {endpoint} accessible")
                
                except Exception as e:
                    logger.debug(f"Admin access test error for {endpoint}: {e}")
        
        return findings

    async def run_all_tests(self) -> list[dict[str, Any]]:
        """
        運行所有工作流程繞過測試
            
        Returns:
            list: 所有測試結果
        """
        logger.info(f"Starting workflow bypass tests on {self.target_url}")
        
        all_findings = []
        
        # 並發執行測試
        results = await asyncio.gather(
            self.test_direct_checkout(),
            self.test_payment_bypass(),
            self.test_admin_access_bypass(),
            return_exceptions=True
        )
        
        for result in results:
            if isinstance(result, list):
                all_findings.extend(result)
        
        logger.info(f"✅ Workflow bypass tests completed: {len(all_findings)} findings")
        
        return all_findings

