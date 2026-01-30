"""價格操控掃描器 - 檢測電商/支付系統的價格操控漏洞

這是實際的安全掃描能力，用於檢測應用程式中的價格篡改風險。
"""

import asyncio
import hashlib
import logging
from typing import Any, Optional
from decimal import Decimal
import json
import time

import httpx

from aiva_common.enums import VulnerabilityType, Severity, Confidence
from .finding_helper import create_bizlogic_finding

logger = logging.getLogger(__name__)


class PriceManipulationScanner:
    """價格操控掃描器 - 檢測價格篡改漏洞"""

    def __init__(self, target_url: str, authorization_token: str | None = None, user_role: str = "customer"):
        """
        初始化價格操控測試器
        
        Args:
            target_url: 目標 URL
            authorization_token: 授權令牌
            user_role: 用戶角色 (customer/merchant/admin)
        """
        self.target_url = target_url
        self.authorization_token = authorization_token
        self.user_role = user_role
        self.findings = []
        
        # 授權檢查
        if not authorization_token or len(authorization_token) < 32:
            logger.warning("No valid authorization token - running in limited mode")
    
    def _verify_actual_price_change(
        self, 
        response_data: dict[str, Any], 
        expected_price: float
    ) -> dict[str, Any]:
        """
        驗證價格是否真正被改變
        
        Args:
            response_data: API 響應數據
            expected_price: 預期的測試價格
            
        Returns:
            dict: 驗證結果
        """
        # 提取實際價格
        actual_price = response_data.get("total") or response_data.get("amount") or response_data.get("price")
        
        if actual_price is None:
            return {
                "vulnerable": False,
                "reason": "no_price_in_response",
                "details": "Response does not contain price information"
            }
        
        # 檢查是否真的改變了價格
        tolerance = 0.01  # 允許浮點誤差
        if abs(float(actual_price) - float(expected_price)) < tolerance:
            # 價格確實改變了，但需要進一步驗證是否合法
            return {
                "vulnerable": True,
                "actual_price": actual_price,
                "expected_price": expected_price,
                "price_changed": True
            }
        else:
            return {
                "vulnerable": False,
                "reason": "price_not_changed",
                "actual_price": actual_price,
                "expected_price": expected_price,
                "details": "Price was not changed to test value"
            }
    
    async def _verify_transaction_completed(
        self, 
        response_data: dict[str, Any],
        client: httpx.AsyncClient,
        transaction_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        驗證交易是否真正完成
        
        Args:
            response_data: API 響應數據
            client: HTTP client
            transaction_id: 交易 ID（如果有）
            
        Returns:
            dict: 驗證結果
        """
        # 檢查交易狀態
        status = response_data.get("status") or response_data.get("state")
        
        # 完成狀態
        completed_statuses = ["completed", "success", "paid", "confirmed", "approved"]
        # 待處理狀態
        pending_statuses = ["pending", "processing", "review", "awaiting_payment"]
        
        if status and status.lower() in completed_statuses:
            return {
                "completed": True,
                "status": status,
                "immediate": True
            }
        elif status and status.lower() in pending_statuses:
            # 待處理，需要再次查詢
            transaction_id = transaction_id or response_data.get("id") or response_data.get("transaction_id") or response_data.get("order_id")
            
            if transaction_id:
                # 等待 5 秒後再次查詢
                await asyncio.sleep(5)
                try:
                    check_response = await client.get(f"{self.target_url}/api/transaction/{transaction_id}")
                    if check_response.status_code == 200:
                        check_data = check_response.json()
                        final_status = check_data.get("status") or check_data.get("state")
                        
                        if final_status and final_status.lower() in completed_statuses:
                            return {
                                "completed": True,
                                "status": final_status,
                                "immediate": False,
                                "verified_after_delay": True
                            }
                except Exception as e:
                    logger.debug(f"Transaction status check error: {e}")
            
            return {
                "completed": False,
                "status": status,
                "reason": "transaction_pending"
            }
        else:
            return {
                "completed": False,
                "status": status or "unknown",
                "reason": "status_unclear"
            }
    
    def _verify_user_privilege(self, operation: str) -> dict[str, Any]:
        """
        驗證用戶是否應該有權限執行該操作
        
        Args:
            operation: 操作類型 (set_price, modify_price, etc.)
            
        Returns:
            dict: 驗證結果
        """
        # 權限矩陣
        permission_matrix = {
            "admin": ["set_any_price", "modify_price", "approve_refund", "set_zero_price"],
            "merchant": ["set_own_price", "request_refund"],
            "customer": ["purchase", "return"]
        }
        
        allowed_operations = permission_matrix.get(self.user_role, [])
        
        if operation in allowed_operations:
            return {
                "has_privilege": True,
                "user_role": self.user_role,
                "operation": operation,
                "reason": "legitimate_privilege"
            }
        else:
            return {
                "has_privilege": False,
                "user_role": self.user_role,
                "operation": operation,
                "reason": "insufficient_privilege"
            }
    
    def _detect_business_limits(
        self, 
        response_data: dict[str, Any], 
        test_value: float
    ) -> dict[str, Any]:
        """
        檢測業務限制是否生效
        
        Args:
            response_data: API 響應數據
            test_value: 測試值
            
        Returns:
            dict: 檢測結果
        """
        # 檢查實際值是否被限制
        actual_value = response_data.get("quantity") or response_data.get("amount") or response_data.get("total")
        
        if actual_value is None:
            return {
                "limits_effective": None,
                "reason": "no_value_in_response"
            }
        
        # 如果測試的是極大值（如 999999），但實際值被限制在合理範圍（如 100）
        if test_value > 1000 and actual_value < 100:
            return {
                "limits_effective": True,
                "test_value": test_value,
                "actual_value": actual_value,
                "reason": "value_capped"
            }
        
        # 如果實際值接近測試值，說明限制失效
        if abs(float(actual_value) - float(test_value)) / max(abs(float(test_value)), 1) < 0.1:
            return {
                "limits_effective": False,
                "test_value": test_value,
                "actual_value": actual_value,
                "reason": "limits_bypassed"
            }
        
        return {
            "limits_effective": True,
            "test_value": test_value,
            "actual_value": actual_value
        }

    async def test_negative_price(self, endpoint: str, price_param: str = "price") -> list[dict[str, Any]]:
        """
        測試負數價格 - 增強版驗證
        
        Args:
            endpoint: API 端點
            price_param: 價格參數名稱
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        negative_prices = [-1, -100, -999.99, -0.01]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for price in negative_prices:
                try:
                    response = await client.post(
                        f"{self.target_url}{endpoint}",
                        json={price_param: price, "quantity": 1},
                        headers={"Authorization": f"Bearer {self.authorization_token}"} if self.authorization_token else {}
                    )
                    
                    if response.status_code in [200, 201]:
                        response_data = response.json()
                        
                        # ✅ 強化驗證 1: 檢查價格是否真正改變
                        price_verification = self._verify_actual_price_change(
                            response_data, price
                        )
                        
                        if not price_verification.get("vulnerable"):
                            logger.debug(f"Negative price {price} not accepted: {price_verification.get('reason')}")
                            continue
                        
                        # ✅ 強化驗證 2: 檢查是否為合法退款功能
                        transaction_type = response_data.get("type") or response_data.get("transaction_type")
                        if transaction_type and transaction_type.lower() in ["refund", "return", "credit"]:
                            logger.info(f"Negative price {price} is legitimate refund/credit feature")
                            continue
                        
                        # ✅ 強化驗證 3: 檢查用戶權限
                        privilege_check = self._verify_user_privilege("set_negative_price")
                        if privilege_check.get("has_privilege"):
                            logger.info(f"User {self.user_role} has legitimate privilege for negative price")
                            continue
                        
                        # ✅ 強化驗證 4: 驗證交易是否完成
                        transaction_verification = await self._verify_transaction_completed(
                            response_data, client
                        )
                        
                        if not transaction_verification.get("completed"):
                            logger.debug(f"Negative price transaction not completed: {transaction_verification.get('reason')}")
                            continue
                        
                        # ✅ 確認漏洞
                        findings.append({
                            "type": "negative_price_accepted",
                            "severity": Severity.CRITICAL,
                            "confidence": Confidence.CERTAIN,
                            "price": price,
                            "actual_price": price_verification.get("actual_price"),
                            "transaction_status": transaction_verification.get("status"),
                            "user_role": self.user_role,
                            "response_code": response.status_code,
                            "evidence": json.dumps(response_data, ensure_ascii=False)[:300],
                            "verification_details": {
                                "price_changed": True,
                                "transaction_completed": True,
                                "not_refund_feature": True,
                                "unauthorized_user": True
                            }
                        })
                        logger.warning(f"⚠️ CONFIRMED: Negative price vulnerability: {price}")
                
                except httpx.HTTPError as e:
                    logger.debug(f"Negative price test HTTP error: {e}")
                except Exception as e:
                    logger.exception(f"Negative price test unexpected error: {e}")
        
        return findings

    async def test_zero_price(self, endpoint: str, price_param: str = "price") -> list[dict[str, Any]]:
        """
        測試零價格 - 增強版驗證
        
        Args:
            endpoint: API 端點
            price_param: 價格參數名稱
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{self.target_url}{endpoint}",
                    json={price_param: 0, "quantity": 1},
                    headers={"Authorization": f"Bearer {self.authorization_token}"} if self.authorization_token else {}
                )
                
                if response.status_code in [200, 201]:
                    response_data = response.json()
                    
                    # ✅ 強化驗證 1: 確認實際價格為零
                    price_verification = self._verify_actual_price_change(response_data, 0)
                    
                    if not price_verification.get("vulnerable"):
                        logger.debug(f"Zero price not accepted: {price_verification.get('reason')}")
                        return findings
                    
                    # ✅ 強化驗證 2: 檢查是否為促銷/免費商品
                    promotion_type = response_data.get("promotion") or response_data.get("discount_type")
                    is_free_item = response_data.get("is_free") or response_data.get("price_type") == "free"
                    
                    if promotion_type or is_free_item:
                        logger.info("Zero price is legitimate promotion/free item")
                        return findings
                    
                    # ✅ 強化驗證 3: 檢查 Admin 權限
                    privilege_check = self._verify_user_privilege("set_zero_price")
                    if privilege_check.get("has_privilege"):
                        logger.info(f"User {self.user_role} has legitimate privilege for zero price")
                        return findings
                    
                    # ✅ 強化驗證 4: 驗證交易完成
                    transaction_verification = await self._verify_transaction_completed(response_data, client)
                    
                    if not transaction_verification.get("completed"):
                        logger.debug(f"Zero price transaction not completed: {transaction_verification.get('reason')}")
                        return findings
                    
                    # ✅ 確認漏洞
                    findings.append({
                        "type": "zero_price_accepted",
                        "severity": Severity.HIGH,
                        "confidence": Confidence.CERTAIN,
                        "actual_price": 0,
                        "transaction_status": transaction_verification.get("status"),
                        "user_role": self.user_role,
                        "response_code": response.status_code,
                        "evidence": json.dumps(response_data, ensure_ascii=False)[:300],
                        "verification_details": {
                            "price_is_zero": True,
                            "not_promotion": True,
                            "transaction_completed": True,
                            "unauthorized_user": True
                        }
                    })
                    logger.warning("⚠️ CONFIRMED: Zero price vulnerability")
            
            except httpx.HTTPError as e:
                logger.debug(f"Zero price test HTTP error: {e}")
            except Exception as e:
                logger.exception(f"Zero price test unexpected error: {e}")
        
        return findings

    async def test_price_tampering(self, endpoint: str) -> list[dict[str, Any]]:
        """
        測試價格篡改 - 增強版驗證
        
        測試在請求中修改價格參數是否能改變最終價格
        
        Args:
            endpoint: API 端點
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        # 測試場景：發送與伺服器不同的價格
        test_cases = [
            {"original_price": 100, "tampered_price": 1},
            {"original_price": 999, "tampered_price": 9.99},
            {"original_price": 50, "tampered_price": 0.01},
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for case in test_cases:
                try:
                    # 嘗試發送被篡改的價格
                    response = await client.post(
                        f"{self.target_url}{endpoint}",
                        json={
                            "product_id": "test_product",
                            "price": case["tampered_price"],
                            "quantity": 1
                        },
                        headers={"Authorization": f"Bearer {self.authorization_token}"} if self.authorization_token else {}
                    )
                    
                    if response.status_code in [200, 201]:
                        response_data = response.json()
                        
                        # ✅ 強化驗證 1: 確認實際價格
                        price_verification = self._verify_actual_price_change(
                            response_data, 
                            case["tampered_price"]
                        )
                        
                        if not price_verification.get("vulnerable"):
                            logger.debug(f"Price tampering failed: {price_verification.get('reason')}")
                            continue
                        
                        # ✅ 強化驗證 2: 檢查用戶權限
                        privilege_check = self._verify_user_privilege("modify_price")
                        if privilege_check.get("has_privilege"):
                            logger.info(f"User {self.user_role} has legitimate privilege to modify price")
                            continue
                        
                        # ✅ 強化驗證 3: 驗證交易完成
                        transaction_verification = await self._verify_transaction_completed(response_data, client)
                        
                        if not transaction_verification.get("completed"):
                            logger.debug(f"Tampered price transaction not completed: {transaction_verification.get('reason')}")
                            continue
                        
                        # ✅ 確認漏洞
                        actual_total = price_verification.get("actual_price")
                        savings = case["original_price"] - actual_total
                        
                        findings.append({
                            "type": "price_tampering",
                            "severity": Severity.CRITICAL,
                            "confidence": Confidence.CERTAIN,
                            "original_price": case["original_price"],
                            "tampered_price": case["tampered_price"],
                            "accepted_price": actual_total,
                            "savings": savings,
                            "savings_percentage": (savings / case["original_price"]) * 100,
                            "transaction_status": transaction_verification.get("status"),
                            "user_role": self.user_role,
                            "evidence": json.dumps(response_data, ensure_ascii=False)[:300],
                            "verification_details": {
                                "price_tampered": True,
                                "transaction_completed": True,
                                "unauthorized_user": True
                            }
                        })
                        logger.warning(f"⚠️ CONFIRMED: Price tampering vulnerability (${case['original_price']} → ${actual_total})")
                
                except httpx.HTTPError as e:
                    logger.debug(f"Price tampering test HTTP error: {e}")
                except Exception as e:
                    logger.exception(f"Price tampering test unexpected error: {e}")
        
        return findings

    async def test_overflow_price(self, endpoint: str) -> list[dict[str, Any]]:
        """
        測試價格溢出
        
        Args:
            endpoint: API 端點
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        overflow_prices = [
            999999999999999,
            float('inf'),
            1e308,
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for price in overflow_prices:
                try:
                    response = await client.post(
                        f"{self.target_url}{endpoint}",
                        json={"price": price, "quantity": 1}
                    )
                    
                    if response.status_code == 500:
                        findings.append({
                            "type": "price_overflow",
                            "severity": Severity.MEDIUM,
                            "confidence": Confidence.POSSIBLE,
                            "price": price,
                            "response_code": 500,
                            "evidence": "Server error indicates potential overflow"
                        })
                        logger.warning(f"⚠️ Price overflow detected: {price}")
                
                except Exception as e:
                    logger.debug(f"Overflow test error: {e}")
        
        return findings

    async def run_all_tests(self, endpoint: str = "/api/checkout") -> list[dict[str, Any]]:
        """
        運行所有價格操控測試
        
        Args:
            endpoint: API 端點
            
        Returns:
            list: 所有測試結果
        """
        logger.info(f"Starting price manipulation tests on {self.target_url}{endpoint}")
        
        all_findings = []
        
        # 並發執行所有測試
        results = await asyncio.gather(
            self.test_negative_price(endpoint),
            self.test_zero_price(endpoint),
            self.test_price_tampering(endpoint),
            self.test_overflow_price(endpoint),
            return_exceptions=True
        )
        
        for result in results:
            if isinstance(result, list):
                all_findings.extend(result)
        
        logger.info(f"✅ Price manipulation tests completed: {len(all_findings)} findings")
        
        return all_findings

