"""競態條件掃描器 - 檢測並發請求中的競態條件漏洞

這是實際的安全掃描能力，用於檢測應用程式在並發請求下的安全問題。
"""

import asyncio
import logging
from typing import Any
from datetime import datetime

import httpx

from aiva_common.enums import VulnerabilityType, Severity, Confidence
from .finding_helper import create_bizlogic_finding

logger = logging.getLogger(__name__)


class RaceConditionScanner:
    """競態條件掃描器 - 檢測並發請求漏洞"""

    def __init__(self, target_url: str, authorization_token: str | None = None):
        """
        初始化競態條件測試器
        
        Args:
            target_url: 目標 URL
            authorization_token: 授權令牌
        """
        self.target_url = target_url
        self.authorization_token = authorization_token
        
        if not authorization_token or len(authorization_token) < 32:
            logger.warning("No valid authorization token - running in limited mode")

    async def test_concurrent_requests(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict | None = None,
        concurrent_count: int = 10
    ) -> list[dict[str, Any]]:
        """
        測試並發請求
        
        Args:
            endpoint: API 端點
            method: HTTP 方法
            payload: 請求數據
            concurrent_count: 並發數量
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 創建並發請求
            tasks = []
            start_time = datetime.now()
            
            for _ in range(concurrent_count):
                if method.upper() == "POST":
                    task = client.post(f"{self.target_url}{endpoint}", json=payload or {})
                elif method.upper() == "GET":
                    task = client.get(f"{self.target_url}{endpoint}")
                else:
                    task = client.request(method, f"{self.target_url}{endpoint}", json=payload)
                
                tasks.append(task)
            
            try:
                # 同時發送所有請求
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # 分析響應
                success_count = sum(
                    1 for r in responses 
                    if isinstance(r, httpx.Response) and r.status_code in [200, 201]
                )
                
                # 檢查是否存在競態條件
                if success_count > 1:
                    # 如果多個請求都成功，可能存在競態條件
                    findings.append({
                        "type": "race_condition_detected",
                        "severity": Severity.HIGH,
                        "confidence": Confidence.POSSIBLE,
                        "concurrent_requests": concurrent_count,
                        "successful_requests": success_count,
                        "duration": duration,
                        "evidence": f"{success_count}/{concurrent_count} concurrent requests succeeded"
                    })
                    logger.warning(f"⚠️ Potential race condition: {success_count}/{concurrent_count} succeeded")
            
            except Exception as e:
                logger.error(f"Concurrent request test error: {e}")
        
        return findings

    async def test_balance_manipulation(
        self,
        withdrawal_endpoint: str,
        balance_endpoint: str,
        withdrawal_amount: float = 100.0
    ) -> list[dict[str, Any]]:
        """
        測試餘額操控競態條件
        
        測試場景：同時發送多個取款請求，檢查是否能超額提取
        
        Args:
            withdrawal_endpoint: 取款 API 端點
            balance_endpoint: 餘額查詢端點
            withdrawal_amount: 單次取款金額
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. 獲取初始餘額
            try:
                balance_response = await client.get(f"{self.target_url}{balance_endpoint}")
                if balance_response.status_code == 200:
                    initial_balance = balance_response.json().get("balance", 0)
                else:
                    logger.warning("Failed to get initial balance")
                    return findings
            except Exception as e:
                logger.error(f"Balance check error: {e}")
                return findings
            
            # 2. 同時發送多個取款請求
            concurrent_count = 5
            tasks = [
                client.post(
                    f"{self.target_url}{withdrawal_endpoint}",
                    json={"amount": withdrawal_amount}
                )
                for _ in range(concurrent_count)
            ]
            
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(
                    1 for r in responses 
                    if isinstance(r, httpx.Response) and r.status_code in [200, 201]
                )
                
                # 3. 檢查最終餘額
                final_balance_response = await client.get(f"{self.target_url}{balance_endpoint}")
                if final_balance_response.status_code == 200:
                    final_balance = final_balance_response.json().get("balance", 0)
                    
                    expected_balance = initial_balance - (withdrawal_amount * success_count)
                    
                    # 檢查是否超額提取
                    if final_balance < expected_balance - 0.01:
                        findings.append({
                            "type": "balance_race_condition",
                            "severity": Severity.CRITICAL,
                            "confidence": Confidence.CERTAIN,
                            "initial_balance": initial_balance,
                            "final_balance": final_balance,
                            "expected_balance": expected_balance,
                            "over_withdrawal": expected_balance - final_balance,
                            "evidence": f"Withdrew {success_count * withdrawal_amount} but balance only reduced by {initial_balance - final_balance}"
                        })
                        logger.warning(f"⚠️ Balance race condition detected: over-withdrawal of {expected_balance - final_balance}")
            
            except Exception as e:
                logger.error(f"Balance manipulation test error: {e}")
        
        return findings

    async def test_coupon_reuse(self, coupon_endpoint: str, coupon_code: str = "TEST2024") -> list[dict[str, Any]]:
        """
        測試優惠券重複使用競態條件
        
        Args:
            coupon_endpoint: 優惠券使用端點
            coupon_code: 優惠券代碼
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 同時發送多個使用優惠券的請求
            concurrent_count = 5
            tasks = [
                client.post(
                    f"{self.target_url}{coupon_endpoint}",
                    json={"coupon_code": coupon_code}
                )
                for _ in range(concurrent_count)
            ]
            
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(
                    1 for r in responses 
                    if isinstance(r, httpx.Response) and r.status_code in [200, 201]
                )
                
                # 如果多個請求都成功使用了同一個優惠券
                if success_count > 1:
                    findings.append({
                        "type": "coupon_reuse_race_condition",
                        "severity": Severity.HIGH,
                        "confidence": Confidence.CERTAIN,
                        "coupon_code": coupon_code,
                        "reuse_count": success_count,
                        "evidence": f"Coupon '{coupon_code}' used {success_count} times simultaneously"
                    })
                    logger.warning(f"⚠️ Coupon reuse race condition: {coupon_code} used {success_count} times")
            
            except Exception as e:
                logger.error(f"Coupon reuse test error: {e}")
        
        return findings

    async def test_inventory_depletion(
        self,
        purchase_endpoint: str,
        product_id: str,
        quantity: int = 1
    ) -> list[dict[str, Any]]:
        """
        測試庫存耗盡競態條件
        
        測試場景：同時購買超過庫存數量的商品
        
        Args:
            purchase_endpoint: 購買端點
            product_id: 商品 ID
            quantity: 購買數量
            
        Returns:
            list: 測試結果
        """
        findings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 同時發送多個購買請求
            concurrent_count = 10
            tasks = [
                client.post(
                    f"{self.target_url}{purchase_endpoint}",
                    json={"product_id": product_id, "quantity": quantity}
                )
                for _ in range(concurrent_count)
            ]
            
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(
                    1 for r in responses 
                    if isinstance(r, httpx.Response) and r.status_code in [200, 201]
                )
                
                # 如果超過預期數量的請求成功
                if success_count > 5:  # 假設庫存小於 5
                    findings.append({
                        "type": "inventory_race_condition",
                        "severity": Severity.HIGH,
                        "confidence": Confidence.POSSIBLE,
                        "product_id": product_id,
                        "successful_purchases": success_count,
                        "total_quantity": success_count * quantity,
                        "evidence": f"{success_count} concurrent purchases succeeded"
                    })
                    logger.warning(f"⚠️ Inventory race condition: {success_count} purchases succeeded")
            
            except Exception as e:
                logger.error(f"Inventory depletion test error: {e}")
        
        return findings

    async def run_all_tests(
        self,
        test_endpoints: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """
        運行所有競態條件測試
        
        Args:
            test_endpoints: 測試端點配置
            
        Returns:
            list: 所有測試結果
        """
        logger.info(f"Starting race condition tests on {self.target_url}")
        
        all_findings = []
        
        # 默認端點
        if not test_endpoints:
            test_endpoints = {
                "generic": "/api/action",
                "withdrawal": "/api/withdraw",
                "balance": "/api/balance",
                "coupon": "/api/coupon/apply",
                "purchase": "/api/purchase"
            }
        
        # 並發執行測試
        results = await asyncio.gather(
            self.test_concurrent_requests(test_endpoints.get("generic", "/api/action")),
            self.test_coupon_reuse(test_endpoints.get("coupon", "/api/coupon/apply")),
            return_exceptions=True
        )
        
        for result in results:
            if isinstance(result, list):
                all_findings.extend(result)
        
        logger.info(f"✅ Race condition tests completed: {len(all_findings)} findings")
        
        return all_findings

