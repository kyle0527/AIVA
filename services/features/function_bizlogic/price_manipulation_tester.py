"""價格操控測試器 - Real Implementation

測試電商/支付系統的價格操控漏洞
"""

import asyncio
import hashlib
import logging
from typing import Any
from decimal import Decimal

import httpx

from services.aiva_common.enums import VulnerabilityType, Severity, Confidence
from .finding_helper import create_bizlogic_finding

logger = logging.getLogger(__name__)


class PriceManipulationTester:
    """價格操控測試器 - 真實實現"""

    def __init__(self, target_url: str, authorization_token: str | None = None):
        """
        初始化價格操控測試器
        
        Args:
            target_url: 目標 URL
            authorization_token: 授權令牌
        """
        self.target_url = target_url
        self.authorization_token = authorization_token
        self.findings = []
        
        # 授權檢查
        if not authorization_token or len(authorization_token) < 32:
            logger.warning("No valid authorization token - running in limited mode")

    async def test_negative_price(self, endpoint: str, price_param: str = "price") -> list[dict[str, Any]]:
        """
        測試負數價格
        
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
                        json={price_param: price, "quantity": 1}
                    )
                    
                    if response.status_code in [200, 201]:
                        # 檢查響應中是否接受了負價格
                        response_data = response.json()
                        if "total" in response_data or "amount" in response_data:
                            findings.append({
                                "type": "negative_price_accepted",
                                "severity": Severity.HIGH,
                                "confidence": Confidence.CERTAIN,
                                "price": price,
                                "response_code": response.status_code,
                                "evidence": str(response_data)[:200]
                            })
                            logger.warning(f"⚠️ Negative price accepted: {price}")
                
                except Exception as e:
                    logger.debug(f"Negative price test error: {e}")
        
        return findings

    async def test_zero_price(self, endpoint: str, price_param: str = "price") -> list[dict[str, Any]]:
        """
        測試零價格
        
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
                    json={price_param: 0, "quantity": 1}
                )
                
                if response.status_code in [200, 201]:
                    response_data = response.json()
                    if "total" in response_data or "amount" in response_data:
                        total = response_data.get("total", response_data.get("amount", 0))
                        if total == 0:
                            findings.append({
                                "type": "zero_price_accepted",
                                "severity": Severity.MEDIUM,
                                "confidence": Confidence.CERTAIN,
                                "response_code": response.status_code,
                                "evidence": str(response_data)[:200]
                            })
                            logger.warning("⚠️ Zero price accepted")
            
            except Exception as e:
                logger.debug(f"Zero price test error: {e}")
        
        return findings

    async def test_price_tampering(self, endpoint: str) -> list[dict[str, Any]]:
        """
        測試價格篡改
        
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
                        }
                    )
                    
                    if response.status_code in [200, 201]:
                        response_data = response.json()
                        total = response_data.get("total", response_data.get("amount", 0))
                        
                        # 檢查是否使用了篡改的價格
                        if abs(total - case["tampered_price"]) < 0.01:
                            findings.append({
                                "type": "price_tampering",
                                "severity": Severity.CRITICAL,
                                "confidence": Confidence.CERTAIN,
                                "original_price": case["original_price"],
                                "tampered_price": case["tampered_price"],
                                "accepted_price": total,
                                "evidence": str(response_data)[:200]
                            })
                            logger.warning(f"⚠️ Price tampering successful: {case['tampered_price']}")
                
                except Exception as e:
                    logger.debug(f"Price tampering test error: {e}")
        
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
