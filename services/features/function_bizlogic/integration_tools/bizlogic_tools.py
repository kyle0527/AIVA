#!/usr/bin/env python3
"""
AIVA BizLogic Integration Tools
整合業務邏輯漏洞檢測能力

功能:
- 競態條件檢測 (Race Condition Detection)
- 價格操控檢測 (Price Manipulation Detection)  
- 工作流程繞過檢測 (Workflow Bypass Detection)
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from aiva_common.schemas.tasks import FunctionTaskPayload

from ..price_manipulation_scanner import PriceManipulationScanner

# 導入掃描器
from ..race_condition_scanner import RaceConditionScanner
from ..workflow_bypass_scanner import WorkflowBypassScanner

logger = logging.getLogger(__name__)


@dataclass
class BizLogicTarget:
    """業務邏輯檢測目標"""
    url: str
    method: str = "POST"
    parameters: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    authorization_token: str | None = None
    user_role: str = "customer"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.headers is None:
            self.headers = {}


@dataclass
class BizLogicVulnerability:
    """業務邏輯漏洞"""
    vuln_type: str  # race_condition, price_manipulation, workflow_bypass
    sub_type: str  # concurrent_requests, negative_price, step_skipping, etc.
    severity: str
    confidence: str
    target_url: str
    evidence: dict[str, Any]
    description: str
    business_impact: str
    remediation: str
    timestamp: str | None = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BizLogicScanOptions:
    """業務邏輯掃描選項"""
    # 競態條件選項
    test_race_condition: bool = True
    concurrent_count: int = 10
    race_test_endpoints: list[str] = None
    
    # 價格操控選項
    test_price_manipulation: bool = True
    test_negative_price: bool = True
    test_zero_price: bool = True
    test_overflow_price: bool = True
    
    # 工作流程繞過選項
    test_workflow_bypass: bool = True
    workflow_steps: list[str] = None
    checkout_endpoint: str = "/checkout"
    
    def __post_init__(self):
        if self.race_test_endpoints is None:
            self.race_test_endpoints = []
        if self.workflow_steps is None:
            self.workflow_steps = []


class BizLogicManager:
    """業務邏輯漏洞管理器
    
    統一管理業務邏輯漏洞檢測的主類別，協調多個掃描器協同工作。
    """
    
    def __init__(self):
        """初始化管理器"""
        self.logger = logger
        self.logger.info("✅ BizLogic Manager 已初始化")
    
    async def comprehensive_scan(
        self,
        target_url: str,
        options: dict[str, Any]
    ) -> dict[str, Any]:
        """執行全面的業務邏輯掃描
        
        Args:
            target_url: 目標 URL
            options: 掃描選項
                - use_race_condition: 是否檢測競態條件
                - use_price_manipulation: 是否檢測價格操控
                - use_workflow_bypass: 是否檢測工作流程繞過
                - authorization_token: 授權令牌
                - user_role: 用戶角色
                - concurrent_count: 並發請求數量
                - race_endpoints: 競態條件測試端點列表
                - workflow_steps: 工作流程步驟列表
                
        Returns:
            Dict: 掃描結果
                {
                    "race_conditions": [...],
                    "price_manipulations": [...],
                    "workflow_bypasses": [...],
                    "total_vulnerabilities": 10,
                    "scan_info": {...}
                }
        """
        start_time = datetime.now()
        
        # 提取通用參數
        auth_token = options.get("authorization_token")
        user_role = options.get("user_role", "customer")
        
        # 初始化結果
        results = {
            "race_conditions": [],
            "price_manipulations": [],
            "workflow_bypasses": [],
            "total_vulnerabilities": 0,
            "scan_info": {
                "target_url": target_url,
                "start_time": start_time.isoformat(),
                "tests_performed": []
            }
        }
        
        # 執行檢測任務
        tasks = []
        
        # 1. 競態條件檢測
        if options.get("use_race_condition", True):
            results["scan_info"]["tests_performed"].append("race_condition")
            race_scanner = RaceConditionScanner(target_url, auth_token)
            
            # 測試並發請求
            concurrent_count = options.get("concurrent_count", 10)
            race_endpoints = options.get("race_endpoints", ["/api/coupon", "/checkout"])
            
            for endpoint in race_endpoints:
                tasks.append(
                    self._wrap_race_condition_test(
                        race_scanner, endpoint, concurrent_count
                    )
                )
        
        # 2. 價格操控檢測
        if options.get("use_price_manipulation", True):
            results["scan_info"]["tests_performed"].append("price_manipulation")
            price_scanner = PriceManipulationScanner(target_url, auth_token, user_role)
            
            checkout_endpoint = options.get("checkout_endpoint", "/api/checkout")
            
            # 負數價格測試
            if options.get("test_negative_price", True):
                tasks.append(
                    self._wrap_price_test(
                        price_scanner.test_negative_price,
                        checkout_endpoint
                    )
                )
            
            # 零價格測試
            if options.get("test_zero_price", True):
                tasks.append(
                    self._wrap_price_test(
                        price_scanner.test_zero_price,
                        checkout_endpoint
                    )
                )
            
            # 溢出價格測試
            if options.get("test_overflow_price", True):
                tasks.append(
                    self._wrap_price_test(
                        price_scanner.test_overflow_price,
                        checkout_endpoint
                    )
                )
        
        # 3. 工作流程繞過檢測
        if options.get("use_workflow_bypass", True):
            results["scan_info"]["tests_performed"].append("workflow_bypass")
            workflow_scanner = WorkflowBypassScanner(target_url, auth_token)
            
            # 步驟跳過測試
            workflow_steps = options.get("workflow_steps", [])
            if workflow_steps:
                tasks.append(
                    self._wrap_workflow_test(
                        workflow_scanner.test_step_skipping,
                        workflow_steps
                    )
                )
            
            # 直接結帳測試
            checkout_endpoint = options.get("checkout_endpoint", "/checkout")
            tasks.append(
                self._wrap_workflow_test(
                    workflow_scanner.test_direct_checkout,
                    checkout_endpoint
                )
            )
        
        # 執行所有測試
        if tasks:
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 整合結果
            for result in test_results:
                if isinstance(result, Exception):
                    self.logger.error(f"測試執行錯誤: {result}")
                    continue
                
                if isinstance(result, list):
                    for finding in result:
                        vuln_type = finding.get("type", "").lower()
                        
                        if "race" in vuln_type:
                            results["race_conditions"].append(finding)
                        elif "price" in vuln_type:
                            results["price_manipulations"].append(finding)
                        elif "workflow" in vuln_type or "bypass" in vuln_type:
                            results["workflow_bypasses"].append(finding)
        
        # 計算總漏洞數
        results["total_vulnerabilities"] = (
            len(results["race_conditions"]) +
            len(results["price_manipulations"]) +
            len(results["workflow_bypasses"])
        )
        
        # 完成時間
        end_time = datetime.now()
        results["scan_info"]["end_time"] = end_time.isoformat()
        results["scan_info"]["duration"] = (end_time - start_time).total_seconds()
        
        self.logger.info(
            f"✅ BizLogic 掃描完成: {results['total_vulnerabilities']} 個漏洞"
        )
        
        return results
    
    def scan_sync(
        self,
        target_url: str,
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """執行掃描 (同步接口)
        
        這是功能模組的主要接口，提供同步調用方式。
        AI Commander 將使用 asyncio.to_thread() 來包裝此方法。
        
        Args:
            target_url: 目標 URL
            options: 掃描選項字典（可選）
                - use_race_condition: bool = True
                - use_price_manipulation: bool = True
                - use_workflow_bypass: bool = True
                - authorization_token: str = None
                - user_role: str = "customer"
                - concurrent_count: int = 10
                - race_endpoints: List[str] = []
                - workflow_steps: List[str] = []
                - checkout_endpoint: str = "/checkout"
                
        Returns:
            Dict: 掃描結果
                {
                    "race_conditions": [...],
                    "price_manipulations": [...],
                    "workflow_bypasses": [...],
                    "total_vulnerabilities": int,
                    "scan_info": {...}
                }
        
        Example:
            >>> manager = BizLogicManager()
            >>> result = manager.scan_sync(
            ...     "https://example.com",
            ...     {"authorization_token": "Bearer xxx"}
            ... )
            >>> print(f"Found {result['total_vulnerabilities']} vulnerabilities")
        """
        if options is None:
            options = {}
        
        # 使用 asyncio.run 在同步環境中運行異步代碼
        try:
            return asyncio.run(self.comprehensive_scan(target_url, options))
        except RuntimeError as e:
            # 如果已經在事件循環中，使用 run_until_complete
            if "cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.comprehensive_scan(target_url, options))
            raise
    
    async def scan(
        self,
        task: FunctionTaskPayload,
        options: BizLogicScanOptions
    ) -> dict[str, Any]:
        """執行掃描 (FunctionTaskPayload 接口 - 異步)
        
        Args:
            task: 任務載荷
            options: 掃描選項
            
        Returns:
            Dict: 掃描結果
        """
        # 轉換為 comprehensive_scan 格式
        scan_options = {
            "use_race_condition": options.test_race_condition,
            "use_price_manipulation": options.test_price_manipulation,
            "use_workflow_bypass": options.test_workflow_bypass,
            "authorization_token": task.metadata.get("authorization_token"),
            "user_role": task.metadata.get("user_role", "customer"),
            "concurrent_count": options.concurrent_count,
            "race_endpoints": options.race_test_endpoints,
            "test_negative_price": options.test_negative_price,
            "test_zero_price": options.test_zero_price,
            "test_overflow_price": options.test_overflow_price,
            "workflow_steps": options.workflow_steps,
            "checkout_endpoint": options.checkout_endpoint,
        }
        
        return await self.comprehensive_scan(task.target_url, scan_options)
    
    # 輔助方法 - 包裝測試以處理異常
    
    async def _wrap_race_condition_test(
        self,
        scanner: RaceConditionScanner,
        endpoint: str,
        concurrent_count: int
    ) -> list[dict[str, Any]]:
        """包裝競態條件測試"""
        try:
            return await scanner.test_concurrent_requests(
                endpoint=endpoint,
                concurrent_count=concurrent_count
            )
        except Exception as e:
            self.logger.error(f"競態條件測試失敗 ({endpoint}): {e}")
            return []
    
    async def _wrap_price_test(
        self,
        test_func,
        endpoint: str
    ) -> list[dict[str, Any]]:
        """包裝價格測試"""
        try:
            return await test_func(endpoint)
        except Exception as e:
            self.logger.error(f"價格測試失敗: {e}")
            return []
    
    async def _wrap_workflow_test(
        self,
        test_func,
        *args
    ) -> list[dict[str, Any]]:
        """包裝工作流程測試"""
        try:
            return await test_func(*args)
        except Exception as e:
            self.logger.error(f"工作流程測試失敗: {e}")
            return []

