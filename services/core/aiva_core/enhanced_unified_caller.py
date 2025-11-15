"""升級版統一功能調用器 - 使用 ProtocolAdapter 設計模式

此模組實現了 Gang of Four Adapter 模式，提供統一的跨語言模組調用接口。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from services.core.aiva_core.adapters import HttpProtocolAdapter, create_http_adapter
from services.aiva_common.utils import get_logger
from services.aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context,
)

if TYPE_CHECKING:
    from services.core.aiva_core.adapters import ProtocolAdapter

logger = get_logger(__name__)

# 模組常量
MODULE_NAME = "enhanced_unified_caller"


@dataclass
class FunctionCallResult:
    """功能調用結果"""

    success: bool
    language: str
    module_name: str
    function_name: str
    result: Any
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class ModuleEndpoint:
    """模組端點配置"""

    name: str
    language: str
    protocol: str  # http, grpc
    host: str
    port: int
    available_functions: list[str]
    adapter: ProtocolAdapter | None = None  # 協議適配器


class EnhancedUnifiedFunctionCaller:
    """增強型統一功能調用器 - 使用協議適配器模式
    
    實現優勢：
    1. 遵循 Gang of Four Adapter 設計模式
    2. 支持多種協議（HTTP、gRPC、WebSocket）
    3. 易於擴展新的協議類型
    4. 統一的錯誤處理和重試機制
    """

    def __init__(self):
        self.logger = logger
        self.endpoints: Dict[str, ModuleEndpoint] = {}
        self.adapters: Dict[str, ProtocolAdapter] = {}
        
    def initialize(self):
        """初始化調用器和協議適配器"""
        self._setup_protocol_adapters()
        self._init_endpoints()
        
    def _setup_protocol_adapters(self):
        """設置協議適配器"""
        # HTTP 協議適配器
        http_adapter = create_http_adapter()
        self.adapters["http"] = http_adapter
        
    def _init_endpoints(self):
        """初始化所有模組端點 - 使用適配器模式"""
        self.endpoints = {
            # Python 模組
            "function_sqli": ModuleEndpoint(
                name="function_sqli",
                language="python",
                protocol="http",
                host="localhost",
                port=8001,
                available_functions=["test_sql_injection"],
                adapter=self.adapters.get("http")
            ),
            
            "function_xss": ModuleEndpoint(
                name="function_xss",
                language="python", 
                protocol="http",
                host="localhost",
                port=8002,
                available_functions=["test_xss"],
                adapter=self.adapters.get("http")
            ),
            
            "function_auth": ModuleEndpoint(
                name="function_auth",
                language="python",
                protocol="http", 
                host="localhost",
                port=8003,
                available_functions=["test_auth_bypass", "test_weak_auth"],
                adapter=self.adapters.get("http")
            ),
            
            # TypeScript 模組
            "function_crawler": ModuleEndpoint(
                name="function_crawler",
                language="typescript",
                protocol="http",
                host="localhost", 
                port=9001,
                available_functions=["crawl_website", "extract_forms"],
                adapter=self.adapters.get("http")
            ),
            
            "function_fuzzing": ModuleEndpoint(
                name="function_fuzzing", 
                language="typescript",
                protocol="http",
                host="localhost",
                port=9002, 
                available_functions=["fuzz_parameters"],
                adapter=self.adapters.get("http")
            ),
            
            # Go 模組
            "function_network": ModuleEndpoint(
                name="function_network",
                language="go",
                protocol="http",
                host="localhost",
                port=10001,
                available_functions=["scan_ports", "test_ssl"],
                adapter=self.adapters.get("http")
            ),
            
            # Rust 模組  
            "function_performance": ModuleEndpoint(
                name="function_performance",
                language="rust",
                protocol="http", 
                host="localhost",
                port=11001,
                available_functions=["stress_test", "benchmark"],
                adapter=self.adapters.get("http")
            )
        }
        
    async def call_function(
        self,
        module_name: str,
        function_name: str,
        parameters: Dict[str, Any]
    ) -> FunctionCallResult:
        """調用指定功能模組的函數 - 使用協議適配器
        
        Args:
            module_name: 模組名稱
            function_name: 函數名稱  
            parameters: 函數參數
            
        Returns:
            功能調用結果
        """
        if module_name not in self.endpoints:
            return FunctionCallResult(
                success=False,
                language="unknown",
                module_name=module_name,
                function_name=function_name,
                result=None,
                error=f"Unknown module: {module_name}"
            )
            
        endpoint = self.endpoints[module_name]
        
        if function_name not in endpoint.available_functions:
            return FunctionCallResult(
                success=False,
                language=endpoint.language,
                module_name=module_name,
                function_name=function_name,
                result=None,
                error=f"Function {function_name} not available in {module_name}"
            )
            
        try:
            # 使用協議適配器進行調用
            start_time = asyncio.get_event_loop().time()
            
            url = f"http://{endpoint.host}:{endpoint.port}/{function_name}"
            request_data = {
                "function": function_name,
                "parameters": parameters,
                "module": module_name
            }
            
            # 通過適配器發送請求 - 關鍵的適配器模式應用
            if endpoint.adapter:
                response_data = await endpoint.adapter.send_request(url, request_data)
            else:
                context = create_error_context(
                    module=MODULE_NAME,
                    function="call_function"
                )
                raise AIVAError(
                    message=f"No adapter available for protocol: {endpoint.protocol}",
                    error_type=ErrorType.CONFIGURATION,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                )
                
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return FunctionCallResult(
                success=True,
                language=endpoint.language,
                module_name=module_name,
                function_name=function_name,
                result=response_data.get("data"),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Function call failed: {module_name}.{function_name} - {e}")
            return FunctionCallResult(
                success=False,
                language=endpoint.language,
                module_name=module_name,
                function_name=function_name,
                result=None,
                error=str(e)
            )
            
    async def call_multiple_functions(
        self,
        calls: list[dict],
        concurrent: bool = True
    ) -> Sequence[FunctionCallResult | BaseException]:
        """並行調用多個功能模組
        
        Args:
            calls: 調用列表，格式: [{"module": "name", "function": "name", "parameters": {}}]
            concurrent: 是否並行執行
            
        Returns:
            功能調用結果列表（並行時可能包含異常）
        """
        if concurrent:
            tasks = [
                self.call_function(
                    call["module"],
                    call["function"], 
                    call.get("parameters", {})
                )
                for call in calls
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for call in calls:
                result = await self.call_function(
                    call["module"],
                    call["function"],
                    call.get("parameters", {})
                )
                results.append(result)
            return results
            
    async def health_check(self) -> Dict[str, bool]:
        """檢查所有模組的健康狀態
        
        Returns:
            模組健康狀態字典
        """
        health_results = {}
        
        for module_name, endpoint in self.endpoints.items():
            try:
                url = f"http://{endpoint.host}:{endpoint.port}/health"
                if endpoint.adapter:
                    await endpoint.adapter.send_request(url, {})
                    health_results[module_name] = True
                else:
                    health_results[module_name] = False
            except Exception:
                health_results[module_name] = False
                
        return health_results
        
    async def cleanup(self):
        """清理資源"""
        for adapter in self.adapters.values():
            if isinstance(adapter, HttpProtocolAdapter):
                if hasattr(adapter.client, 'aclose'):
                    await adapter.client.aclose()


# 全局實例 - 單例模式
_unified_caller_instance = None

def get_unified_caller() -> EnhancedUnifiedFunctionCaller:
    """獲取統一功能調用器實例 - 單例模式"""
    global _unified_caller_instance
    
    if _unified_caller_instance is None:
        _unified_caller_instance = EnhancedUnifiedFunctionCaller()
        _unified_caller_instance.initialize()
        
    return _unified_caller_instance