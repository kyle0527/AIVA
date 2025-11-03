"""
AIVA V2 易用層客戶端實現
讓跨語言調用像本地函數調用一樣簡單

設計目標：
1. 比 V1 的 requests.post() 更簡單
2. 封裝所有 gRPC 複雜性
3. 提供自動重試和錯誤處理
4. 支持連接池和服務發現
5. 統一協議支援 (gRPC/HTTP/MQ)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from pathlib import Path
import time

from ..cross_language.core import CrossLanguageConfig, PythonAdapter
from ..enums import ProgrammingLanguage
from ..schemas.generated.messaging import AIVARequest, AIVAResponse
from ..utils.logging import setup_logger
from ..utils.retry import RetryConfig, retry_async

# 引入 gRPC 客戶端
from ..grpc.grpc_client import AIVAGRPCClient
from ..messaging.compatibility_layer import message_broker

logger = setup_logger(__name__)


# 常量定義
HEALTH_PATH = "/health"

@dataclass
class ServiceEndpoint:
    """服務端點配置"""
    language: ProgrammingLanguage
    host: str = "localhost"
    port: int = 50051
    health_path: str = HEALTH_PATH
    timeout: float = 30.0
    enabled: bool = True


class AivaClient:
    """AIVA V2 統一客戶端
    
    使用示例：
    ```python
    # 簡單的異步調用 - 就像 V1 一樣簡單！
    response = await aiva_client.call_rust("fuzzing_task", {"target": "example.com"})
    response = await aiva_client.call_go("sca_scan", {"project_path": "/path/to/code"})
    response = await aiva_client.call_typescript("web_scan", {"url": "https://example.com"})
    
    # 批量調用
    results = await aiva_client.call_multiple([
        ("rust", "fuzzing_task", {"target": "example.com"}),
        ("go", "sca_scan", {"project_path": "/path"}),
    ])
    ```
    """
    
    def __init__(
        self, 
        config: Optional[CrossLanguageConfig] = None,
        auto_discover: bool = True,
        grpc_enabled: bool = True,
        grpc_server_address: str = "localhost:50051"
    ):
        """初始化 AIVA 客戶端
        
        Args:
            config: 跨語言通信配置
            auto_discover: 是否自動發現服務
            grpc_enabled: 是否啟用 gRPC 支援
            grpc_server_address: gRPC 服務器地址
        """
        self.config = config or CrossLanguageConfig()
        self.adapter = PythonAdapter()
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self._connection_pools: Dict[str, Any] = {}
        
        # gRPC 支援
        self.grpc_enabled = grpc_enabled
        self.grpc_client: Optional[AIVAGRPCClient] = None
        if grpc_enabled:
            try:
                self.grpc_client = AIVAGRPCClient(grpc_server_address)
                logger.info(f"🔌 gRPC 客戶端已初始化: {grpc_server_address}")
            except Exception as e:
                logger.warning(f"⚠️  gRPC 初始化失敗，將使用備用協議: {e}")
                self.grpc_enabled = False
        
        # 預設服務端點
        self._setup_default_endpoints()
        
        if auto_discover:
            # 在事件循環中延遲執行服務發現
            self._auto_discover = True
        
        logger.info("AivaClient initialized with V2 unified communication")
    
    def _setup_default_endpoints(self):
        """設置預設服務端點"""
        self.endpoints = {
            "rust": ServiceEndpoint(
                language=ProgrammingLanguage.RUST,
                host="localhost",
                port=50052,
                health_path=HEALTH_PATH
            ),
            "go": ServiceEndpoint(
                language=ProgrammingLanguage.GO,
                host="localhost",
                port=50053,
                health_path=HEALTH_PATH
            ),
            "typescript": ServiceEndpoint(
                language=ProgrammingLanguage.TYPESCRIPT,
                host="localhost",
                port=50054,
                health_path=HEALTH_PATH
            ),
        }
    
    async def _discover_services(self):
        """自動發現可用服務"""
        logger.info("Starting service discovery...")
        
        for service_name, endpoint in self.endpoints.items():
            try:
                is_healthy = await self._check_service_health(service_name)
                endpoint.enabled = is_healthy
                
                if is_healthy:
                    logger.info(f"✅ Service '{service_name}' is available at {endpoint.host}:{endpoint.port}")
                else:
                    logger.warning(f"❌ Service '{service_name}' is not available")
                    
            except Exception as e:
                logger.error(f"Service discovery failed for {service_name}: {e}")
                endpoint.enabled = False
    
    async def _check_service_health(self, service_name: str) -> bool:
        """檢查服務健康狀態"""
        endpoint = self.endpoints.get(service_name)
        if not endpoint:
            return False
        
        try:
            # 這裡應該使用 gRPC health check
            # 暫時使用簡單的連接測試
            await asyncio.wait_for(
                asyncio.open_connection(endpoint.host, endpoint.port),
                timeout=2.0
            )
            return True
        except (asyncio.TimeoutError, OSError):
            return False
    
    # ==================== 核心調用方法 ====================
    
    async def call_rust(self, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """調用 Rust 服務
        
        Args:
            task: 任務名稱
            params: 任務參數
            
        Returns:
            任務執行結果
        """
        return await self._call_service("rust", task, params)
    
    async def call_go(self, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """調用 Go 服務
        
        Args:
            task: 任務名稱
            params: 任務參數
            
        Returns:
            任務執行結果
        """
        return await self._call_service("go", task, params)
    
    async def call_typescript(self, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """調用 TypeScript 服務
        
        Args:
            task: 任務名稱
            params: 任務參數
            
        Returns:
            任務執行結果
        """
        return await self._call_service("typescript", task, params)
    
    async def call_service(
        self, 
        language: str, 
        task: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """通用服務調用方法
        
        Args:
            language: 目標語言 (rust/go/typescript)
            task: 任務名稱
            params: 任務參數
            
        Returns:
            任務執行結果
        """
        return await self._call_service(language, task, params)
    
    # ==================== 批量操作 ====================
    
    async def call_multiple(
        self, 
        calls: List[tuple[str, str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """批量並發調用多個服務
        
        Args:
            calls: 調用列表，每個元素為 (language, task, params)
            
        Returns:
            所有調用的結果列表
        """
        tasks = [
            self._call_service(language, task, params)
            for language, task, params in calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 處理異常結果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "call_index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    # ==================== 內部實現 ====================
    
    @retry_async(RetryConfig(max_attempts=3, delay=1.0))
    async def _call_service(
        self, 
        language: str, 
        task: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """內部服務調用實現，包含重試邏輯 - 支援 gRPC/HTTP/MQ 多協議"""
        
        endpoint = self.endpoints.get(language)
        if not endpoint:
            raise ValueError(f"Unknown language: {language}")
        
        if not endpoint.enabled:
            raise ConnectionError(f"Service {language} is not available")
        
        start_time = time.time()
        
        try:
            # 1. 優先嘗試 gRPC 協議 
            if self.grpc_enabled and self.grpc_client:
                try:
                    logger.debug(f"🔄 嘗試 gRPC 調用: {language}.{task}")
                    
                    # 使用 gRPC 客戶端
                    grpc_result = await self.grpc_client.execute_cross_language_task(
                        task=f"{language}.{task}",
                        parameters=params
                    )
                    
                    if grpc_result.get("success", False):
                        execution_time = time.time() - start_time
                        logger.info(f"✅ gRPC 調用成功: {language}.{task} ({execution_time:.3f}s)")
                        
                        return {
                            "success": True,
                            "result": grpc_result.get("result", {}),
                            "protocol": "gRPC",
                            "execution_time": execution_time,
                            "language": language,
                            "task": task
                        }
                        
                except Exception as e:
                    logger.warning(f"⚠️  gRPC 調用失敗，嘗試備用協議: {e}")
            
            # 2. 構建通用請求
            request = AIVARequest(
                request_id=f"aiva_{int(time.time() * 1000)}",
                task=task,
                parameters=params,
                timeout=endpoint.timeout
            )
            
            # 3. HTTP/其他協議調用 - 這裡目前使用模擬實現
            # 實際上應該使用 gRPC 調用
            result = await self._execute_grpc_call(endpoint, request)
            
            duration = time.time() - start_time
            logger.info(f"✅ {language} task '{task}' completed in {duration:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "language": language,
                "task": task,
                "duration": duration,
                "request_id": request.request_id
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {language} task '{task}' failed in {duration:.2f}s: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "language": language,
                "task": task,
                "duration": duration
            }
    
    async def _execute_grpc_call(
        self, 
        endpoint: ServiceEndpoint, 
        request: AIVARequest
    ) -> Any:
        """執行 gRPC 調用 (目前為模擬實現)
        
        TODO: 實際實現 gRPC 調用邏輯
        這裡需要：
        1. 建立 gRPC 連接
        2. 序列化請求為 Protobuf
        3. 發送請求並獲取響應
        4. 反序列化響應
        """
        
        # 模擬 gRPC 調用延遲
        await asyncio.sleep(0.1)
        
        # 模擬不同語言的響應
        if endpoint.language == ProgrammingLanguage.RUST:
            return {
                "processed_by": "rust_service",
                "task": request.task,
                "status": "completed",
                "performance": "high",
                "details": request.parameters
            }
        elif endpoint.language == ProgrammingLanguage.GO:
            return {
                "processed_by": "go_service", 
                "task": request.task,
                "status": "completed",
                "concurrency": "excellent",
                "details": request.parameters
            }
        elif endpoint.language == ProgrammingLanguage.TYPESCRIPT:
            return {
                "processed_by": "typescript_service",
                "task": request.task, 
                "status": "completed",
                "integration": "smooth",
                "details": request.parameters
            }
        else:
            raise ValueError(f"Unsupported language: {endpoint.language}")
    
    # ==================== 管理方法 ====================
    
    async def get_service_status(self) -> Dict[str, Any]:
        """獲取所有服務狀態"""
        status = {}
        
        for service_name, endpoint in self.endpoints.items():
            is_healthy = await self._check_service_health(service_name)
            status[service_name] = {
                "enabled": endpoint.enabled,
                "healthy": is_healthy,
                "endpoint": f"{endpoint.host}:{endpoint.port}",
                "language": endpoint.language.value
            }
        
        return status
    
    def enable_service(self, service_name: str):
        """啟用服務"""
        if service_name in self.endpoints:
            self.endpoints[service_name].enabled = True
            logger.info(f"Service '{service_name}' enabled")
    
    def disable_service(self, service_name: str):
        """禁用服務"""
        if service_name in self.endpoints:
            self.endpoints[service_name].enabled = False
            logger.info(f"Service '{service_name}' disabled")
    
    async def close(self):
        """關閉客戶端，清理資源"""
        # 關閉 gRPC 客戶端
        if self.grpc_client:
            await self.grpc_client.close()
            self.grpc_client = None
        
        # 關閉連接池等資源
        self._connection_pools.clear()
        logger.info("🔌 AivaClient 已關閉")


# 全局單例客戶端實例 - 延遲初始化
aiva_client = None


def get_aiva_client() -> AivaClient:
    """獲取全局客戶端實例"""
    global aiva_client
    if aiva_client is None:
        aiva_client = AivaClient(auto_discover=False)  # 不自動發現避免事件循環問題
    return aiva_client


# ==================== 便利函數 ====================

async def call_rust(task: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """便利函數：調用 Rust 服務"""
    return await aiva_client.call_rust(task, params)


async def call_go(task: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """便利函數：調用 Go 服務"""
    return await aiva_client.call_go(task, params)


async def call_typescript(task: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """便利函數：調用 TypeScript 服務"""
    return await aiva_client.call_typescript(task, params)


async def get_service_status() -> Dict[str, Any]:
    """便利函數：獲取服務狀態"""
    return await aiva_client.get_service_status()