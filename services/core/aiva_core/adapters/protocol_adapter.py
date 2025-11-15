"""協議適配器 - Gang of Four Adapter 設計模式實現

此模組實現了協議適配器模式，允許不同協議間的轉換，
遵循開閉原則和依賴反轉原則。

參考：
- Gang of Four Design Patterns
- https://refactoring.guru/design-patterns/adapter
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from services.aiva_common.utils import get_logger
from services.aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context,
)

if TYPE_CHECKING:
    import httpx

logger = get_logger(__name__)

# 模組常量
MODULE_NAME = "adapters.protocol_adapter"


class ProtocolAdapter(ABC):
    """協議適配器抽象基類
    
    實現 Gang of Four 的 Adapter 模式，定義統一的協議轉換接口。
    """
    
    @abstractmethod
    async def send_request(self, endpoint: str, data: Any) -> Dict[str, Any]:
        """發送請求
        
        Args:
            endpoint: 目標端點
            data: 請求數據
            
        Returns:
            響應數據
        """
        pass
    
    @abstractmethod
    async def handle_response(self, response: Any) -> Dict[str, Any]:
        """處理響應
        
        Args:
            response: 原始響應
            
        Returns:
            標準化響應數據
        """
        pass


class HttpProtocolAdapter(ProtocolAdapter):
    """HTTP 協議適配器
    
    將內部請求格式適配為 HTTP 協議，並將 HTTP 響應
    轉換為統一的內部格式。
    """
    
    def __init__(self, client: httpx.AsyncClient):
        """初始化 HTTP 適配器
        
        Args:
            client: HTTP 客戶端實例
        """
        self.client = client
    
    async def send_request(self, endpoint: str, data: Any) -> Dict[str, Any]:
        """發送 HTTP 請求
        
        Args:
            endpoint: HTTP 端點 URL
            data: 請求數據（將轉換為 JSON）
            
        Returns:
            標準化的響應數據
            
        Raises:
            AIVAError: 當請求失敗時
        """
        try:
            # 適配內部數據格式為 HTTP JSON 請求
            json_data = self._adapt_request_data(data)
            
            logger.debug(f"Sending HTTP request to {endpoint}")
            response = await self.client.post(
                endpoint,
                json=json_data,
                headers={"Content-Type": "application/json"}
            )
            
            return await self.handle_response(response)
            
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            context = create_error_context(
                module=MODULE_NAME,
                function="send_request"
            )
            raise AIVAError(
                message=f"HTTP 請求失敗: {e}",
                error_type=ErrorType.NETWORK,
                severity=ErrorSeverity.HIGH,
                context=context,
                original_exception=e,
            ) from e
    
    async def handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """處理 HTTP 響應
        
        Args:
            response: HTTP 響應對象
            
        Returns:
            標準化的內部響應格式
            
        Raises:
            AIVAError: 當響應無效時
        """
        try:
            if response.status_code == 200:
                # 適配 HTTP JSON 響應為內部格式
                response_data = response.json()
                return self._adapt_response_data(response_data)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"HTTP error response: {error_msg}")
                context = create_error_context(
                    module=MODULE_NAME,
                    function="handle_response"
                )
                raise AIVAError(
                    message=error_msg,
                    error_type=ErrorType.NETWORK,
                    severity=ErrorSeverity.HIGH,
                    context=context,
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            context = create_error_context(
                module=MODULE_NAME,
                function="handle_response"
            )
            raise AIVAError(
                message=f"無效的 JSON 響應: {e}",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                original_exception=e,
            ) from e
    
    def _adapt_request_data(self, internal_data: Any) -> Dict[str, Any]:
        """適配內部請求數據為 HTTP 格式
        
        Args:
            internal_data: 內部數據格式
            
        Returns:
            HTTP JSON 請求格式
        """
        if isinstance(internal_data, dict):
            return internal_data
        elif hasattr(internal_data, 'model_dump'):
            # Pydantic 模型
            return internal_data.model_dump()
        else:
            return {"data": internal_data}
    
    def _adapt_response_data(self, http_data: Dict[str, Any]) -> Dict[str, Any]:
        """適配 HTTP 響應數據為內部格式
        
        Args:
            http_data: HTTP JSON 響應數據
            
        Returns:
            標準化的內部響應格式
        """
        return {
            "status": "success",
            "data": http_data,
            "protocol": "http",
            "timestamp": http_data.get("timestamp")
        }


# 工廠函數 - 簡化適配器創建
def create_http_adapter() -> HttpProtocolAdapter:
    """創建 HTTP 協議適配器實例
    
    Returns:
        配置好的 HTTP 適配器
    """
    import httpx
    
    client = httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True
    )
    
    return HttpProtocolAdapter(client)