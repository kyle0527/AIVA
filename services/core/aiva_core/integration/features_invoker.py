"""Features Invoker - 統一 Features 調用層 (P0)

實現 Core → Features 調用打通：
1. 統一調用接口（Python/Rust/Go/TypeScript Features）
2. 使用 aiva_common 數據合約（FeatureRequest/FeatureResult）
3. 異步執行與錯誤處理
4. 性能監控與超時控制

遵循 AI自動化閉環完整實施計劃.md 第一章：Core → Features調用打通
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC
from enum import Enum

from pydantic import BaseModel, Field

from aiva_common.enums import ModuleName, TaskStatus
from aiva_common.schemas import APIResponse
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Feature 類型枚舉"""
    PYTHON = "python"
    RUST = "rust"
    GO = "go"
    TYPESCRIPT = "typescript"


class FeatureRequest(BaseModel):
    """Feature 請求標準合約"""
    request_id: str = Field(default_factory=lambda: f"req_{int(datetime.now(UTC).timestamp() * 1000)}")
    feature_module: ModuleName
    feature_type: FeatureType = FeatureType.PYTHON
    
    # 請求參數
    target_url: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # 執行配置
    timeout_ms: int = 30000  # 30 秒默認超時
    max_retries: int = 3
    priority: int = 5  # 1-10, 10最高
    
    # 元數據
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FeatureResponse(BaseModel):
    """Feature 響應標準合約"""
    request_id: str
    feature_module: ModuleName
    
    # 執行結果
    status: TaskStatus
    success: bool
    duration_ms: float
    
    # 數據
    data: Dict[str, Any] = Field(default_factory=dict)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 錯誤信息
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # 性能指標
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # 元數據
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FeaturesInvoker:
    """Features 統一調用器
    
    職責：
    1. 統一調用接口（支持多語言 Features）
    2. 數據合約驗證（FeatureRequest → FeatureResponse）
    3. 異步執行與超時控制
    4. 錯誤處理與重試機制
    5. 性能監控與日誌記錄
    
    使用場景：
    - ScannerPlugin → FeaturesInvoker → func_xss
    - ExploiterPlugin → FeaturesInvoker → func_sqli
    - AICommanderV2 → FeaturesInvoker → 任意 Feature
    """
    
    def __init__(self):
        self._feature_registry: Dict[ModuleName, Any] = {}
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_duration_ms": 0.0
        }
        logger.info("FeaturesInvoker initialized")
    
    def register_feature(
        self, 
        module_name: ModuleName, 
        feature_instance: Any
    ) -> None:
        """註冊 Feature 模組
        
        Args:
            module_name: Feature 模組名稱（使用 aiva_common.ModuleName 枚舉）
            feature_instance: Feature 實例對象（需實現 execute() 方法）
        """
        if module_name in self._feature_registry:
            logger.warning(f"Feature {module_name} already registered, overwriting")
        
        self._feature_registry[module_name] = feature_instance
        logger.info(f"Registered feature: {module_name}")
    
    def is_registered(self, module_name: ModuleName) -> bool:
        """檢查 Feature 是否已註冊"""
        return module_name in self._feature_registry
    
    async def invoke(
        self, 
        request: FeatureRequest
    ) -> FeatureResponse:
        """調用 Feature 並返回結果
        
        Args:
            request: Feature 請求（標準合約）
        
        Returns:
            Feature 響應（標準合約）
        
        Raises:
            AIVAError: 調用失敗時拋出
        """
        start_time = asyncio.get_event_loop().time()
        self._stats["total_requests"] += 1
        
        # 驗證 Feature 是否已註冊
        if not self.is_registered(request.feature_module):
            error_msg = f"Feature {request.feature_module} not registered"
            logger.error(error_msg)
            self._stats["failed_requests"] += 1
            
            return FeatureResponse(
                request_id=request.request_id,
                feature_module=request.feature_module,
                status=TaskStatus.FAILED,
                success=False,
                duration_ms=0.0,
                error_message=error_msg,
                error_code="FEATURE_NOT_REGISTERED"
            )
        
        # 獲取 Feature 實例
        feature_instance = self._feature_registry[request.feature_module]
        
        # 執行調用（帶超時控制）
        try:
            logger.info(
                f"Invoking feature {request.feature_module} "
                f"for target {request.target_url} "
                f"(timeout: {request.timeout_ms}ms)"
            )
            
            # 超時控制
            result = await asyncio.wait_for(
                self._execute_feature(feature_instance, request),
                timeout=request.timeout_ms / 1000.0
            )
            
            # 計算執行時間
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # 構建響應
            response = FeatureResponse(
                request_id=request.request_id,
                feature_module=request.feature_module,
                status=TaskStatus.COMPLETED,
                success=True,
                duration_ms=duration_ms,
                data=result.get("data", {}),
                findings=result.get("findings", []),
                performance_metrics=result.get("performance_metrics", {}),
                metadata=result.get("metadata", {})
            )
            
            self._stats["successful_requests"] += 1
            logger.info(
                f"Feature {request.feature_module} completed successfully "
                f"in {duration_ms:.2f}ms, {len(response.findings)} findings"
            )
            
            return response
            
        except asyncio.TimeoutError:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            error_msg = f"Feature {request.feature_module} timeout after {request.timeout_ms}ms"
            logger.error(error_msg)
            self._stats["failed_requests"] += 1
            
            return FeatureResponse(
                request_id=request.request_id,
                feature_module=request.feature_module,
                status=TaskStatus.FAILED,
                success=False,
                duration_ms=duration_ms,
                error_message=error_msg,
                error_code="TIMEOUT"
            )
            
        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            error_msg = f"Feature {request.feature_module} execution error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._stats["failed_requests"] += 1
            
            return FeatureResponse(
                request_id=request.request_id,
                feature_module=request.feature_module,
                status=TaskStatus.FAILED,
                success=False,
                duration_ms=duration_ms,
                error_message=error_msg,
                error_code="EXECUTION_ERROR"
            )
    
    async def _execute_feature(
        self, 
        feature_instance: Any, 
        request: FeatureRequest
    ) -> Dict[str, Any]:
        """執行 Feature（支持多種調用方式）
        
        Args:
            feature_instance: Feature 實例
            request: Feature 請求
        
        Returns:
            Feature 執行結果（字典格式）
        """
        # 判斷 Feature 類型並調用對應方法
        if hasattr(feature_instance, 'execute'):
            # Python Feature（標準 execute 方法）
            result = await feature_instance.execute(
                target_url=request.target_url,
                parameters=request.parameters
            )
        elif hasattr(feature_instance, 'scan'):
            # Scanner Feature（scan 方法）
            result = await feature_instance.scan(
                target=request.target_url,
                config=request.parameters
            )
        elif hasattr(feature_instance, 'run'):
            # Generic run 方法
            result = await feature_instance.run(request)
        else:
            raise AIVAError(
                message=f"Feature {request.feature_module} has no callable method (execute/scan/run)",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH
            )
        
        # 確保返回字典格式
        if isinstance(result, dict):
            return result
        elif hasattr(result, 'dict'):
            return result.dict()
        elif hasattr(result, '__dict__'):
            return result.__dict__
        else:
            return {"data": result}
    
    async def invoke_batch(
        self, 
        requests: List[FeatureRequest]
    ) -> List[FeatureResponse]:
        """批量調用 Features（並行執行）
        
        Args:
            requests: Feature 請求列表
        
        Returns:
            Feature 響應列表
        """
        logger.info(f"Invoking {len(requests)} features in batch")
        
        # 並行執行所有請求
        tasks = [self.invoke(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 處理異常結果
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 構建錯誤響應
                responses.append(FeatureResponse(
                    request_id=requests[i].request_id,
                    feature_module=requests[i].feature_module,
                    status=TaskStatus.FAILED,
                    success=False,
                    duration_ms=0.0,
                    error_message=str(result),
                    error_code="BATCH_EXECUTION_ERROR"
                ))
            else:
                responses.append(result)
        
        logger.info(
            f"Batch invocation completed: "
            f"{sum(r.success for r in responses)}/{len(responses)} successful"
        )
        
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        if self._stats["total_requests"] > 0:
            self._stats["success_rate"] = (
                self._stats["successful_requests"] / self._stats["total_requests"]
            )
        else:
            self._stats["success_rate"] = 0.0
        
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """重置統計信息"""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_duration_ms": 0.0
        }


# 全局單例實例（方便模組間共享）
_global_invoker: Optional[FeaturesInvoker] = None


def get_global_invoker() -> FeaturesInvoker:
    """獲取全局 FeaturesInvoker 實例"""
    global _global_invoker
    if _global_invoker is None:
        _global_invoker = FeaturesInvoker()
    return _global_invoker
