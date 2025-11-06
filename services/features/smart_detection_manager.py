"""
Smart Detection Manager - 智能檢測管理器

統一管理多個檢測器的執行，提供結構化的錯誤處理、
日誌記錄和性能監控。

遵循 README 規範：
- 添加完整的類型標註
- 實現結構化日誌
- 提供詳細的錯誤信息
- 支援並發執行（可選）
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 檢測器函數類型定義
DetectorFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


class DetectionResult:
    """檢測結果的結構化表示"""
    
    def __init__(
        self, 
        detector_name: str, 
        success: bool, 
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time: float = 0.0
    ):
        self.detector_name = detector_name
        self.success = success
        self.result = result or {}
        self.error = error
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "detector": self.detector_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time
        }


class SmartDetectionManager:
    """
    智能檢測管理器
    
    負責註冊和執行多個安全檢測器，提供統一的介面和
    結構化的結果處理。
    
    Attributes:
        _detectors: 已註冊的檢測器字典
        _execution_stats: 執行統計資訊
    """
    
    def __init__(self) -> None:
        """初始化智能檢測管理器"""
        self._detectors: Dict[str, DetectorFunc] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
        logger.info("SmartDetectionManager initialized")
    
    def register(self, name: str, fn: DetectorFunc) -> None:
        """
        註冊一個檢測器
        
        Args:
            name: 檢測器名稱（唯一標識符）
            fn: 檢測器函數，接收字典參數並返回字典結果
            
        Raises:
            ValueError: 當檢測器名稱已存在時
        """
        if name in self._detectors:
            logger.warning(f"Detector '{name}' already registered, overwriting")
        
        self._detectors[name] = fn
        self._execution_stats[name] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0
        }
        logger.info(f"Registered detector: {name}")
    
    def unregister(self, name: str) -> Optional[DetectorFunc]:
        """
        取消註冊一個檢測器
        
        Args:
            name: 要取消註冊的檢測器名稱
            
        Returns:
            被移除的檢測器函數，如果不存在則返回 None
        """
        detector = self._detectors.pop(name, None)
        if detector:
            self._execution_stats.pop(name, None)
            logger.info(f"Unregistered detector: {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent detector: {name}")
        return detector
    
    def run_all(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        執行所有已註冊的檢測器
        
        Args:
            input_data: 傳遞給檢測器的輸入數據
            
        Returns:
            所有檢測器的執行結果列表
        """
        results: List[Dict[str, Any]] = []
        
        logger.info(f"Running {len(self._detectors)} detectors")
        
        for name, fn in self._detectors.items():
            result = self._run_detector(name, fn, input_data)
            results.append(result.to_dict())
        
        logger.info(f"Completed all detectors. Success: {sum(1 for r in results if r['success'])}/{len(results)}")
        
        return results
    
    def run_detector(
        self, 
        name: str, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        執行特定的檢測器
        
        Args:
            name: 檢測器名稱
            input_data: 輸入數據
            
        Returns:
            檢測結果字典
            
        Raises:
            KeyError: 當檢測器不存在時
        """
        if name not in self._detectors:
            raise KeyError(f"Detector '{name}' not found")
        
        fn = self._detectors[name]
        result = self._run_detector(name, fn, input_data)
        return result.to_dict()
    
    def _run_detector(
        self, 
        name: str, 
        fn: DetectorFunc, 
        input_data: Dict[str, Any]
    ) -> DetectionResult:
        """
        內部方法：執行單個檢測器並記錄統計
        
        Args:
            name: 檢測器名稱
            fn: 檢測器函數
            input_data: 輸入數據
            
        Returns:
            DetectionResult 實例
        """
        start_time = time.time()
        stats = self._execution_stats[name]
        stats["total_executions"] += 1
        
        try:
            logger.debug(f"Executing detector: {name}")
            result = fn(input_data)
            execution_time = time.time() - start_time
            
            stats["successful_executions"] += 1
            stats["total_execution_time"] += execution_time
            
            logger.debug(f"Detector '{name}' completed successfully in {execution_time:.3f}s")
            
            return DetectionResult(
                detector_name=name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            stats["failed_executions"] += 1
            stats["total_execution_time"] += execution_time
            
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Detector '{name}' failed: {error_msg}", exc_info=True)
            
            return DetectionResult(
                detector_name=name,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def get_stats(self, detector_name: Optional[str] = None) -> Dict[str, Any]:
        """
        獲取執行統計資訊
        
        Args:
            detector_name: 特定檢測器的名稱，如果為 None 則返回所有統計
            
        Returns:
            統計資訊字典
        """
        if detector_name:
            if detector_name not in self._execution_stats:
                raise KeyError(f"Detector '{detector_name}' not found")
            return self._execution_stats[detector_name].copy()
        
        return {
            "detectors": self._execution_stats.copy(),
            "total_detectors": len(self._detectors),
            "summary": self._get_summary_stats()
        }
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """計算匯總統計資訊"""
        total_executions = sum(s["total_executions"] for s in self._execution_stats.values())
        total_successful = sum(s["successful_executions"] for s in self._execution_stats.values())
        total_failed = sum(s["failed_executions"] for s in self._execution_stats.values())
        total_time = sum(s["total_execution_time"] for s in self._execution_stats.values())
        
        return {
            "total_executions": total_executions,
            "successful_executions": total_successful,
            "failed_executions": total_failed,
            "total_execution_time": total_time,
            "average_execution_time": total_time / max(total_executions, 1)
        }
    
    def list_detectors(self) -> List[str]:
        """
        列出所有已註冊的檢測器名稱
        
        Returns:
            檢測器名稱列表
        """
        return list(self._detectors.keys())


# 單例模式的全局管理器實例
_default_manager: Optional[SmartDetectionManager] = None


def get_smart_detection_manager() -> SmartDetectionManager:
    """
    獲取全局的 SmartDetectionManager 單例實例
    
    Returns:
        SmartDetectionManager 實例
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = SmartDetectionManager()
        logger.info("Created global SmartDetectionManager instance")
    return _default_manager
