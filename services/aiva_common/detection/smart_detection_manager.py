"""
Unified Smart Detection Manager

統一智能檢測管理器
整合自適應超時、速率限制、早期停止、進度追蹤等功能

基於以下最佳實踐:
- Python asyncio patterns
- Token Bucket rate limiting
- Adaptive timeout management
- Circuit Breaker pattern
- Progressive metrics collection
"""

import asyncio
import httpx
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Awaitable
from enum import Enum

from .timeout_manager import AdaptiveTimeoutManager, TimeoutConfig
from .rate_limiter import SmartRateLimiter, RateLimitConfig
from .metrics_collector import MetricsCollector, DetectionPhase, DetectionMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StopCondition(str, Enum):
    """早期停止條件"""
    MAX_ERRORS = "max_errors"  # 錯誤數過多
    MAX_TIMEOUTS = "max_timeouts"  # 超時數過多
    VULNERABILITY_FOUND = "vulnerability_found"  # 已發現漏洞
    PROGRESS_STALLED = "progress_stalled"  # 進度停滯
    USER_CANCELLED = "user_cancelled"  # 用戶取消


@dataclass
class DetectionConfig:
    """檢測配置"""
    
    # 超時配置
    timeout_config: Optional[TimeoutConfig] = None
    
    # 速率限制配置
    rate_limit_config: Optional[RateLimitConfig] = None
    
    # 早期停止配置
    max_consecutive_errors: int = 10  # 連續錯誤數閾值
    max_error_rate: float = 0.5  # 錯誤率閾值 (0-1)
    max_timeout_rate: float = 0.3  # 超時率閾值 (0-1)
    stop_on_first_vulnerability: bool = False  # 發現首個漏洞後停止
    progress_stall_threshold: int = 50  # 進度停滯閾值(請求數)
    
    # 並發控制
    max_concurrent_requests: int = 10  # 最大並發請求數
    
    # HTTP 配置
    follow_redirects: bool = True
    verify_ssl: bool = True
    default_headers: Optional[Dict[str, str]] = None


class UnifiedSmartDetectionManager:
    """
    統一智能檢測管理器
    
    提供以下核心功能:
    1. 自適應超時管理 - 根據歷史響應時間動態調整
    2. 智能速率限制 - Token Bucket + 自適應調整
    3. 早期停止機制 - 多種停止條件檢測
    4. 進度追蹤 - 實時進度和指標收集
    5. 智能請求執行 - 錯誤處理、重試、熔斷
    6. 並發控制 - Semaphore 限制並發數
    
    使用示例:
    ```python
    manager = UnifiedSmartDetectionManager(
        detection_id="xss_001",
        detection_type="xss",
        target_url="https://example.com"
    )
    
    async with manager.start_detection():
        for payload in payloads:
            if manager.should_continue_testing():
                result = await manager.execute_smart_request(
                    lambda: client.post(url, data={"input": payload})
                )
                
                if is_vulnerable(result):
                    manager.report_vulnerability_found()
    
    metrics = manager.get_metrics()
    ```
    """
    
    def __init__(
        self,
        detection_id: str,
        detection_type: str,
        target_url: str,
        config: Optional[DetectionConfig] = None
    ):
        """
        初始化統一智能檢測管理器
        
        Args:
            detection_id: 檢測唯一標識
            detection_type: 檢測類型 (xss, sqli, idor, etc.)
            target_url: 目標URL
            config: 檢測配置
        """
        self.detection_id = detection_id
        self.detection_type = detection_type
        self.target_url = target_url
        self.config = config or DetectionConfig()
        
        # 初始化子組件
        self.timeout_manager = AdaptiveTimeoutManager(
            self.config.timeout_config or TimeoutConfig()
        )
        self.rate_limiter = SmartRateLimiter(
            self.config.rate_limit_config or RateLimitConfig()
        )
        self.metrics = MetricsCollector(detection_id, detection_type, target_url)
        
        # 並發控制
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # 早期停止狀態
        self._should_stop = False
        self._stop_reason: Optional[StopCondition] = None
        self._consecutive_errors = 0
        self._last_progress_requests = 0
        
        logger.info(
            f"UnifiedSmartDetectionManager initialized: "
            f"id={detection_id}, type={detection_type}, target={target_url}"
        )
    
    @asynccontextmanager
    async def start_detection(self):
        """
        啟動檢測會話（上下文管理器）
        
        使用方式:
        ```python
        async with manager.start_detection():
            # 執行檢測邏輯
            pass
        ```
        """
        logger.info(f"Starting detection: {self.detection_id}")
        self.metrics.start_phase(DetectionPhase.INITIALIZATION)
        
        try:
            yield self
        finally:
            # 自動完成指標收集
            final_metrics = self.metrics.finalize()
            logger.info(
                f"Detection completed: {self.detection_id}, "
                f"duration={final_metrics.total_duration:.2f}s, "
                f"requests={final_metrics.total_requests}, "
                f"vulnerabilities={final_metrics.vulnerabilities_found}"
            )
    
    async def execute_smart_request(
        self,
        request_func: Callable[[], Awaitable[httpx.Response]],
        payload_info: Optional[str] = None,
        retries: int = 3
    ) -> Optional[httpx.Response]:
        """
        智能請求執行
        
        整合以下功能:
        - 速率限制
        - 自適應超時
        - 錯誤處理和重試
        - 指標收集
        - 早期停止檢測
        
        Args:
            request_func: 請求執行函數（返回 httpx.Response）
            payload_info: Payload 信息（用於日誌）
            retries: 重試次數
        
        Returns:
            HTTP 響應對象，失敗返回 None
        """
        # 檢查是否應該停止
        if not self.should_continue_testing():
            logger.debug(f"Skipping request due to stop condition: {self._stop_reason}")
            return None
        
        # 並發控制
        async with self._semaphore:
            # 速率限制
            await self.rate_limiter.acquire()
            
            # 重試邏輯
            for attempt in range(retries):
                try:
                    # 使用自適應超時執行請求
                    response = await self.timeout_manager.execute_with_timeout(
                        request_func(),
                        task_name=f"{self.detection_type}_request"
                    )
                    
                    # 記錄成功
                    self.metrics.record_request(
                        success=True,
                        duration=response.elapsed.total_seconds()
                    )
                    self.rate_limiter.record_success()
                    self._consecutive_errors = 0  # 重置連續錯誤計數
                    
                    # 處理 429 (Too Many Requests)
                    if response.status_code == 429:
                        self.rate_limiter.record_429_error()
                        logger.warning("Received 429 response, adjusting rate limit")
                        await asyncio.sleep(5)  # 額外等待
                        continue
                    
                    return response
                
                except TimeoutError:
                    self.metrics.record_timeout()
                    self._check_timeout_rate()
                    
                    if attempt < retries - 1:
                        logger.warning(f"Request timeout, retrying ({attempt + 1}/{retries})")
                        await asyncio.sleep(2 ** attempt)  # 指數退避
                    else:
                        logger.error(f"Request failed after {retries} timeout attempts")
                        return None
                
                except httpx.HTTPError as e:
                    self.metrics.record_request(
                        success=False,
                        duration=0.0,
                        error_type=type(e).__name__
                    )
                    self._consecutive_errors += 1
                    self._check_error_rate()
                    
                    if attempt < retries - 1:
                        logger.warning(
                            f"HTTP error: {e}, retrying ({attempt + 1}/{retries})"
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Request failed after {retries} attempts: {e}")
                        return None
                
                except Exception as e:
                    logger.exception(f"Unexpected error in execute_smart_request: {e}")
                    self.metrics.record_request(
                        success=False,
                        duration=0.0,
                        error_type="unexpected_error"
                    )
                    return None
        
        return None
    
    def should_continue_testing(self) -> bool:
        """
        檢查是否應該繼續測試
        
        檢查各種早期停止條件
        
        Returns:
            True 如果應該繼續，False 如果應該停止
        """
        if self._should_stop:
            return False
        
        # 檢查進度停滯
        current_requests = self.metrics.metrics.total_requests
        if current_requests - self._last_progress_requests > self.config.progress_stall_threshold:
            if self.metrics.metrics.vulnerabilities_found == 0:
                self._trigger_stop(StopCondition.PROGRESS_STALLED)
                return False
            self._last_progress_requests = current_requests
        
        return True
    
    def _check_error_rate(self) -> None:
        """檢查錯誤率是否過高"""
        # 連續錯誤檢查
        if self._consecutive_errors >= self.config.max_consecutive_errors:
            self._trigger_stop(StopCondition.MAX_ERRORS)
            return
        
        # 總體錯誤率檢查
        total = self.metrics.metrics.total_requests
        if total > 20:  # 至少20個請求後才檢查
            error_rate = self.metrics.metrics.failed_requests / total
            if error_rate > self.config.max_error_rate:
                self._trigger_stop(StopCondition.MAX_ERRORS)
    
    def _check_timeout_rate(self) -> None:
        """檢查超時率是否過高"""
        total = self.metrics.metrics.total_requests
        if total > 20:
            timeout_rate = self.metrics.metrics.timeout_requests / total
            if timeout_rate > self.config.max_timeout_rate:
                self._trigger_stop(StopCondition.MAX_TIMEOUTS)
    
    def _trigger_stop(self, reason: StopCondition) -> None:
        """
        觸發早期停止
        
        Args:
            reason: 停止原因
        """
        self._should_stop = True
        self._stop_reason = reason
        self.metrics.mark_early_stopped(reason.value)
        
        logger.warning(
            f"Early stop triggered: {reason.value}, "
            f"requests={self.metrics.metrics.total_requests}, "
            f"errors={self.metrics.metrics.failed_requests}, "
            f"timeouts={self.metrics.metrics.timeout_requests}"
        )
    
    def report_vulnerability_found(self, is_false_positive: bool = False) -> None:
        """
        報告發現漏洞
        
        Args:
            is_false_positive: 是否為誤報
        """
        self.metrics.record_vulnerability(is_false_positive)
        
        # 如果配置了發現首個漏洞後停止
        if (
            not is_false_positive
            and self.config.stop_on_first_vulnerability
            and self.metrics.metrics.vulnerabilities_found == 1
        ):
            self._trigger_stop(StopCondition.VULNERABILITY_FOUND)
    
    def update_progress(
        self,
        current: int,
        total: int,
        phase: Optional[DetectionPhase] = None
    ) -> None:
        """
        更新進度
        
        Args:
            current: 當前進度
            total: 總數
            phase: 當前階段（可選）
        """
        if phase:
            self.metrics.start_phase(phase)
        
        percentage = (current / total * 100) if total > 0 else 0.0
        
        # 估算剩餘時間
        if current > 0 and self.metrics.metrics.avg_request_time > 0:
            remaining_requests = total - current
            estimated_time = remaining_requests * self.metrics.metrics.avg_request_time
        else:
            estimated_time = None
        
        self.metrics.update_progress(percentage, estimated_time)
    
    def get_metrics(self) -> DetectionMetrics:
        """
        獲取當前指標
        
        Returns:
            當前的檢測指標
        """
        return self.metrics.get_current_metrics()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取所有組件的統計信息
        
        Returns:
            統計信息字典
        """
        return {
            "detection": {
                "id": self.detection_id,
                "type": self.detection_type,
                "target": self.target_url,
                "should_stop": self._should_stop,
                "stop_reason": self._stop_reason.value if self._stop_reason else None,
            },
            "metrics": self.metrics.get_current_metrics().to_dict(),
            "timeout": self.timeout_manager.get_stats(),
            "rate_limiter": self.rate_limiter.get_stats(),
        }
    
    def request_stop(self) -> None:
        """請求停止檢測（用戶取消）"""
        self._trigger_stop(StopCondition.USER_CANCELLED)
