"""
統一智能檢測管理器 - 基於 SQLi 模組成功經驗
為所有功能模組提供統一的智能檢測能力
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from services.aiva_common.utils import get_logger

from .detection_config import BaseDetectionConfig

logger = get_logger(__name__)


@dataclass
class DetectionMetrics:
    """檢測指標統計 - 統一的性能監控"""

    total_requests: int = 0
    successful_requests: int = 0
    timeout_count: int = 0
    rate_limited_count: int = 0
    total_time: float = 0.0
    vulnerabilities_found: int = 0

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def avg_response_time(self) -> float:
        """平均回應時間"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_time / self.successful_requests


class AdaptiveTimeoutManager:
    """自適應超時管理器 - 統一版本"""

    def __init__(self, base_timeout: float = 5.0, max_timeout: float = 15.0):
        self.base_timeout = base_timeout
        self.max_timeout = max_timeout
        self.failure_count = 0
        self.success_count = 0
        self.recent_response_times: list[float] = []

    def get_timeout(self) -> float:
        """獲取當前應該使用的超時時間"""
        # 基於失敗次數調整
        if self.failure_count > 3:
            failure_multiplier = 1 + (self.failure_count - 3) * 0.3
            adaptive_timeout = min(
                self.max_timeout, self.base_timeout * failure_multiplier
            )
            return adaptive_timeout

        # 基於歷史回應時間調整
        if len(self.recent_response_times) >= 3:
            avg_time = sum(self.recent_response_times) / len(self.recent_response_times)
            # 設置為平均時間的2倍，但不超過最大值
            adaptive_timeout = max(
                self.base_timeout, min(self.max_timeout, avg_time * 2)
            )
            return adaptive_timeout

        return self.base_timeout

    def on_success(self, response_time: float):
        """記錄成功的請求"""
        self.success_count += 1
        self.failure_count = max(0, self.failure_count - 1)

        # 記錄最近的回應時間（保留最多10個）
        self.recent_response_times.append(response_time)
        if len(self.recent_response_times) > 10:
            self.recent_response_times.pop(0)

    def on_timeout(self):
        """記錄超時失敗"""
        self.failure_count += 1


class RateLimiter:
    """智能速率限制器 - 防止觸發目標防護機制"""

    def __init__(self, requests_per_second: float = 0.5):
        self.base_interval = 1.0 / requests_per_second
        self.current_interval = self.base_interval
        self.last_request_time = 0.0
        self.adaptive_factor = 1.0
        self.consecutive_rate_limits = 0

    async def wait(self):
        """等待合適的時間間隔"""
        now = time.time()
        elapsed = now - self.last_request_time
        required_interval = self.current_interval * self.adaptive_factor

        if elapsed < required_interval:
            wait_time = required_interval - elapsed
            if wait_time > 0.1:  # 只對較長的等待時間記錄日志
                logger.debug(f"速率控制等待: {wait_time:.2f}秒")
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    def on_rate_limit_detected(self):
        """檢測到速率限制時的處理"""
        self.consecutive_rate_limits += 1
        self.adaptive_factor = min(5.0, self.adaptive_factor * 1.5)
        logger.warning(
            f"檢測到速率限制 (連續{self.consecutive_rate_limits}次)，"
            f"調整間隔係數至 {self.adaptive_factor:.2f}"
        )

    def on_success(self):
        """成功請求時逐步恢復正常速率"""
        if self.consecutive_rate_limits > 0:
            self.consecutive_rate_limits = max(0, self.consecutive_rate_limits - 1)
        self.adaptive_factor = max(1.0, self.adaptive_factor * 0.95)


class EarlyStopController:
    """早期停止控制器 - 避免過度測試"""

    def __init__(self, max_vulnerabilities: int = 3, max_consecutive_failures: int = 5):
        self.max_vulnerabilities = max_vulnerabilities
        self.max_consecutive_failures = max_consecutive_failures
        self.vulnerabilities_found = 0
        self.consecutive_failures = 0
        self.total_tests = 0
        self.should_stop = False

    def should_continue(self) -> bool:
        """判斷是否應該繼續檢測"""
        if self.should_stop:
            return False

        # 如果發現足夠的漏洞
        if self.vulnerabilities_found >= self.max_vulnerabilities:
            logger.info(f"已發現 {self.vulnerabilities_found} 個漏洞，達到設定上限")
            self.should_stop = True
            return False

        # 如果連續失敗太多次
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"連續失敗 {self.consecutive_failures} 次，可能遇到防護機制")
            self.should_stop = True
            return False

        return True

    def on_vulnerability_found(self):
        """記錄發現漏洞"""
        self.vulnerabilities_found += 1
        self.consecutive_failures = 0  # 重置失敗計數
        logger.info(f"發現漏洞 #{self.vulnerabilities_found}")

    def on_test_failure(self):
        """記錄測試失敗"""
        self.consecutive_failures += 1
        self.total_tests += 1

    def on_test_success_no_vuln(self):
        """記錄測試成功但無漏洞"""
        self.consecutive_failures = 0
        self.total_tests += 1

    def force_stop(self, reason: str = ""):
        """強制停止檢測"""
        self.should_stop = True
        if reason:
            logger.info(f"強制停止檢測: {reason}")


class ProgressTracker:
    """統一進度追蹤器"""

    def __init__(self, total_steps: int, task_name: str = "檢測"):
        self.total_steps = total_steps
        self.current_step = 0
        self.task_name = task_name
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.update_interval = 2.0  # 最少間隔2秒更新一次

    def update(self, step_description: str = "", force_update: bool = False):
        """更新進度"""
        self.current_step += 1
        now = time.time()

        # 控制更新頻率
        if (
            not force_update
            and now - self.last_update_time < self.update_interval
            and self.current_step < self.total_steps
        ):
            return

        elapsed = now - self.start_time
        progress_percent = (self.current_step / self.total_steps) * 100

        if self.current_step > 0 and self.current_step < self.total_steps:
            # 估算剩餘時間
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_time_per_step

            logger.info(
                f"{self.task_name}: {progress_percent:.1f}% "
                f"({self.current_step}/{self.total_steps}) | "
                f"{step_description} | "
                f"預計剩餘: {eta:.1f}秒"
            )
        elif self.current_step >= self.total_steps:
            total_time = elapsed
            logger.info(f"{self.task_name} 完成! 總耗時: {total_time:.1f}秒")

        self.last_update_time = now


class ProtectionDetector:
    """統一防護機制檢測器"""

    @staticmethod
    async def analyze_response(response, response_time: float) -> dict[str, Any]:
        """分析回應以檢測各種防護機制"""
        protection_signals: dict[str, Any] = {
            "waf_detected": False,
            "rate_limited": False,
            "blocking_detected": False,
            "slow_response": False,
            "protection_type": "none",
            "confidence": 0.0,
        }

        # 檢測 WAF 狀態碼
        waf_status_codes = [403, 406, 418, 429, 503, 509]
        if response.status_code in waf_status_codes:
            protection_signals["waf_detected"] = True
            protection_signals["confidence"] += 0.3

        # 檢測速率限制
        if response.status_code == 429:
            protection_signals["rate_limited"] = True
            protection_signals["protection_type"] = "rate_limiting"
            protection_signals["confidence"] += 0.5

        # 檢測封鎖
        if response.status_code in [403, 503]:
            protection_signals["blocking_detected"] = True
            protection_signals["protection_type"] = "blocking"
            protection_signals["confidence"] += 0.4

        # 檢測異常回應時間
        if response_time > 10.0:
            protection_signals["slow_response"] = True
            protection_signals["confidence"] += 0.2

        # 檢測回應內容中的 WAF 特徵
        try:
            response_text = response.text.lower()
            waf_signatures = [
                ("cloudflare", 0.8),
                ("access denied", 0.6),
                ("blocked", 0.5),
                ("firewall", 0.7),
                ("security policy", 0.6),
                ("suspicious activity", 0.5),
                ("rate limit", 0.7),
                ("mod_security", 0.9),
                ("imperva", 0.9),
                ("f5 big-ip", 0.9),
            ]

            for signature, weight in waf_signatures:
                if signature in response_text:
                    protection_signals["waf_detected"] = True
                    protection_signals["protection_type"] = "waf"
                    protection_signals["confidence"] += weight
                    break

        except Exception:
            # 如果無法讀取回應內容，跳過內容分析
            pass

        # 標準化置信度
        protection_signals["confidence"] = min(1.0, protection_signals["confidence"])

        return protection_signals


class UnifiedSmartDetectionManager:
    """統一智能檢測管理器 - 適用於所有功能模組"""

    def __init__(self, module_name: str, config: BaseDetectionConfig):
        self.module_name = module_name
        self.config = config

        # 初始化各個組件
        self.timeout_manager = AdaptiveTimeoutManager(
            config.timeout_base, config.timeout_max
        )
        self.rate_limiter = RateLimiter(config.requests_per_second)
        self.early_stop = EarlyStopController(
            config.max_vulnerabilities, config.max_consecutive_failures
        )
        self.progress: ProgressTracker | None = None
        self.metrics = DetectionMetrics()
        self.protection_detector = ProtectionDetector()

    def start_detection(self, total_steps: int):
        """開始檢測並初始化進度追蹤"""
        self.progress = ProgressTracker(total_steps, f"{self.module_name}檢測")
        logger.info(f"開始 {self.module_name} 檢測，預計 {total_steps} 個步驟")

    async def execute_smart_request(
        self, request_func: Callable, *args, **kwargs
    ) -> tuple[Any, dict[str, Any]]:
        """執行智能請求 - 包含所有優化功能"""
        # 速率控制
        await self.rate_limiter.wait()

        # 獲取動態超時
        timeout = self.timeout_manager.get_timeout()

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            # 執行請求
            response = await asyncio.wait_for(
                request_func(*args, **kwargs), timeout=timeout
            )

            response_time = time.time() - start_time
            self.metrics.successful_requests += 1
            self.metrics.total_time += response_time

            # 記錄成功
            self.timeout_manager.on_success(response_time)
            self.rate_limiter.on_success()

            # 檢測防護機制
            protection_signals = await self.protection_detector.analyze_response(
                response, response_time
            )

            # 處理檢測到的防護機制
            if protection_signals["rate_limited"]:
                self.rate_limiter.on_rate_limit_detected()
                self.metrics.rate_limited_count += 1

            return response, protection_signals

        except TimeoutError:
            self.metrics.timeout_count += 1
            self.timeout_manager.on_timeout()
            self.early_stop.on_test_failure()

            logger.warning(f"請求超時 ({timeout:.1f}秒)")
            raise

        except Exception as e:
            self.early_stop.on_test_failure()
            logger.error(f"請求失敗: {e}")
            raise

    def should_continue_testing(self) -> bool:
        """檢查是否應該繼續測試"""
        return self.early_stop.should_continue()

    def report_vulnerability_found(self):
        """報告發現漏洞"""
        self.early_stop.on_vulnerability_found()
        self.metrics.vulnerabilities_found += 1

    def update_progress(self, description: str = "", force_update: bool = False):
        """更新進度"""
        if self.progress:
            self.progress.update(description, force_update)

    def get_performance_summary(self) -> str:
        """獲取性能摘要報告"""
        return (
            f"\n{self.module_name} 檢測完成統計:\n"
            f"  總請求數: {self.metrics.total_requests}\n"
            f"  成功率: {self.metrics.success_rate:.1f}%\n"
            f"  平均回應時間: {self.metrics.avg_response_time:.2f}秒\n"
            f"  超時次數: {self.metrics.timeout_count}\n"
            f"  速率限制次數: {self.metrics.rate_limited_count}\n"
            f"  發現漏洞數: {self.metrics.vulnerabilities_found}\n"
            f"  總檢測時間: {self.metrics.total_time:.1f}秒"
        )

    def force_stop(self, reason: str = ""):
        """強制停止檢測"""
        self.early_stop.force_stop(reason)
        if self.progress:
            logger.info(f"{self.module_name} 檢測被強制停止: {reason}")
