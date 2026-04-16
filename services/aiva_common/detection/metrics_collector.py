"""
Detection Metrics Collector

檢測指標收集器，用於統計和監控檢測過程
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DetectionPhase(str, Enum):
    """檢測階段"""
    INITIALIZATION = "initialization"
    RECONNAISSANCE = "reconnaissance"
    PAYLOAD_INJECTION = "payload_injection"
    RESPONSE_ANALYSIS = "response_analysis"
    VALIDATION = "validation"
    REPORTING = "reporting"


@dataclass
class DetectionMetrics:
    """
    檢測指標
    
    記錄檢測過程中的各項指標
    """

    # 基本信息
    detection_id: str
    detection_type: str  # xss, sqli, idor, ssrf, etc.
    target_url: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # 執行統計
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    rate_limited_requests: int = 0

    # 檢測結果
    vulnerabilities_found: int = 0
    false_positives: int = 0

    # 性能指標
    total_duration: float = 0.0  # 總執行時間(秒)
    avg_request_time: float = 0.0  # 平均請求時間
    max_request_time: float = 0.0  # 最大請求時間
    min_request_time: float = float('inf')  # 最小請求時間

    # 階段計時
    phase_durations: dict[str, float] = field(default_factory=dict)

    # 錯誤統計
    error_counts: dict[str, int] = field(default_factory=dict)

    # 進度信息
    current_phase: DetectionPhase | None = None
    progress_percentage: float = 0.0
    estimated_remaining_time: float | None = None

    # 標記
    completed_at: datetime | None = None
    early_stopped: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            "detection_id": self.detection_id,
            "detection_type": self.detection_type,
            "target_url": self.target_url,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "timeout_requests": self.timeout_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "vulnerabilities_found": self.vulnerabilities_found,
            "false_positives": self.false_positives,
            "total_duration": self.total_duration,
            "avg_request_time": self.avg_request_time,
            "max_request_time": self.max_request_time,
            "min_request_time": self.min_request_time if self.min_request_time != float('inf') else None,
            "phase_durations": self.phase_durations,
            "error_counts": self.error_counts,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "progress_percentage": self.progress_percentage,
            "estimated_remaining_time": self.estimated_remaining_time,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
        }


class MetricsCollector:
    """
    指標收集器
    
    收集和聚合檢測過程中的各項指標
    """

    def __init__(self, detection_id: str, detection_type: str, target_url: str):
        """
        初始化指標收集器
        
        Args:
            detection_id: 檢測ID
            detection_type: 檢測類型
            target_url: 目標URL
        """
        self.metrics = DetectionMetrics(
            detection_id=detection_id,
            detection_type=detection_type,
            target_url=target_url
        )

        self._phase_start_time: float | None = None
        self._request_times: list[float] = []

        logger.debug(f"MetricsCollector initialized for {detection_id}")

    def start_phase(self, phase: DetectionPhase) -> None:
        """
        開始新的檢測階段
        
        Args:
            phase: 檢測階段
        """
        # 結束上一個階段
        if self.metrics.current_phase and self._phase_start_time:
            duration = time.perf_counter() - self._phase_start_time
            self.metrics.phase_durations[self.metrics.current_phase.value] = duration

        # 開始新階段
        self.metrics.current_phase = phase
        self._phase_start_time = time.perf_counter()

        logger.debug(f"Started phase: {phase.value}")

    def record_request(
        self,
        success: bool,
        duration: float,
        error_type: str | None = None
    ) -> None:
        """
        記錄請求
        
        Args:
            success: 請求是否成功
            duration: 請求耗時(秒)
            error_type: 錯誤類型(如果失敗)
        """
        self.metrics.total_requests += 1

        if success:
            self.metrics.successful_requests += 1
            self._request_times.append(duration)

            # 更新統計
            self.metrics.max_request_time = max(self.metrics.max_request_time, duration)
            self.metrics.min_request_time = min(self.metrics.min_request_time, duration)
            self.metrics.avg_request_time = sum(self._request_times) / len(self._request_times)
        else:
            self.metrics.failed_requests += 1

            if error_type:
                self.metrics.error_counts[error_type] = (
                    self.metrics.error_counts.get(error_type, 0) + 1
                )

    def record_timeout(self) -> None:
        """記錄超時"""
        self.metrics.timeout_requests += 1
        self.record_request(success=False, duration=0.0, error_type="timeout")

    def record_rate_limited(self) -> None:
        """記錄速率限制"""
        self.metrics.rate_limited_requests += 1
        self.record_request(success=False, duration=0.0, error_type="rate_limited")

    def record_vulnerability(self, is_false_positive: bool = False) -> None:
        """
        記錄漏洞發現
        
        Args:
            is_false_positive: 是否為誤報
        """
        if is_false_positive:
            self.metrics.false_positives += 1
        else:
            self.metrics.vulnerabilities_found += 1

        logger.info(
            f"Vulnerability recorded: "
            f"total={self.metrics.vulnerabilities_found}, "
            f"false_positives={self.metrics.false_positives}"
        )

    def update_progress(
        self,
        progress_percentage: float,
        estimated_remaining_time: float | None = None
    ) -> None:
        """
        更新進度信息
        
        Args:
            progress_percentage: 進度百分比 (0-100)
            estimated_remaining_time: 預計剩餘時間(秒)
        """
        self.metrics.progress_percentage = progress_percentage
        self.metrics.estimated_remaining_time = estimated_remaining_time

    def mark_early_stopped(self, reason: str) -> None:
        """
        標記為提前停止
        
        Args:
            reason: 停止原因
        """
        self.metrics.early_stopped = True
        self.metrics.stop_reason = reason
        logger.info(f"Detection early stopped: {reason}")

    def finalize(self) -> DetectionMetrics:
        """
        完成指標收集
        
        Returns:
            最終的指標對象
        """
        # 結束當前階段
        if self.metrics.current_phase and self._phase_start_time:
            duration = time.perf_counter() - self._phase_start_time
            self.metrics.phase_durations[self.metrics.current_phase.value] = duration

        # 記錄完成時間
        self.metrics.completed_at = datetime.now(UTC)

        # 計算總執行時間
        self.metrics.total_duration = (
            self.metrics.completed_at - self.metrics.started_at
        ).total_seconds()

        logger.info(
            f"Metrics finalized: duration={self.metrics.total_duration:.2f}s, "
            f"requests={self.metrics.total_requests}, "
            f"vulnerabilities={self.metrics.vulnerabilities_found}"
        )

        return self.metrics

    def get_current_metrics(self) -> DetectionMetrics:
        """
        獲取當前指標(不結束收集)
        
        Returns:
            當前的指標對象
        """
        return self.metrics
