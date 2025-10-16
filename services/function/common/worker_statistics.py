"""
Enhanced Worker 統計數據收集模組

提供統一的統計數據 Schema 和收集 API，支持：
- OAST 回調統計
- 錯誤收集和分類
- Early stopping 指標
- 檢測效率分析
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """錯誤分類"""

    NETWORK = "network"  # 網絡錯誤
    TIMEOUT = "timeout"  # 超時錯誤
    RATE_LIMIT = "rate_limit"  # 速率限制
    VALIDATION = "validation"  # 驗證錯誤
    PROTECTION = "protection"  # 保護機制檢測到
    PARSING = "parsing"  # 解析錯誤
    UNKNOWN = "unknown"  # 未知錯誤


class StoppingReason(str, Enum):
    """Early Stopping 原因"""

    MAX_VULNERABILITIES = "max_vulnerabilities_reached"  # 達到最大漏洞數
    TIME_LIMIT = "time_limit_exceeded"  # 超過時間限制
    PROTECTION_DETECTED = "protection_detected"  # 檢測到防護
    ERROR_THRESHOLD = "error_threshold_exceeded"  # 錯誤率過高
    RATE_LIMITED = "rate_limited"  # 被速率限制
    NO_RESPONSE = "no_response_timeout"  # 無響應超時


@dataclass
class ErrorRecord:
    """錯誤記錄"""

    category: ErrorCategory
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    request_info: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "category": self.category.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "request_info": self.request_info,
            "stack_trace": self.stack_trace,
        }


@dataclass
class OastCallbackRecord:
    """OAST 回調記錄"""

    probe_token: str
    callback_type: str  # "dns", "http", "smtp", etc.
    source_ip: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload_info: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "probe_token": self.probe_token,
            "callback_type": self.callback_type,
            "source_ip": self.source_ip,
            "timestamp": self.timestamp.isoformat(),
            "payload_info": self.payload_info,
            "success": self.success,
        }


@dataclass
class EarlyStoppingRecord:
    """Early Stopping 記錄"""

    reason: StoppingReason
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    requests_made: int = 0
    findings_found: int = 0
    time_elapsed: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "reason": self.reason.value,
            "triggered_at": self.triggered_at.isoformat(),
            "requests_made": self.requests_made,
            "findings_found": self.findings_found,
            "time_elapsed": self.time_elapsed,
            "details": self.details,
        }


class WorkerStatistics(BaseModel):
    """Worker 統計數據統一 Schema"""

    # 基礎統計
    task_id: str
    worker_type: str  # "ssrf", "idor", "sqli", "xss"
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # 請求統計
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    rate_limited_requests: int = 0

    # 檢測結果
    vulnerabilities_found: int = 0
    false_positives_filtered: int = 0
    payloads_tested: int = 0
    payloads_succeeded: int = 0

    # OAST 統計
    oast_probes_sent: int = 0
    oast_callbacks_received: int = 0
    oast_callback_details: list[dict[str, Any]] = Field(default_factory=list)

    # 錯誤統計
    error_count: int = 0
    errors_by_category: dict[str, int] = Field(default_factory=dict)
    error_details: list[dict[str, Any]] = Field(default_factory=list)

    # Early Stopping
    early_stopping_triggered: bool = False
    early_stopping_reason: str | None = None
    early_stopping_details: dict[str, Any] | None = None

    # 自適應行為
    adaptive_timeout_used: bool = False
    timeout_adjustments: int = 0
    rate_limiting_applied: bool = False
    protection_detected: bool = False

    # 模組特定統計（可擴展）
    module_specific: dict[str, Any] = Field(default_factory=dict)

    def add_error(self, error: ErrorRecord) -> None:
        """添加錯誤記錄"""
        self.error_count += 1
        category_key = error.category.value
        self.errors_by_category[category_key] = (
            self.errors_by_category.get(category_key, 0) + 1
        )
        self.error_details.append(error.to_dict())

    def add_oast_callback(self, callback: OastCallbackRecord) -> None:
        """添加 OAST 回調記錄"""
        self.oast_callbacks_received += 1
        self.oast_callback_details.append(callback.to_dict())

    def record_early_stopping(self, record: EarlyStoppingRecord) -> None:
        """記錄 Early Stopping 事件"""
        self.early_stopping_triggered = True
        self.early_stopping_reason = record.reason.value
        self.early_stopping_details = record.to_dict()

    def finalize(self) -> None:
        """完成統計數據收集"""
        self.end_time = datetime.now(UTC)
        if self.start_time:
            self.duration_seconds = (
                self.end_time - self.start_time
            ).total_seconds()

    def to_summary(self) -> dict[str, Any]:
        """生成摘要報告"""
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0.0
        )

        payload_success_rate = (
            (self.payloads_succeeded / self.payloads_tested * 100)
            if self.payloads_tested > 0
            else 0.0
        )

        oast_success_rate = (
            (self.oast_callbacks_received / self.oast_probes_sent * 100)
            if self.oast_probes_sent > 0
            else 0.0
        )

        return {
            "task_id": self.task_id,
            "worker_type": self.worker_type,
            "duration_seconds": self.duration_seconds,
            "performance": {
                "total_requests": self.total_requests,
                "success_rate": round(success_rate, 2),
                "requests_per_second": (
                    round(self.total_requests / self.duration_seconds, 2)
                    if self.duration_seconds > 0
                    else 0.0
                ),
            },
            "detection": {
                "vulnerabilities_found": self.vulnerabilities_found,
                "payloads_tested": self.payloads_tested,
                "payload_success_rate": round(payload_success_rate, 2),
                "false_positives_filtered": self.false_positives_filtered,
            },
            "oast": {
                "probes_sent": self.oast_probes_sent,
                "callbacks_received": self.oast_callbacks_received,
                "success_rate": round(oast_success_rate, 2),
            },
            "errors": {
                "total": self.error_count,
                "by_category": self.errors_by_category,
                "rate": (
                    round(self.error_count / self.total_requests * 100, 2)
                    if self.total_requests > 0
                    else 0.0
                ),
            },
            "adaptive_behavior": {
                "early_stopping": self.early_stopping_triggered,
                "stopping_reason": self.early_stopping_reason,
                "adaptive_timeout": self.adaptive_timeout_used,
                "rate_limiting": self.rate_limiting_applied,
                "protection_detected": self.protection_detected,
            },
        }


class StatisticsCollector:
    """統計數據收集器"""

    def __init__(self, task_id: str, worker_type: str):
        """
        初始化統計數據收集器

        Args:
            task_id: 任務 ID
            worker_type: Worker 類型（"ssrf", "idor", "sqli", "xss"）
        """
        self.stats = WorkerStatistics(task_id=task_id, worker_type=worker_type)

    def record_request(
        self,
        success: bool = True,
        timeout: bool = False,
        rate_limited: bool = False,
    ) -> None:
        """記錄請求"""
        self.stats.total_requests += 1
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        if timeout:
            self.stats.timeout_requests += 1
        if rate_limited:
            self.stats.rate_limited_requests += 1

    def record_payload_test(self, success: bool = False) -> None:
        """記錄 Payload 測試"""
        self.stats.payloads_tested += 1
        if success:
            self.stats.payloads_succeeded += 1

    def record_vulnerability(self, false_positive: bool = False) -> None:
        """記錄漏洞發現"""
        if false_positive:
            self.stats.false_positives_filtered += 1
        else:
            self.stats.vulnerabilities_found += 1

    def record_oast_probe(self) -> None:
        """記錄 OAST 探針發送"""
        self.stats.oast_probes_sent += 1

    def record_oast_callback(
        self,
        probe_token: str,
        callback_type: str,
        source_ip: str,
        payload_info: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """記錄 OAST 回調"""
        callback = OastCallbackRecord(
            probe_token=probe_token,
            callback_type=callback_type,
            source_ip=source_ip,
            payload_info=payload_info or {},
            success=success,
        )
        self.stats.add_oast_callback(callback)

    def record_error(
        self,
        category: ErrorCategory,
        message: str,
        request_info: dict[str, Any] | None = None,
        stack_trace: str | None = None,
    ) -> None:
        """記錄錯誤"""
        error = ErrorRecord(
            category=category,
            message=message,
            request_info=request_info or {},
            stack_trace=stack_trace,
        )
        self.stats.add_error(error)

    def record_early_stopping(
        self,
        reason: StoppingReason,
        details: dict[str, Any] | None = None,
    ) -> None:
        """記錄 Early Stopping"""
        record = EarlyStoppingRecord(
            reason=reason,
            requests_made=self.stats.total_requests,
            findings_found=self.stats.vulnerabilities_found,
            time_elapsed=self.stats.duration_seconds,
            details=details or {},
        )
        self.stats.record_early_stopping(record)

    def set_adaptive_behavior(
        self,
        adaptive_timeout: bool = False,
        rate_limiting: bool = False,
        protection_detected: bool = False,
    ) -> None:
        """設置自適應行為標記"""
        if adaptive_timeout:
            self.stats.adaptive_timeout_used = True
            self.stats.timeout_adjustments += 1
        if rate_limiting:
            self.stats.rate_limiting_applied = True
        if protection_detected:
            self.stats.protection_detected = True

    def set_module_specific(self, key: str, value: Any) -> None:
        """設置模組特定統計數據"""
        self.stats.module_specific[key] = value

    def get_statistics(self) -> WorkerStatistics:
        """獲取統計數據"""
        return self.stats

    def finalize(self) -> WorkerStatistics:
        """完成統計並返回"""
        self.stats.finalize()
        return self.stats

    def get_summary(self) -> dict[str, Any]:
        """獲取摘要報告"""
        return self.stats.to_summary()
