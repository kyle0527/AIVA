

import time
from typing import Any

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class Metrics:
    """Prometheus 指標收集器"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if self.enabled:
            # HTTP 請求指標
            self.http_requests_total = Counter(
                "aiva_integration_http_requests_total",
                "Total HTTP requests",
                ["method", "endpoint", "status_code"],
            )

            self.http_request_duration = Histogram(
                "aiva_integration_http_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "endpoint"],
                buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )

            # 業務指標
            self.findings_processed_total = Counter(
                "aiva_integration_findings_processed_total",
                "Total findings processed",
                ["status"],
            )

            self.reports_generated_total = Counter(
                "aiva_integration_reports_generated_total",
                "Total reports generated",
                ["format"],
            )

            self.vulnerabilities_found = Counter(
                "aiva_integration_vulnerabilities_found_total",
                "Total vulnerabilities found",
                ["severity", "type"],
            )

            # 系統信息
            self.app_info = Info("aiva_integration_app", "Application information")
            self.app_info.info(
                {
                    "version": "1.0.0",
                    "name": "AIVA Integration Module",
                }
            )

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ) -> None:
        """記錄 HTTP 請求指標"""
        if not self.enabled:
            return

        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

    def record_finding_processed(self, status: str) -> None:
        """記錄處理的發現指標"""
        if not self.enabled:
            return
        self.findings_processed_total.labels(status=status).inc()

    def record_report_generated(self, format_type: str) -> None:
        """記錄生成的報告指標"""
        if not self.enabled:
            return
        self.reports_generated_total.labels(format=format_type).inc()

    def record_vulnerability_found(self, severity: str, vuln_type: str) -> None:
        """記錄發現的漏洞指標"""
        if not self.enabled:
            return
        self.vulnerabilities_found.labels(severity=severity, type=vuln_type).inc()

    def get_metrics(self) -> tuple[str, str]:
        """
        獲取 Prometheus 格式的指標

        Returns:
            (content, content_type) 元組
        """
        if not self.enabled:
            return "# Prometheus metrics not available\n", "text/plain"

        return generate_latest(), CONTENT_TYPE_LATEST

    def create_timer(self) -> MetricsTimer:
        """創建計時器上下文管理器"""
        return MetricsTimer()


class MetricsTimer:
    """計時器上下文管理器"""

    def __init__(self):
        self.start_time = None
        self.duration = None

    def __enter__(self) -> MetricsTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time

    @property
    def elapsed(self) -> float:
        """獲取經過的時間（秒）"""
        if self.duration is not None:
            return self.duration
        if self.start_time is not None:
            return time.perf_counter() - self.start_time
        return 0.0
