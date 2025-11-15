"""系統整合監控模組

從 optimized_core.py 拆分的監控系統部分，提供：
- 效能指標收集
- 組件健康檢查
- 系統資源監控
- 跨組件整合監控
"""

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional
from enum import Enum


class ComponentHealth(str, Enum):
    """組件健康狀態"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """監控指標"""

    name: str
    value: float
    timestamp: float
    labels: Optional[Dict[str, str]] = None


class MetricsCollector:
    """效能指標收集器 - 中央監控服務"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, float] = {}  # component_name -> last_check_time

    def record_duration(
        self, name: str, duration: float, labels: dict[str, str] = None
    ):
        """記錄執行時間"""
        metric = Metric(name, duration, time.time(), labels or {})
        self.metrics[f"{name}_duration"].append(metric)

        # 保持最近的1000筆記錄
        if len(self.metrics[f"{name}_duration"]) > 1000:
            self.metrics[f"{name}_duration"] = self.metrics[f"{name}_duration"][-1000:]

    def increment_counter(self, name: str, labels: dict[str, str] = None):
        """增加計數器"""
        key = self._make_key(name, labels)
        self.counters[key] += 1

    def set_gauge(self, name: str, value: float, labels: dict[str, str] = None):
        """設置儀表值"""
        key = self._make_key(name, labels)
        self.gauges[key] = Metric(name, value, time.time(), labels or {})

    def _make_key(self, name: str, labels: dict[str, str] = None) -> str:
        """生成指標鍵值"""
        if not labels:
            return name

        label_str = "_".join(f"{k}:{v}" for k, v in sorted(labels.items()))
        return f"{name}_{hash(label_str)}"

    def get_metrics_summary(self) -> dict[str, Any]:
        """獲取指標摘要"""
        summary = {
            "counters": dict(self.counters),
            "gauges": {k: v.value for k, v in self.gauges.items()},
            "durations": {},
            "component_health": {k: v.value for k, v in self.component_health.items()},
            "system_health": self.get_system_health_status(),
        }

        # 計算持續時間的統計資訊
        for name, metrics in self.metrics.items():
            if metrics:
                durations = [m.value for m in metrics]
                summary["durations"][name] = {
                    "count": len(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p95": (
                        np.percentile(durations, 95)
                        if len(durations) > 1
                        else durations[0]
                    ),
                }

        return summary

    def update_component_health(
        self, component_name: str, health: ComponentHealth, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """更新組件健康狀態"""
        self.component_health[component_name] = health
        self.health_checks[component_name] = time.time()
        if details:
            self.set_gauge(f"{component_name}_health_details", 1.0, details)

    def get_system_health_status(self) -> str:
        """獲取系統整體健康狀態"""
        if not self.component_health:
            return ComponentHealth.UNKNOWN.value
        
        health_values = list(self.component_health.values())
        if any(h == ComponentHealth.UNHEALTHY for h in health_values):
            return ComponentHealth.UNHEALTHY.value
        elif any(h == ComponentHealth.DEGRADED for h in health_values):
            return ComponentHealth.DEGRADED.value
        elif all(h == ComponentHealth.HEALTHY for h in health_values):
            return ComponentHealth.HEALTHY.value
        else:
            return ComponentHealth.UNKNOWN.value

    def check_component_freshness(self, component_name: str, max_age: float = 300.0) -> bool:
        """檢查組件健康檢查是否過期（默認5分鐘）"""
        if component_name not in self.health_checks:
            return False
        return (time.time() - self.health_checks[component_name]) < max_age


def monitor_performance(metric_name: str):
    """效能監控裝飾器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            from .monitoring import metrics_collector  # 延遲導入避免循環

            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                metrics_collector.record_duration(
                    metric_name, duration, {"status": "success"}
                )
                metrics_collector.increment_counter(
                    f"{metric_name}_total", {"status": "success"}
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                metrics_collector.record_duration(
                    metric_name,
                    duration,
                    {"status": "error", "error_type": type(e).__name__},
                )
                metrics_collector.increment_counter(
                    f"{metric_name}_total",
                    {"status": "error", "error_type": type(e).__name__},
                )

                raise

        return wrapper

    return decorator


# 全域實例
metrics_collector = MetricsCollector()
