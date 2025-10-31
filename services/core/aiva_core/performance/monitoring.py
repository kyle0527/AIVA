"""監控系統模組
拆分自 optimized_core.py 的監控系統部分
"""

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any


@dataclass
class Metric:
    """監控指標"""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = None


class MetricsCollector:
    """效能指標收集器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = {}

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
