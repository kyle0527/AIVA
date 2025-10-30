

from datetime import UTC, datetime
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScanMetadataAnalyzer:
    """掃描元數據分析器 - 分析掃描性能和元數據"""

    def analyze_metadata(self, scan_data: dict[str, Any]) -> dict[str, Any]:
        """分析掃描元數據"""
        return {
            "scan_duration": scan_data.get("duration", 0),
            "targets_scanned": scan_data.get("targets_count", 0),
            "modules_used": scan_data.get("modules", []),
            "performance_score": self._calculate_performance_score(scan_data),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _calculate_performance_score(self, scan_data: dict[str, Any]) -> float:
        """計算性能分數"""
        duration = scan_data.get("duration", 0)
        targets = scan_data.get("targets_count", 1)

        if duration == 0 or targets == 0:
            return 0.0

        # 簡單的性能分數計算 (目標數/秒)
        score = targets / max(duration, 1) * 100
        return min(score, 100.0)
