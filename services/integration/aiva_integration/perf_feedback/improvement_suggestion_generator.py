from __future__ import annotations

from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ImprovementSuggestionGenerator:
    """改進建議生成器 - 基於掃描結果生成改進建議"""

    def generate_suggestions(
        self, performance_data: dict[str, Any], findings: list[dict[str, Any]]
    ) -> list[str]:
        """生成改進建議"""
        suggestions = []

        # 性能改進建議
        performance_score = performance_data.get("performance_score", 0)
        if performance_score < 50:
            suggestions.append("建議優化掃描配置以提升性能")

        # 基於發現數量的建議
        finding_count = len(findings)
        if finding_count > 50:
            suggestions.append("發現大量漏洞，建議分階段修復")
        elif finding_count == 0:
            suggestions.append("未發現漏洞，建議定期重複掃描")

        return suggestions
