from __future__ import annotations

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ReportTemplateSelector:
    """報告模板選擇器 - 根據需求選擇合適的報告模板"""

    def __init__(self) -> None:
        self._templates = {
            "executive": "高層管理報告模板",
            "technical": "技術詳細報告模板",
            "compliance": "合規檢查報告模板",
            "summary": "簡要摘要報告模板",
        }

    def select_template(self, report_type: str = "technical") -> str:
        """選擇報告模板"""
        return self._templates.get(report_type, self._templates["technical"])

    def get_available_templates(self) -> list[str]:
        """獲取可用模板列表"""
        return list(self._templates.keys())
