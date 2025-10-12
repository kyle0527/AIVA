from __future__ import annotations

import json
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class FormatterExporter:
    """
    格式化匯出器

    支援多種格式的報告匯出，包括 JSON、XML、PDF、CSV 等格式。
    """

    def __init__(self) -> None:
        self._supported_formats = ["json", "xml", "csv", "html"]

    def export_to_format(
        self, content: dict[str, Any], format_type: str = "json"
    ) -> str:
        """
        將報告內容匯出為指定格式

        Args:
            content: 報告內容
            format_type: 目標格式

        Returns:
            格式化後的字符串
        """
        format_type = format_type.lower()

        if format_type not in self._supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")

        if format_type == "json":
            return self._export_json(content)
        elif format_type == "xml":
            return self._export_xml(content)
        elif format_type == "csv":
            return self._export_csv(content)
        elif format_type == "html":
            return self._export_html(content)
        else:
            raise ValueError(f"Format {format_type} not implemented")

    def _export_json(self, content: dict[str, Any]) -> str:
        """匯出為 JSON 格式"""
        return json.dumps(content, indent=2, ensure_ascii=False)

    def _export_xml(self, content: dict[str, Any]) -> str:
        """匯出為 XML 格式"""
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append("<SecurityReport>")
        xml_lines.extend(self._dict_to_xml(content, indent=2))
        xml_lines.append("</SecurityReport>")
        return "\n".join(xml_lines)

    def _export_csv(self, content: dict[str, Any]) -> str:
        """匯出為 CSV 格式（主要針對發現列表）"""
        csv_lines = []

        # CSV 標頭
        headers = [
            "Finding ID",
            "Vulnerability Type",
            "Severity",
            "Location",
            "Description",
        ]
        csv_lines.append(",".join(headers))

        # 提取技術細節進行 CSV 匯出
        technical_details = content.get("technical_details", [])
        for finding in technical_details:
            row = [
                finding.get("finding_id", ""),
                finding.get("vulnerability_type", ""),
                finding.get("severity", ""),
                finding.get("location", {}).get("url", ""),
                finding.get("description", "").replace(",", ";"),  # 避免 CSV 分隔符衝突
            ]
            csv_lines.append(",".join(f'"{item}"' for item in row))

        return "\n".join(csv_lines)

    def _export_html(self, content: dict[str, Any]) -> str:
        """匯出為 HTML 格式"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Security Assessment Report</title>",
            "<meta charset='UTF-8'>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; }",
            ".finding { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }",
            ".critical { border-left: 5px solid #dc3545; }",
            ".high { border-left: 5px solid #fd7e14; }",
            ".medium { border-left: 5px solid #ffc107; }",
            ".low { border-left: 5px solid #28a745; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        # 執行摘要
        exec_summary = content.get("executive_summary", {})
        html_parts.extend(
            [
                "<h1>Security Assessment Report</h1>",
                "<div class='summary'>",
                "<h2>Executive Summary</h2>",
                f"<p>Total Findings: {exec_summary.get('total_findings', 0)}</p>",
                f"<p>Overall Risk Level: {exec_summary.get('overall_risk_level', 'Unknown')}</p>",
                f"<p>Compliance Score: {exec_summary.get('compliance_score', 0):.1f}/100</p>",
                "</div>",
            ]
        )

        # 技術細節
        technical_details = content.get("technical_details", [])
        if technical_details:
            html_parts.append("<h2>Technical Findings</h2>")
            for finding in technical_details[:20]:  # 限制顯示數量
                severity = finding.get("severity", "").lower()
                html_parts.extend(
                    [
                        f"<div class='finding {severity}'>",
                        f"<h3>{finding.get('vulnerability_type', 'Unknown')}</h3>",
                        f"<p><strong>Severity:</strong> {finding.get('severity', 'Unknown')}</p>",
                        f"<p><strong>Location:</strong> {finding.get('location', {}).get('url', 'Unknown')}</p>",
                        f"<p><strong>Description:</strong> {finding.get('description', 'No description available')}</p>",
                        "</div>",
                    ]
                )

        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def _dict_to_xml(self, data: dict[str, Any], indent: int = 0) -> list[str]:
        """將字典轉換為 XML 行"""
        lines = []
        spaces = " " * indent

        for key, value in data.items():
            # 清理 XML 標籤名稱
            clean_key = key.replace(" ", "_").replace("-", "_")

            if isinstance(value, dict):
                lines.append(f"{spaces}<{clean_key}>")
                lines.extend(self._dict_to_xml(value, indent + 2))
                lines.append(f"{spaces}</{clean_key}>")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{spaces}<{clean_key}>")
                        lines.extend(self._dict_to_xml(item, indent + 2))
                        lines.append(f"{spaces}</{clean_key}>")
                    else:
                        lines.append(f"{spaces}<{clean_key}>{str(item)}</{clean_key}>")
            else:
                lines.append(f"{spaces}<{clean_key}>{str(value)}</{clean_key}>")

        return lines

    def get_supported_formats(self) -> list[str]:
        """獲取支援的匯出格式"""
        return self._supported_formats.copy()
