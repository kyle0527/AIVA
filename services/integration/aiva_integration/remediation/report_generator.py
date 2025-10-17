"""
Report Generator - 報告生成器

生成詳細的安全掃描和修復報告
支持 HTML, PDF 等多種格式
使用 Jinja2 模板和 ReportLab/WeasyPrint
"""

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
import structlog

logger = structlog.get_logger(__name__)

# 嘗試導入 PDF 庫
try:
    from weasyprint import HTML as WeasyHTML  # type: ignore[import-untyped]

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WeasyHTML = None  # type: ignore[assignment,misc]
    WEASYPRINT_AVAILABLE = False
    logger.warning("weasyprint_not_available")

try:
    from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
    from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import-untyped]
    from reportlab.platypus import (  # type: ignore[import-untyped]
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    SimpleDocTemplate = None  # type: ignore[assignment,misc]
    Paragraph = None  # type: ignore[assignment,misc]
    Spacer = None  # type: ignore[assignment,misc]
    letter = None  # type: ignore[assignment,misc]
    getSampleStyleSheet = None  # type: ignore[assignment,misc]
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab_not_available")


class ReportGenerator:
    """
    多格式報告生成器

    支持 HTML, PDF, Markdown 等格式
    """

    def __init__(
        self,
        template_dir: str | Path | None = None,
        output_dir: str | Path = "./reports",
    ):
        """
        初始化報告生成器

        Args:
            template_dir: 模板目錄
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 設置 Jinja2 環境
        if template_dir and Path(template_dir).exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=True,
            )
            logger.info("report_generator_initialized", template_dir=str(template_dir))
        else:
            # 使用內置模板
            self.jinja_env = None
            logger.info("report_generator_initialized", mode="inline_templates")

        self.reports: list[dict[str, Any]] = []

    def generate_vulnerability_report(
        self,
        vulnerabilities: list[dict[str, Any]],
        scan_info: dict[str, Any] | None = None,
        format: str = "html",
    ) -> dict[str, Any]:
        """
        生成漏洞報告

        Args:
            vulnerabilities: 漏洞列表
            scan_info: 掃描信息
            format: 輸出格式 (html, pdf, markdown)

        Returns:
            報告信息
        """
        logger.info(
            "generating_vulnerability_report",
            vuln_count=len(vulnerabilities),
            format=format,
        )

        report_id = hashlib.sha256(
            f"vuln_report_{datetime.now()}".encode()
        ).hexdigest()[:16]

        report = {
            "report_id": report_id,
            "timestamp": datetime.now().isoformat(),
            "type": "vulnerability",
            "format": format,
            "vulnerabilities": vulnerabilities,
            "scan_info": scan_info or {},
            "statistics": self._calculate_vuln_statistics(vulnerabilities),
        }

        # 根據格式生成報告
        if format == "html":
            output_file = self._generate_html_report(report)
        elif format == "pdf":
            output_file = self._generate_pdf_report(report)
        elif format == "markdown":
            output_file = self._generate_markdown_report(report)
        else:
            logger.error("unsupported_format", format=format)
            return {"success": False, "error": f"Unsupported format: {format}"}

        report["output_file"] = str(output_file)
        report["success"] = True

        self.reports.append(report)
        logger.info("vulnerability_report_generated", report_id=report_id, file=str(output_file))
        return report

    def generate_remediation_report(
        self,
        fixes: list[dict[str, Any]],
        format: str = "html",
    ) -> dict[str, Any]:
        """
        生成修復報告

        Args:
            fixes: 修復列表
            format: 輸出格式

        Returns:
            報告信息
        """
        logger.info("generating_remediation_report", fixes_count=len(fixes), format=format)

        report_id = hashlib.sha256(
            f"remediation_report_{datetime.now()}".encode()
        ).hexdigest()[:16]

        report = {
            "report_id": report_id,
            "timestamp": datetime.now().isoformat(),
            "type": "remediation",
            "format": format,
            "fixes": fixes,
            "statistics": self._calculate_fix_statistics(fixes),
        }

        if format == "html":
            output_file = self._generate_remediation_html(report)
        elif format == "markdown":
            output_file = self._generate_remediation_markdown(report)
        else:
            output_file = self._generate_remediation_html(report)

        report["output_file"] = str(output_file)
        report["success"] = True

        self.reports.append(report)
        logger.info("remediation_report_generated", report_id=report_id)
        return report

    def _calculate_vuln_statistics(
        self,
        vulnerabilities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """計算漏洞統計"""
        stats = {
            "total": len(vulnerabilities),
            "by_severity": {},
            "by_type": {},
        }

        for vuln in vulnerabilities:
            severity = vuln.get("severity", "UNKNOWN")
            vuln_type = vuln.get("type", "UNKNOWN")

            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            stats["by_type"][vuln_type] = stats["by_type"].get(vuln_type, 0) + 1

        return stats

    def _calculate_fix_statistics(
        self,
        fixes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """計算修復統計"""
        stats = {
            "total": len(fixes),
            "successful": 0,
            "failed": 0,
            "pending": 0,
        }

        for fix in fixes:
            status = fix.get("status", "pending")
            if status == "completed":
                stats["successful"] += 1
            elif status == "failed":
                stats["failed"] += 1
            else:
                stats["pending"] += 1

        return stats

    def _generate_html_report(self, report: dict[str, Any]) -> Path:
        """生成 HTML 報告"""
        output_file = self.output_dir / f"{report['report_id']}_vulnerability.html"

        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("vulnerability_report.html")
                html_content = template.render(report=report)
            except Exception as e:
                logger.warning("template_load_failed", error=str(e))
                html_content = self._generate_inline_html(report)
        else:
            html_content = self._generate_inline_html(report)

        output_file.write_text(html_content, encoding="utf-8")
        return output_file

    def _generate_inline_html(self, report: dict[str, Any]) -> str:
        """生成內置 HTML 模板"""
        stats = report["statistics"]
        vulns = report["vulnerabilities"]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Vulnerability Report - {report['report_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ flex: 1; padding: 20px; background: #ecf0f1; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #e74c3c; }}
        .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
        .vulnerability {{ border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .severity-CRITICAL {{ border-left-color: #c0392b; background: #fadbd8; }}
        .severity-HIGH {{ border-left-color: #e74c3c; background: #f5b7b1; }}
        .severity-MEDIUM {{ border-left-color: #f39c12; background: #fdebd0; }}
        .severity-LOW {{ border-left-color: #3498db; background: #d6eaf8; }}
        .vuln-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 5px; }}
        .vuln-meta {{ color: #7f8c8d; font-size: 0.9em; }}
        .timestamp {{ text-align: right; color: #95a5a6; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>[LOCK] Vulnerability Report</h1>
        <p><strong>Report ID:</strong> {report['report_id']}</p>
        <p><strong>Generated:</strong> {report['timestamp']}</p>

        <h2>[STATS] Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{stats['total']}</div>
                <div class="stat-label">Total Vulnerabilities</div>
            </div>
"""

        # 按嚴重程度統計
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = stats["by_severity"].get(severity, 0)
            if count > 0:
                html += f"""
            <div class="stat-card">
                <div class="stat-number">{count}</div>
                <div class="stat-label">{severity}</div>
            </div>
"""

        html += """
        </div>

        <h2>[SEARCH] Vulnerabilities</h2>
"""

        # 列出所有漏洞
        for vuln in vulns:
            severity = vuln.get("severity", "UNKNOWN")
            title = vuln.get("title", "Unknown Vulnerability")
            vuln_type = vuln.get("type", "Unknown")
            location = vuln.get("location", "Unknown")

            html += f"""
        <div class="vulnerability severity-{severity}">
            <div class="vuln-title">{title}</div>
            <div class="vuln-meta">
                <strong>Type:</strong> {vuln_type} |
                <strong>Severity:</strong> {severity} |
                <strong>Location:</strong> {location}
            </div>
        </div>
"""

        html += f"""
        <div class="timestamp">Report generated: {report['timestamp']}</div>
    </div>
</body>
</html>
"""
        return html

    def _generate_pdf_report(self, report: dict[str, Any]) -> Path:
        """生成 PDF 報告"""
        output_file = self.output_dir / f"{report['report_id']}_vulnerability.pdf"

        if WEASYPRINT_AVAILABLE and WeasyHTML is not None:
            # 使用 WeasyPrint
            html_content = self._generate_inline_html(report)
            WeasyHTML(string=html_content).write_pdf(output_file)  # type: ignore[misc]
            logger.info("pdf_generated_weasyprint")

        elif REPORTLAB_AVAILABLE and SimpleDocTemplate is not None:
            # 使用 ReportLab
            self._generate_reportlab_pdf(report, output_file)
            logger.info("pdf_generated_reportlab")

        else:
            # 降級到 HTML
            logger.warning("no_pdf_library", message="Falling back to HTML")
            return self._generate_html_report(report)

        return output_file

    def _generate_reportlab_pdf(self, report: dict[str, Any], output_file: Path) -> None:
        """使用 ReportLab 生成 PDF"""
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab not available")

        # 這些檢查確保類型檢查器知道這些不是 None
        assert SimpleDocTemplate is not None
        assert getSampleStyleSheet is not None
        assert Paragraph is not None
        assert Spacer is not None
        assert letter is not None

        doc = SimpleDocTemplate(str(output_file), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # 標題
        story.append(Paragraph("Vulnerability Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Report ID: {report['report_id']}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # 統計
        stats = report["statistics"]
        story.append(Paragraph(f"Total Vulnerabilities: {stats['total']}", styles["Heading2"]))
        story.append(Spacer(1, 12))

        # 漏洞列表
        for vuln in report["vulnerabilities"]:
            story.append(
                Paragraph(
                    f"{vuln.get('severity', 'UNKNOWN')}: {vuln.get('title', 'Unknown')}",
                    styles["Heading3"],
                )
            )
            story.append(Paragraph(f"Type: {vuln.get('type', 'Unknown')}", styles["Normal"]))
            story.append(Spacer(1, 6))

        doc.build(story)

    def _generate_markdown_report(self, report: dict[str, Any]) -> Path:
        """生成 Markdown 報告"""
        output_file = self.output_dir / f"{report['report_id']}_vulnerability.md"

        stats = report["statistics"]
        vulns = report["vulnerabilities"]

        md = f"""# [LOCK] Vulnerability Report

**Report ID:** `{report['report_id']}`
**Generated:** {report['timestamp']}

## [STATS] Statistics

- **Total Vulnerabilities:** {stats['total']}
"""

        for severity, count in stats["by_severity"].items():
            md += f"- **{severity}:** {count}\n"

        md += "\n## [SEARCH] Vulnerabilities\n\n"

        for vuln in vulns:
            severity = vuln.get("severity", "UNKNOWN")
            title = vuln.get("title", "Unknown")
            vuln_type = vuln.get("type", "Unknown")
            location = vuln.get("location", "Unknown")

            md += f"""### [{severity}] {title}

- **Type:** {vuln_type}
- **Location:** {location}

---

"""

        output_file.write_text(md, encoding="utf-8")
        return output_file

    def _generate_remediation_html(self, report: dict[str, Any]) -> Path:
        """生成修復報告 HTML"""
        output_file = self.output_dir / f"{report['report_id']}_remediation.html"

        stats = report["statistics"]
        fixes = report["fixes"]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Remediation Report - {report['report_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; }}
        h1 {{ color: #27ae60; }}
        .fix {{ border-left: 4px solid #27ae60; padding: 15px; margin: 15px 0; background: #d5f4e6; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ flex: 1; padding: 20px; background: #ecf0f1; border-radius: 8px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>[OK] Remediation Report</h1>
        <p><strong>Report ID:</strong> {report['report_id']}</p>

        <h2>Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div>{stats['total']}</div>
                <div>Total Fixes</div>
            </div>
            <div class="stat-card">
                <div>{stats['successful']}</div>
                <div>Successful</div>
            </div>
            <div class="stat-card">
                <div>{stats['failed']}</div>
                <div>Failed</div>
            </div>
        </div>

        <h2>Fixes</h2>
"""

        for fix in fixes:
            fix_id = fix.get("fix_id", "unknown")
            status = fix.get("status", "pending")
            html += f"""
        <div class="fix">
            <strong>Fix ID:</strong> {fix_id}<br>
            <strong>Status:</strong> {status}
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        output_file.write_text(html, encoding="utf-8")
        return output_file

    def _generate_remediation_markdown(self, report: dict[str, Any]) -> Path:
        """生成修復報告 Markdown"""
        output_file = self.output_dir / f"{report['report_id']}_remediation.md"

        stats = report["statistics"]
        md = f"""# [OK] Remediation Report

**Report ID:** `{report['report_id']}`

## Statistics

- Total Fixes: {stats['total']}
- Successful: {stats['successful']}
- Failed: {stats['failed']}
- Pending: {stats['pending']}

## Fixes

"""

        for fix in report["fixes"]:
            md += f"- **{fix.get('fix_id')}**: {fix.get('status')}\n"

        output_file.write_text(md, encoding="utf-8")
        return output_file

    def get_reports(self) -> list[dict[str, Any]]:
        """獲取所有報告"""
        return self.reports


def main():
    """測試範例"""
    print("[U+1F4C4] Report Generator Demo")
    print("=" * 60)

    generator = ReportGenerator(output_dir="./demo_reports")

    # 生成漏洞報告
    test_vulns = [
        {
            "title": "SQL Injection in login form",
            "type": "SQL Injection",
            "severity": "CRITICAL",
            "location": "app/views.py:42",
        },
        {
            "title": "XSS in user profile",
            "type": "XSS",
            "severity": "HIGH",
            "location": "app/templates/profile.html:15",
        },
        {
            "title": "Weak password policy",
            "type": "Weak Configuration",
            "severity": "MEDIUM",
            "location": "config/security.yaml",
        },
    ]

    report = generator.generate_vulnerability_report(
        vulnerabilities=test_vulns,
        scan_info={"target": "demo-app", "scanner": "AIVA"},
        format="html",
    )

    print("\n[OK] Vulnerability report generated:")
    print(f"   Report ID: {report['report_id']}")
    print(f"   File: {report['output_file']}")
    print(f"   Total vulnerabilities: {report['statistics']['total']}")

    print("\n[OK] Demo completed")


if __name__ == "__main__":
    main()
