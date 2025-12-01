"""Report Generator - Bug Bounty 報告生成器 (P3)

實現：
1. Markdown 格式報告
2. PDF 格式報告（可選）
3. 整合 SQLite 數據生成報告
4. OWASP 分類與 CWE 映射
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from pathlib import Path
import json

from aiva_common.utils.logging import get_logger
from aiva_common.enums import Severity, VulnerabilityType

logger = get_logger(__name__)


class ReportGenerator:
    """Bug Bounty 報告生成器
    
    職責：
    1. 從 SQLite 讀取掃描結果
    2. 生成結構化 Markdown 報告
    3. 包含 OWASP 分類、CWE 映射、CVSS 評分
    4. 支持導出 JSON 格式（API 友好）
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OWASP Top 10 2021 映射
        self.owasp_mapping = {
            VulnerabilityType.XSS: "A03:2021-Injection",
            VulnerabilityType.SQLI: "A03:2021-Injection",
            VulnerabilityType.AUTHENTICATION_BYPASS: "A07:2021-Identification and Authentication Failures",
            VulnerabilityType.WEAK_AUTH: "A07:2021-Identification and Authentication Failures",
            VulnerabilityType.INFO_LEAK: "A02:2021-Cryptographic Failures",
            VulnerabilityType.SSRF: "A10:2021-Server-Side Request Forgery"
        }
        
        # CWE 映射
        self.cwe_mapping = {
            VulnerabilityType.XSS: "CWE-79",
            VulnerabilityType.SQLI: "CWE-89",
            VulnerabilityType.AUTHENTICATION_BYPASS: "CWE-287",
            VulnerabilityType.WEAK_AUTH: "CWE-287",
            VulnerabilityType.INFO_LEAK: "CWE-200",
            VulnerabilityType.SSRF: "CWE-918"
        }
        
        logger.info(f"ReportGenerator initialized (output_dir={output_dir})")
    
    def generate_markdown_report(
        self,
        target_url: str,
        findings: List[Dict[str, Any]],
        scan_metadata: Dict[str, Any]
    ) -> str:
        """生成 Markdown 格式報告
        
        Args:
            target_url: 目標 URL
            findings: 漏洞發現列表
            scan_metadata: 掃描元數據（掃描時間、配置等）
        
        Returns:
            報告文件路徑
        """
        # 統計數據
        total_findings = len(findings)
        severity_count = self._count_by_severity(findings)
        owasp_coverage = self._analyze_owasp_coverage(findings)
        
        # 生成報告內容
        report_lines = []
        
        # 標題
        report_lines.append("# Security Assessment Report")
        report_lines.append(f"## Target: {target_url}")
        report_lines.append(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("**Scanner:** AIVA Bug Bounty Hunter v6.3")
        report_lines.append("")
        
        # 執行摘要
        report_lines.append("## Executive Summary")
        report_lines.append(f"- **Total Findings:** {total_findings}")
        report_lines.append(f"- **Critical:** {severity_count.get(Severity.CRITICAL, 0)}")
        report_lines.append(f"- **High:** {severity_count.get(Severity.HIGH, 0)}")
        report_lines.append(f"- **Medium:** {severity_count.get(Severity.MEDIUM, 0)}")
        report_lines.append(f"- **Low:** {severity_count.get(Severity.LOW, 0)}")
        report_lines.append("")
        
        # OWASP 覆蓋
        report_lines.append("## OWASP Top 10 2021 Coverage")
        for category, count in owasp_coverage.items():
            report_lines.append(f"- **{category}:** {count} findings")
        report_lines.append("")
        
        # 掃描配置
        report_lines.append("## Scan Configuration")
        report_lines.append(f"- **Scan Duration:** {scan_metadata.get('duration_seconds', 'N/A')} seconds")
        report_lines.append(f"- **Payloads Tested:** {scan_metadata.get('payloads_tested', 'N/A')}")
        report_lines.append(f"- **Requests Sent:** {scan_metadata.get('requests_sent', 'N/A')}")
        report_lines.append("")
        
        # 詳細發現
        report_lines.append("## Detailed Findings")
        report_lines.append("")
        
        # 按嚴重程度排序
        sorted_findings = sorted(
            findings,
            key=lambda f: self._severity_score(f.get('severity', Severity.LOW)),
            reverse=True
        )
        
        for idx, finding in enumerate(sorted_findings, 1):
            report_lines.append(f"### {idx}. {finding.get('title', 'Untitled Vulnerability')}")
            report_lines.append(f"**Severity:** {finding.get('severity', 'UNKNOWN')}")
            report_lines.append(f"**Type:** {finding.get('vulnerability_type', 'UNKNOWN')}")
            
            vuln_type = finding.get('vulnerability_type')
            if vuln_type:
                owasp = self.owasp_mapping.get(vuln_type, "N/A")
                cwe = self.cwe_mapping.get(vuln_type, "N/A")
                report_lines.append(f"**OWASP:** {owasp}")
                report_lines.append(f"**CWE:** {cwe}")
            
            report_lines.append(f"**Confidence:** {finding.get('confidence', 'N/A')}")
            report_lines.append("")
            
            report_lines.append("**Description:**")
            report_lines.append(finding.get('description', 'No description available'))
            report_lines.append("")
            
            # 證據
            evidence = finding.get('evidence', [])
            if evidence:
                report_lines.append("**Evidence:**")
                for ev in evidence[:3]:  # 限制顯示前 3 個證據
                    report_lines.append(f"- **Payload:** `{ev.get('payload', 'N/A')}`")
                    report_lines.append(f"  **Request:** `{ev.get('request', 'N/A')[:100]}...`")
                    report_lines.append(f"  **Response:** `{ev.get('response', 'N/A')[:100]}...`")
                report_lines.append("")
            
            # 修復建議
            report_lines.append("**Remediation:**")
            report_lines.append(finding.get('remediation', 'No remediation guidance available'))
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        # 附錄
        report_lines.append("## Appendix")
        report_lines.append("### Scan Logs")
        report_lines.append("Detailed scan logs are available in the database.")
        report_lines.append("")
        
        # 寫入文件
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"✅ Markdown report generated: {filepath}")
        return str(filepath)
    
    def generate_json_report(
        self,
        target_url: str,
        findings: List[Dict[str, Any]],
        scan_metadata: Dict[str, Any]
    ) -> str:
        """生成 JSON 格式報告
        
        Args:
            target_url: 目標 URL
            findings: 漏洞發現列表
            scan_metadata: 掃描元數據
        
        Returns:
            報告文件路徑
        """
        report_data = {
            "target": target_url,
            "generated_at": datetime.now(UTC).isoformat(),
            "scanner_version": "AIVA v6.3",
            "summary": {
                "total_findings": len(findings),
                "severity_breakdown": self._count_by_severity(findings),
                "owasp_coverage": self._analyze_owasp_coverage(findings)
            },
            "scan_metadata": scan_metadata,
            "findings": findings
        }
        
        # 寫入文件
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ JSON report generated: {filepath}")
        return str(filepath)
    
    def _count_by_severity(self, findings: List[Dict[str, Any]]) -> Dict[Severity, int]:
        """統計各嚴重程度的漏洞數量"""
        count: Dict[Severity, int] = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 0,
            Severity.MEDIUM: 0,
            Severity.LOW: 0
        }
        
        for finding in findings:
            severity = finding.get('severity', Severity.LOW)
            if severity in count:
                count[severity] += 1
        
        return count
    
    def _analyze_owasp_coverage(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析 OWASP 覆蓋"""
        coverage: Dict[str, int] = {}
        
        for finding in findings:
            vuln_type = finding.get('vulnerability_type')
            if vuln_type:
                owasp = self.owasp_mapping.get(vuln_type, "Other")
                coverage[owasp] = coverage.get(owasp, 0) + 1
        
        return coverage
    
    def _severity_score(self, severity: Severity) -> int:
        """將嚴重程度轉換為數值（用於排序）"""
        score_map = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1
        }
        return score_map.get(severity, 0)
