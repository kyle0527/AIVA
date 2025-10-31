#!/usr/bin/env python3
"""
AIVA Bug Bounty Reporting System
Comprehensive vulnerability reporting and PoC generation for all severity levels
Supporting both high-value and low-value findings for Bug Bounty hunters
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import base64
import hashlib

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.theme import Theme
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns

# Local imports
from ...core.base_capability import BaseCapability
from ...core.registry import CapabilityRegistry

# Setup theme and console
_theme = Theme({"purple": "#7B61FF"})
console = Console(theme=_theme)
logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """漏洞嚴重程度分類 - 對應 Bug Bounty 報酬等級"""
    CRITICAL = ("Critical", "💀", "$5000-$50000", "RCE, SQLi with data access")
    HIGH = ("High", "🔥", "$1000-$10000", "XSS, Authentication bypass")
    MEDIUM = ("Medium", "⚠️", "$100-$2000", "CSRF, Information disclosure")
    LOW = ("Low", "📝", "$25-$500", "Missing headers, Configuration issues")
    INFO = ("Informational", "ℹ️", "$0-$100", "Version disclosure, Debug info")
    
    def __init__(self, label, emoji, bounty_range, examples):
        self.label = label
        self.emoji = emoji
        self.bounty_range = bounty_range
        self.examples = examples


class VulnerabilityCategory(Enum):
    """漏洞分類 - OWASP Top 10 + Bug Bounty 常見"""
    # High-Value Categories
    INJECTION = ("SQL/NoSQL/Command Injection", "💉")
    BROKEN_AUTH = ("Broken Authentication", "🔐")
    SENSITIVE_DATA = ("Sensitive Data Exposure", "📊")
    XXE = ("XML External Entities", "🔗")
    BROKEN_ACCESS = ("Broken Access Control", "🚪")
    SECURITY_MISCONFIG = ("Security Misconfiguration", "⚙️")
    XSS = ("Cross-Site Scripting", "🕸️")
    DESERIALIZATION = ("Insecure Deserialization", "📦")
    
    # Medium-Value Categories  
    COMPONENTS = ("Components with Known Vulnerabilities", "🧩")
    LOGGING = ("Insufficient Logging & Monitoring", "📋")
    CSRF = ("Cross-Site Request Forgery", "🔄")
    CLICKJACKING = ("Clickjacking", "👆")
    
    # Low-Value but Important Categories
    INFO_DISCLOSURE = ("Information Disclosure", "📰")
    MISSING_HEADERS = ("Missing Security Headers", "📄")
    SSL_TLS = ("SSL/TLS Issues", "🔒")
    RATE_LIMITING = ("Missing Rate Limiting", "⏱️")
    
    def __init__(self, description, emoji):
        self.description = description
        self.emoji = emoji


@dataclass
class VulnerabilityFinding:
    """漏洞發現記錄"""
    id: str
    title: str
    severity: VulnerabilitySeverity
    category: VulnerabilityCategory
    target_url: str
    description: str
    impact: str
    reproduction_steps: List[str]
    poc_code: Optional[str] = None
    screenshot_paths: Optional[List[str]] = None
    discovered_by_tool: str = ""
    discovery_time: str = ""
    cvss_score: float = 0.0
    estimated_bounty: str = ""
    status: str = "New"  # New, Reported, Accepted, Duplicate, N/A
    
    def __post_init__(self):
        if not self.discovery_time:
            self.discovery_time = datetime.now().isoformat()
        if not self.id:
            self.id = self._generate_id()
        if not self.estimated_bounty:
            self.estimated_bounty = self.severity.bounty_range
        if self.screenshot_paths is None:
            self.screenshot_paths = []
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        content = f"{self.title}{self.target_url}{self.discovery_time}"
        return hashlib.md5(content.encode()).hexdigest()[:8].upper()


class BugBountyReportGenerator:
    """Bug Bounty 報告生成器"""
    
    def __init__(self):
        self.reports_dir = Path("reports/bug_bounty")
        self.templates_dir = Path("reports/templates")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self, finding: VulnerabilityFinding) -> str:
        """生成 Markdown 格式報告"""
        template = f"""# {finding.severity.emoji} {finding.title}

## Summary
**Severity:** {finding.severity.label} {finding.severity.emoji}
**Category:** {finding.category.emoji} {finding.category.description}
**Target:** `{finding.target_url}`
**Discovery Tool:** {finding.discovered_by_tool}
**CVSS Score:** {finding.cvss_score}
**Estimated Bounty:** {finding.estimated_bounty}

## Description
{finding.description}

## Impact
{finding.impact}

## Reproduction Steps
"""
        
        for i, step in enumerate(finding.reproduction_steps, 1):
            template += f"{i}. {step}\n"
        
        if finding.poc_code:
            template += f"""
## Proof of Concept
```bash
{finding.poc_code}
```
"""
        
        if finding.screenshot_paths:
            template += "\n## Screenshots\n"
            for screenshot in finding.screenshot_paths:
                template += f"![Screenshot]({screenshot})\n"
        
        template += f"""
## Remediation
Please implement appropriate security measures to address this vulnerability.

## Timeline
- **Discovery:** {finding.discovery_time}
- **Report ID:** {finding.id}

---
*Generated by AIVA Bug Bounty Toolkit*
"""
        
        return template
    
    def generate_hackerone_template(self, finding: VulnerabilityFinding) -> str:
        """生成 HackerOne 專用模板"""
        return f"""**Summary:** {finding.title}

**Description:**
{finding.description}

**Steps To Reproduce:**
{chr(10).join(f"{i}. {step}" for i, step in enumerate(finding.reproduction_steps, 1))}

**Impact:**
{finding.impact}

**Proof of Concept:**
{finding.poc_code or 'See attached screenshots'}

**Recommended Fix:**
Implement proper input validation and security controls.

**Supporting Material/References:**
- Tool used: {finding.discovered_by_tool}
- Discovery time: {finding.discovery_time}
"""
    
    def save_report(self, finding: VulnerabilityFinding, format_type: str = "markdown") -> str:
        """保存報告文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{finding.id}_{finding.severity.label.lower()}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        if format_type == "hackerone":
            content = self.generate_hackerone_template(finding)
        else:
            content = self.generate_markdown_report(finding)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(filepath)


class PoCGenerator:
    """PoC 代碼生成器"""
    
    @staticmethod
    def generate_xss_poc(url: str, payload: str, parameter: str) -> str:
        """生成 XSS PoC"""
        return f"""# XSS Proof of Concept
curl -X GET "{url}" \\
  -G \\
  --data-urlencode "{parameter}={payload}" \\
  -H "User-Agent: Mozilla/5.0 (Bug Bounty PoC)" \\
  -v

# Alternative with browser automation
python3 -c "
import requests
from selenium import webdriver

url = '{url}'
payload = '{payload}'
response = requests.get(url, params={{'{parameter}': payload}})
print('XSS Triggered!' if payload in response.text else 'XSS Failed')
"
"""
    
    @staticmethod
    def generate_sqli_poc(url: str, parameter: str, injection_type: str = "UNION") -> str:
        """生成 SQL 注入 PoC"""
        payloads = {
            "UNION": "' UNION SELECT 1,2,3,version(),5-- -",
            "BOOLEAN": "' AND 1=1-- -",
            "TIME": "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)-- -"
        }
        
        payload = payloads.get(injection_type, payloads["UNION"])
        
        return f"""# SQL Injection Proof of Concept ({injection_type})
curl -X POST "{url}" \\
  --data "{parameter}={payload}" \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -v

# Using sqlmap for verification
sqlmap -u "{url}" \\
  --data "{parameter}=test" \\
  --batch \\
  --level 2 \\
  --risk 2
"""
    
    @staticmethod
    def generate_lfi_poc(url: str, parameter: str) -> str:
        """生成 LFI PoC"""
        return f"""# Local File Inclusion Proof of Concept
# Test for /etc/passwd
curl -X GET "{url}" \\
  -G \\
  --data-urlencode "{parameter}=/etc/passwd" \\
  -v

# Test with path traversal
curl -X GET "{url}" \\
  -G \\
  --data-urlencode "{parameter}=../../../etc/passwd" \\
  -v

# Windows equivalent
curl -X GET "{url}" \\
  -G \\
  --data-urlencode "{parameter}=C:\\Windows\\System32\\drivers\\etc\\hosts" \\
  -v
"""
    
    @staticmethod
    def generate_csrf_poc(target_url: str, action_url: str, parameters: Dict[str, str]) -> str:
        """生成 CSRF PoC"""
        form_fields = ""
        for name, value in parameters.items():
            form_fields += f'    <input type="hidden" name="{name}" value="{value}">\n'
        
        return f"""# CSRF Proof of Concept
<!-- Save as csrf_poc.html -->
<!DOCTYPE html>
<html>
<head>
    <title>CSRF PoC - {target_url}</title>
</head>
<body>
    <h1>CSRF Vulnerability Demonstration</h1>
    <p>This form will automatically submit when the page loads.</p>
    
    <form action="{action_url}" method="POST" id="csrf_form">
{form_fields}
    </form>
    
    <script>
        // Auto-submit form
        document.getElementById('csrf_form').submit();
    </script>
</body>
</html>

<!-- Python requests equivalent -->
"""


class BugBountyTracker:
    """Bug Bounty 漏洞追蹤管理"""
    
    def __init__(self):
        self.findings_file = Path("reports/bug_bounty/findings.json")
        self.findings_file.parent.mkdir(parents=True, exist_ok=True)
        self.findings: List[VulnerabilityFinding] = []
        self.report_generator = BugBountyReportGenerator()
        self.poc_generator = PoCGenerator()
        self.load_findings()
    
    def load_findings(self):
        """載入已保存的漏洞發現"""
        if self.findings_file.exists():
            try:
                with open(self.findings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # Convert dict back to dataclass
                        item['severity'] = VulnerabilitySeverity[item['severity']]
                        item['category'] = VulnerabilityCategory[item['category']]
                        self.findings.append(VulnerabilityFinding(**item))
            except Exception as e:
                logger.error(f"Failed to load findings: {e}")
    
    def save_findings(self):
        """保存漏洞發現"""
        try:
            data = []
            for finding in self.findings:
                finding_dict = asdict(finding)
                finding_dict['severity'] = finding.severity.name
                finding_dict['category'] = finding.category.name
                data.append(finding_dict)
            
            with open(self.findings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save findings: {e}")
    
    def add_finding(self, finding: VulnerabilityFinding) -> str:
        """添加新漏洞發現"""
        self.findings.append(finding)
        self.save_findings()
        
        # 生成報告
        report_path = self.report_generator.save_report(finding)
        
        console.print(f"[green]✅ 新漏洞已記錄: {finding.id}[/green]")
        console.print(f"[blue]📄 報告已保存: {report_path}[/blue]")
        
        return finding.id
    
    def get_findings_by_severity(self, severity: VulnerabilitySeverity) -> List[VulnerabilityFinding]:
        """按嚴重程度獲取漏洞"""
        return [f for f in self.findings if f.severity == severity]
    
    def get_bounty_estimate(self) -> Dict[str, Any]:
        """獲取獎金估算"""
        total_findings = len(self.findings)
        by_severity = {}
        
        for severity in VulnerabilitySeverity:
            count = len(self.get_findings_by_severity(severity))
            by_severity[severity.label] = {
                'count': count,
                'emoji': severity.emoji,
                'range': severity.bounty_range,
                'examples': severity.examples
            }
        
        return {
            'total_findings': total_findings,
            'by_severity': by_severity,
            'estimated_total': "Varies by program"
        }


class BugBountyCapability(BaseCapability):
    """Bug Bounty 報告系統 - AIVA 整合"""
    
    def __init__(self):
        super().__init__()
        self.name = "bug_bounty_reporting"
        self.version = "1.0.0"
        self.description = "Bug Bounty Vulnerability Reporting and PoC Generation System"
        self.dependencies = ["curl", "python3-requests"]
        self.tracker = BugBountyTracker()
    
    async def initialize(self) -> bool:
        """初始化功能"""
        try:
            console.print("[yellow]初始化 Bug Bounty 報告系統...[/yellow]")
            console.print("[green]💰 支援高中低價值漏洞發現與報告[/green]")
            console.print("[cyan]🎯 為了生活，每個漏洞都重要！[/cyan]")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """執行命令"""
        try:
            if command == "interactive_menu":
                self._show_main_menu()
                return {"success": True, "message": "Interactive menu completed"}
            
            elif command == "add_finding":
                return self._handle_add_finding(parameters)
            
            elif command == "list_findings":
                return self._handle_list_findings(parameters)
            
            elif command == "generate_poc":
                return self._handle_generate_poc(parameters)
            
            elif command == "bounty_estimate":
                estimate = self.tracker.get_bounty_estimate()
                return {"success": True, "data": estimate}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _show_main_menu(self):
        """顯示主選單"""
        while True:
            console.print("\n")
            panel = Panel.fit("[bold magenta]💰 AIVA Bug Bounty 報告系統[/bold magenta]\n"
                              "支援各種價值等級的漏洞發現與報告\n"
                              "🎯 高價值漏洞賺大錢，低價值漏洞維持生活",
                              border_style="green")
            console.print(panel)
            
            table = Table(title="[bold cyan]功能選單[/bold cyan]", show_lines=True)
            table.add_column("選項", style="yellow", width=8)
            table.add_column("功能", style="green")
            table.add_column("描述", style="white")
            
            table.add_row("1", "📝 添加漏洞發現", "記錄新發現的漏洞")
            table.add_row("2", "📋 查看漏洞列表", "瀏覽已發現的漏洞")
            table.add_row("3", "🔧 生成 PoC", "為漏洞生成概念驗證代碼")
            table.add_row("4", "💰 獎金估算", "查看潛在獎金估算")
            table.add_row("5", "📊 統計報告", "查看漏洞統計信息")
            table.add_row("6", "🎯 漏洞分類說明", "了解漏洞價值分類")
            table.add_row("99", "🚪 退出", "返回主選單")
            
            console.print(table)
            
            choice = Prompt.ask("[bold cyan]選擇功能[/bold cyan]", default="99")
            
            if choice == "1":
                self._add_finding_interactive()
            elif choice == "2":
                self._show_findings_list()
            elif choice == "3":
                self._generate_poc_interactive()
            elif choice == "4":
                self._show_bounty_estimate()
            elif choice == "5":
                self._show_statistics()
            elif choice == "6":
                self._show_vulnerability_guide()
            elif choice == "99":
                break
            else:
                console.print("[red]無效選擇，請重試[/red]")
    
    def _add_finding_interactive(self):
        """交互式添加漏洞發現"""
        console.print("\n[bold green]📝 添加新漏洞發現[/bold green]")
        
        # 基本信息
        title = Prompt.ask("漏洞標題")
        target_url = Prompt.ask("目標 URL")
        description = Prompt.ask("漏洞描述")
        impact = Prompt.ask("影響說明")
        
        # 選擇嚴重程度
        console.print("\n[cyan]選擇漏洞嚴重程度:[/cyan]")
        severity_table = Table()
        severity_table.add_column("編號", style="yellow")
        severity_table.add_column("等級", style="green")
        severity_table.add_column("獎金範圍", style="blue")
        severity_table.add_column("示例", style="white")
        
        severities = list(VulnerabilitySeverity)
        for i, sev in enumerate(severities, 1):
            severity_table.add_row(str(i), f"{sev.emoji} {sev.label}", sev.bounty_range, sev.examples)
        
        console.print(severity_table)
        
        sev_choice = IntPrompt.ask("選擇嚴重程度", choices=list(range(1, len(severities)+1)))
        severity = severities[sev_choice - 1]
        
        # 選擇分類
        console.print("\n[cyan]選擇漏洞分類:[/cyan]")
        category_table = Table()
        category_table.add_column("編號", style="yellow")
        category_table.add_column("分類", style="green")
        
        categories = list(VulnerabilityCategory)
        for i, cat in enumerate(categories, 1):
            category_table.add_row(str(i), f"{cat.emoji} {cat.description}")
        
        console.print(category_table)
        
        cat_choice = IntPrompt.ask("選擇漏洞分類", choices=list(range(1, len(categories)+1)))
        category = categories[cat_choice - 1]
        
        # 復現步驟
        console.print("\n[cyan]輸入復現步驟 (每行一個步驟，輸入空行結束):[/cyan]")
        reproduction_steps = []
        while True:
            step = Prompt.ask(f"步驟 {len(reproduction_steps) + 1}", default="")
            if not step:
                break
            reproduction_steps.append(step)
        
        # 發現工具
        discovered_by_tool = Prompt.ask("發現工具", default="Manual Testing")
        
        # PoC 代碼 (可選)
        poc_code = Prompt.ask("PoC 代碼 (可選)", default="")
        
        # 創建漏洞發現記錄
        finding = VulnerabilityFinding(
            id="",  # Will be auto-generated
            title=title,
            severity=severity,
            category=category,
            target_url=target_url,
            description=description,
            impact=impact,
            reproduction_steps=reproduction_steps,
            poc_code=poc_code if poc_code else None,
            discovered_by_tool=discovered_by_tool
        )
        
        # 添加到追蹤器
        finding_id = self.tracker.add_finding(finding)
        
        console.print(f"[green]✅ 漏洞 {finding_id} 已成功添加![/green]")
        console.print(f"[blue]💰 預估獎金: {severity.bounty_range}[/blue]")
    
    def _show_findings_list(self):
        """顯示漏洞列表"""
        if not self.tracker.findings:
            console.print("[yellow]暫無漏洞發現記錄[/yellow]")
            return
        
        table = Table(title="🐛 漏洞發現列表", show_lines=True)
        table.add_column("ID", style="yellow", width=10)
        table.add_column("標題", style="green", width=25)
        table.add_column("嚴重程度", style="red", width=12)
        table.add_column("分類", style="blue", width=20)
        table.add_column("目標", style="cyan", width=30)
        table.add_column("發現時間", style="magenta", width=12)
        table.add_column("狀態", style="white", width=10)
        
        for finding in self.tracker.findings:
            discovery_time = finding.discovery_time.split('T')[0]  # Just date
            target_short = finding.target_url[:27] + "..." if len(finding.target_url) > 30 else finding.target_url
            title_short = finding.title[:22] + "..." if len(finding.title) > 25 else finding.title
            
            table.add_row(
                finding.id,
                title_short,
                f"{finding.severity.emoji} {finding.severity.label}",
                f"{finding.category.emoji} {finding.category.description[:15]}",
                target_short,
                discovery_time,
                finding.status
            )
        
        console.print(table)
    
    def _show_bounty_estimate(self):
        """顯示獎金估算"""
        estimate = self.tracker.get_bounty_estimate()
        
        console.print("\n[bold green]💰 Bug Bounty 獎金估算報告[/bold green]")
        console.print(f"[cyan]總漏洞數量: {estimate['total_findings']}[/cyan]")
        
        if estimate['total_findings'] == 0:
            console.print("[yellow]暫無漏洞發現，開始尋找漏洞賺取獎金吧！[/yellow]")
            return
        
        table = Table(title="按嚴重程度分類", show_lines=True)
        table.add_column("嚴重程度", style="green")
        table.add_column("數量", style="yellow")
        table.add_column("獎金範圍", style="blue")
        table.add_column("漏洞示例", style="white")
        
        for severity_label, data in estimate['by_severity'].items():
            if data['count'] > 0:
                table.add_row(
                    f"{data['emoji']} {severity_label}",
                    str(data['count']),
                    data['range'],
                    data['examples']
                )
        
        console.print(table)
        console.print("\n[bold yellow]💡 提醒: 低價值漏洞雖然獎金不高，但積少成多也是重要收入來源！[/bold yellow]")
    
    def _show_vulnerability_guide(self):
        """顯示漏洞價值指南"""
        console.print("\n[bold green]🎯 Bug Bounty 漏洞價值指南[/bold green]")
        
        # 高價值漏洞
        high_value_panel = Panel(
            "[bold red]💀 高價值漏洞 ($1000-$50000+)[/bold red]\n"
            "• RCE (遠程代碼執行)\n"
            "• SQL 注入 (有數據訪問)\n"
            "• 身份驗證繞過\n"
            "• 權限提升\n"
            "• SSRF (內網訪問)",
            title="🔥 重點關注",
            border_style="red"
        )
        
        # 中等價值漏洞  
        medium_value_panel = Panel(
            "[bold yellow]⚠️ 中等價值漏洞 ($100-$2000)[/bold yellow]\n"
            "• XSS (存儲型優於反射型)\n"
            "• CSRF (重要功能)\n"
            "• 信息洩露 (敏感數據)\n"
            "• 不安全直接對象引用\n"
            "• 業務邏輯漏洞",
            title="💡 穩定收入",
            border_style="yellow"
        )
        
        # 低價值漏洞
        low_value_panel = Panel(
            "[bold blue]📝 低價值漏洞 ($25-$500)[/bold blue]\n"
            "• 缺失安全標頭\n"
            "• 版本信息洩露\n"
            "• 配置錯誤\n"
            "• 弱密碼策略\n"
            "• 調試信息洩露\n"
            "[green]💰 積少成多，維持生活的重要來源！[/green]",
            title="🏠 生活保障",
            border_style="blue"
        )
        
        console.print(Columns([high_value_panel, medium_value_panel, low_value_panel]))
        
        console.print("\n[bold cyan]🔍 尋找策略:[/bold cyan]")
        console.print("1. 先掃描高價值漏洞 (SQL注入、RCE)")
        console.print("2. 檢查常見配置錯誤 (安全標頭、版本洩露)")
        console.print("3. 測試業務邏輯漏洞")
        console.print("4. 不要忽視低價值漏洞，積少成多!")
    
    def _generate_poc_interactive(self):
        """交互式生成 PoC"""
        if not self.tracker.findings:
            console.print("[yellow]請先添加漏洞發現記錄[/yellow]")
            return
        
        finding = self._select_finding_for_poc()
        if not finding:
            return
        
        poc = self._generate_poc_by_type(finding)
        if poc:
            self._save_and_display_poc(finding, poc)
    
    def _select_finding_for_poc(self) -> Optional[VulnerabilityFinding]:
        """選擇漏洞進行 PoC 生成"""
        console.print("\n[cyan]選擇要生成 PoC 的漏洞:[/cyan]")
        for i, finding in enumerate(self.tracker.findings, 1):
            console.print(f"{i}. [{finding.severity.emoji}] {finding.title} - {finding.target_url}")
        
        try:
            choice = IntPrompt.ask("選擇漏洞", choices=list(range(1, len(self.tracker.findings)+1)))
            return self.tracker.findings[choice - 1]
        except Exception:
            return None
    
    def _generate_poc_by_type(self, finding: VulnerabilityFinding) -> Optional[str]:
        """根據漏洞類型生成 PoC"""
        try:
            if "XSS" in finding.category.description or "Cross-Site Scripting" in finding.category.description:
                return self._generate_xss_poc_interactive(finding)
            elif "SQL" in finding.category.description or "Injection" in finding.category.description:
                return self._generate_sql_poc_interactive(finding)
            elif "CSRF" in finding.category.description:
                return self._generate_csrf_poc_interactive(finding)
            else:
                console.print("[yellow]暫不支持此類型的 PoC 生成，請手動添加[/yellow]")
                return None
        except Exception as e:
            console.print(f"[red]生成 PoC 時出錯: {e}[/red]")
            return None
    
    def _generate_xss_poc_interactive(self, finding: VulnerabilityFinding) -> str:
        """生成 XSS PoC"""
        payload = Prompt.ask("XSS Payload", default="<script>alert('XSS')</script>")
        parameter = Prompt.ask("參數名稱", default="search")
        return self.tracker.poc_generator.generate_xss_poc(finding.target_url, payload, parameter)
    
    def _generate_sql_poc_interactive(self, finding: VulnerabilityFinding) -> str:
        """生成 SQL 注入 PoC"""
        injection_type = Prompt.ask("注入類型", choices=["UNION", "BOOLEAN", "TIME"], default="UNION")
        parameter = Prompt.ask("參數名稱", default="id")
        return self.tracker.poc_generator.generate_sqli_poc(finding.target_url, parameter, injection_type)
    
    def _generate_csrf_poc_interactive(self, finding: VulnerabilityFinding) -> str:
        """生成 CSRF PoC"""
        action_url = Prompt.ask("表單提交 URL")
        console.print("輸入表單參數 (格式: name=value，空行結束):")
        parameters = {}
        while True:
            param = Prompt.ask("參數", default="")
            if not param:
                break
            if "=" in param:
                name, value = param.split("=", 1)
                parameters[name] = value
        return self.tracker.poc_generator.generate_csrf_poc(finding.target_url, action_url, parameters)
    
    def _save_and_display_poc(self, finding: VulnerabilityFinding, poc: str):
        """保存並顯示 PoC"""
        finding.poc_code = poc
        self.tracker.save_findings()
        
        console.print(f"\n[green]✅ PoC 已生成並保存到漏洞記錄 {finding.id}[/green]")
        console.print(Panel(poc, title=f"[green]PoC for {finding.title}[/green]", border_style="green"))
    
    def _show_statistics(self):
        """顯示統計信息"""
        stats = self.tracker.get_bounty_estimate()
        
        if stats['total_findings'] == 0:
            console.print("[yellow]暫無統計數據[/yellow]")
            return
        
        console.print("\n[bold green]📊 Bug Bounty 統計報告[/bold green]")
        
        # 按分類統計
        category_stats = {}
        for finding in self.tracker.findings:
            cat_name = finding.category.description
            if cat_name not in category_stats:
                category_stats[cat_name] = 0
            category_stats[cat_name] += 1
        
        table = Table(title="按漏洞分類統計", show_lines=True)
        table.add_column("漏洞類型", style="cyan")
        table.add_column("數量", style="yellow")
        table.add_column("百分比", style="green")
        
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_findings']) * 100
            table.add_row(category, str(count), f"{percentage:.1f}%")
        
        console.print(table)
        
        # 按狀態統計
        status_stats = {}
        for finding in self.tracker.findings:
            status = finding.status
            if status not in status_stats:
                status_stats[status] = 0
            status_stats[status] += 1
        
        console.print("\n[cyan]按狀態統計:[/cyan]")
        for status, count in status_stats.items():
            console.print(f"• {status}: {count}")
    
    def _handle_add_finding(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """處理添加漏洞發現 API 調用"""
        required_fields = ['title', 'severity', 'category', 'target_url', 'description', 'impact']
        for field in required_fields:
            if field not in parameters:
                return {"success": False, "error": f"Missing required field: {field}"}
        
        try:
            severity = VulnerabilitySeverity[parameters['severity']]
            category = VulnerabilityCategory[parameters['category']]
            
            finding = VulnerabilityFinding(
                id="",
                title=parameters['title'],
                severity=severity,
                category=category,
                target_url=parameters['target_url'],
                description=parameters['description'],
                impact=parameters['impact'],
                reproduction_steps=parameters.get('reproduction_steps', []),
                poc_code=parameters.get('poc_code'),
                discovered_by_tool=parameters.get('discovered_by_tool', 'API'),
            )
            
            finding_id = self.tracker.add_finding(finding)
            return {"success": True, "data": {"finding_id": finding_id}}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_list_findings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """處理列出漏洞發現 API 調用"""
        findings_data = []
        for finding in self.tracker.findings:
            findings_data.append({
                "id": finding.id,
                "title": finding.title,
                "severity": finding.severity.label,
                "category": finding.category.description,
                "target_url": finding.target_url,
                "discovery_time": finding.discovery_time,
                "status": finding.status,
                "estimated_bounty": finding.estimated_bounty
            })
        
        return {"success": True, "data": {"findings": findings_data}}
    
    def _handle_generate_poc(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """處理生成 PoC API 調用"""
        finding_id = parameters.get('finding_id')
        if not finding_id:
            return {"success": False, "error": "Missing finding_id"}
        
        finding = next((f for f in self.tracker.findings if f.id == finding_id), None)
        if not finding:
            return {"success": False, "error": f"Finding {finding_id} not found"}
        
        # 簡單的 PoC 生成邏輯
        if "XSS" in finding.category.description:
            poc = self.tracker.poc_generator.generate_xss_poc(
                finding.target_url,
                parameters.get('payload', "<script>alert('XSS')</script>"),
                parameters.get('parameter', 'search')
            )
        else:
            poc = f"# PoC for {finding.title}\n# Manual testing required for this vulnerability type"
        
        finding.poc_code = poc
        self.tracker.save_findings()
        
        return {"success": True, "data": {"poc": poc}}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            self.tracker.save_findings()
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False


# 註冊功能
CapabilityRegistry.register("bug_bounty_reporting", BugBountyCapability)


if __name__ == "__main__":
    # 測試用例
    async def test_bug_bounty_system():
        capability = BugBountyCapability()
        await capability.initialize()
        
        console.print("[bold red]💰 Bug Bounty 報告系統測試[/bold red]")
        console.print("[yellow]支援各種價值等級的漏洞發現與報告！[/yellow]")
        
        # 顯示互動選單
        capability._show_main_menu()
        
        await capability.cleanup()
    
    # 運行測試
    asyncio.run(test_bug_bounty_system())