"""
AIVA Function Reconnaissance Module
==================================

Based on HackingTool's information_gathering_tools.py, this module provides
comprehensive reconnaissance and information gathering capabilities.

Author: AIVA Development Team
License: MIT
"""

import os
import socket
import subprocess
import webbrowser
import asyncio
import ipaddress
import dns.resolver
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.layout import Layout
from rich.live import Live

from services.integration.capability.registry import CapabilityRegistry, CapabilityType
from services.integration.capability.toolkit import CapabilityToolkit
import logging
LOGGER = logging.getLogger(__name__)

console = Console()
RECON_STYLE = "bold cyan"
SUCCESS_STYLE = "bold green"
ERROR_STYLE = "bold red"
INFO_STYLE = "bold yellow"


class ReconTargetType(Enum):
    """偵察目標類型"""
    IP_ADDRESS = auto()
    HOSTNAME = auto()
    DOMAIN = auto()
    URL = auto()
    EMAIL = auto()
    NETWORK_RANGE = auto()


class ReconStatus(Enum):
    """偵察狀態"""
    PENDING = auto()
    SCANNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ReconTarget:
    """偵察目標"""
    target: str
    target_type: ReconTargetType
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """驗證目標格式"""
        if self.target_type == ReconTargetType.IP_ADDRESS:
            try:
                ipaddress.ip_address(self.target)
            except ValueError:
                raise ValueError(f"無效的IP地址: {self.target}")
        elif self.target_type == ReconTargetType.EMAIL:
            if "@" not in self.target:
                raise ValueError(f"無效的電子郵件格式: {self.target}")


@dataclass
class ReconResult:
    """偵察結果"""
    target: ReconTarget
    scan_type: str
    status: ReconStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    
    def mark_completed(self, data: Dict[str, Any]):
        """標記為完成"""
        self.status = ReconStatus.COMPLETED
        self.data = data
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str):
        """標記為失敗"""
        self.status = ReconStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()


class NetworkScanner:
    """網絡掃描器 - 基於HackingTool的NMAP和PortScan"""
    
    def __init__(self):
        self.console = Console()
    
    async def nmap_scan(self, target: str, scan_type: str = "basic") -> Dict[str, Any]:
        """執行nmap掃描"""
        try:
            if scan_type == "basic":
                cmd = ["nmap", "-sV", target]
            elif scan_type == "aggressive":
                cmd = ["nmap", "-A", "-T4", target]
            elif scan_type == "stealth":
                cmd = ["nmap", "-sS", "-T2", target]
            elif scan_type == "os_detection":
                cmd = ["nmap", "-O", "-Pn", target]
            else:
                cmd = ["nmap", target]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    "output": stdout.decode(),
                    "scan_type": scan_type,
                    "target": target,
                    "success": True
                }
            else:
                return {
                    "error": stderr.decode(),
                    "success": False
                }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def port_scan(self, target: str, ports: str = "1-1000") -> Dict[str, Any]:
        """執行端口掃描"""
        try:
            cmd = ["nmap", "-p", ports, target]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    "output": stdout.decode(),
                    "ports": ports,
                    "target": target,
                    "success": True
                }
            else:
                return {
                    "error": stderr.decode(),
                    "success": False
                }
        except Exception as e:
            return {"error": str(e), "success": False}


class DNSRecon:
    """DNS偵察 - 基於HackingTool的Host2IP功能"""
    
    def __init__(self):
        self.console = Console()
    
    def host_to_ip(self, hostname: str) -> Dict[str, Any]:
        """主機名轉IP地址"""
        try:
            ip = socket.gethostbyname(hostname)
            return {
                "hostname": hostname,
                "ip": ip,
                "success": True
            }
        except socket.gaierror as e:
            return {
                "hostname": hostname,
                "error": str(e),
                "success": False
            }
    
    def dns_lookup(self, domain: str, record_type: str = "A") -> Dict[str, Any]:
        """DNS查詢"""
        try:
            resolver = dns.resolver.Resolver()
            answers = resolver.resolve(domain, record_type)
            
            records = []
            for answer in answers:
                records.append(str(answer))
            
            return {
                "domain": domain,
                "record_type": record_type,
                "records": records,
                "success": True
            }
        except Exception as e:
            return {
                "domain": domain,
                "record_type": record_type,
                "error": str(e),
                "success": False
            }
    
    def reverse_dns(self, ip: str) -> Dict[str, Any]:
        """反向DNS查詢"""
        try:
            hostname = socket.gethostbyaddr(ip)
            return {
                "ip": ip,
                "hostname": hostname[0],
                "aliases": hostname[1],
                "success": True
            }
        except socket.herror as e:
            return {
                "ip": ip,
                "error": str(e),
                "success": False
            }


class WebRecon:
    """Web偵察 - 基於HackingTool的Striker和Breacher功能"""
    
    def __init__(self):
        self.console = Console()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AIVA-Recon/1.0 (Information Gathering)'
        })
    
    def website_info(self, url: str) -> Dict[str, Any]:
        """獲取網站基本信息"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = f"http://{url}"
            
            response = self.session.get(url, timeout=10)
            
            return {
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "server": response.headers.get('Server', 'Unknown'),
                "content_type": response.headers.get('Content-Type', 'Unknown'),
                "content_length": len(response.content),
                "success": True
            }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "success": False
            }
    
    def check_admin_panels(self, domain: str) -> Dict[str, Any]:
        """檢查常見管理面板 - 基於Breacher功能"""
        admin_paths = [
            "admin", "administrator", "admin.php", "admin.html",
            "login", "login.php", "wp-admin", "cpanel",
            "control", "admin/login", "admin/admin",
            "administrator/login", "manager", "admin_area"
        ]
        
        found_panels = []
        checked_paths = []
        
        for path in admin_paths:
            try:
                url = f"http://{domain}/{path}"
                response = self.session.get(url, timeout=5)
                
                checked_paths.append({
                    "path": path,
                    "url": url,
                    "status_code": response.status_code,
                    "accessible": response.status_code == 200
                })
                
                if response.status_code == 200:
                    found_panels.append({
                        "path": path,
                        "url": url,
                        "title": self._extract_title(response.text)
                    })
                    
            except Exception as e:
                checked_paths.append({
                    "path": path,
                    "url": f"http://{domain}/{path}",
                    "error": str(e),
                    "accessible": False
                })
        
        return {
            "domain": domain,
            "found_panels": found_panels,
            "checked_paths": checked_paths,
            "total_found": len(found_panels),
            "success": True
        }
    
    def _extract_title(self, html_content: str) -> str:
        """提取HTML標題"""
        try:
            import re
            title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
            return title_match.group(1).strip() if title_match else "No Title"
        except Exception:
            return "No Title"


class OSINTRecon:
    """開源情報收集 - 基於HackingTool的Infoga和SecretFinder功能"""
    
    def __init__(self):
        self.console = Console()
    
    def email_osint(self, email: str) -> Dict[str, Any]:
        """電子郵件開源情報收集"""
        try:
            domain = email.split('@')[1]
            
            # 基本信息收集
            info = {
                "email": email,
                "domain": domain,
                "mx_records": [],
                "domain_info": {}
            }
            
            # MX記錄查詢
            try:
                import dns.resolver
                mx_records = dns.resolver.resolve(domain, 'MX')
                info["mx_records"] = [str(mx) for mx in mx_records]
            except Exception:
                pass
            
            # 域名基本信息
            try:
                ip = socket.gethostbyname(domain)
                info["domain_info"] = {
                    "ip": ip,
                    "resolved": True
                }
            except Exception:
                info["domain_info"] = {"resolved": False}
            
            return {**info, "success": True}
            
        except Exception as e:
            return {
                "email": email,
                "error": str(e),
                "success": False
            }
    
    def search_secrets(self, domain: str) -> Dict[str, Any]:
        """搜索敏感信息 - 簡化版SecretFinder功能"""
        try:
            # 獲取robots.txt
            robots_info = self._check_robots_txt(domain)
            
            # 檢查常見敏感文件
            sensitive_files = self._check_sensitive_files(domain)
            
            return {
                "domain": domain,
                "robots_txt": robots_info,
                "sensitive_files": sensitive_files,
                "success": True
            }
        except Exception as e:
            return {
                "domain": domain,
                "error": str(e),
                "success": False
            }
    
    def _check_robots_txt(self, domain: str) -> Dict[str, Any]:
        """檢查robots.txt"""
        try:
            url = f"http://{domain}/robots.txt"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return {
                    "exists": True,
                    "content": response.text,
                    "disallowed_paths": self._parse_robots_disallow(response.text)
                }
            else:
                return {"exists": False, "status_code": response.status_code}
        except Exception:
            return {"exists": False, "error": "Request failed"}
    
    def _check_sensitive_files(self, domain: str) -> List[Dict[str, Any]]:
        """檢查敏感文件"""
        sensitive_paths = [
            ".env", ".git/config", "config.php", "database.php",
            "wp-config.php", "settings.py", "web.config"
        ]
        
        results = []
        for path in sensitive_paths:
            try:
                url = f"http://{domain}/{path}"
                response = requests.get(url, timeout=3)
                results.append({
                    "path": path,
                    "url": url,
                    "status_code": response.status_code,
                    "accessible": response.status_code == 200,
                    "size": len(response.content) if response.status_code == 200 else 0
                })
            except Exception:
                results.append({
                    "path": path,
                    "url": f"http://{domain}/{path}",
                    "accessible": False,
                    "error": "Request failed"
                })
        
        return results
    
    def _parse_robots_disallow(self, robots_content: str) -> List[str]:
        """解析robots.txt中的禁止路徑"""
        disallowed = []
        for line in robots_content.split('\n'):
            if line.strip().lower().startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    disallowed.append(path)
        return disallowed


class FunctionReconManager:
    """功能偵察管理器 - 整合所有偵察工具"""
    
    def __init__(self):
        self.console = Console()
        self.network_scanner = NetworkScanner()
        self.dns_recon = DNSRecon()
        self.web_recon = WebRecon()
        self.osint_recon = OSINTRecon()
        self.results: List[ReconResult] = []
        
        LOGGER.info("功能偵察管理器已初始化")
    
    def create_target(self, target: str, target_type: ReconTargetType, description: str = None) -> ReconTarget:
        """創建偵察目標"""
        return ReconTarget(
            target=target,
            target_type=target_type,
            description=description
        )
    
    async def comprehensive_scan(self, target: ReconTarget) -> List[ReconResult]:
        """綜合掃描"""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            
            # 根據目標類型選擇掃描方法
            if target.target_type == ReconTargetType.IP_ADDRESS:
                task = progress.add_task("正在掃描IP地址...", total=3)
                
                # 網絡掃描
                progress.update(task, description="執行網絡掃描...")
                network_result = await self._scan_network(target)
                results.append(network_result)
                progress.advance(task)
                
                # 端口掃描
                progress.update(task, description="執行端口掃描...")
                port_result = await self._scan_ports(target)
                results.append(port_result)
                progress.advance(task)
                
                # 反向DNS
                progress.update(task, description="執行反向DNS查詢...")
                dns_result = self._scan_reverse_dns(target)
                results.append(dns_result)
                progress.advance(task)
                
            elif target.target_type == ReconTargetType.DOMAIN:
                task = progress.add_task("正在掃描域名...", total=4)
                
                # DNS查詢
                progress.update(task, description="執行DNS查詢...")
                dns_result = self._scan_dns(target)
                results.append(dns_result)
                progress.advance(task)
                
                # Web信息收集  
                progress.update(task, description="收集Web信息...")
                web_result = self._scan_web(target)
                results.append(web_result)
                progress.advance(task)
                
                # 管理面板檢查
                progress.update(task, description="檢查管理面板...")
                admin_result = self._check_admin_panels(target)
                results.append(admin_result)
                progress.advance(task)
                
                # OSINT收集
                progress.update(task, description="執行OSINT收集...")
                osint_result = self._scan_osint(target)
                results.append(osint_result)
                progress.advance(task)
                
            elif target.target_type == ReconTargetType.EMAIL:
                task = progress.add_task("正在分析電子郵件...", total=1)
                
                progress.update(task, description="執行電子郵件OSINT...")
                email_result = self._scan_email_osint(target)
                results.append(email_result)
                progress.advance(task)
        
        self.results.extend(results)
        return results
    
    async def _scan_network(self, target: ReconTarget) -> ReconResult:
        """網絡掃描"""
        result = ReconResult(target=target, scan_type="network_scan", status=ReconStatus.SCANNING)
        
        try:
            scan_data = await self.network_scanner.nmap_scan(target.target, "basic")
            result.mark_completed(scan_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    async def _scan_ports(self, target: ReconTarget) -> ReconResult:
        """端口掃描"""
        result = ReconResult(target=target, scan_type="port_scan", status=ReconStatus.SCANNING)
        
        try:
            port_data = await self.network_scanner.port_scan(target.target)
            result.mark_completed(port_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _scan_dns(self, target: ReconTarget) -> ReconResult:
        """DNS掃描"""
        result = ReconResult(target=target, scan_type="dns_scan", status=ReconStatus.SCANNING)
        
        try:
            dns_data = self.dns_recon.dns_lookup(target.target)
            result.mark_completed(dns_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _scan_reverse_dns(self, target: ReconTarget) -> ReconResult:
        """反向DNS掃描"""
        result = ReconResult(target=target, scan_type="reverse_dns", status=ReconStatus.SCANNING)
        
        try:
            reverse_data = self.dns_recon.reverse_dns(target.target)
            result.mark_completed(reverse_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _scan_web(self, target: ReconTarget) -> ReconResult:
        """Web掃描"""
        result = ReconResult(target=target, scan_type="web_scan", status=ReconStatus.SCANNING)
        
        try:
            web_data = self.web_recon.website_info(target.target)
            result.mark_completed(web_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _check_admin_panels(self, target: ReconTarget) -> ReconResult:
        """管理面板檢查"""
        result = ReconResult(target=target, scan_type="admin_panel_check", status=ReconStatus.SCANNING)
        
        try:
            admin_data = self.web_recon.check_admin_panels(target.target)
            result.mark_completed(admin_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _scan_osint(self, target: ReconTarget) -> ReconResult:
        """OSINT掃描"""
        result = ReconResult(target=target, scan_type="osint_scan", status=ReconStatus.SCANNING)
        
        try:
            osint_data = self.osint_recon.search_secrets(target.target)
            result.mark_completed(osint_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _scan_email_osint(self, target: ReconTarget) -> ReconResult:
        """電子郵件OSINT掃描"""
        result = ReconResult(target=target, scan_type="email_osint", status=ReconStatus.SCANNING)
        
        try:
            email_data = self.osint_recon.email_osint(target.target)
            result.mark_completed(email_data)
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def display_results(self, results: List[ReconResult]):
        """顯示掃描結果"""
        for result in results:
            panel_title = f"[{RECON_STYLE}]{result.scan_type.upper()}[/]"
            
            if result.status == ReconStatus.COMPLETED:
                content = self._format_result_data(result)
                style = SUCCESS_STYLE
            elif result.status == ReconStatus.FAILED:
                content = f"[{ERROR_STYLE}]錯誤: {result.error_message}[/]"
                style = ERROR_STYLE
            else:
                content = f"[{INFO_STYLE}]狀態: {result.status.name}[/]"
                style = INFO_STYLE
            
            panel = Panel(
                content,
                title=panel_title,
                border_style=style,
                padding=(1, 2)
            )
            
            self.console.print(panel)
            self.console.print()
    
    def _format_result_data(self, result: ReconResult) -> str:
        """格式化結果數據"""
        if not result.data:
            return "[dim]無數據[/dim]"
        
        lines = []
        
        # 根據掃描類型格式化輸出
        if result.scan_type == "network_scan":
            if result.data.get("success"):
                lines.append("[green]✓[/green] 網絡掃描完成")
                lines.append(f"目標: {result.data.get('target')}")
                lines.append(f"掃描類型: {result.data.get('scan_type')}")
                if "output" in result.data:
                    # 截取輸出的前幾行
                    output_lines = result.data["output"].split('\n')[:10]
                    lines.extend([f"  {line}" for line in output_lines if line.strip()])
            else:
                lines.append(f"[red]✗[/red] 掃描失敗: {result.data.get('error', '未知錯誤')}")
        
        elif result.scan_type == "dns_scan":
            if result.data.get("success"):
                lines.append("[green]✓[/green] DNS查詢完成")
                lines.append(f"域名: {result.data.get('domain')}")
                lines.append(f"記錄類型: {result.data.get('record_type')}")
                records = result.data.get('records', [])
                for record in records[:5]:  # 顯示前5個記錄
                    lines.append(f"  {record}")
            else:
                lines.append(f"[red]✗[/red] DNS查詢失敗: {result.data.get('error', '未知錯誤')}")
        
        elif result.scan_type == "web_scan":
            if result.data.get("success"):
                lines.append("[green]✓[/green] Web信息收集完成")
                lines.append(f"URL: {result.data.get('url')}")
                lines.append(f"狀態碼: {result.data.get('status_code')}")
                lines.append(f"服務器: {result.data.get('server', 'Unknown')}")
                lines.append(f"內容類型: {result.data.get('content_type', 'Unknown')}")
                lines.append(f"內容長度: {result.data.get('content_length', 0)} bytes")
            else:
                lines.append(f"[red]✗[/red] Web掃描失敗: {result.data.get('error', '未知錯誤')}")
        
        elif result.scan_type == "admin_panel_check":
            if result.data.get("success"):
                found = result.data.get('total_found', 0)
                lines.append("[green]✓[/green] 管理面板檢查完成")
                lines.append(f"域名: {result.data.get('domain')}")
                lines.append(f"找到面板: {found} 個")
                
                for panel in result.data.get('found_panels', [])[:3]:  # 顯示前3個
                    lines.append(f"  [yellow]→[/yellow] {panel['path']} - {panel.get('title', 'No Title')}")
            else:
                lines.append("[red]✗[/red] 管理面板檢查失敗")
        
        else:
            # 通用格式化
            lines.append(f"[green]✓[/green] {result.scan_type} 完成")
            for key, value in result.data.items():
                if key not in ['success', 'error']:
                    lines.append(f"{key}: {value}")
        
        # 添加執行時間
        if result.duration:
            lines.append(f"[dim]執行時間: {result.duration:.2f}秒[/dim]")
        
        return '\n'.join(lines)
    
    def get_scan_summary(self) -> Dict[str, Any]:
        """獲取掃描摘要"""
        total = len(self.results)
        completed = len([r for r in self.results if r.status == ReconStatus.COMPLETED])
        failed = len([r for r in self.results if r.status == ReconStatus.FAILED])
        
        return {
            "total_scans": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "scan_types": list({r.scan_type for r in self.results})
        }


class ReconCLI:
    """偵察命令行界面 - 基於HackingTool的Rich UI設計"""
    
    def __init__(self):
        self.console = Console()
        self.manager = FunctionReconManager()
    
    def show_main_menu(self):
        """顯示主菜單"""
        self.console.clear()
        
        title = Panel.fit(
            "[bold cyan]AIVA 功能偵察模組[/bold cyan]\n"
            "基於 HackingTool 設計的綜合信息收集工具",
            border_style=RECON_STYLE
        )
        self.console.print(title)
        self.console.print()
        
        table = Table(title="[bold cyan]偵察工具選單[/bold cyan]", show_lines=True, expand=True)
        table.add_column("選項", justify="center", style="bold yellow")
        table.add_column("功能", justify="left", style="bold green")
        table.add_column("說明", justify="left", style="white")
        
        table.add_row("1", "IP地址掃描", "對單個IP地址進行綜合掃描")
        table.add_row("2", "域名偵察", "對域名進行全面信息收集")
        table.add_row("3", "電子郵件OSINT", "電子郵件開源情報收集")
        table.add_row("4", "自定義掃描", "自定義掃描目標和類型")
        table.add_row("5", "查看歷史記錄", "查看之前的掃描結果")
        table.add_row("6", "掃描統計", "顯示掃描統計信息")
        table.add_row("[red]0[/red]", "[bold red]退出[/bold red]", "退出偵察模組")
        
        self.console.print(table)
        self.console.print()
        
        choice = Prompt.ask("[bold cyan]請選擇功能[/bold cyan]", default="0")
        return choice
    
    async def run_interactive(self):
        """運行交互式界面"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "1":
                    await self.scan_ip_address()
                elif choice == "2":
                    await self.scan_domain()
                elif choice == "3":
                    await self.scan_email()
                elif choice == "4":
                    await self.custom_scan()
                elif choice == "5":
                    self.show_history()
                elif choice == "6":
                    self.show_statistics()
                elif choice == "0":
                    self.console.print("[bold green]感謝使用 AIVA 偵察模組！[/bold green]")
                    break
                else:
                    self.console.print("[bold red]無效選項，請重新選擇[/bold red]")
                    
                if choice != "0":
                    await asyncio.to_thread(input, "\n按 Enter 繼續...")
                    
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]用戶中斷操作[/bold yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]發生錯誤: {e}[/bold red]")
                await asyncio.to_thread(input, "\n按 Enter 繼續...")
    
    async def scan_ip_address(self):
        """IP地址掃描"""
        self.console.print(Panel("IP地址綜合掃描", style=RECON_STYLE))
        
        ip = Prompt.ask("[bold]請輸入IP地址[/bold]", default="8.8.8.8")
        
        try:
            target = self.manager.create_target(ip, ReconTargetType.IP_ADDRESS, "IP地址掃描")
            results = await self.manager.comprehensive_scan(target)
            
            self.console.print(f"\n[bold green]✓ 掃描完成！共執行了 {len(results)} 項掃描[/bold green]\n")
            self.manager.display_results(results)
            
        except Exception as e:
            self.console.print(f"[bold red]掃描失敗: {e}[/bold red]")
    
    async def scan_domain(self):
        """域名偵察"""
        self.console.print(Panel("域名綜合偵察", style=RECON_STYLE))
        
        domain = Prompt.ask("[bold]請輸入域名[/bold]", default="example.com")
        
        try:
            target = self.manager.create_target(domain, ReconTargetType.DOMAIN, "域名偵察")
            results = await self.manager.comprehensive_scan(target)
            
            self.console.print(f"\n[bold green]✓ 偵察完成！共執行了 {len(results)} 項掃描[/bold green]\n")
            self.manager.display_results(results)
            
        except Exception as e:
            self.console.print(f"[bold red]偵察失敗: {e}[/bold red]")
    
    async def scan_email(self):
        """電子郵件OSINT"""
        self.console.print(Panel("電子郵件開源情報收集", style=RECON_STYLE))
        
        email = Prompt.ask("[bold]請輸入電子郵件地址[/bold]", default="user@example.com")
        
        try:
            target = self.manager.create_target(email, ReconTargetType.EMAIL, "電子郵件OSINT")
            results = await self.manager.comprehensive_scan(target)
            
            self.console.print(f"\n[bold green]✓ OSINT收集完成！共執行了 {len(results)} 項掃描[/bold green]\n")
            self.manager.display_results(results)
            
        except Exception as e:
            self.console.print(f"[bold red]OSINT收集失敗: {e}[/bold red]")
    
    async def custom_scan(self):
        """自定義掃描"""
        self.console.print(Panel("自定義掃描配置", style=RECON_STYLE))
        
        target_input = Prompt.ask("[bold]請輸入掃描目標[/bold]")
        
        # 自動檢測目標類型
        target_type = self._detect_target_type(target_input)
        
        self.console.print(f"[bold yellow]檢測到目標類型: {target_type.name}[/bold yellow]")
        
        try:
            target = self.manager.create_target(target_input, target_type, "自定義掃描")
            results = await self.manager.comprehensive_scan(target)
            
            self.console.print(f"\n[bold green]✓ 自定義掃描完成！共執行了 {len(results)} 項掃描[/bold green]\n")
            self.manager.display_results(results)
            
        except Exception as e:
            self.console.print(f"[bold red]掃描失敗: {e}[/bold red]")
    
    def _detect_target_type(self, target: str) -> ReconTargetType:
        """自動檢測目標類型"""
        try:
            ipaddress.ip_address(target)
            return ReconTargetType.IP_ADDRESS
        except ValueError:
            pass
        
        if "@" in target:
            return ReconTargetType.EMAIL
        
        if target.startswith(('http://', 'https://')):
            return ReconTargetType.URL
        
        # 默認為域名
        return ReconTargetType.DOMAIN
    
    def show_history(self):
        """顯示歷史記錄"""
        self.console.print(Panel("掃描歷史記錄", style=RECON_STYLE))
        
        if not self.manager.results:
            self.console.print("[bold yellow]暫無掃描記錄[/bold yellow]")
            return
        
        table = Table(title="歷史掃描記錄", show_lines=True)
        table.add_column("時間", style="cyan")
        table.add_column("目標", style="green")
        table.add_column("類型", style="yellow")
        table.add_column("掃描類型", style="blue")
        table.add_column("狀態", style="magenta")
        table.add_column("耗時", style="white")
        
        for result in self.manager.results[-20:]:  # 顯示最近20條記錄
            status_style = SUCCESS_STYLE if result.status == ReconStatus.COMPLETED else ERROR_STYLE
            duration = f"{result.duration:.2f}s" if result.duration else "N/A"
            
            table.add_row(
                result.started_at.strftime("%H:%M:%S"),
                result.target.target,
                result.target.target_type.name,
                result.scan_type,
                f"[{status_style}]{result.status.name}[/]",
                duration
            )
        
        self.console.print(table)
    
    def show_statistics(self):
        """顯示統計信息"""
        self.console.print(Panel("掃描統計信息", style=RECON_STYLE))
        
        summary = self.manager.get_scan_summary()
        
        stats_table = Table(title="統計數據", show_lines=True)
        stats_table.add_column("項目", style="cyan")
        stats_table.add_column("數值", style="green")
        
        stats_table.add_row("總掃描次數", str(summary["total_scans"]))
        stats_table.add_row("成功掃描", str(summary["completed"]))
        stats_table.add_row("失敗掃描", str(summary["failed"]))
        stats_table.add_row("成功率", f"{summary['success_rate']:.1f}%")
        
        self.console.print(stats_table)
        
        if summary["scan_types"]:
            types_table = Table(title="掃描類型", show_lines=True)
            types_table.add_column("掃描類型", style="yellow")
            
            for scan_type in summary["scan_types"]:
                types_table.add_row(scan_type)
            
            self.console.print(types_table)


# 註冊到能力系統
async def register_recon_capabilities():
    """註冊偵察能力到系統"""
    from .models import CapabilityRecord, ProgrammingLanguage
    
    registry = CapabilityRegistry()
    
    # 註冊網絡掃描能力
    network_capability = CapabilityRecord(
        id="recon.network_scan",
        name="網絡掃描",
        description="使用Nmap進行網絡發現和端口掃描",
        module="function_recon",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="NetworkScanner",
        capability_type=CapabilityType.SCANNER,
        dependencies=["nmap"],
        tags=["reconnaissance", "network", "security"]
    )
    await registry.register_capability(network_capability)
    
    # 註冊DNS偵察能力
    dns_capability = CapabilityRecord(
        id="recon.dns",
        name="DNS偵察",
        description="DNS查詢和反向解析",
        module="function_recon",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="DNSRecon",
        capability_type=CapabilityType.ANALYZER,
        dependencies=["dnspython"],
        tags=["reconnaissance", "dns", "network"]
    )
    await registry.register_capability(dns_capability)
    
    # 註冊Web偵察能力
    web_capability = CapabilityRecord(
        id="recon.web",
        name="Web偵察",
        description="網站信息收集和管理面板發現",
        module="function_recon",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="WebRecon",
        capability_type=CapabilityType.SCANNER,
        dependencies=["requests"],
        tags=["reconnaissance", "web", "osint"]
    )
    await registry.register_capability(web_capability)
    
    # 註冊OSINT能力
    osint_capability = CapabilityRecord(
        id="recon.osint",
        name="開源情報收集",
        description="電子郵件和敏感信息收集",
        module="function_recon",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="OSINTRecon",
        capability_type=CapabilityType.ANALYZER,
        dependencies=["requests", "dnspython"],
        tags=["reconnaissance", "osint", "intelligence"]
    )
    await registry.register_capability(osint_capability)
    
    # 註冊綜合偵察管理器
    manager_capability = CapabilityRecord(
        id="recon.manager",
        name="偵察管理器",
        description="綜合偵察和信息收集管理",
        module="function_recon",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="FunctionReconManager",
        capability_type=CapabilityType.UTILITY,
        dependencies=["nmap", "requests", "dnspython"],
        tags=["reconnaissance", "management", "orchestration"]
    )
    await registry.register_capability(manager_capability)
    
    LOGGER.info("偵察能力已註冊到系統")


# 主程序入口
async def main():
    """主程序"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]AIVA 功能偵察模組[/bold cyan]\n"
        "基於 HackingTool 的綜合信息收集工具",
        border_style=RECON_STYLE
    ))
    
    # 註冊能力
    register_recon_capabilities()
    
    # 創建CLI並運行
    cli = ReconCLI()
    await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())