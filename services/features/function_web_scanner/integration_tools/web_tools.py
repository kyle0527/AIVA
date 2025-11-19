#!/usr/bin/env python3
"""
AIVA Web Attack Module - Task 11
基於 HackingTool webattack.py 實現的網絡攻擊工具集成
包含網站安全掃描、子域名發現、目錄掃描、弱點檢測等功能

✅ **已移至 Features 模組** ✅
本檔案提供 Web 掃描工具整合功能（子域名發現、目錄掃描等）。
符合 AIVA 五大模組架構原則：
- Features 模組負責**實際執行測試**
- 提供與外部工具的整合接口
"""

import asyncio
import json
import os
import re
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import dns.resolver
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from services.aiva_common.schemas import APIResponse
from services.config.settings_manager import SettingsManager
from utilities.logger_setup import setup_logger


logger = setup_logger(__name__)
console = Console()


@dataclass
class WebTarget:
    """網絡攻擊目標資訊"""
    url: str
    domain: str = ""
    ip: str = ""
    ports: List[int] = field(default_factory=list)
    subdomains: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.domain:
            parsed = urllib.parse.urlparse(self.url)
            self.domain = parsed.netloc or parsed.path


@dataclass
class ScanResult:
    """掃描結果資料結構"""
    target: str
    scan_type: str
    timestamp: datetime
    status: str  # success, failed, partial
    data: Dict[str, Any]
    error: Optional[str] = None


class SubdomainEnumerator:
    """子域名枚舉器 - 基於 Sublist3r 模式"""
    
    def __init__(self):
        self.found_subdomains: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def enumerate_subdomains(self, domain: str, timeout: int = 30) -> List[str]:
        """枚舉子域名"""
        try:
            self.found_subdomains.clear()
            
            # 並行運行多種方法
            tasks = [
                self._enumerate_crt_sh(domain),
                self._enumerate_dns_brute(domain),
                self._enumerate_search_engines(domain),
                self._enumerate_common_subdomains(domain)
            ]
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                self.session = session
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return sorted(self.found_subdomains)
            
        except Exception as e:
            logger.error(f"子域名枚舉失敗: {e}")
            return []
    
    async def _enumerate_crt_sh(self, domain: str) -> None:
        """使用 crt.sh 枚舉子域名"""
        try:
            url = f"https://crt.sh/?q=%25.{domain}&output=json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for entry in data:
                        name = entry.get('name_value', '')
                        if name and name.endswith(domain):
                            self.found_subdomains.add(name.strip())
        except Exception as e:
            logger.debug(f"crt.sh 枚舉失敗: {e}")
    
    async def _enumerate_dns_brute(self, domain: str) -> None:
        """DNS 暴力破解枚舉"""
        common_prefixes = [
            'www', 'mail', 'ftp', 'admin', 'test', 'dev', 'api', 'app',
            'blog', 'shop', 'forum', 'support', 'help', 'news', 'portal',
            'secure', 'vpn', 'remote', 'proxy', 'gateway', 'server'
        ]
        
        resolver = dns.resolver.Resolver()
        resolver.timeout = 2
        resolver.lifetime = 5
        
        for prefix in common_prefixes:
            subdomain = f"{prefix}.{domain}"
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, resolver.resolve, subdomain
                )
                self.found_subdomains.add(subdomain)
            except Exception:
                continue
    
    async def _enumerate_search_engines(self, domain: str) -> None:
        """搜索引擎枚舉 (模擬)"""
        # 這裡可以實現 Google、Bing 等搜索引擎的查詢邏輯
        # 由於 API 限制，這裡僅作為示例結構
        pass
    
    async def _enumerate_common_subdomains(self, domain: str) -> None:
        """枚舉常見子域名"""
        common_subs = [
            'www', 'mail', 'webmail', 'ftp', 'localhost', 'www1', 'www2',
            'ns1', 'ns2', 'mx', 'pop', 'smtp', 'imap', 'blog', 'forum'
        ]
        
        for sub in common_subs:
            subdomain = f"{sub}.{domain}"
            try:
                # 簡單的 HTTP 檢測
                async with self.session.get(f"http://{subdomain}", allow_redirects=False) as response:
                    if response.status < 400:
                        self.found_subdomains.add(subdomain)
            except Exception:
                continue


class DirectoryScanner:
    """目錄掃描器 - 基於 Dirb 模式"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.found_directories: List[Dict[str, Any]] = []
        
    async def scan_directories(self, target_url: str, wordlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """掃描目錄和文件"""
        try:
            self.found_directories.clear()
            
            if not wordlist:
                wordlist = self._get_default_wordlist()
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5),
                connector=aiohttp.TCPConnector(limit=20)
            ) as session:
                self.session = session
                
                # 創建任務
                tasks = []
                for path in wordlist:
                    url = urllib.parse.urljoin(target_url, path)
                    tasks.append(self._check_path(url, path))
                
                # 並行執行掃描
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return self.found_directories
            
        except Exception as e:
            logger.error(f"目錄掃描失敗: {e}")
            return []
    
    async def _check_path(self, url: str, path: str) -> None:
        """檢查單個路徑"""
        try:
            async with self.session.get(url, allow_redirects=False) as response:
                result = {
                    'path': path,
                    'url': url,
                    'status': response.status,
                    'size': len(await response.read()) if response.status == 200 else 0,
                    'headers': dict(response.headers)
                }
                
                if response.status in [200, 301, 302, 403]:
                    self.found_directories.append(result)
                    
        except Exception as e:
            logger.debug(f"路徑檢查失敗 {url}: {e}")
    
    def _get_default_wordlist(self) -> List[str]:
        """獲取默認詞典"""
        return [
            'admin/', 'administrator/', 'login/', 'wp-admin/', 'phpmyadmin/',
            'uploads/', 'images/', 'img/', 'css/', 'js/', 'api/', 'docs/',
            'test/', 'demo/', 'config/', 'backup/', 'db/', 'database/',
            'robots.txt', 'sitemap.xml', '.htaccess', 'phpinfo.php',
            'readme.txt', 'changelog.txt', 'install.php', 'setup.php'
        ]


class VulnerabilityScanner:
    """漏洞掃描器"""
    
    def __init__(self):
        self.vulnerabilities: List[Dict[str, Any]] = []
        
    async def scan_vulnerabilities(self, target_url: str) -> List[Dict[str, Any]]:
        """掃描常見漏洞"""
        try:
            self.vulnerabilities.clear()
            
            # 並行執行不同類型的漏洞掃描
            tasks = [
                self._scan_xss(target_url),
                self._scan_sql_injection(target_url),
                self._scan_directory_traversal(target_url),
                self._scan_security_headers(target_url),
                self._scan_clickjacking(target_url)
            ]
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                self.session = session
                await asyncio.gather(*tasks, return_exceptions=True)
            
            return self.vulnerabilities
            
        except Exception as e:
            logger.error(f"漏洞掃描失敗: {e}")
            return []
    
    async def _scan_xss(self, target_url: str) -> None:
        """XSS 漏洞掃描"""
        payloads = [
            "<script>alert('XSS')</script>",
            "'\"><script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ]
        
        for payload in payloads:
            try:
                # 嘗試 GET 參數注入
                test_url = f"{target_url}?test={urllib.parse.quote(payload)}"
                async with self.session.get(test_url) as response:
                    content = await response.text()
                    if payload in content or 'alert(' in content:
                        self.vulnerabilities.append({
                            'type': 'XSS',
                            'severity': 'Medium',
                            'location': test_url,
                            'payload': payload,
                            'description': 'Possible XSS vulnerability detected'
                        })
            except Exception:
                continue
    
    async def _scan_sql_injection(self, target_url: str) -> None:
        """SQL 注入漏洞掃描"""
        payloads = ["'", "1' OR '1'='1", "'; DROP TABLE users; --", "1' UNION SELECT NULL--"]
        
        for payload in payloads:
            try:
                test_url = f"{target_url}?id={urllib.parse.quote(payload)}"
                async with self.session.get(test_url) as response:
                    content = await response.text().lower()
                    error_indicators = ['sql', 'mysql', 'oracle', 'postgresql', 'syntax error']
                    
                    if any(indicator in content for indicator in error_indicators):
                        self.vulnerabilities.append({
                            'type': 'SQL Injection',
                            'severity': 'High',
                            'location': test_url,
                            'payload': payload,
                            'description': 'Possible SQL injection vulnerability detected'
                        })
            except Exception:
                continue
    
    async def _scan_directory_traversal(self, target_url: str) -> None:
        """目錄遍歷漏洞掃描"""
        payloads = ["../../../etc/passwd", "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"]
        
        for payload in payloads:
            try:
                test_url = f"{target_url}?file={urllib.parse.quote(payload)}"
                async with self.session.get(test_url) as response:
                    content = await response.text()
                    if 'root:' in content or '[drivers]' in content:
                        self.vulnerabilities.append({
                            'type': 'Directory Traversal',
                            'severity': 'High',
                            'location': test_url,
                            'payload': payload,
                            'description': 'Directory traversal vulnerability detected'
                        })
            except Exception:
                continue
    
    async def _scan_security_headers(self, target_url: str) -> None:
        """安全標頭檢查"""
        try:
            async with self.session.get(target_url) as response:
                headers = response.headers
                missing_headers = []
                
                security_headers = {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'DENY',
                    'X-XSS-Protection': '1; mode=block',
                    'Strict-Transport-Security': 'max-age=31536000',
                    'Content-Security-Policy': '*'
                }
                
                for header, expected in security_headers.items():
                    if header not in headers:
                        missing_headers.append(header)
                
                if missing_headers:
                    self.vulnerabilities.append({
                        'type': 'Missing Security Headers',
                        'severity': 'Low',
                        'location': target_url,
                        'description': f'Missing headers: {", ".join(missing_headers)}'
                    })
        except Exception:
            pass
    
    async def _scan_clickjacking(self, target_url: str) -> None:
        """點擊劫持檢查"""
        try:
            async with self.session.get(target_url) as response:
                headers = response.headers
                if 'X-Frame-Options' not in headers and 'Content-Security-Policy' not in headers:
                    self.vulnerabilities.append({
                        'type': 'Clickjacking',
                        'severity': 'Medium',
                        'location': target_url,
                        'description': 'Website may be vulnerable to clickjacking attacks'
                    })
        except Exception:
            pass


class TechnologyDetector:
    """技術檢測器"""
    
    def __init__(self):
        self.technologies: List[str] = []
        
    async def detect_technologies(self, target_url: str) -> List[str]:
        """檢測網站使用的技術"""
        try:
            self.technologies.clear()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(target_url) as response:
                    headers = response.headers
                    content = await response.text()
                    
                    # 檢測服務器技術
                    server = headers.get('Server', '')
                    if server:
                        self.technologies.append(f"Server: {server}")
                    
                    # 檢測框架和CMS
                    self._detect_frameworks(content)
                    
                    # 檢測JavaScript庫
                    self._detect_js_libraries(content)
                    
                    # 檢測CSS框架
                    self._detect_css_frameworks(content)
            
            return self.technologies
            
        except Exception as e:
            logger.error(f"技術檢測失敗: {e}")
            return []
    
    def _detect_frameworks(self, content: str) -> None:
        """檢測框架和CMS"""
        framework_patterns = {
            'WordPress': ['wp-content', 'wp-includes', 'wordpress'],
            'Drupal': ['drupal', 'sites/default'],
            'Joomla': ['joomla', 'option=com_'],
            'Laravel': ['laravel_session', 'laravel'],
            'Django': ['djangoproject', 'csrfmiddlewaretoken'],
            'React': ['react', 'reactjs'],
            'Angular': ['angular', 'ng-'],
            'Vue.js': ['vue.js', 'vuejs']
        }
        
        content_lower = content.lower()
        for framework, patterns in framework_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                self.technologies.append(f"Framework: {framework}")
    
    def _detect_js_libraries(self, content: str) -> None:
        """檢測JavaScript庫"""
        js_patterns = {
            'jQuery': ['jquery', 'jQuery'],
            'Bootstrap': ['bootstrap'],
            'Lodash': ['lodash'],
            'Moment.js': ['moment.js'],
            'Chart.js': ['chart.js']
        }
        
        for library, patterns in js_patterns.items():
            if any(pattern in content for pattern in patterns):
                self.technologies.append(f"JS Library: {library}")
    
    def _detect_css_frameworks(self, content: str) -> None:
        """檢測CSS框架"""
        css_patterns = {
            'Bootstrap': ['bootstrap.css', 'bootstrap.min.css'],
            'Foundation': ['foundation.css'],
            'Bulma': ['bulma.css'],
            'Semantic UI': ['semantic.css']
        }
        
        for framework, patterns in css_patterns.items():
            if any(pattern in content for pattern in patterns):
                self.technologies.append(f"CSS Framework: {framework}")


class WebAttackManager:
    """網絡攻擊管理器 - 主要控制類"""
    
    def __init__(self):
        self.subdomain_enumerator = SubdomainEnumerator()
        self.directory_scanner = DirectoryScanner()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.technology_detector = TechnologyDetector()
        self.scan_results: List[ScanResult] = []
        
    async def comprehensive_scan(self, target_url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """執行綜合掃描"""
        options = options or {}
        target = WebTarget(url=target_url)
        
        console.print(f"\n[bold cyan]開始綜合掃描: {target_url}[/bold cyan]")
        
        results = {
            'target': target_url,
            'timestamp': datetime.now().isoformat(),
            'subdomains': [],
            'directories': [],
            'vulnerabilities': [],
            'technologies': [],
            'scan_summary': {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False
        ) as progress:
            
            # 子域名枚舉
            if options.get('subdomain_scan', True):
                task = progress.add_task("枚舉子域名...", total=None)
                try:
                    subdomains = await self.subdomain_enumerator.enumerate_subdomains(target.domain)
                    results['subdomains'] = subdomains
                    progress.update(task, description=f"找到 {len(subdomains)} 個子域名")
                except Exception as e:
                    logger.error(f"子域名掃描失敗: {e}")
                finally:
                    progress.remove_task(task)
            
            # 目錄掃描
            if options.get('directory_scan', True):
                task = progress.add_task("掃描目錄結構...", total=None)
                try:
                    directories = await self.directory_scanner.scan_directories(target_url)
                    results['directories'] = directories
                    progress.update(task, description=f"找到 {len(directories)} 個目錄/文件")
                except Exception as e:
                    logger.error(f"目錄掃描失敗: {e}")
                finally:
                    progress.remove_task(task)
            
            # 漏洞掃描
            if options.get('vulnerability_scan', True):
                task = progress.add_task("掃描安全漏洞...", total=None)
                try:
                    vulnerabilities = await self.vulnerability_scanner.scan_vulnerabilities(target_url)
                    results['vulnerabilities'] = vulnerabilities
                    progress.update(task, description=f"發現 {len(vulnerabilities)} 個潛在漏洞")
                except Exception as e:
                    logger.error(f"漏洞掃描失敗: {e}")
                finally:
                    progress.remove_task(task)
            
            # 技術檢測
            if options.get('technology_scan', True):
                task = progress.add_task("檢測使用技術...", total=None)
                try:
                    technologies = await self.technology_detector.detect_technologies(target_url)
                    results['technologies'] = technologies
                    progress.update(task, description=f"識別 {len(technologies)} 種技術")
                except Exception as e:
                    logger.error(f"技術檢測失敗: {e}")
                finally:
                    progress.remove_task(task)
        
        # 生成掃描摘要
        results['scan_summary'] = {
            'total_subdomains': len(results['subdomains']),
            'total_directories': len(results['directories']),
            'total_vulnerabilities': len(results['vulnerabilities']),
            'total_technologies': len(results['technologies']),
            'high_severity_vulns': len([v for v in results['vulnerabilities'] if v.get('severity') == 'High']),
            'medium_severity_vulns': len([v for v in results['vulnerabilities'] if v.get('severity') == 'Medium']),
            'low_severity_vulns': len([v for v in results['vulnerabilities'] if v.get('severity') == 'Low'])
        }
        
        # 保存結果
        scan_result = ScanResult(
            target=target_url,
            scan_type='comprehensive',
            timestamp=datetime.now(),
            status='success',
            data=results
        )
        self.scan_results.append(scan_result)
        
        return results


class WebAttackCLI:
    """網絡攻擊命令行介面"""
    
    def __init__(self, manager: WebAttackManager):
        self.manager = manager
        
    def show_main_menu(self):
        """顯示主選單"""
        console.print("\n")
        panel = Panel.fit(
            "[bold magenta]AIVA 網絡攻擊工具 (Task 11)[/bold magenta]\n"
            "基於 HackingTool webattack.py 實現的網站安全掃描工具集",
            border_style="purple"
        )
        console.print(panel)
        
        table = Table(title="[bold cyan]可用功能[/bold cyan]", show_lines=True)
        table.add_column("選項", justify="center", style="bold yellow")
        table.add_column("功能", justify="left", style="bold green")
        table.add_column("描述", justify="left", style="white")
        
        options = [
            ("1", "綜合掃描", "執行完整的網站安全掃描"),
            ("2", "子域名枚舉", "發現目標域名的子域名"),
            ("3", "目錄掃描", "掃描網站目錄和文件"),
            ("4", "漏洞掃描", "檢測常見安全漏洞"),
            ("5", "技術檢測", "識別網站使用的技術棧"),
            ("6", "查看掃描歷史", "查看之前的掃描結果"),
            ("7", "導出結果", "導出掃描結果到文件"),
            ("99", "退出", "返回上級選單")
        ]
        
        for option, function, description in options:
            table.add_row(option, function, description)
        
        console.print(table)
        
        try:
            choice = console.input("\n[bold cyan]請選擇功能 (1-7, 99): [/bold cyan]")
            return choice.strip()
        except KeyboardInterrupt:
            return "99"
    
    async def run_interactive(self):
        """運行交互式介面"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "1":
                    await self._comprehensive_scan()
                elif choice == "2":
                    await self._subdomain_enumeration()
                elif choice == "3":
                    await self._directory_scan()
                elif choice == "4":
                    await self._vulnerability_scan()
                elif choice == "5":
                    await self._technology_detection()
                elif choice == "6":
                    self._show_scan_history()
                elif choice == "7":
                    await self._export_results()
                elif choice == "99":
                    console.print("[bold yellow]感謝使用 AIVA 網絡攻擊工具![/bold yellow]")
                    break
                else:
                    console.print("[bold red]無效選項，請重新選擇[/bold red]")
                    
            except KeyboardInterrupt:
                console.print("\n[bold yellow]程序已中斷[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]錯誤: {e}[/bold red]")
    
    async def _comprehensive_scan(self):
        """執行綜合掃描"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        if not target_url.startswith(('http://', 'https://')):
            target_url = f"http://{target_url}"
        
        try:
            results = await self.manager.comprehensive_scan(target_url)
            self._display_scan_results(results)
        except Exception as e:
            console.print(f"[bold red]掃描失敗: {e}[/bold red]")
    
    async def _subdomain_enumeration(self):
        """子域名枚舉"""
        domain = console.input("[bold cyan]請輸入目標域名: [/bold cyan]").strip()
        if not domain:
            console.print("[bold red]域名不能為空[/bold red]")
            return
        
        try:
            subdomains = await self.manager.subdomain_enumerator.enumerate_subdomains(domain)
            
            if subdomains:
                table = Table(title=f"[bold green]找到的子域名 ({len(subdomains)})[/bold green]")
                table.add_column("子域名", style="cyan")
                
                for subdomain in subdomains:
                    table.add_row(subdomain)
                
                console.print(table)
            else:
                console.print("[bold yellow]未找到子域名[/bold yellow]")
                
        except Exception as e:
            console.print(f"[bold red]子域名枚舉失敗: {e}[/bold red]")
    
    async def _directory_scan(self):
        """目錄掃描"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        if not target_url.startswith(('http://', 'https://')):
            target_url = f"http://{target_url}"
        
        try:
            directories = await self.manager.directory_scanner.scan_directories(target_url)
            
            if directories:
                table = Table(title=f"[bold green]發現的目錄/文件 ({len(directories)})[/bold green]")
                table.add_column("路徑", style="cyan")
                table.add_column("狀態碼", style="yellow")
                table.add_column("大小", style="green")
                
                for dir_info in directories:
                    size_str = f"{dir_info['size']} bytes" if dir_info['size'] > 0 else "-"
                    table.add_row(dir_info['path'], str(dir_info['status']), size_str)
                
                console.print(table)
            else:
                console.print("[bold yellow]未發現可訪問的目錄/文件[/bold yellow]")
                
        except Exception as e:
            console.print(f"[bold red]目錄掃描失敗: {e}[/bold red]")
    
    async def _vulnerability_scan(self):
        """漏洞掃描"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        if not target_url.startswith(('http://', 'https://')):
            target_url = f"http://{target_url}"
        
        try:
            vulnerabilities = await self.manager.vulnerability_scanner.scan_vulnerabilities(target_url)
            
            if vulnerabilities:
                table = Table(title=f"[bold red]發現的漏洞 ({len(vulnerabilities)})[/bold red]")
                table.add_column("類型", style="red")
                table.add_column("嚴重程度", style="yellow")
                table.add_column("描述", style="white")
                
                for vuln in vulnerabilities:
                    severity_style = {
                        'High': 'bold red',
                        'Medium': 'bold yellow',
                        'Low': 'bold green'
                    }.get(vuln['severity'], 'white')
                    
                    table.add_row(
                        vuln['type'],
                        Text(vuln['severity'], style=severity_style),
                        vuln['description']
                    )
                
                console.print(table)
            else:
                console.print("[bold green]未發現明顯漏洞[/bold green]")
                
        except Exception as e:
            console.print(f"[bold red]漏洞掃描失敗: {e}[/bold red]")
    
    async def _technology_detection(self):
        """技術檢測"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        if not target_url.startswith(('http://', 'https://')):
            target_url = f"http://{target_url}"
        
        try:
            technologies = await self.manager.technology_detector.detect_technologies(target_url)
            
            if technologies:
                table = Table(title=f"[bold blue]檢測到的技術 ({len(technologies)})[/bold blue]")
                table.add_column("技術", style="blue")
                
                for tech in technologies:
                    table.add_row(tech)
                
                console.print(table)
            else:
                console.print("[bold yellow]未檢測到特定技術[/bold yellow]")
                
        except Exception as e:
            console.print(f"[bold red]技術檢測失敗: {e}[/bold red]")
    
    def _show_scan_history(self):
        """顯示掃描歷史"""
        if not self.manager.scan_results:
            console.print("[bold yellow]暫無掃描歷史[/bold yellow]")
            return
        
        table = Table(title="[bold cyan]掃描歷史[/bold cyan]")
        table.add_column("時間", style="green")
        table.add_column("目標", style="cyan")
        table.add_column("類型", style="yellow")
        table.add_column("狀態", style="white")
        
        for result in self.manager.scan_results[-10:]:  # 只顯示最近10次
            table.add_row(
                result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                result.target,
                result.scan_type,
                result.status
            )
        
        console.print(table)
    
    async def _export_results(self):
        """導出結果"""
        if not self.manager.scan_results:
            console.print("[bold yellow]暫無掃描結果可導出[/bold yellow]")
            return
        
        output_dir = Path("reports/web_attack")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"web_attack_results_{timestamp}.json"
        
        try:
            export_data = {
                'export_time': datetime.now().isoformat(),
                'total_scans': len(self.manager.scan_results),
                'results': []
            }
            
            for result in self.manager.scan_results:
                export_data['results'].append({
                    'target': result.target,
                    'scan_type': result.scan_type,
                    'timestamp': result.timestamp.isoformat(),
                    'status': result.status,
                    'data': result.data,
                    'error': result.error
                })
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            console.print(f"[bold green]結果已導出到: {filename}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]導出失敗: {e}[/bold red]")
    
    def _display_scan_results(self, results: Dict[str, Any]):
        """顯示掃描結果"""
        summary = results['scan_summary']
        
        # 掃描摘要
        summary_table = Table(title="[bold cyan]掃描摘要[/bold cyan]")
        summary_table.add_column("項目", style="yellow")
        summary_table.add_column("數量", style="green")
        
        summary_table.add_row("子域名", str(summary['total_subdomains']))
        summary_table.add_row("目錄/文件", str(summary['total_directories']))
        summary_table.add_row("總漏洞", str(summary['total_vulnerabilities']))
        summary_table.add_row("高危漏洞", str(summary['high_severity_vulns']))
        summary_table.add_row("中危漏洞", str(summary['medium_severity_vulns']))
        summary_table.add_row("低危漏洞", str(summary['low_severity_vulns']))
        summary_table.add_row("檢測技術", str(summary['total_technologies']))
        
        console.print(summary_table)
        
        # 如果有漏洞，顯示詳細信息
        if results['vulnerabilities']:
            vuln_table = Table(title="[bold red]發現的漏洞[/bold red]")
            vuln_table.add_column("類型", style="red")
            vuln_table.add_column("嚴重程度", style="yellow")
            vuln_table.add_column("位置", style="cyan")
            
            for vuln in results['vulnerabilities'][:10]:  # 只顯示前10個
                vuln_table.add_row(vuln['type'], vuln['severity'], vuln.get('location', ''))
            
            console.print(vuln_table)


class WebAttackCapability(BaseCapability):
    """網絡攻擊能力類"""
    
    def __init__(self):
        super().__init__()
        self.manager = WebAttackManager()
        self.cli = WebAttackCLI(self.manager)
        
    @property
    def name(self) -> str:
        return "web_attack"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "網絡攻擊工具集 - 網站安全掃描、漏洞檢測、子域名枚舉"
    
    @property
    def dependencies(self) -> List[str]:
        return ["aiohttp", "dnspython", "requests", "rich"]
    
    async def initialize(self) -> bool:
        """初始化網絡攻擊能力"""
        try:
            logger.info("初始化網絡攻擊能力...")
            
            # 檢查必要的依賴
            required_modules = ['aiohttp', 'dns.resolver', 'requests']
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    logger.warning(f"缺少依賴模組: {module}")
            
            logger.info("網絡攻擊能力初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"網絡攻擊能力初始化失敗: {e}")
            return False
    
    async def execute(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """執行網絡攻擊命令"""
        try:
            parameters = parameters or {}
            
            if command == "comprehensive_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                options = parameters.get('options', {})
                results = await self.manager.comprehensive_scan(target_url, options)
                return {'success': True, 'data': results}
            
            elif command == "subdomain_scan":
                domain = parameters.get('domain')
                if not domain:
                    return {'success': False, 'error': 'Missing domain parameter'}
                
                subdomains = await self.manager.subdomain_enumerator.enumerate_subdomains(domain)
                return {'success': True, 'data': {'subdomains': subdomains}}
            
            elif command == "directory_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                directories = await self.manager.directory_scanner.scan_directories(target_url)
                return {'success': True, 'data': {'directories': directories}}
            
            elif command == "vulnerability_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                vulnerabilities = await self.manager.vulnerability_scanner.scan_vulnerabilities(target_url)
                return {'success': True, 'data': {'vulnerabilities': vulnerabilities}}
            
            elif command == "technology_detection":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                technologies = await self.manager.technology_detector.detect_technologies(target_url)
                return {'success': True, 'data': {'technologies': technologies}}
            
            elif command == "interactive":
                await self.cli.run_interactive()
                return {'success': True, 'message': 'Interactive session completed'}
            
            else:
                return {'success': False, 'error': f'Unknown command: {command}'}
                
        except Exception as e:
            logger.error(f"執行網絡攻擊命令失敗: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            # 清理掃描結果
            self.manager.scan_results.clear()
            logger.info("網絡攻擊能力清理完成")
            return True
        except Exception as e:
            logger.error(f"網絡攻擊能力清理失敗: {e}")
            return False


# 註冊能力到系統
async def register_capability():
    """註冊網絡攻擊能力"""
    try:
        capability = WebAttackCapability()
        success = await CapabilityRegistry.register_capability(capability)
        if success:
            logger.info("網絡攻擊能力註冊成功")
        else:
            logger.error("網絡攻擊能力註冊失敗")
        return success
    except Exception as e:
        logger.error(f"註冊網絡攻擊能力時發生錯誤: {e}")
        return False


if __name__ == "__main__":
    async def main():
        """主函數 - 用於測試"""
        capability = WebAttackCapability()
        await capability.initialize()
        
        # 測試交互式介面
        await capability.execute('interactive')
    
    asyncio.run(main())