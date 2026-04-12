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

from aiva_common.schemas import APIResponse
from aiva_common.utils import get_logger

logger = get_logger(__name__)
console = Console()

# UI 常量
PROMPT_ENTER_URL = "[bold cyan]請輸入目標 URL: [/bold cyan]"
ERROR_EMPTY_URL = "[bold red]URL 不能為空[/bold red]"
PROTOCOL_HTTP = 'http://'
PROTOCOL_HTTPS = 'https://'

# 錯誤訊息常量
ERROR_MISSING_TARGET_URL = 'Missing target_url parameter'


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
class SubdomainResult:
    """子域名枚舉結果"""
    domain: str
    subdomain: str
    ip_addresses: List[str] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.0


@dataclass
class DirectoryScanResult:
    """目錄掃描結果"""
    path: str
    url: str
    status_code: int = 0
    content_length: int = 0
    content_type: str = ""
    redirect_url: str = ""
    severity: str = "info"


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
        
    async def enumerate_subdomains(self, domain: str) -> List[str]:
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
            
            # 使用 asyncio.timeout 來控制總超時（30秒）
            async with aiohttp.ClientSession() as session:
                self.session = session
                try:
                    async with asyncio.timeout(30):
                        await asyncio.gather(*tasks, return_exceptions=True)
                except TimeoutError:
                    logger.warning("子域名枚舉超時 (30秒)")
            
            return sorted(self.found_subdomains)
            
        except Exception as e:
            logger.error(f"子域名枚舉失敗: {e}")
            return []
    
    async def _enumerate_crt_sh(self, domain: str) -> None:
        """使用 crt.sh 枚舉子域名"""
        try:
            if not self.session:
                return
            url = f"https://crt.sh/?q=%25.{domain}&output=json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for entry in data:
                        name = entry.get('name_value', '') if isinstance(entry, dict) else ''
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
        """搜索引擎枚舉 — 透過公開搜尋 API 發現子域名"""
        if not self.session:
            return

        # 1. Bing Search (不需要 API key 的公開搜尋)
        try:
            bing_url = f"https://www.bing.com/search?q=site%3A{domain}&count=50"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            async with self.session.get(bing_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    # 從搜尋結果中萃取子域名
                    pattern = re.compile(
                        r'https?://([a-zA-Z0-9][-a-zA-Z0-9]*\.' + re.escape(domain) + r')'
                    )
                    matches = pattern.findall(content)
                    for match in matches:
                        self.found_subdomains.add(match)
        except Exception as e:
            logger.debug(f"Bing 搜尋枚舉失敗: {e}")

        # 2. DuckDuckGo HTML search
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q=site%3A{domain}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            async with self.session.get(ddg_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    pattern = re.compile(
                        r'https?://([a-zA-Z0-9][-a-zA-Z0-9]*\.' + re.escape(domain) + r')'
                    )
                    matches = pattern.findall(content)
                    for match in matches:
                        self.found_subdomains.add(match)
        except Exception as e:
            logger.debug(f"DuckDuckGo 搜尋枚舉失敗: {e}")

        # 3. RapidDNS (免費子域名資料庫)
        try:
            rapiddns_url = f"https://rapiddns.io/subdomain/{domain}?full=1"
            async with self.session.get(rapiddns_url) as response:
                if response.status == 200:
                    content = await response.text()
                    pattern = re.compile(
                        r'([a-zA-Z0-9][-a-zA-Z0-9]*\.' + re.escape(domain) + r')'
                    )
                    matches = pattern.findall(content)
                    for match in matches:
                        if match.endswith(domain):
                            self.found_subdomains.add(match)
        except Exception as e:
            logger.debug(f"RapidDNS 枚舉失敗: {e}")
    
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
                if self.session:
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
            if not self.session:
                return
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
    """漏洞掃描器 — 執行多類型 Web 漏洞檢測並分析回應"""

    def __init__(self):
        self.vulnerabilities: List[Dict[str, Any]] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self._baseline_content: str = ""
        self._baseline_status: int = 0

    async def scan_vulnerabilities(self, target_url: str) -> List[Dict[str, Any]]:
        """掃描常見漏洞"""
        try:
            self.vulnerabilities.clear()

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                self.session = session

                # 取得基線回應（用於比較）
                await self._get_baseline(target_url)

                # 並行執行不同類型的漏洞掃描
                tasks = [
                    self._scan_xss(target_url),
                    self._scan_sql_injection(target_url),
                    self._scan_directory_traversal(target_url),
                    self._scan_security_headers(target_url),
                    self._scan_clickjacking(target_url),
                    self._scan_cors_misconfiguration(target_url),
                    self._scan_open_redirect(target_url),
                    self._scan_sensitive_files(target_url),
                ]

                await asyncio.gather(*tasks, return_exceptions=True)

            # 去重（同類型同 payload 只保留一筆）
            seen = set()
            deduped = []
            for vuln in self.vulnerabilities:
                key = (vuln.get('type', ''), vuln.get('payload', ''))
                if key not in seen:
                    seen.add(key)
                    deduped.append(vuln)
            self.vulnerabilities = deduped

            return self.vulnerabilities

        except Exception as e:
            logger.error(f"漏洞掃描失敗: {e}")
            return []

    async def _get_baseline(self, target_url: str) -> None:
        """取得基線回應用於比較"""
        try:
            if not self.session:
                return
            async with self.session.get(target_url) as response:
                self._baseline_content = await response.text()
                self._baseline_status = response.status
        except Exception:
            self._baseline_content = ""
            self._baseline_status = 0

    def _is_reflected(self, payload: str, content: str) -> bool:
        """檢查 payload 是否在回應中被反射（未編碼）"""
        return payload in content

    def _response_differs(self, content: str) -> bool:
        """檢查回應是否與基線有顯著差異"""
        if not self._baseline_content:
            return False
        baseline_len = len(self._baseline_content)
        if baseline_len == 0:
            return len(content) > 0
        diff_ratio = abs(len(content) - baseline_len) / baseline_len
        return diff_ratio > 0.15

    async def _scan_xss(self, target_url: str) -> None:
        """XSS 漏洞掃描 — 驗證 payload 是否在回應中未編碼反射"""
        payloads = [
            "<script>alert('XSS')</script>",
            "'\"><script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
        ]

        for payload in payloads:
            try:
                test_url = f"{target_url}?test={urllib.parse.quote(payload)}"
                if not self.session:
                    continue
                async with self.session.get(test_url) as response:
                    content = await response.text()
                    # 真正的反射驗證：payload 必須在回應中出現（未被編碼）
                    if self._is_reflected(payload, content):
                        # 排除誤報：檢查是否在 HTML 屬性中被正確轉義
                        html_encoded = payload.replace('<', '&lt;').replace('>', '&gt;')
                        if html_encoded not in content:
                            self.vulnerabilities.append({
                                'type': 'Reflected XSS',
                                'severity': 'High',
                                'confidence': 'High',
                                'location': test_url,
                                'payload': payload,
                                'description': f'XSS payload reflected without encoding in response (HTTP {response.status})',
                                'recommendation': 'Implement output encoding and Content-Security-Policy'
                            })
                            break  # 一個反射點足以確認漏洞
            except Exception:
                continue

    async def _scan_sql_injection(self, target_url: str) -> None:
        """SQL 注入漏洞掃描 — 基於錯誤訊息與回應差異分析"""
        # Error-based detection
        error_payloads = ["'", "\"", "1' OR '1'='1", "1 AND 1=CONVERT(int,@@version)--"]

        sql_error_patterns = [
            'you have an error in your sql syntax',
            'unclosed quotation mark',
            'quoted string not properly terminated',
            'mysql_fetch', 'mysql_num_rows', 'mysql_query',
            'pg_query', 'pg_exec',
            'ora-01756', 'ora-00933',
            'microsoft ole db provider for sql server',
            'syntax error at or near',
            'sqlstate[',
            'sqlite3::',
        ]

        for payload in error_payloads:
            try:
                test_url = f"{target_url}?id={urllib.parse.quote(payload)}"
                if not self.session:
                    continue
                async with self.session.get(test_url) as response:
                    content = await response.text()
                    content_lower = content.lower()

                    matched_errors = [p for p in sql_error_patterns if p in content_lower]
                    if matched_errors:
                        self.vulnerabilities.append({
                            'type': 'SQL Injection (Error-based)',
                            'severity': 'Critical',
                            'confidence': 'High',
                            'location': test_url,
                            'payload': payload,
                            'description': f'SQL error message detected: {matched_errors[0]}',
                            'recommendation': 'Use parameterized queries / prepared statements'
                        })
                        return  # 一個錯誤就足以確認

                    # Boolean-based: 回應長度與基線比較
                    if self._response_differs(content) and response.status != self._baseline_status:
                        self.vulnerabilities.append({
                            'type': 'SQL Injection (Boolean-based)',
                            'severity': 'High',
                            'confidence': 'Medium',
                            'location': test_url,
                            'payload': payload,
                            'description': f'Response differs significantly from baseline (status {response.status} vs {self._baseline_status})',
                            'recommendation': 'Use parameterized queries / prepared statements'
                        })
                        return
            except Exception:
                continue

    async def _scan_directory_traversal(self, target_url: str) -> None:
        """目錄遍歷漏洞掃描"""
        payloads = [
            ("../../../etc/passwd", ["root:", "/bin/bash", "/bin/sh"]),
            ("....//....//....//etc/passwd", ["root:", "/bin/bash"]),
            ("..\\..\\..\\windows\\system32\\drivers\\etc\\hosts", ["127.0.0.1", "localhost"]),
            ("..%2f..%2f..%2fetc%2fpasswd", ["root:", "/bin/bash"]),
        ]

        for payload, indicators in payloads:
            try:
                test_url = f"{target_url}?file={urllib.parse.quote(payload)}"
                if not self.session:
                    continue
                async with self.session.get(test_url) as response:
                    content = await response.text()
                    matched = [i for i in indicators if i in content]
                    if len(matched) >= 2:
                        self.vulnerabilities.append({
                            'type': 'Directory Traversal / LFI',
                            'severity': 'Critical',
                            'confidence': 'High',
                            'location': test_url,
                            'payload': payload,
                            'description': f'File content indicators found: {", ".join(matched)}',
                            'recommendation': 'Validate and sanitize file path inputs; use allowlists'
                        })
                        return
            except Exception:
                continue

    async def _scan_security_headers(self, target_url: str) -> None:
        """安全標頭檢查"""
        try:
            if not self.session:
                return
            async with self.session.get(target_url) as response:
                headers = response.headers

                security_headers = {
                    'X-Content-Type-Options': ('nosniff', 'Low', 'Prevents MIME type sniffing'),
                    'X-Frame-Options': ('DENY/SAMEORIGIN', 'Medium', 'Prevents clickjacking'),
                    'Strict-Transport-Security': ('max-age=...', 'Medium', 'Enforces HTTPS'),
                    'Content-Security-Policy': ('policy', 'Medium', 'Prevents XSS and data injection'),
                    'Referrer-Policy': ('no-referrer', 'Low', 'Controls referrer information'),
                    'Permissions-Policy': ('policy', 'Low', 'Controls browser features'),
                }

                for header, (expected, severity, desc) in security_headers.items():
                    if header not in headers:
                        self.vulnerabilities.append({
                            'type': 'Missing Security Header',
                            'severity': severity,
                            'confidence': 'High',
                            'location': target_url,
                            'payload': '',
                            'description': f'Missing {header} header - {desc}',
                            'recommendation': f'Add {header}: {expected}'
                        })
        except Exception:
            pass

    async def _scan_clickjacking(self, target_url: str) -> None:
        """點擊劫持檢查"""
        try:
            if not self.session:
                return
            async with self.session.get(target_url) as response:
                headers = response.headers
                has_xfo = 'X-Frame-Options' in headers
                has_csp_fa = False
                csp = headers.get('Content-Security-Policy', '')
                if 'frame-ancestors' in csp:
                    has_csp_fa = True

                if not has_xfo and not has_csp_fa:
                    self.vulnerabilities.append({
                        'type': 'Clickjacking',
                        'severity': 'Medium',
                        'confidence': 'High',
                        'location': target_url,
                        'payload': '',
                        'description': 'No X-Frame-Options or CSP frame-ancestors directive found',
                        'recommendation': "Add X-Frame-Options: DENY or CSP frame-ancestors 'self'"
                    })
        except Exception:
            pass

    async def _scan_cors_misconfiguration(self, target_url: str) -> None:
        """CORS 配置錯誤檢查"""
        try:
            if not self.session:
                return
            evil_origin = "https://evil-attacker.com"
            headers_to_send = {"Origin": evil_origin}
            async with self.session.get(target_url, headers=headers_to_send) as response:
                acao = response.headers.get("Access-Control-Allow-Origin", "")
                acac = response.headers.get("Access-Control-Allow-Credentials", "")

                if acao == "*" and acac.lower() == "true":
                    self.vulnerabilities.append({
                        'type': 'CORS Misconfiguration',
                        'severity': 'High',
                        'confidence': 'High',
                        'location': target_url,
                        'payload': evil_origin,
                        'description': 'CORS allows all origins with credentials (wildcard + credentials)',
                        'recommendation': 'Restrict Access-Control-Allow-Origin to specific trusted domains'
                    })
                elif acao == evil_origin:
                    self.vulnerabilities.append({
                        'type': 'CORS Misconfiguration',
                        'severity': 'High',
                        'confidence': 'High',
                        'location': target_url,
                        'payload': evil_origin,
                        'description': f'CORS reflects arbitrary origin: {evil_origin}',
                        'recommendation': 'Validate Origin header against allowlist'
                    })
                elif acao == "null":
                    self.vulnerabilities.append({
                        'type': 'CORS Misconfiguration',
                        'severity': 'Medium',
                        'confidence': 'Medium',
                        'location': target_url,
                        'payload': 'null',
                        'description': 'CORS allows null origin (exploitable via sandboxed iframes)',
                        'recommendation': 'Do not allow null origin in CORS configuration'
                    })
        except Exception:
            pass

    async def _scan_open_redirect(self, target_url: str) -> None:
        """開放重定向檢查"""
        redirect_params = ["url", "redirect", "next", "return", "returnTo", "goto", "target", "rurl", "dest"]
        evil_url = "https://evil-attacker.com/"

        for param in redirect_params:
            try:
                test_url = f"{target_url}?{param}={urllib.parse.quote(evil_url)}"
                if not self.session:
                    continue
                async with self.session.get(test_url, allow_redirects=False) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location", "")
                        if evil_url in location or location.startswith("//evil-attacker.com"):
                            self.vulnerabilities.append({
                                'type': 'Open Redirect',
                                'severity': 'Medium',
                                'confidence': 'High',
                                'location': test_url,
                                'payload': f'{param}={evil_url}',
                                'description': f'Open redirect via {param} parameter (redirects to {location})',
                                'recommendation': 'Validate redirect targets against allowlist of trusted domains'
                            })
                            return
            except Exception:
                continue

    async def _scan_sensitive_files(self, target_url: str) -> None:
        """敏感檔案暴露檢查"""
        sensitive_paths = [
            (".env", ["DB_PASSWORD", "SECRET_KEY", "API_KEY", "DATABASE_URL"]),
            (".git/config", ["[core]", "[remote", "repositoryformatversion"]),
            ("wp-config.php.bak", ["DB_NAME", "DB_USER", "DB_PASSWORD"]),
            (".htpasswd", [":"]),
            ("server-status", ["Apache Server Status", "Server Version"]),
            ("debug/", ["Traceback", "DEBUG", "Stack Trace"]),
            ("phpinfo.php", ["PHP Version", "php.ini", "Configuration"]),
        ]

        parsed = urllib.parse.urlparse(target_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        for path, indicators in sensitive_paths:
            try:
                test_url = f"{base_url}/{path}"
                if not self.session:
                    continue
                async with self.session.get(test_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        matched = [i for i in indicators if i in content]
                        if matched:
                            self.vulnerabilities.append({
                                'type': 'Sensitive File Exposure',
                                'severity': 'High' if any(k in path for k in ['.env', '.git', 'wp-config', '.htpasswd']) else 'Medium',
                                'confidence': 'High',
                                'location': test_url,
                                'payload': path,
                                'description': f'Sensitive file accessible: {path} (indicators: {", ".join(matched[:3])})',
                                'recommendation': f'Block access to {path} via web server configuration'
                            })
            except Exception:
                continue


# 向後相容別名
WebVulnerabilityScanner = VulnerabilityScanner


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
                    self._export_results()
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
        target_url = console.input(PROMPT_ENTER_URL).strip()
        if not target_url:
            console.print(ERROR_EMPTY_URL)
            return
        
        if not target_url.startswith((PROTOCOL_HTTP, PROTOCOL_HTTPS)):
            target_url = f"{PROTOCOL_HTTP}{target_url}"
        
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
        target_url = console.input(PROMPT_ENTER_URL).strip()
        if not target_url:
            console.print(ERROR_EMPTY_URL)
            return
        
        if not target_url.startswith((PROTOCOL_HTTP, PROTOCOL_HTTPS)):
            target_url = f"{PROTOCOL_HTTP}{target_url}"
        
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
        target_url = console.input(PROMPT_ENTER_URL).strip()
        if not target_url:
            console.print(ERROR_EMPTY_URL)
            return
        
        if not target_url.startswith((PROTOCOL_HTTP, PROTOCOL_HTTPS)):
            target_url = f"{PROTOCOL_HTTP}{target_url}"
        
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
        target_url = console.input(PROMPT_ENTER_URL).strip()
        if not target_url:
            console.print(ERROR_EMPTY_URL)
            return
        
        if not target_url.startswith((PROTOCOL_HTTP, PROTOCOL_HTTPS)):
            target_url = f"{PROTOCOL_HTTP}{target_url}"
        
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
    
    def _export_results(self):
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



