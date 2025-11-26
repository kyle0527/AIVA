"""
Payload Generator 主管理器

協調所有 Payload 生成引擎,處理授權檢查和結果管理
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from services.aiva_common.enums import Severity, Confidence
from services.core.aiva_core.service_backbone.authz.permission_matrix import authorize_operation

from .models import (
    PayloadConfig,
    PayloadResult,
    PayloadType,
    PayloadPlatform,
    PayloadFormat,
    ReverseShellLanguage,
    WebShellType,
    PoCType,
    DeliveryMethod,
    ListenerType,
)

logger = logging.getLogger(__name__)


class PayloadGeneratorManager:
    """
    Payload 生成器主管理器
    
    功能:
    - MSFVenom Payload 生成
    - Reverse Shell 生成
    - Web Shell 生成
    - PoC 自動生成
    - 混淆與編碼
    - 交付機制管理
    
    授權: L2+ (High-Critical Risk)
    """

    def __init__(self, authorization_token: Optional[str] = None):
        """
        初始化 Payload Generator Manager
        
        Args:
            authorization_token: 授權令牌 (可選)
        """
        self.authorization_token = authorization_token
        self.environment = os.getenv("AIVA_ENVIRONMENT", "development")
        
        # 延遲導入避免循環依賴
        self._msfvenom_engine = None
        self._reverse_shell_engine = None
        self._webshell_engine = None
        self._poc_engine = None
        
        logger.info(f"PayloadGeneratorManager initialized in {self.environment} environment")

    def _check_authorization(self, operation_name: str) -> bool:
        """
        檢查授權
        
        Args:
            operation_name: 操作名稱
            
        Returns:
            bool: 是否有權限
        """
        # 如果有 Authorization Token,優先使用
        if self.authorization_token:
            return True
        
        # 否則使用 RiskGuard 授權系統
        try:
            return authorize_operation(
                operation_name=operation_name,
                risk_level="L2",
                tags=["payload", "exploitation", "post_exploitation"],
                environment=self.environment
            )
        except Exception as e:
            logger.error(f"Authorization check failed: {e}")
            return False

    def _validate_environment(self) -> bool:
        """
        驗證環境是否允許 Payload 生成
        
        Returns:
            bool: 環境是否有效
        """
        allowed_envs = ["development", "controlled_pentest", "testing"]
        if self.environment not in allowed_envs:
            logger.error(f"Payload generation not allowed in {self.environment}")
            return False
        return True

    async def generate_msfvenom_payload(
        self,
        payload_type: str,
        lhost: str,
        lport: int,
        format: str = "exe",
        encoder: Optional[str] = None,
        iterations: int = 1,
        **kwargs
    ) -> PayloadResult:
        """
        生成 MSFVenom Payload
        
        Args:
            payload_type: MSFVenom payload 類型 (e.g., "windows/meterpreter/reverse_tcp")
            lhost: 監聽主機 IP
            lport: 監聽端口
            format: 輸出格式 (exe, elf, dll, etc.)
            encoder: 編碼器 (e.g., "x86/shikata_ga_nai")
            iterations: 編碼迭代次數
            **kwargs: 額外參數
            
        Returns:
            PayloadResult: 生成結果
        """
        # 授權檢查
        if not self._check_authorization("msfvenom_payload_generation"):
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.MSFVENOM,
                platform=PayloadPlatform.WINDOWS,
                format=PayloadFormat.EXE,
                authorized=False,
                error_message="Unauthorized: Requires L2+ authorization or token"
            )
        
        # 環境檢查
        if not self._validate_environment():
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.MSFVENOM,
                platform=PayloadPlatform.WINDOWS,
                format=PayloadFormat.EXE,
                authorized=True,
                error_message=f"Environment error: Payload generation not allowed in {self.environment}"
            )
        
        logger.info(f"Generating MSFVenom payload: {payload_type}")
        
        try:
            # 延遲導入
            if self._msfvenom_engine is None:
                from .engines.msfvenom_wrapper import MSFVenomWrapper
                self._msfvenom_engine = MSFVenomWrapper()
            
            config = PayloadConfig(
                payload_type=PayloadType.MSFVENOM,
                platform=self._get_platform_from_payload_type(payload_type),
                format=PayloadFormat(format),
                lhost=lhost,
                lport=lport,
                msfvenom_payload=payload_type,
                encoder=encoder,
                iterations=iterations,
                authorization_token=self.authorization_token,
                **kwargs
            )
            
            result = await self._msfvenom_engine.generate(config)
            return result
            
        except Exception as e:
            logger.error(f"MSFVenom payload generation failed: {e}", exc_info=True)
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.MSFVENOM,
                platform=PayloadPlatform.WINDOWS,
                format=PayloadFormat.EXE,
                authorized=True,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__}
            )

    async def generate_reverse_shell(
        self,
        language: str,
        lhost: str,
        lport: int,
        obfuscate: bool = False,
        **kwargs
    ) -> PayloadResult:
        """
        生成 Reverse Shell
        
        Args:
            language: 程式語言 (bash, python, powershell, php, ruby, perl, java, c)
            lhost: 監聽主機 IP
            lport: 監聽端口
            obfuscate: 是否混淆
            **kwargs: 額外參數
            
        Returns:
            PayloadResult: 生成結果
        """
        # 授權檢查
        if not self._check_authorization("reverse_shell_generation"):
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.REVERSE_SHELL,
                platform=PayloadPlatform.MULTI,
                format=self._get_format_from_language(language),
                authorized=False,
                error_message="Unauthorized: Requires L2+ authorization or token"
            )
        
        if not self._validate_environment():
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.REVERSE_SHELL,
                platform=PayloadPlatform.MULTI,
                format=self._get_format_from_language(language),
                authorized=True,
                error_message=f"Environment error: not allowed in {self.environment}"
            )
        
        logger.info(f"Generating {language} reverse shell")
        
        try:
            if self._reverse_shell_engine is None:
                from .engines.reverse_shell_generator import ReverseShellGenerator
                self._reverse_shell_engine = ReverseShellGenerator()
            
            config = PayloadConfig(
                payload_type=PayloadType.REVERSE_SHELL,
                platform=PayloadPlatform.MULTI,
                format=self._get_format_from_language(language),
                lhost=lhost,
                lport=lport,
                shell_language=ReverseShellLanguage(language),
                obfuscation=kwargs.get("obfuscation_method", "base64" if obfuscate else "none"),
                authorization_token=self.authorization_token,
                **kwargs
            )
            
            result = await self._reverse_shell_engine.generate(config)
            return result
            
        except Exception as e:
            logger.error(f"Reverse shell generation failed: {e}", exc_info=True)
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.REVERSE_SHELL,
                platform=PayloadPlatform.MULTI,
                format=self._get_format_from_language(language),
                authorized=True,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__}
            )

    async def generate_webshell(
        self,
        type: str,
        password: str,
        obfuscate: bool = False,
        **kwargs
    ) -> PayloadResult:
        """
        生成 Web Shell
        
        Args:
            type: Web Shell 類型 (php_simple, php_advanced, aspx, jsp)
            password: 訪問密碼
            obfuscate: 是否混淆
            **kwargs: 額外參數
            
        Returns:
            PayloadResult: 生成結果
        """
        if not self._check_authorization("webshell_generation"):
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.WEBSHELL,
                platform=PayloadPlatform.WEB,
                format=self._get_format_from_webshell_type(type),
                authorized=False,
                error_message="Unauthorized"
            )
        
        if not self._validate_environment():
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.WEBSHELL,
                platform=PayloadPlatform.WEB,
                format=self._get_format_from_webshell_type(type),
                authorized=True,
                error_message=f"Environment error: not allowed in {self.environment}"
            )
        
        logger.info(f"Generating {type} webshell")
        
        try:
            if self._webshell_engine is None:
                from .engines.webshell_generator import WebShellGenerator
                self._webshell_engine = WebShellGenerator()
            
            config = PayloadConfig(
                payload_type=PayloadType.WEBSHELL,
                platform=PayloadPlatform.WEB,
                format=self._get_format_from_webshell_type(type),
                lhost="",
                lport=0,
                webshell_type=WebShellType(type),
                password=password,
                obfuscation=kwargs.get("obfuscation_method", "base64" if obfuscate else "none"),
                authorization_token=self.authorization_token,
                **kwargs
            )
            
            result = await self._webshell_engine.generate(config)
            return result
            
        except Exception as e:
            logger.error(f"WebShell generation failed: {e}", exc_info=True)
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.WEBSHELL,
                platform=PayloadPlatform.WEB,
                format=self._get_format_from_webshell_type(type),
                authorized=True,
                error_message=str(e)
            )

    async def generate_poc(
        self,
        vulnerability_type: str,
        target_url: str,
        cve_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PayloadResult:
        """
        生成 PoC (Proof of Concept)
        
        Args:
            vulnerability_type: 漏洞類型 (rce, sqli, xss, lfi, etc.)
            target_url: 目標 URL
            cve_id: CVE ID (可選)
            parameters: 額外參數
            **kwargs: 其他參數
            
        Returns:
            PayloadResult: 生成結果
        """
        if not self._check_authorization("poc_generation"):
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.POC,
                platform=PayloadPlatform.WEB,
                format=PayloadFormat.PYTHON,
                authorized=False,
                error_message="Unauthorized"
            )
        
        if not self._validate_environment():
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.POC,
                platform=PayloadPlatform.WEB,
                format=PayloadFormat.PYTHON,
                authorized=True,
                error_message=f"Environment error"
            )
        
        logger.info(f"Generating PoC for {vulnerability_type} at {target_url}")
        
        try:
            if self._poc_engine is None:
                from .generators.poc_generator import PoCGenerator
                self._poc_engine = PoCGenerator()
            
            config = PayloadConfig(
                payload_type=PayloadType.POC,
                platform=PayloadPlatform.WEB,
                format=PayloadFormat.PYTHON,
                lhost="",
                lport=0,
                poc_type=PoCType(vulnerability_type),
                target_url=target_url,
                cve_id=cve_id,
                parameters=parameters or {},
                authorization_token=self.authorization_token,
                **kwargs
            )
            
            result = await self._poc_engine.generate(config)
            return result
            
        except Exception as e:
            logger.error(f"PoC generation failed: {e}", exc_info=True)
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.POC,
                platform=PayloadPlatform.WEB,
                format=PayloadFormat.PYTHON,
                authorized=True,
                error_message=str(e)
            )

    # Helper methods
    def _get_platform_from_payload_type(self, payload_type: str) -> PayloadPlatform:
        """從 MSFVenom payload 類型推導平台"""
        if "windows" in payload_type.lower():
            return PayloadPlatform.WINDOWS
        elif "linux" in payload_type.lower():
            return PayloadPlatform.LINUX
        elif "osx" in payload_type.lower() or "macos" in payload_type.lower():
            return PayloadPlatform.MACOS
        elif "android" in payload_type.lower():
            return PayloadPlatform.ANDROID
        else:
            return PayloadPlatform.MULTI

    def _get_format_from_language(self, language: str) -> PayloadFormat:
        """從語言推導格式"""
        format_map = {
            "bash": PayloadFormat.BASH,
            "python": PayloadFormat.PYTHON,
            "powershell": PayloadFormat.POWERSHELL,
            "php": PayloadFormat.PHP,
            "ruby": PayloadFormat.RUBY,
            "perl": PayloadFormat.PERL,
            "java": PayloadFormat.JAVA,
            "c": PayloadFormat.C,
        }
        return format_map.get(language.lower(), PayloadFormat.RAW)

    def _get_format_from_webshell_type(self, type: str) -> PayloadFormat:
        """從 WebShell 類型推導格式"""
        if "php" in type.lower():
            return PayloadFormat.PHP
        elif "aspx" in type.lower():
            return PayloadFormat.ASPX
        elif "jsp" in type.lower():
            return PayloadFormat.JSP
        elif "python" in type.lower():
            return PayloadFormat.PYTHON
        else:
            return PayloadFormat.PHP
