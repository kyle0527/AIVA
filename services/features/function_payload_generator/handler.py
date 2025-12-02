"""
Payload Generator 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的 Payload 生成功能,使其可以被 AI 通過 AICommandCenter 調用。

Usage:
    from aiva_common import get_command_center
    from services.features.function_payload_generator.handler import PayloadGeneratorCommandHandler
    
    command_center = get_command_center()
    payload_handler = PayloadGeneratorCommandHandler()
    command_center.register_module("features.payload_generator", payload_handler)
"""

import asyncio
import time
from typing import Any, Dict, Optional
from datetime import datetime

# aiva_common 標準導入
from aiva_common.command_center import CommandHandler
from aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandType,
)
from aiva_common.utils import get_logger

# 現有 Payload Generator 功能導入
from .manager import PayloadGeneratorManager
from .models import (
    PayloadConfig,
    PayloadType,
    PayloadPlatform,
    PayloadFormat,
    ReverseShellLanguage,
    WebShellType,
    PoCType,
)

logger = get_logger(__name__)


class PayloadGeneratorCommandHandler(CommandHandler):
    """Payload Generator 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理 Payload 生成相關命令。
    包裝現有的 PayloadGeneratorManager 功能,提供標準化接口。
    """
    
    def __init__(self, authorization_token: Optional[str] = None):
        """初始化 Payload Generator 命令處理器
        
        Args:
            authorization_token: 授權令牌 (可選)
        """
        self.manager = PayloadGeneratorManager(authorization_token=authorization_token)
        self.logger = logger
        self.logger.info("✅ Payload Generator 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: CommandType.FEATURE_PAYLOAD_GENERATE
                - payload: {
                    "operation": "generate_msfvenom" | "generate_reverse_shell" | 
                                 "generate_webshell" | "generate_poc",
                    "config": {
                        "payload_type": "windows/meterpreter/reverse_tcp",
                        "lhost": "192.168.1.100",
                        "lport": 4444,
                        "format": "exe",
                        "encoder": "x86/shikata_ga_nai",
                        "iterations": 3
                    }
                }
                
            context: 執行上下文
                - user_id: 用戶 ID
                - authorization_level: 授權級別
                
        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()
        
        try:
            # 1. 驗證命令類型
            if not self._is_supported_command_type(command.command_type):
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}"
                )
            
            # 2. 提取參數
            payload = command.payload or {}
            operation = payload.get("operation")
            if not operation:
                raise ValueError("缺少必要參數: operation")
            
            config = payload.get("config", {})
            
            self.logger.info(
                f"🎯 開始 Payload 生成: {operation}"
            )
            
            # 3. 執行對應操作
            result_data = await self._execute_operation(
                operation=operation,
                config=config,
                context=context
            )
            
            # 4. 計算執行時間
            execution_time = time.time() - start_time
            
            # 5. 構建結果
            result = AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=result_data.get("success", True),
                result=result_data,
                execution_time=execution_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=result_data.get("error_message"),
                error_code=result_data.get("error_code"),
                error_details=result_data.get("error_details"),
                metrics={
                    "operation": operation,
                    "authorized": result_data.get("authorized", True),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if result_data.get("success", True):
                self.logger.info(
                    f"✅ Payload 生成完成: {operation} (耗時 {execution_time:.2f}s)"
                )
            else:
                self.logger.warning(
                    f"⚠️ Payload 生成失敗: {operation} - {result_data.get('error_message')}"
                )
            
            return result
            
        except ValueError as e:
            # 參數錯誤
            self.logger.error(f"❌ Payload 生成參數錯誤: {e}")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="VALIDATION_ERROR",
                error_details={"type": "ValueError", "message": str(e)}
            )
            
        except Exception as e:
            # 其他錯誤
            self.logger.error(f"❌ Payload 生成執行錯誤: {e}", exc_info=True)
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="EXECUTION_ERROR",
                error_details={"type": type(e).__name__, "message": str(e)}
            )
    
    def _is_supported_command_type(self, command_type: CommandType) -> bool:
        """檢查是否支持該命令類型"""
        supported_types = [
            CommandType.FEATURE_PAYLOAD_GENERATE,
            # 可擴展支持更多命令類型
        ]
        return command_type in supported_types
    
    async def _execute_operation(
        self,
        operation: str,
        config: Dict[str, Any],
        context: Optional[CommandContext] = None
    ) -> Dict[str, Any]:
        """執行具體的 Payload 生成操作
        
        Args:
            operation: 操作類型
            config: 配置參數
            context: 執行上下文
            
        Returns:
            操作結果
        """
        try:
            if operation == "generate_msfvenom":
                return await self._generate_msfvenom(config)
            elif operation == "generate_reverse_shell":
                return await self._generate_reverse_shell(config)
            elif operation == "generate_webshell":
                return await self._generate_webshell(config)
            elif operation == "generate_poc":
                return await self._generate_poc(config)
            else:
                return {
                    "success": False,
                    "error_message": f"不支持的操作類型: {operation}",
                    "error_code": "UNSUPPORTED_OPERATION"
                }
        except Exception as e:
            self.logger.error(f"操作執行失敗: {operation} - {e}", exc_info=True)
            return {
                "success": False,
                "error_message": str(e),
                "error_code": "OPERATION_FAILED",
                "error_details": {"type": type(e).__name__, "message": str(e)}
            }
    
    async def _generate_msfvenom(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成 MSFVenom Payload"""
        payload_result = await self.manager.generate_msfvenom_payload(
            payload_type=config.get("payload_type", "windows/meterpreter/reverse_tcp"),
            lhost=config.get("lhost", "127.0.0.1"),
            lport=config.get("lport", 4444),
            format=config.get("format", "exe"),
            encoder=config.get("encoder"),
            iterations=config.get("iterations", 1),
            **config.get("extra_options", {})
        )
        
        return {
            "success": payload_result.success,
            "payload": payload_result.payload,
            "payload_type": payload_result.payload_type.value if payload_result.payload_type else None,
            "platform": payload_result.platform.value if payload_result.platform else None,
            "format": payload_result.format.value if payload_result.format else None,
            "output_file": payload_result.output_file,
            "file_size": payload_result.file_size,
            "authorized": payload_result.authorized,
            "error_message": payload_result.error_message
        }
    
    async def _generate_reverse_shell(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成 Reverse Shell"""
        # 實現 Reverse Shell 生成邏輯
        return {
            "success": False,
            "error_message": "Reverse Shell 生成功能待實現",
            "error_code": "NOT_IMPLEMENTED"
        }
    
    async def _generate_webshell(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成 Web Shell"""
        # 實現 Web Shell 生成邏輯
        return {
            "success": False,
            "error_message": "Web Shell 生成功能待實現",
            "error_code": "NOT_IMPLEMENTED"
        }
    
    async def _generate_poc(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成 PoC"""
        # 實現 PoC 生成邏輯
        return {
            "success": False,
            "error_message": "PoC 生成功能待實現",
            "error_code": "NOT_IMPLEMENTED"
        }


# 註冊命令類型 (如果需要擴展 CommandType)
# 這裡記錄需要添加到 aiva_common 的命令類型
REQUIRED_COMMAND_TYPES = [
    ("FEATURE_PAYLOAD_GENERATE", "feature_payload_generate", "生成各類 Payload"),
]
