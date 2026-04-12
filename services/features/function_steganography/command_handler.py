"""
Steganography 模組命令處理器
"""
import time
from typing import Any, Dict, Optional
from datetime import datetime

from aiva_common.core.command_center import CommandHandler
from aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandType,
)
from aiva_common.utils.logging import get_logger

from .manager import SteganographyManager
from .models import SteganographyMethod, CarrierType

logger = get_logger(__name__)


class SteganographyCommandHandler(CommandHandler):
    """Steganography 命令處理器"""

    def __init__(self):
        self.manager = SteganographyManager()
        self.logger = logger
        self.logger.info("✅ Steganography 命令處理器已初始化")

    async def handle_command(
        self,
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        start_time = time.time()

        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_STEGANOGRAPHY:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_STEGANOGRAPHY}"
                )

            # 2. 提取參數
            payload = command.payload or {}
            operation = payload.get("operation")
            if not operation:
                raise ValueError("缺少必要參數: operation")

            config = payload.get("config", {})

            self.logger.info(f"🎯 開始隱寫術操作: {operation}")

            # 3. 執行對應操作
            result_data = await self._execute_operation(operation, config, context)

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
                error=result_data.get("error"),
                error_code=result_data.get("error_code"),
                error_details=result_data.get("error_details"),
                metrics={
                    "operation": operation,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ 隱寫術執行錯誤: {e}", exc_info=True)
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

    async def _execute_operation(self, operation: str, config: Dict[str, Any], context: Optional[CommandContext] = None) -> Dict[str, Any]:
        if operation == "detect_hidden_data":
            file_path = config.get("target_url") or config.get("file_path")
            if not file_path:
                raise ValueError("缺少必要參數: target_url 或 file_path")
            result = await self.manager.detect_hidden_data(file_path)
            return {
                "success": True,
                "has_hidden_data": result.has_hidden_data,
                "confidence": result.confidence,
                "method_detected": result.method_detected.value if result.method_detected else None,
            }

        elif operation == "extract_data":
            stego_file = config.get("target_url") or config.get("stego_file")
            output_file = config.get("output_path") or config.get("output_file")
            password = config.get("password")

            if not stego_file or not output_file:
                raise ValueError("缺少必要參數: target_url/stego_file 或 output_path/output_file")

            result = await self.manager.stegx_extract_file(stego_file, output_file, password)
            return {
                "success": result.success,
                "output_file": result.output_file,
                "extracted_size": result.extracted_size,
                "error": result.error
            }

        elif operation == "embed_data":
            carrier_image = config.get("carrier_image")
            secret_file = config.get("secret_file")
            output_image = config.get("output_image")
            password = config.get("password")

            if not carrier_image or not secret_file or not output_image or not password:
                raise ValueError("缺少必要參數: carrier_image, secret_file, output_image, 或 password")

            result = await self.manager.stegx_hide_file(carrier_image, secret_file, output_image, password)
            return {
                "success": result.success,
                "output_file": result.output_file,
                "secret_size": result.secret_size,
                "error": result.error
            }

        else:
            return {
                "success": False,
                "error": f"不支持的操作類型: {operation}",
                "error_code": "UNSUPPORTED_OPERATION"
            }
