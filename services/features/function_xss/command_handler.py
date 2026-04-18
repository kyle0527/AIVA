"""
XSS 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
使用原生的 XSS 檢測功能 (XssScanner)。

Usage:
    from .command_handler import XSSCommandHandler
    from aiva_common.schemas.commands import AICommand, CommandType

    # 直接創建處理器
    xss_handler = XSSCommandHandler()

    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_XSS_TEST,
        payload={"target_url": "https://example.com"},
        target_module="features.xss"
    )

    # 直接執行
    result = await xss_handler.handle_command(command)
"""

from datetime import datetime
import time

# aiva_common 標準導入
from aiva_common.core.command_center import CommandHandler
from aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandContext,
    CommandStatus,
    CommandType,
)
from aiva_common.utils import get_logger

# 導入新的統一掃描器
from .scanner import XssScanner

logger = get_logger(__name__)


class XSSCommandHandler(CommandHandler):
    """XSS 命令處理器

    實現 aiva_common.CommandHandler 協議,處理 FEATURE_XSS_TEST 命令。
    使用 XssScanner 執行原生的 XSS 檢測。
    """

    def __init__(self):
        """初始化 XSS 命令處理器"""
        self.scanner = XssScanner()
        self.logger = logger
        self.logger.info("✅ XSS 命令處理器已初始化 (Native Mode)")

    async def handle_command(
        self,
        command: AICommand,
        context: CommandContext | None = None
    ) -> AICommandResult:
        """處理 AI 命令

        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_XSS_TEST
                - payload: {
                    "target_url": "https://example.com",
                    "scan_type": "comprehensive",  # comprehensive/reflected/stored/dom
                    "options": {
                        "timeout": 30,
                        "method": "GET",
                        "params": {"q": "search"},
                        "view_url": "http://..." # For stored XSS
                    }
                }

            context: 執行上下文

        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()

        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_XSS_TEST:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_XSS_TEST}"
                )

            # 2. 提取參數
            payload = command.payload or {}
            target_url = payload.get("target_url")
            if not target_url:
                raise ValueError("缺少必要參數: target_url")

            scan_type = payload.get("scan_type", "comprehensive")
            options = payload.get("options", {})

            self.logger.info(
                f"🎯 開始 XSS 測試: {target_url} (類型: {scan_type})"
            )

            # 3. 執行 XSS 掃描 (調用 Scanner)
            scan_result = await self.scanner.scan(
                target_url=target_url,
                scan_type=scan_type,
                options=options
            )

            # 4. 計算執行時間
            execution_time_ms = int((time.time() - start_time) * 1000)

            # 5. 構建結果
            result = AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result=scan_result,
                execution_time=execution_time_ms / 1000.0,  # 轉換為秒
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=None,
                error_code=None,
                error_details=None,
                metrics={
                    "scan_type": scan_type,
                    "target_url": target_url,
                    "vulnerabilities_found": scan_result.get("findings_count", 0),
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.logger.info(
                f"✅ XSS 測試完成: {target_url} "
                f"(發現 {scan_result.get('findings_count', 0)} 個漏洞, "
                f"耗時 {execution_time_ms}ms)"
            )

            return result

        except ValueError as e:
            # 參數錯誤
            self.logger.error(f"❌ XSS 測試參數錯誤: {e}")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                execution_time=(time.time() - start_time),
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=f"參數錯誤: {str(e)}",
                error_code="INVALID_PARAMETER",
                error_details={"exception_type": "ValueError", "parameter_error": str(e)}
            )

        except TimeoutError:
            # 超時
            self.logger.error("⏱️  XSS 測試超時")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.TIMEOUT,
                success=False,
                execution_time=(time.time() - start_time),
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error="執行超時",
                error_code="EXECUTION_TIMEOUT",
                error_details={"timeout_seconds": command.timeout if hasattr(command, "timeout") else 300}
            )

        except Exception as e:
            # 其他錯誤
            self.logger.exception(f"❌ XSS 測試失敗: {e}")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                execution_time=(time.time() - start_time),
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="EXECUTION_ERROR",
                error_details={"exception_type": type(e).__name__, "traceback": str(e)}
            )