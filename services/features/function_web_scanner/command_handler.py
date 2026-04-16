"""
Web Scanner 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
使用原生的 Web Scanner 檢測功能 (WebAttackManager)。

Usage:
    from services.features.function_web_scanner.command_handler import WebScannerCommandHandler
    from aiva_common.schemas.commands import AICommand, CommandType

    # 直接創建處理器
    scanner_handler = WebScannerCommandHandler()

    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_WEB_SCAN,
        payload={"target_url": "https://example.com"},
        target_module="features.web_scanner"
    )

    # 直接執行
    result = await scanner_handler.handle_command(command)
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

# 導入統一掃描器
from services.features.function_web_scanner.integration_tools.web_tools import (
    WebAttackManager,
)

logger = get_logger(__name__)


class WebScannerCommandHandler(CommandHandler):
    """Web Scanner 命令處理器

    實現 aiva_common.CommandHandler 協議,處理 FEATURE_WEB_SCAN 命令。
    使用 WebAttackManager 執行原生的 Web 掃描。
    """

    def __init__(self):
        """初始化 Web Scanner 命令處理器"""
        self.manager = WebAttackManager()
        self.logger = logger
        self.logger.info("✅ Web Scanner 命令處理器已初始化")

    async def handle_command(
        self,
        command: AICommand,
        context: CommandContext | None = None
    ) -> AICommandResult:
        """處理 AI 命令

        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_WEB_SCAN
                - payload: {
                    "target_url": "https://example.com",
                    "options": {
                        "subdomain_scan": True,
                        "directory_scan": True,
                        "vulnerability_scan": True,
                        "technology_scan": True,
                    }
                }

            context: 執行上下文

        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()

        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_WEB_SCAN:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_WEB_SCAN}"
                )

            # 2. 提取參數
            payload = command.payload or {}
            target_url = payload.get("target_url")
            if not target_url:
                raise ValueError("缺少必要參數: target_url")

            options = payload.get("options", {})

            self.logger.info(
                f"🎯 開始 Web 掃描: {target_url}"
            )

            # 3. 執行 Web 掃描 (調用 Manager)
            scan_result = await self.manager.comprehensive_scan(
                target_url=target_url,
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
                    "target_url": target_url,
                    "vulnerabilities_found": len(scan_result.get("vulnerabilities", [])),
                    "subdomains_found": len(scan_result.get("subdomains", [])),
                    "directories_found": len(scan_result.get("directories", [])),
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.logger.info(
                f"✅ Web 掃描完成: {target_url} "
                f"(發現 {len(scan_result.get('vulnerabilities', []))} 個漏洞, "
                f"耗時 {execution_time_ms}ms)"
            )

            return result

        except ValueError as e:
            # 參數錯誤
            self.logger.error(f"❌ Web 掃描參數錯誤: {e}")
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
            self.logger.error("⏱️  Web 掃描超時")
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
            self.logger.exception(f"❌ Web 掃描失敗: {e}")
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
