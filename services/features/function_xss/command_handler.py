"""
XSS 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的 XSS 檢測功能,使其可以被 AI 通過 AICommandCenter 調用。

Usage:
    from services.features.function_xss.command_handler import XSSCommandHandler
    from services.aiva_common.schemas.commands import AICommand, CommandType
    
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

import asyncio
import time
from typing import Any, Dict, Optional
from datetime import datetime

# aiva_common 標準導入
from services.aiva_common.command_center import CommandHandler
from services.aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandType,
)
from services.aiva_common.utils import get_logger

# 現有 XSS 功能導入
try:
    from .integration_tools.xss_tools import (
        XSSManager,
        XSSTarget,
        XSSVulnerability,
    )
except ImportError:
    # 如果相對導入失敗，使用絕對導入
    from features.function_xss.integration_tools.xss_tools import (
        XSSManager,
        XSSTarget,
        XSSVulnerability,
    )

logger = get_logger(__name__)


class XSSCommandHandler(CommandHandler):
    """XSS 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理 FEATURE_XSS_TEST 命令。
    包裝現有的 XSSManager 功能,提供標準化接口。
    """
    
    def __init__(self):
        """初始化 XSS 命令處理器"""
        self.xss_manager = XSSManager()
        self.logger = logger
        self.logger.info("✅ XSS 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_XSS_TEST
                - payload: {
                    "target_url": "https://example.com",
                    "scan_type": "comprehensive",  # comprehensive/dalfox/dom/stored/blind/custom
                    "options": {
                        "use_dalfox": true,
                        "scan_dom": true,
                        "scan_stored": true,
                        "scan_blind": false,
                        "custom_scan": true,
                        "callback_server": "http://your-server.com"  # 盲 XSS 用
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
            
            # 3. 執行 XSS 掃描
            scan_result = await self._execute_xss_scan(
                target_url=target_url,
                scan_type=scan_type,
                options=options,
                context=context
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
                    "vulnerabilities_found": scan_result["summary"]["total_vulnerabilities"],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(
                f"✅ XSS 測試完成: {target_url} "
                f"(發現 {scan_result['summary']['total_vulnerabilities']} 個漏洞, "
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
            
        except asyncio.TimeoutError:
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
    
    async def _execute_xss_scan(
        self,
        target_url: str,
        scan_type: str,
        options: Dict[str, Any],
        context: Optional[CommandContext]
    ) -> Dict[str, Any]:
        """執行 XSS 掃描
        
        包裝現有的 XSSManager.comprehensive_scan() 方法
        """
        # 根據掃描類型設置選項
        scan_options = self._build_scan_options(scan_type, options)
        
        # 檢查授權級別 (盲 XSS 需要更高授權)
        if scan_options.get("scan_blind") and context:
            # 從 custom_config 檢查用戶角色
            user_role = context.custom_config.get("user_role", "guest")
            if user_role not in ["admin", "security_tester", "pentester"]:
                self.logger.warning(f"⚠️  盲 XSS 檢測需要高級授權,當前角色: {user_role},已禁用")
                scan_options["scan_blind"] = False
        
        # 調用現有功能
        result = await self.xss_manager.comprehensive_scan(
            target_url=target_url,
            options=scan_options
        )
        
        return result
    
    def _build_scan_options(
        self,
        scan_type: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """構建掃描選項
        
        根據 scan_type 和 options 構建 XSSManager.comprehensive_scan() 的參數
        """
        # 默認選項
        default_options = {
            "use_dalfox": True,
            "scan_dom": True,
            "scan_stored": True,
            "scan_blind": False,  # 默認關閉,需要外部服務器
            "custom_scan": True,
            "dalfox_options": {},
        }
        
        # 根據掃描類型調整
        if scan_type == "comprehensive":
            # 綜合掃描: 啟用所有功能 (除了盲 XSS)
            pass
        elif scan_type == "dalfox":
            # 只用 Dalfox
            default_options.update({
                "use_dalfox": True,
                "scan_dom": False,
                "scan_stored": False,
                "custom_scan": False,
            })
        elif scan_type == "dom":
            # 只檢測 DOM XSS
            default_options.update({
                "use_dalfox": False,
                "scan_dom": True,
                "scan_stored": False,
                "custom_scan": False,
            })
        elif scan_type == "stored":
            # 只檢測存儲型 XSS
            default_options.update({
                "use_dalfox": False,
                "scan_dom": False,
                "scan_stored": True,
                "custom_scan": False,
            })
        elif scan_type == "blind":
            # 只檢測盲 XSS
            default_options.update({
                "use_dalfox": False,
                "scan_dom": False,
                "scan_stored": False,
                "scan_blind": True,
                "custom_scan": False,
            })
        elif scan_type == "custom":
            # 自定義掃描
            default_options.update({
                "use_dalfox": False,
                "scan_dom": False,
                "scan_stored": False,
                "custom_scan": True,
            })
        
        # 合併用戶選項
        default_options.update(options)
        
        return default_options



