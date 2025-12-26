"""
業務邏輯命令處理器

符合 aiva_common 命令系統規範，實現統一的命令處理接口。
包裝業務邏輯檢測功能，使其可以被 AI Commander 調用。

Usage:
    from services.features.function_bizlogic.command_handler import BizLogicCommandHandler
    from services.aiva_common.schemas.commands import AICommand, CommandType
    
    # 創建處理器
    handler = BizLogicCommandHandler()
    
    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_BIZLOGIC_TEST,
        payload={"target_url": "https://example.com"},
        target_module="features.bizlogic"
    )
    
    # 執行命令
    result = await handler.handle_command(command)
"""

import time
from typing import Any, Dict, Optional
from datetime import datetime

from services.aiva_common.command_center import CommandHandler
from services.aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandType,
)
from services.aiva_common.schemas.tasks import FunctionTaskPayload
from services.aiva_common.utils import get_logger

# 現有業務邏輯功能導入
from .integration_tools.bizlogic_tools import (
    BizLogicManager,
    BizLogicScanOptions,
)

logger = get_logger(__name__)


class BizLogicCommandHandler(CommandHandler):
    """業務邏輯命令處理器
    
    實現 aiva_common.CommandHandler 協議，處理 FEATURE_BIZLOGIC_TEST 命令。
    包裝 BizLogicManager 功能，提供標準化接口。
    
    使用方式:
        handler = BizLogicCommandHandler()
        
        command = AICommand(
            command_type=CommandType.FEATURE_BIZLOGIC_TEST,
            payload={
                "target_url": "https://example.com",
                "options": {
                    "test_race_condition": True,
                    "test_price_manipulation": True,
                    "test_workflow_bypass": True
                }
            }
        )
        
        result = await handler.handle_command(command)
    """
    
    def __init__(self):
        """初始化業務邏輯命令處理器"""
        self.manager = BizLogicManager()
        self.logger = logger
        self.logger.info("✅ 業務邏輯命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_BIZLOGIC_TEST
                - payload: {
                    "target_url": "https://example.com",
                    "options": {
                        "test_race_condition": true,
                        "test_price_manipulation": true,
                        "test_workflow_bypass": true,
                        "concurrent_count": 10,
                        "race_test_endpoints": ["/api/coupon", "/checkout"],
                        "workflow_steps": ["/step1", "/step2", "/checkout"]
                    },
                    "authorization_token": "...",
                    "user_role": "customer"
                }
                
            context: 執行上下文
                - user_id: 用戶 ID
                - authorization_level: 授權級別
                
        Returns:
            AICommandResult: 標準命令結果
                - success: 是否成功
                - result: {
                    "race_conditions": [...],
                    "price_manipulations": [...],
                    "workflow_bypasses": [...],
                    "total_vulnerabilities": 5,
                    "scan_info": {...}
                }
        """
        start_time = time.time()
        
        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_BIZLOGIC_TEST:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_BIZLOGIC_TEST}"
                )
            
            # 2. 提取參數
            payload = command.payload or {}
            target_url = payload.get("target_url")
            if not target_url:
                raise ValueError("缺少必要參數: target_url")
            
            options_dict = payload.get("options", {})
            auth_token = payload.get("authorization_token")
            user_role = payload.get("user_role", "customer")
            
            self.logger.info(
                f"🎯 開始業務邏輯測試: {target_url}"
            )
            
            # 3. 構建掃描選項
            scan_options = {
                "use_race_condition": options_dict.get("test_race_condition", True),
                "use_price_manipulation": options_dict.get("test_price_manipulation", True),
                "use_workflow_bypass": options_dict.get("test_workflow_bypass", True),
                "authorization_token": auth_token,
                "user_role": user_role,
                "concurrent_count": options_dict.get("concurrent_count", 10),
                "race_endpoints": options_dict.get("race_test_endpoints", []),
                "test_negative_price": options_dict.get("test_negative_price", True),
                "test_zero_price": options_dict.get("test_zero_price", True),
                "test_overflow_price": options_dict.get("test_overflow_price", True),
                "workflow_steps": options_dict.get("workflow_steps", []),
                "checkout_endpoint": options_dict.get("checkout_endpoint", "/checkout")
            }
            
            # 4. 執行掃描
            scan_result = await self.manager.comprehensive_scan(
                target_url=target_url,
                options=scan_options
            )
            
            # 5. 計算執行時間
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # 6. 構建結果
            result = AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result=scan_result,
                execution_time=execution_time_ms / 1000.0,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=None,
                error_code=None,
                error_details=None,
                metrics={
                    "target_url": target_url,
                    "total_vulnerabilities": scan_result.get("total_vulnerabilities", 0),
                    "tests_performed": len(scan_result.get("scan_info", {}).get("tests_performed", [])),
                    "race_conditions_found": len(scan_result.get("race_conditions", [])),
                    "price_manipulations_found": len(scan_result.get("price_manipulations", [])),
                    "workflow_bypasses_found": len(scan_result.get("workflow_bypasses", []))
                }
            )
            
            self.logger.info(
                f"✅ 業務邏輯測試完成: 發現 {scan_result.get('total_vulnerabilities', 0)} 個漏洞 "
                f"(執行時間: {execution_time_ms}ms)"
            )
            
            return result
            
        except ValueError as e:
            # 參數驗證錯誤
            self.logger.error(f"❌ 參數錯誤: {e}")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},  # AICommandResult 的 result 不能是 None
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="INVALID_PARAMETERS",
                error_details={"parameter_error": str(e)}
            )
            
        except Exception as e:
            # 執行錯誤
            self.logger.error(f"❌ 業務邏輯測試失敗: {e}", exc_info=True)
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},  # AICommandResult 的 result 不能是 None
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=f"業務邏輯測試執行失敗: {str(e)}",
                error_code="EXECUTION_ERROR",
                error_details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
