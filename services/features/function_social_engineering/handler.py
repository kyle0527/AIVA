"""
Social Engineering 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的社交工程攻擊功能,使其可以被 AI 通過 AICommandCenter 調用。

Usage:
    from aiva_common import get_command_center
    from services.features.function_social_engineering.handler import SocialEngineeringCommandHandler
    
    command_center = get_command_center()
    se_handler = SocialEngineeringCommandHandler()
    command_center.register_module("features.social_engineering", se_handler)
"""

import asyncio
import time
from typing import Any, Dict, Optional, List
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

# 現有 Social Engineering 功能導入
from .manager import SocialEngineeringManager
from .models import (
    PhishingConfig,
    PhishingResult,
    CampaignConfig,
    TargetInfo,
    CredentialData,
    AnalyticsData,
    PhishingType,
    TargetPlatform,
    DeliveryMethod,
    CampaignStatus
)

logger = get_logger(__name__)


class SocialEngineeringCommandHandler(CommandHandler):
    """Social Engineering 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理社交工程攻擊相關命令。
    包裝現有的 SocialEngineeringManager 功能,提供標準化接口。
    
    注意: 所有社交工程操作都需要 L2 授權和適當的環境設置。
    """
    
    def __init__(self, authorization_token: Optional[str] = None, environment: Optional[str] = None):
        """初始化 Social Engineering 命令處理器
        
        Args:
            authorization_token: 授權 Token (可選)
            environment: 執行環境 (development/controlled_pentest/testing)
        """
        self.manager = SocialEngineeringManager(
            authorization_token=authorization_token,
            environment=environment
        )
        self.logger = logger
        self.logger.info("✅ Social Engineering 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: CommandType.FEATURE_PHISHING_CAMPAIGN
                - payload: {
                    "operation": "launch_phishing" | "collect_osint" | 
                                 "analyze_target" | "harvest_credentials",
                    "config": {
                        # launch_phishing
                        "phishing_type": "email" | "sms" | "web" | "qr_code",
                        "target_list": ["email1@example.com", "email2@example.com"],
                        "template_id": "template_001",
                        "sender_profile": {...},
                        "landing_page_url": "https://fake-site.com",
                        
                        # collect_osint
                        "target_name": "John Doe",
                        "target_email": "john@example.com",
                        "search_depth": "comprehensive",
                        
                        # analyze_target
                        "target_domain": "example.com",
                        "analysis_type": "behavioral" | "technical",
                        
                        # harvest_credentials
                        "campaign_id": "campaign_001"
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
                f"🎯 開始社交工程操作: {operation}"
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
                error=result_data.get("error"),
                error_code=result_data.get("error_code"),
                error_details=result_data.get("error_details"),
                metrics={
                    "operation": operation,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if result_data.get("success", True):
                self.logger.info(
                    f"✅ 社交工程操作完成: {operation} (耗時 {execution_time:.2f}s)"
                )
            else:
                self.logger.warning(
                    f"⚠️ 社交工程操作失敗: {operation} - {result_data.get('error')}"
                )
            
            return result
            
        except ValueError as e:
            # 參數錯誤
            self.logger.error(f"❌ 社交工程操作參數錯誤: {e}")
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
            self.logger.error(f"❌ 社交工程操作執行錯誤: {e}", exc_info=True)
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
            CommandType.FEATURE_PHISHING_CAMPAIGN,
            CommandType.FEATURE_OSINT_COLLECT,
            # 可擴展支持更多命令類型
        ]
        return command_type in supported_types
    
    async def _execute_operation(
        self,
        operation: str,
        config: Dict[str, Any],
        context: Optional[CommandContext] = None
    ) -> Dict[str, Any]:
        """執行具體的社交工程操作
        
        Args:
            operation: 操作類型
            config: 配置參數
            context: 執行上下文
            
        Returns:
            操作結果
        """
        try:
            if operation == "launch_phishing":
                return await self._launch_phishing(config)
            elif operation == "collect_osint":
                return await self._collect_osint(config)
            elif operation == "analyze_target":
                return await self._analyze_target(config)
            elif operation == "harvest_credentials":
                return await self._harvest_credentials(config)
            else:
                return {
                    "success": False,
                    "error": f"不支持的操作類型: {operation}",
                    "error_code": "UNSUPPORTED_OPERATION"
                }
        except Exception as e:
            self.logger.error(f"操作執行失敗: {operation} - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_code": "OPERATION_FAILED",
                "error_details": {"type": type(e).__name__, "message": str(e)}
            }
    
    async def _launch_phishing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """啟動釣魚攻擊活動"""
        try:
            # 構建 PhishingConfig - 使用正確的參數名稱
            phishing_config = PhishingConfig(
                phishing_type=PhishingType(config.get("phishing_type", "email_phishing")),
                target_platform=TargetPlatform(config.get("target_platform", "generic")),
                target_emails=config.get("target_emails", config.get("target_list", [])),
                sender_email=config.get("sender_email", "noreply@example.com"),
                subject=config.get("subject", "Important Security Alert"),
                template_name=config.get("template_name", config.get("template_id", "default")),
                callback_url=config.get("callback_url", config.get("landing_page_url", "http://localhost:8080")),
                smtp_host=config.get("smtp_host"),
                smtp_port=config.get("smtp_port"),
                smtp_username=config.get("smtp_username"),
                smtp_password=config.get("smtp_password"),
                smtp_use_tls=config.get("smtp_use_tls", True),
                personalization=config.get("personalization", False),
                use_url_shortener=config.get("use_url_shortener", False),
                tracking_pixel=config.get("tracking_enabled", config.get("tracking_pixel", True)),
                schedule_time=config.get("schedule_time"),
                send_rate=config.get("send_rate", 10),
                batch_size=config.get("batch_size", 50),
                target_filters=config.get("target_filters", {}),
                template_vars=config.get("template_vars", config.get("sender_profile", {}))
            )
            
            result = await self.manager.launch_phishing_campaign(phishing_config)
            
            return {
                "success": result.success,
                "campaign_id": result.campaign_id,
                "emails_sent": result.emails_sent,
                "emails_failed": result.emails_failed,
                "public_url": result.public_url,
                "log_file": result.log_file,
                "delivery_method": result.delivery_method.value if result.delivery_method else None,
                "server_port": result.server_port,
                "status": result.status.value,
                "error": result.error
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "PHISHING_LAUNCH_FAILED"
            }
    
    async def _collect_osint(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """收集目標 OSINT 資訊"""
        # 實現 OSINT 收集邏輯
        return {
            "success": False,
            "error": "OSINT 收集功能待實現",
            "error_code": "NOT_IMPLEMENTED"
        }
    
    async def _analyze_target(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """分析目標行為和特徵"""
        # 實現目標分析邏輯
        return {
            "success": False,
            "error": "目標分析功能待實現",
            "error_code": "NOT_IMPLEMENTED"
        }
    
    async def _harvest_credentials(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """收集捕獲的憑證"""
        # 實現憑證收集邏輯
        return {
            "success": False,
            "error": "憑證收集功能待實現",
            "error_code": "NOT_IMPLEMENTED"
        }


# 註冊命令類型 (如果需要擴展 CommandType)
REQUIRED_COMMAND_TYPES = [
    ("FEATURE_PHISHING_CAMPAIGN", "feature_phishing_campaign", "釣魚攻擊活動"),
    ("FEATURE_OSINT_COLLECT", "feature_osint_collect", "OSINT 資訊收集"),
]
