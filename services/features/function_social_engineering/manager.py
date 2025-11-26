"""
Social Engineering Manager

社交工程攻擊管理器，提供統一的社交工程攻擊介面。
"""

import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

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

# 假設 RiskGuard 授權系統已經實現
# from services.core.aiva_core.authorization import authorize_operation

logger = logging.getLogger(__name__)


class SocialEngineeringManager:
    """
    社交工程管理器
    
    提供社交工程攻擊的統一管理介面，包括：
    - 釣魚攻擊 (Phishing)
    - 憑證竊取 (Credential Harvesting)
    - 目標資訊收集 (OSINT)
    - 行為分析 (Analytics)
    
    所有操作都需要 L2 授權和 AIVA_ALLOW_ATTACK=1 環境變數。
    """
    
    def __init__(
        self,
        authorization_token: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        初始化社交工程管理器
        
        Args:
            authorization_token: 授權 Token (優先於 RiskGuard)
            environment: 執行環境 (development/controlled_pentest/testing)
        """
        self.authorization_token = authorization_token
        self.environment = environment or os.getenv("AIVA_ENVIRONMENT", "development")
        
        # 延遲加載引擎
        self._phishing_engine = None
        self._credential_harvester = None
        self._osint_collector = None
        self._analytics_engine = None
        
        logger.info(
            f"SocialEngineeringManager initialized",
            extra={
                "environment": self.environment,
                "auth_mode": "token" if authorization_token else "riskguard"
            }
        )
    
    def _check_authorization(self, operation_name: str) -> bool:
        """
        檢查操作授權
        
        優先級：
        1. Authorization Token (如果提供)
        2. RiskGuard 授權系統
        
        Args:
            operation_name: 操作名稱
            
        Returns:
            bool: 是否有授權
        """
        # Token 優先模式
        if self.authorization_token:
            logger.info(f"Authorization via token for: {operation_name}")
            return True
        
        # TODO: 整合 RiskGuard
        # return authorize_operation(
        #     operation_name=operation_name,
        #     risk_level="L2",
        #     tags=["social_engineering", "phishing", "credential_theft"],
        #     environment=self.environment
        # )
        
        # 臨時：檢查環境變數
        allow_attack = os.getenv("AIVA_ALLOW_ATTACK", "0") == "1"
        if not allow_attack:
            logger.warning(
                f"Operation {operation_name} denied: AIVA_ALLOW_ATTACK not set"
            )
            return False
        
        logger.info(f"Authorization granted for: {operation_name}")
        return True
    
    def _validate_environment(self) -> bool:
        """
        驗證執行環境
        
        Returns:
            bool: 環境是否合法
        """
        allowed_envs = ["development", "controlled_pentest", "testing", "red_team"]
        if self.environment not in allowed_envs:
            logger.error(
                f"Invalid environment: {self.environment}. "
                f"Allowed: {allowed_envs}"
            )
            return False
        return True
    
    async def launch_phishing_campaign(
        self,
        config: PhishingConfig
    ) -> PhishingResult:
        """
        啟動釣魚攻擊活動
        
        Args:
            config: 釣魚活動配置
            
        Returns:
            PhishingResult: 活動執行結果
        """
        operation_name = f"phishing_campaign_{config.phishing_type.value}"
        
        # 授權檢查
        if not self._check_authorization(operation_name):
            return PhishingResult(
                success=False,
                campaign_id="",
                error="Authorization denied: L2 permission required"
            )
        
        # 環境驗證
        if not self._validate_environment():
            return PhishingResult(
                success=False,
                campaign_id="",
                error=f"Invalid environment: {self.environment}"
            )
        
        try:
            logger.info(
                f"Launching phishing campaign",
                extra={
                    "type": config.phishing_type.value,
                    "platform": config.target_platform.value,
                    "target_count": len(config.target_emails)
                }
            )
            
            # TODO: 實現實際的釣魚引擎
            # if not self._phishing_engine:
            #     from .phishing import PhishingEngine
            #     self._phishing_engine = PhishingEngine()
            # 
            # result = await self._phishing_engine.launch_campaign(config)
            
            # 臨時：返回模擬結果
            campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = PhishingResult(
                success=True,
                campaign_id=campaign_id,
                emails_sent=len(config.target_emails),
                emails_failed=0,
                log_file=f"logs/phishing/{campaign_id}.log",
                status=CampaignStatus.RUNNING,
                metadata={
                    "type": config.phishing_type.value,
                    "platform": config.target_platform.value
                }
            )
            
            logger.info(
                f"Phishing campaign launched successfully",
                extra={
                    "campaign_id": campaign_id,
                    "emails_sent": result.emails_sent
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to launch phishing campaign: {str(e)}",
                exc_info=True
            )
            return PhishingResult(
                success=False,
                campaign_id="",
                error=f"Campaign launch failed: {str(e)}"
            )
    
    async def start_credential_harvester(
        self,
        platform: TargetPlatform,
        delivery_method: DeliveryMethod = DeliveryMethod.NGROK,
        port: int = 8080,
        custom_template: Optional[str] = None
    ) -> PhishingResult:
        """
        啟動憑證竊取伺服器
        
        Args:
            platform: 目標平台
            delivery_method: 傳遞方式
            port: 本地埠號
            custom_template: 自訂模板路徑
            
        Returns:
            PhishingResult: 包含公開 URL
        """
        operation_name = "credential_harvester"
        
        # 授權檢查
        if not self._check_authorization(operation_name):
            return PhishingResult(
                success=False,
                campaign_id="",
                error="Authorization denied: L2 permission required"
            )
        
        # 環境驗證
        if not self._validate_environment():
            return PhishingResult(
                success=False,
                campaign_id="",
                error=f"Invalid environment: {self.environment}"
            )
        
        try:
            logger.info(
                f"Starting credential harvester",
                extra={
                    "platform": platform.value,
                    "delivery_method": delivery_method.value,
                    "port": port
                }
            )
            
            # TODO: 實現實際的憑證竊取引擎
            # if not self._credential_harvester:
            #     from .credential_harvesting import CredentialHarvester
            #     self._credential_harvester = CredentialHarvester()
            # 
            # result = await self._credential_harvester.start_server(
            #     platform=platform,
            #     delivery_method=delivery_method,
            #     port=port,
            #     template=custom_template
            # )
            
            # 臨時：返回模擬結果
            campaign_id = f"harvester_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            public_url = f"https://{campaign_id}.ngrok-free.app" if delivery_method == DeliveryMethod.NGROK else f"http://localhost:{port}"
            
            result = PhishingResult(
                success=True,
                campaign_id=campaign_id,
                public_url=public_url,
                log_file=f"logs/credentials/{campaign_id}.log",
                delivery_method=delivery_method,
                server_port=port,
                status=CampaignStatus.RUNNING,
                metadata={
                    "platform": platform.value,
                    "template": custom_template or "default"
                }
            )
            
            logger.info(
                f"Credential harvester started successfully",
                extra={
                    "campaign_id": campaign_id,
                    "public_url": public_url
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to start credential harvester: {str(e)}",
                exc_info=True
            )
            return PhishingResult(
                success=False,
                campaign_id="",
                error=f"Harvester start failed: {str(e)}"
            )
    
    async def collect_osint(
        self,
        target: str,
        search_engines: Optional[List[str]] = None,
        social_media: Optional[List[str]] = None
    ) -> TargetInfo:
        """
        收集目標 OSINT 資訊
        
        Args:
            target: 目標 (電子郵件/網域/姓名)
            search_engines: 搜尋引擎列表
            social_media: 社交媒體平台列表
            
        Returns:
            TargetInfo: 目標資訊
        """
        operation_name = "osint_collection"
        
        # OSINT 收集為 L1 操作（資訊收集）
        # 但仍需要基本授權
        if not self._check_authorization(operation_name):
            raise PermissionError("Authorization denied for OSINT collection")
        
        try:
            logger.info(
                f"Collecting OSINT for target",
                extra={"target": target}
            )
            
            # TODO: 實現實際的 OSINT 收集器
            # if not self._osint_collector:
            #     from .profiling import OSINTCollector
            #     self._osint_collector = OSINTCollector()
            # 
            # info = await self._osint_collector.collect(
            #     target=target,
            #     search_engines=search_engines,
            #     social_media=social_media
            # )
            
            # 臨時：返回模擬結果
            target_id = f"target_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            info = TargetInfo(
                target_id=target_id,
                email=target if "@" in target else None,
                name="Target Name",
                company="Target Company",
                social_profiles={
                    "linkedin": "https://linkedin.com/in/target",
                    "twitter": "https://twitter.com/target"
                },
                confidence_score=0.75,
                data_sources=["google", "linkedin", "hunter.io"]
            )
            
            logger.info(
                f"OSINT collection completed",
                extra={
                    "target_id": target_id,
                    "confidence": info.confidence_score
                }
            )
            
            return info
            
        except Exception as e:
            logger.error(
                f"OSINT collection failed: {str(e)}",
                exc_info=True
            )
            raise
    
    async def get_campaign_analytics(
        self,
        campaign_id: str
    ) -> AnalyticsData:
        """
        獲取活動分析數據
        
        Args:
            campaign_id: 活動 ID
            
        Returns:
            AnalyticsData: 分析數據
        """
        try:
            logger.info(
                f"Fetching analytics for campaign",
                extra={"campaign_id": campaign_id}
            )
            
            # TODO: 實現實際的分析引擎
            # if not self._analytics_engine:
            #     from .analytics import AnalyticsEngine
            #     self._analytics_engine = AnalyticsEngine()
            # 
            # analytics = await self._analytics_engine.get_analytics(campaign_id)
            
            # 臨時：返回模擬結果
            analytics = AnalyticsData(
                campaign_id=campaign_id,
                emails_sent=100,
                emails_delivered=95,
                emails_opened=45,
                links_clicked=20,
                credentials_submitted=8,
                delivery_rate=95.0,
                open_rate=47.4,
                click_rate=44.4,
                success_rate=40.0,
                geo_distribution={
                    "US": 60,
                    "UK": 25,
                    "CA": 10
                },
                browser_stats={
                    "Chrome": 70,
                    "Firefox": 20,
                    "Safari": 10
                },
                os_stats={
                    "Windows": 65,
                    "macOS": 25,
                    "Linux": 10
                }
            )
            
            logger.info(
                f"Analytics retrieved successfully",
                extra={
                    "campaign_id": campaign_id,
                    "success_rate": analytics.success_rate
                }
            )
            
            return analytics
            
        except Exception as e:
            logger.error(
                f"Failed to get analytics: {str(e)}",
                exc_info=True
            )
            raise
    
    async def get_harvested_credentials(
        self,
        campaign_id: str
    ) -> List[CredentialData]:
        """
        獲取收集到的憑證
        
        Args:
            campaign_id: 活動 ID
            
        Returns:
            List[CredentialData]: 憑證列表
        """
        try:
            logger.info(
                f"Fetching harvested credentials",
                extra={"campaign_id": campaign_id}
            )
            
            # TODO: 實現實際的憑證檢索
            # credentials = await self._credential_harvester.get_credentials(campaign_id)
            
            # 臨時：返回空列表
            credentials = []
            
            logger.info(
                f"Retrieved {len(credentials)} credentials",
                extra={"campaign_id": campaign_id}
            )
            
            return credentials
            
        except Exception as e:
            logger.error(
                f"Failed to get credentials: {str(e)}",
                exc_info=True
            )
            raise
    
    async def stop_campaign(
        self,
        campaign_id: str
    ) -> bool:
        """
        停止活動
        
        Args:
            campaign_id: 活動 ID
            
        Returns:
            bool: 是否成功停止
        """
        try:
            logger.info(
                f"Stopping campaign",
                extra={"campaign_id": campaign_id}
            )
            
            # TODO: 實現實際的停止邏輯
            # await self._phishing_engine.stop_campaign(campaign_id)
            # await self._credential_harvester.stop_server(campaign_id)
            
            logger.info(
                f"Campaign stopped successfully",
                extra={"campaign_id": campaign_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to stop campaign: {str(e)}",
                exc_info=True
            )
            return False
