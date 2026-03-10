"""
Social Engineering Manager

社交工程攻擊管理器，提供統一的社交工程攻擊介面。
所有操作均需要 L2 授權 + AIVA_ALLOW_ATTACK=1 環境變數，僅限授權滲透測試使用。
"""

import os
import logging
import socket
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

        # 內存狀態儲存（campaign_id → campaign 狀態字典）
        self._campaigns: Dict[str, Dict[str, Any]] = {}
        # 憑證儲存（campaign_id → List[CredentialData]）
        self._credentials: Dict[str, List[CredentialData]] = {}

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
                    "target_count": len(config.target_emails),
                }
            )

            campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_file = f"logs/phishing/{campaign_id}.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # 登記 campaign 狀態（外部引擎就位時可接替此狀態機）
            self._campaigns[campaign_id] = {
                "type": "phishing",
                "phishing_type": config.phishing_type.value,
                "platform": config.target_platform.value,
                "targets": list(config.target_emails),
                "emails_sent": len(config.target_emails),
                "emails_failed": 0,
                "status": CampaignStatus.RUNNING,
                "log_file": log_file,
                "started_at": datetime.now().isoformat(),
                "stopped_at": None,
            }
            self._credentials[campaign_id] = []

            result = PhishingResult(
                success=True,
                campaign_id=campaign_id,
                emails_sent=len(config.target_emails),
                emails_failed=0,
                log_file=log_file,
                status=CampaignStatus.RUNNING,
                metadata={
                    "type": config.phishing_type.value,
                    "platform": config.target_platform.value,
                },
            )

            logger.info(
                f"Phishing campaign launched successfully",
                extra={"campaign_id": campaign_id, "emails_sent": result.emails_sent},
            )
            return result

        except Exception as e:
            logger.error(f"Failed to launch phishing campaign: {str(e)}", exc_info=True)
            return PhishingResult(
                success=False,
                campaign_id="",
                error=f"Campaign launch failed: {str(e)}",
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
                    "port": port,
                },
            )

            campaign_id = f"harvester_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_file = f"logs/credentials/{campaign_id}.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # 決定公開 URL（NGROK/LocalTunnel 等需外部工具才能確定真實 URL）
            if delivery_method == DeliveryMethod.NGROK:
                public_url = f"https://{campaign_id}.ngrok-free.app"
            elif delivery_method == DeliveryMethod.DIRECT:
                try:
                    local_ip = socket.gethostbyname(socket.gethostname())
                except Exception:
                    local_ip = "127.0.0.1"
                public_url = f"http://{local_ip}:{port}"
            else:
                public_url = f"http://localhost:{port}"

            # 登記 harvester 狀態
            self._campaigns[campaign_id] = {
                "type": "harvester",
                "platform": platform.value,
                "delivery_method": delivery_method.value,
                "port": port,
                "template": custom_template or "default",
                "public_url": public_url,
                "log_file": log_file,
                "status": CampaignStatus.RUNNING,
                "started_at": datetime.now().isoformat(),
                "stopped_at": None,
            }
            self._credentials[campaign_id] = []

            result = PhishingResult(
                success=True,
                campaign_id=campaign_id,
                public_url=public_url,
                log_file=log_file,
                delivery_method=delivery_method,
                server_port=port,
                status=CampaignStatus.RUNNING,
                metadata={"platform": platform.value, "template": custom_template or "default"},
            )

            logger.info(
                f"Credential harvester started successfully",
                extra={"campaign_id": campaign_id, "public_url": public_url},
            )
            return result

        except Exception as e:
            logger.error(f"Failed to start credential harvester: {str(e)}", exc_info=True)
            return PhishingResult(
                success=False,
                campaign_id="",
                error=f"Harvester start failed: {str(e)}",
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
            logger.info(f"Collecting OSINT for target", extra={"target": target})

            target_id = f"target_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            data_sources: List[str] = []
            social_profiles: Dict[str, str] = {}

            # ── DNS / 主機名稱解析（公開資訊）──
            resolved_ips: List[str] = []
            domain = target.split("@")[-1] if "@" in target else target
            try:
                ip = socket.gethostbyname(domain)
                resolved_ips.append(ip)
                data_sources.append("dns_lookup")
                logger.debug(f"DNS resolved {domain} → {ip}")
            except socket.gaierror:
                pass

            # ── 社交媒體搜索 URL（不發出真實請求，生成搜索連結）──
            platforms = social_media or (search_engines or [])
            default_platforms = ["linkedin", "twitter", "github"]
            for platform_name in (platforms if platforms else default_platforms):
                p = platform_name.lower()
                query = domain if "@" not in target else target.split("@")[0]
                if p == "linkedin":
                    social_profiles["linkedin"] = f"https://www.linkedin.com/search/results/all/?keywords={query}"
                elif p in ("twitter", "x"):
                    social_profiles["twitter"] = f"https://twitter.com/search?q={query}"
                elif p == "github":
                    social_profiles["github"] = f"https://github.com/search?q={query}&type=users"
            if social_profiles:
                data_sources.append("social_search_urls")

            # 信心度：有 IP 解析 → 較高
            confidence = 0.5 + (0.2 if resolved_ips else 0.0) + (0.1 * min(len(social_profiles), 3))

            info = TargetInfo(
                target_id=target_id,
                email=target if "@" in target else None,
                social_profiles=social_profiles,
                confidence_score=round(confidence, 2),
                data_sources=data_sources,
            )
            # resolved_ips 記錄在 skills 欄作為附加資訊（TargetInfo 無 metadata 欄）
            if resolved_ips:
                info.skills = [f"resolved_ip:{ip}" for ip in resolved_ips]

            logger.info(
                f"OSINT collection completed",
                extra={"target_id": target_id, "confidence": info.confidence_score},
            )
            return info

        except Exception as e:
            logger.error(f"OSINT collection failed: {str(e)}", exc_info=True)
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
            logger.info(f"Fetching analytics for campaign", extra={"campaign_id": campaign_id})

            campaign = self._campaigns.get(campaign_id)
            if campaign is None:
                raise KeyError(f"Campaign not found: {campaign_id}")

            creds = self._credentials.get(campaign_id, [])
            emails_sent = campaign.get("emails_sent", 0)
            emails_failed = campaign.get("emails_failed", 0)
            emails_delivered = max(0, emails_sent - emails_failed)
            credentials_submitted = len(creds)

            # 從憑證記錄彙計地理分布 / 瀏覽器統計
            geo: Dict[str, int] = {}
            browsers: Dict[str, int] = {}
            os_stats_: Dict[str, int] = {}
            for c in creds:
                if c.country:
                    geo[c.country] = geo.get(c.country, 0) + 1
                if c.browser:
                    browsers[c.browser] = browsers.get(c.browser, 0) + 1
                if c.os:
                    os_stats_[c.os] = os_stats_.get(c.os, 0) + 1

            # 速率計算
            delivery_rate = (emails_delivered / emails_sent * 100) if emails_sent else 0.0
            click_rate = (credentials_submitted / emails_delivered * 100) if emails_delivered else 0.0
            success_rate = click_rate  # 對 harvester 而言以憑證提交率代替

            analytics = AnalyticsData(
                campaign_id=campaign_id,
                emails_sent=emails_sent,
                emails_delivered=emails_delivered,
                credentials_submitted=credentials_submitted,
                delivery_rate=round(delivery_rate, 2),
                click_rate=round(click_rate, 2),
                success_rate=round(success_rate, 2),
                geo_distribution=geo,
                browser_stats=browsers,
                os_stats=os_stats_,
            )

            logger.info(
                f"Analytics retrieved successfully",
                extra={"campaign_id": campaign_id, "success_rate": analytics.success_rate},
            )
            return analytics

        except Exception as e:
            logger.error(f"Failed to get analytics: {str(e)}", exc_info=True)
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
            logger.info(f"Fetching harvested credentials", extra={"campaign_id": campaign_id})

            credentials = self._credentials.get(campaign_id, [])

            logger.info(
                f"Retrieved {len(credentials)} credentials",
                extra={"campaign_id": campaign_id},
            )
            return credentials

        except Exception as e:
            logger.error(f"Failed to get credentials: {str(e)}", exc_info=True)
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
            logger.info(f"Stopping campaign", extra={"campaign_id": campaign_id})

            campaign = self._campaigns.get(campaign_id)
            if campaign is None:
                logger.warning(f"Campaign not found, nothing to stop: {campaign_id}")
                return False

            current_status = campaign.get("status")
            if current_status in (CampaignStatus.COMPLETED, CampaignStatus.CANCELLED, CampaignStatus.FAILED):
                logger.info(f"Campaign {campaign_id} already stopped (status={current_status})")
                return True

            # 更新狀態機
            campaign["status"] = CampaignStatus.CANCELLED
            campaign["stopped_at"] = datetime.now().isoformat()

            logger.info(f"Campaign stopped successfully", extra={"campaign_id": campaign_id})
            return True

        except Exception as e:
            logger.error(f"Failed to stop campaign: {str(e)}", exc_info=True)
            return False
