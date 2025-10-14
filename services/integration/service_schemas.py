"""
AIVA 整合服務模式定義

包含威脅情報、SIEM整合、資產生命週期管理等外部服務整合相關的數據模式。
屬於 integration 模組的業務特定定義。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aiva_common.enums import DataSource, IntegrationStatus, Severity
from pydantic import BaseModel, Field

# ==================== 威脅情報整合 ====================


class IOCRecord(BaseModel):
    """威脅指標記錄 (Indicator of Compromise)"""

    ioc_id: str = Field(description="IOC唯一標識符")
    ioc_type: str = Field(description="IOC類型")  # "ip", "domain", "url", "hash", "email"
    value: str = Field(description="IOC值")

    # 威脅信息
    threat_type: str | None = Field(default=None, description="威脅類型")
    malware_family: str | None = Field(default=None, description="惡意軟體家族")
    campaign: str | None = Field(default=None, description="攻擊活動")

    # 評級信息
    severity: Severity = Field(description="嚴重程度")
    confidence: int = Field(ge=0, le=100, description="可信度 0-100")
    reputation_score: int = Field(ge=0, le=100, description="聲譽分數")

    # 來源信息
    source: DataSource = Field(description="數據來源")
    source_reference: str | None = Field(default=None, description="來源引用")

    # 時間信息
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後發現時間")
    expires_at: datetime | None = Field(default=None, description="過期時間")

    # 標籤和分類
    tags: list[str] = Field(default_factory=list, description="標籤")
    mitre_techniques: list[str] = Field(default_factory=list, description="MITRE ATT&CK技術")

    # 元數據
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ThreatIndicator(BaseModel):
    """威脅指標"""

    indicator_id: str = Field(description="指標ID")
    pattern: str = Field(description="匹配模式")
    indicator_type: str = Field(description="指標類型")

    # 威脅評級
    threat_level: Severity = Field(description="威脅級別")
    accuracy: float = Field(ge=0.0, le=1.0, description="準確度")

    # 檢測配置
    detection_logic: str = Field(description="檢測邏輯")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="誤報率")

    # 關聯信息
    related_iocs: list[str] = Field(default_factory=list, description="相關IOC")
    kill_chain_phase: str | None = Field(default=None, description="殺傷鏈階段")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ThreatIntelPayload(BaseModel):
    """威脅情報查詢載荷"""

    task_id: str = Field(description="任務ID")
    query_type: str = Field(description="查詢類型")  # "ip", "domain", "hash", "batch"
    indicators: list[str] = Field(description="查詢指標列表")

    # 查詢配置
    include_passive_dns: bool = Field(default=True, description="包含被動DNS")
    include_whois: bool = Field(default=True, description="包含WHOIS信息")
    include_geolocation: bool = Field(default=True, description="包含地理位置")

    # 數據源配置
    sources: list[DataSource] = Field(description="數據源列表")
    max_age_days: int = Field(default=30, ge=1, description="最大數據年齡(天)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== SIEM 整合 ====================


class SIEMEvent(BaseModel):
    """SIEM事件"""

    event_id: str = Field(description="事件ID")
    source_system: str = Field(description="來源系統")
    event_type: str = Field(description="事件類型")

    # 時間信息
    timestamp: datetime = Field(description="事件時間戳")
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 事件詳情
    severity: Severity = Field(description="嚴重程度")
    category: str = Field(description="事件分類")
    description: str = Field(description="事件描述")

    # 來源信息
    source_ip: str | None = Field(default=None, description="來源IP")
    source_host: str | None = Field(default=None, description="來源主機")
    source_user: str | None = Field(default=None, description="來源用戶")

    # 目標信息
    target_ip: str | None = Field(default=None, description="目標IP")
    target_host: str | None = Field(default=None, description="目標主機")
    target_user: str | None = Field(default=None, description="目標用戶")

    # 事件數據
    raw_data: dict[str, Any] = Field(default_factory=dict, description="原始數據")
    processed_data: dict[str, Any] = Field(default_factory=dict, description="處理後數據")

    # 關聯信息
    correlation_id: str | None = Field(default=None, description="關聯ID")
    related_events: list[str] = Field(default_factory=list, description="相關事件")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class SIEMIntegrationConfig(BaseModel):
    """SIEM整合配置"""

    integration_name: str = Field(description="整合名稱")
    siem_type: str = Field(description="SIEM類型")  # "splunk", "qradar", "arcsight", "elastic"

    # 連接配置
    endpoint: str = Field(description="SIEM端點")
    api_key: str | None = Field(default=None, description="API密鑰")
    username: str | None = Field(default=None, description="用戶名")
    password: str | None = Field(default=None, description="密碼")

    # 同步配置
    sync_interval: int = Field(default=300, ge=60, description="同步間隔(秒)")
    batch_size: int = Field(default=100, ge=1, le=1000, description="批量大小")

    # 過濾配置
    severity_filter: list[Severity] = Field(default_factory=list, description="嚴重程度過濾")
    category_filter: list[str] = Field(default_factory=list, description="分類過濾")

    # 狀態
    status: IntegrationStatus = Field(description="整合狀態")
    last_sync: datetime | None = Field(default=None, description="最後同步時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 資產生命週期管理 ====================


class AssetLifecycleEvent(BaseModel):
    """資產生命週期事件"""

    event_id: str = Field(description="事件ID")
    asset_id: str = Field(description="資產ID")
    event_type: str = Field(description="事件類型")  # "discovered", "updated", "deprecated", "removed"

    # 變更信息
    old_state: dict[str, Any] | None = Field(default=None, description="變更前狀態")
    new_state: dict[str, Any] | None = Field(default=None, description="變更後狀態")
    change_reason: str | None = Field(default=None, description="變更原因")

    # 操作信息
    initiated_by: str | None = Field(default=None, description="發起者")
    automated: bool = Field(default=False, description="是否自動化")

    # 時間戳
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AssetVulnerabilityManager(BaseModel):
    """資產漏洞管理器"""

    asset_id: str = Field(description="資產ID")
    vulnerability_count: int = Field(ge=0, description="漏洞總數")

    # 嚴重程度統計
    critical_count: int = Field(ge=0, description="嚴重漏洞數")
    high_count: int = Field(ge=0, description="高危漏洞數")
    medium_count: int = Field(ge=0, description="中危漏洞數")
    low_count: int = Field(ge=0, description="低危漏洞數")

    # 狀態統計
    open_count: int = Field(ge=0, description="開放漏洞數")
    fixed_count: int = Field(ge=0, description="已修復漏洞數")
    mitigated_count: int = Field(ge=0, description="已緩解漏洞數")

    # 風險評估
    risk_score: float = Field(ge=0.0, le=10.0, description="風險評分")
    exposure_level: Severity = Field(description="暴露級別")

    # 時間信息
    last_scan: datetime | None = Field(default=None, description="最後掃描時間")
    next_scan: datetime | None = Field(default=None, description="下次掃描時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 第三方服務整合 ====================


class ThirdPartyAPIConfig(BaseModel):
    """第三方API配置"""

    service_name: str = Field(description="服務名稱")
    api_type: str = Field(description="API類型")  # "rest", "graphql", "soap", "webhook"

    # 認證配置
    auth_type: str = Field(description="認證類型")  # "api_key", "oauth", "basic", "bearer"
    credentials: dict[str, str] = Field(default_factory=dict, description="認證憑據")

    # 端點配置
    base_url: str = Field(description="基礎URL")
    endpoints: dict[str, str] = Field(default_factory=dict, description="端點映射")

    # 限制配置
    rate_limit: int = Field(default=100, description="速率限制(每分鐘)")
    timeout: int = Field(default=30, ge=1, description="超時時間(秒)")
    retry_attempts: int = Field(default=3, ge=0, description="重試次數")

    # 狀態
    status: IntegrationStatus = Field(description="配置狀態")
    last_test: datetime | None = Field(default=None, description="最後測試時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class WebhookPayload(BaseModel):
    """Webhook載荷"""

    webhook_id: str = Field(description="Webhook ID")
    event_type: str = Field(description="事件類型")
    source_system: str = Field(description="來源系統")

    # 載荷數據
    payload_data: dict[str, Any] = Field(description="載荷數據")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP標頭")

    # 驗證信息
    signature: str | None = Field(default=None, description="簽名")
    timestamp: datetime = Field(description="時間戳")

    # 處理狀態
    processed: bool = Field(default=False, description="是否已處理")
    processing_error: str | None = Field(default=None, description="處理錯誤")

    # 時間戳
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = Field(default=None, description="處理時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== EASM (外部攻擊面管理) ====================


class EASMAsset(BaseModel):
    """外部攻擊面資產"""

    asset_id: str = Field(description="資產ID")
    asset_type: str = Field(description="資產類型")  # "domain", "subdomain", "ip", "port", "service"
    value: str = Field(description="資產值")

    # 發現信息
    discovery_method: str = Field(description="發現方式")
    discovery_source: str = Field(description="發現來源")
    first_discovered: datetime = Field(description="首次發現")
    last_verified: datetime = Field(description="最後驗證")

    # 狀態信息
    is_active: bool = Field(description="是否活躍")
    is_monitored: bool = Field(default=True, description="是否監控")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")

    # 風險評估
    risk_level: Severity = Field(description="風險級別")
    exposure_score: float = Field(ge=0.0, le=10.0, description="暴露評分")

    # 技術信息
    technologies: list[str] = Field(default_factory=list, description="使用的技術")
    services: list[str] = Field(default_factory=list, description="運行的服務")
    certificates: list[dict] = Field(default_factory=list, description="SSL證書")

    # 關聯信息
    parent_asset: str | None = Field(default=None, description="父資產")
    child_assets: list[str] = Field(default_factory=list, description="子資產")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EASMDiscoveryJob(BaseModel):
    """EASM發現任務"""

    job_id: str = Field(description="任務ID")
    job_type: str = Field(description="任務類型")  # "domain_enum", "port_scan", "service_discovery"

    # 目標配置
    targets: list[str] = Field(description="目標列表")
    scope: dict[str, Any] = Field(description="掃描範圍")

    # 執行配置
    discovery_depth: int = Field(default=3, ge=1, le=10, description="發現深度")
    passive_only: bool = Field(default=True, description="僅被動發現")

    # 狀態信息
    status: str = Field(description="任務狀態")  # "queued", "running", "completed", "failed"
    progress: float = Field(ge=0.0, le=1.0, description="進度")

    # 結果統計
    assets_found: int = Field(ge=0, description="發現的資產數")
    new_assets: int = Field(ge=0, description="新資產數")
    updated_assets: int = Field(ge=0, description="更新的資產數")

    # 時間信息
    started_at: datetime | None = Field(default=None, description="開始時間")
    completed_at: datetime | None = Field(default=None, description="完成時間")
    next_run: datetime | None = Field(default=None, description="下次運行時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


__all__ = [
    "IOCRecord",
    "ThreatIndicator",
    "ThreatIntelPayload",
    "SIEMEvent",
    "SIEMIntegrationConfig",
    "AssetLifecycleEvent",
    "AssetVulnerabilityManager",
    "ThirdPartyAPIConfig",
    "WebhookPayload",
    "EASMAsset",
    "EASMDiscoveryJob",
]
