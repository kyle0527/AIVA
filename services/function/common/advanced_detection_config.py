"""
進階檢測配置系統 - 功能增強和優化
基於現有 detection_config.py，提供更靈活的配置選項
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from datetime import datetime, timedelta

from .detection_config import (
    BaseDetectionConfig, SSRFConfig, XSSConfig, IDORConfig, 
    DetectionStrategy, DEFAULT_SSRF_CONFIG, DEFAULT_XSS_CONFIG, DEFAULT_IDOR_CONFIG
)


class DetectionMode(Enum):
    """檢測模式枚舉"""
    STEALTH = "stealth"          # 隱蔽模式：最小化檢測痕跡
    STANDARD = "standard"        # 標準模式：平衡檢測和隱蔽
    COMPREHENSIVE = "comprehensive"  # 全面模式：完整檢測覆蓋
    COMPLIANCE = "compliance"    # 合規模式：遵循安全標準
    CUSTOM = "custom"           # 自定義模式


class ReportLevel(Enum):
    """報告詳細度級別"""
    MINIMAL = "minimal"         # 最小化報告
    SUMMARY = "summary"         # 摘要報告  
    DETAILED = "detailed"       # 詳細報告
    VERBOSE = "verbose"         # 詳細日志報告


@dataclass
class SQLiConfig(BaseDetectionConfig):
    """SQLi 檢測專用配置 - 新增功能"""
    
    # 檢測引擎控制
    engines_enabled: list[str] = field(
        default_factory=lambda: ["sqlmap", "ghauri", "nosqlmap", "commix", "custom"]
    )
    
    # 數據庫類型優先級
    database_priority: list[str] = field(
        default_factory=lambda: ["mysql", "postgresql", "mssql", "oracle", "sqlite", "mongodb", "redis"]
    )
    
    # 注入技術控制
    injection_techniques: dict[str, bool] = field(
        default_factory=lambda: {
            "boolean_blind": True,
            "time_blind": True, 
            "error_based": True,
            "union_based": True,
            "stacked_queries": True,
            "inline_queries": False,
        }
    )
    
    # WAF 繞過策略
    waf_bypass_enabled: bool = True
    waf_bypass_techniques: list[str] = field(
        default_factory=lambda: [
            "encoding", "chunking", "comment_insertion", 
            "case_variation", "whitespace_manipulation"
        ]
    )
    
    # NoSQL 特殊配置
    nosql_databases: list[str] = field(
        default_factory=lambda: ["mongodb", "redis", "cassandra", "elasticsearch"]
    )
    
    # 載荷生成控制
    payload_complexity: str = "adaptive"  # simple, medium, complex, adaptive
    custom_payloads_enabled: bool = True
    payload_mutation_enabled: bool = True
    
    # 結果驗證
    result_verification_enabled: bool = True
    false_positive_reduction: bool = True


@dataclass  
class AdvancedSSRFConfig(SSRFConfig):
    """增強的 SSRF 配置"""
    
    # 高級檢測功能
    blind_ssrf_detection: bool = True
    callback_validation_timeout: float = 45.0
    
    # 繞過技術
    bypass_techniques_enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "url_encoding": True,
            "unicode_normalization": True, 
            "ip_obfuscation": True,
            "redirect_chains": True,
            "protocol_smuggling": True,
            "hostname_confusion": True,
        }
    )
    
    # 雲服務專項檢測
    cloud_service_detection: dict[str, bool] = field(
        default_factory=lambda: {
            "aws_metadata": True,
            "gcp_metadata": True, 
            "azure_metadata": True,
            "alibaba_metadata": True,
            "kubernetes_api": True,
            "docker_socket": True,
        }
    )
    
    # 內網服務掃描
    internal_service_scan: bool = True
    common_internal_ports: list[int] = field(
        default_factory=lambda: [22, 23, 25, 53, 80, 135, 139, 443, 445, 993, 995, 1433, 3306, 3389, 5432, 6379, 27017]
    )
    
    # OAST (Out-of-Band Application Security Testing)
    oast_providers: list[str] = field(
        default_factory=lambda: ["burpcollaborator", "interact.sh", "canarytokens", "custom"]
    )
    
    # 響應分析
    response_analysis_enabled: bool = True
    response_time_analysis: bool = True
    response_content_analysis: bool = True


@dataclass
class AdvancedXSSConfig(XSSConfig):
    """增強的 XSS 配置"""
    
    # 進階檢測技術
    mutation_based_testing: bool = True
    semantic_analysis_enabled: bool = True
    
    # CSP 繞過技術
    csp_bypass_enabled: bool = True
    csp_bypass_techniques: list[str] = field(
        default_factory=lambda: [
            "jsonp_injection", "angular_template", "vue_template",
            "react_dangerouslySetInnerHTML", "data_uri", "javascript_uri"
        ]
    )
    
    # 框架特定檢測
    framework_specific_tests: dict[str, bool] = field(
        default_factory=lambda: {
            "angular": True,
            "react": True,
            "vue": True, 
            "jquery": True,
            "backbone": True,
            "ember": True,
        }
    )
    
    # WAF 繞過載荷
    waf_bypass_payloads: bool = True
    filter_bypass_techniques: list[str] = field(
        default_factory=lambda: [
            "html_encoding", "js_encoding", "unicode_normalization",
            "tag_breaking", "attribute_breaking", "event_handler_variation"
        ]
    )
    
    # 盲 XSS 增強
    blind_xss_advanced: dict[str, Any] = field(
        default_factory=lambda: {
            "screenshot_capture": True,
            "keylogger_simulation": True,
            "cookie_extraction": True,
            "dom_manipulation": True,
        }
    )
    
    # 多步驟檢測
    multi_step_detection: bool = True
    form_interaction_enabled: bool = True
    ajax_endpoint_testing: bool = True


@dataclass
class AdvancedIDORConfig(IDORConfig):
    """增強的 IDOR 配置"""
    
    # 智能 ID 分析
    intelligent_id_analysis: bool = True
    id_pattern_learning: bool = True
    
    # API 特定檢測
    api_endpoint_analysis: dict[str, bool] = field(
        default_factory=lambda: {
            "rest_api": True,
            "graphql_api": True,
            "grpc_api": True,
            "websocket_api": True,
        }
    )
    
    # 權限模型檢測
    rbac_testing_enabled: bool = True
    acl_testing_enabled: bool = True
    
    # 高級枚舉技術
    advanced_enumeration: dict[str, bool] = field(
        default_factory=lambda: {
            "machine_learning_prediction": True,
            "pattern_based_generation": True,
            "temporal_analysis": True,
            "statistical_inference": True,
        }
    )
    
    # 多租戶測試
    multi_tenant_scenarios: list[str] = field(
        default_factory=lambda: [
            "organization_isolation", "user_isolation", 
            "role_isolation", "department_isolation"
        ]
    )
    
    # 對象關係檢測
    object_relationship_analysis: bool = True
    dependency_chain_testing: bool = True
    
    # 時間相關測試
    temporal_idor_testing: bool = True
    session_fixation_testing: bool = True


@dataclass
class PerformanceConfig:
    """性能配置"""
    
    # 並發控制
    max_concurrent_requests: int = 10
    connection_pool_size: int = 50
    
    # 資源限制
    max_memory_usage_mb: int = 512
    max_cpu_usage_percent: int = 80
    
    # 緩存配置
    result_cache_enabled: bool = True
    cache_ttl_minutes: int = 30
    
    # 負載控制
    adaptive_throttling: bool = True
    circuit_breaker_enabled: bool = True
    
    # 監控指標
    performance_monitoring: bool = True
    metrics_collection_interval: int = 10


@dataclass
class SecurityConfig:
    """安全配置"""
    
    # 通信安全
    tls_verification: bool = True
    certificate_pinning: bool = False
    
    # 認證配置
    authentication_methods: list[str] = field(
        default_factory=lambda: ["basic", "bearer", "oauth2", "api_key"]
    )
    
    # 數據保護
    sensitive_data_masking: bool = True
    log_sanitization: bool = True
    
    # 合規要求
    gdpr_compliance: bool = True
    pci_compliance: bool = False
    
    # 審計日志
    audit_logging_enabled: bool = True
    audit_log_retention_days: int = 90


@dataclass
class EnvironmentConfig:
    """環境配置"""
    
    # 環境類型
    environment_type: str = "development"  # development, staging, production
    
    # 代理設置
    proxy_enabled: bool = False
    proxy_config: dict[str, str] = field(default_factory=dict)
    
    # 用戶代理設置
    user_agents: list[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        ]
    )
    
    # 網絡配置
    dns_servers: list[str] = field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"]
    )
    
    # 測試環境隔離
    sandbox_enabled: bool = True
    network_isolation: bool = True


@dataclass
class UnifiedDetectionConfig:
    """統一檢測配置 - 整合所有模組配置"""
    
    # 模組配置
    sqli_config: SQLiConfig = field(default_factory=SQLiConfig)
    ssrf_config: AdvancedSSRFConfig = field(default_factory=AdvancedSSRFConfig)
    xss_config: AdvancedXSSConfig = field(default_factory=AdvancedXSSConfig) 
    idor_config: AdvancedIDORConfig = field(default_factory=AdvancedIDORConfig)
    
    # 全局配置
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # 檢測策略
    detection_mode: DetectionMode = DetectionMode.STANDARD
    report_level: ReportLevel = ReportLevel.DETAILED
    
    # 排程配置
    schedule_enabled: bool = False
    schedule_cron: str = ""
    
    # 通知配置
    notifications_enabled: bool = True
    notification_channels: list[str] = field(
        default_factory=lambda: ["email", "slack", "webhook"]
    )
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "sqli": self.sqli_config.__dict__,
            "ssrf": self.ssrf_config.__dict__,
            "xss": self.xss_config.__dict__,
            "idor": self.idor_config.__dict__,
            "performance": self.performance_config.__dict__,
            "security": self.security_config.__dict__,
            "environment": self.environment_config.__dict__,
            "detection_mode": self.detection_mode.value,
            "report_level": self.report_level.value,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "UnifiedDetectionConfig":
        """從字典創建配置"""
        # 實現從字典反序列化的邏輯
        pass
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "UnifiedDetectionConfig":
        """從文件加載配置"""
        import json
        with open(file_path, encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class ConfigurationManager:
    """配置管理器 - 動態配置管理"""
    
    def __init__(self):
        self.config = UnifiedDetectionConfig()
        self.config_history: list[dict[str, Any]] = []
        
    def apply_preset(self, preset_name: str) -> None:
        """應用預設配置"""
        presets = {
            "penetration_test": self._get_penetration_test_preset(),
            "compliance_audit": self._get_compliance_audit_preset(),
            "continuous_monitoring": self._get_continuous_monitoring_preset(),
            "red_team_simulation": self._get_red_team_preset(),
        }
        
        if preset_name in presets:
            self.config = presets[preset_name]
            self._save_config_history()
    
    def _get_penetration_test_preset(self) -> UnifiedDetectionConfig:
        """滲透測試預設配置"""
        config = UnifiedDetectionConfig()
        config.detection_mode = DetectionMode.COMPREHENSIVE
        config.report_level = ReportLevel.VERBOSE
        
        # 啟用所有高級功能
        config.sqli_config.waf_bypass_enabled = True
        config.ssrf_config.blind_ssrf_detection = True
        config.xss_config.csp_bypass_enabled = True
        config.idor_config.intelligent_id_analysis = True
        
        return config
    
    def _get_compliance_audit_preset(self) -> UnifiedDetectionConfig:
        """合規審計預設配置"""
        config = UnifiedDetectionConfig()
        config.detection_mode = DetectionMode.COMPLIANCE
        config.report_level = ReportLevel.DETAILED
        
        # 啟用合規相關功能
        config.security_config.gdpr_compliance = True
        config.security_config.audit_logging_enabled = True
        
        return config
    
    def _get_continuous_monitoring_preset(self) -> UnifiedDetectionConfig:
        """持續監控預設配置"""
        config = UnifiedDetectionConfig()
        config.detection_mode = DetectionMode.STEALTH
        config.report_level = ReportLevel.SUMMARY
        config.schedule_enabled = True
        config.schedule_cron = "0 2 * * *"  # 每日凌晨 2 點
        
        return config
    
    def _get_red_team_preset(self) -> UnifiedDetectionConfig:
        """紅隊模擬預設配置"""
        config = UnifiedDetectionConfig()
        config.detection_mode = DetectionMode.COMPREHENSIVE
        config.report_level = ReportLevel.VERBOSE
        
        # 啟用所有繞過技術
        config.sqli_config.waf_bypass_enabled = True
        config.ssrf_config.bypass_techniques_enabled = dict.fromkeys(config.ssrf_config.bypass_techniques_enabled, True)
        config.xss_config.waf_bypass_payloads = True
        config.idor_config.advanced_enumeration = dict.fromkeys(config.idor_config.advanced_enumeration, True)
        
        return config
    
    def _save_config_history(self) -> None:
        """保存配置歷史"""
        self.config_history.append({
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict()
        })
    
    def get_effective_config(self, module_name: str) -> BaseDetectionConfig:
        """獲取特定模組的有效配置"""
        module_configs = {
            "sqli": self.config.sqli_config,
            "ssrf": self.config.ssrf_config,
            "xss": self.config.xss_config,
            "idor": self.config.idor_config,
        }
        
        return module_configs.get(module_name)
    
    def validate_config(self) -> list[str]:
        """驗證配置有效性"""
        errors = []
        
        # 驗證超時設置
        if self.config.sqli_config.timeout_base > self.config.sqli_config.timeout_max:
            errors.append("SQLi: timeout_base cannot be greater than timeout_max")
        
        # 驗證性能配置
        if self.config.performance_config.max_concurrent_requests > 100:
            errors.append("Performance: max_concurrent_requests too high (>100)")
        
        return errors


# 預設配置管理器實例
DEFAULT_CONFIG_MANAGER = ConfigurationManager()

# 預設統一配置
DEFAULT_UNIFIED_CONFIG = UnifiedDetectionConfig()
