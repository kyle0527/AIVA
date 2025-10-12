"""
統一檢測配置系統 - 基於 SQLi 模組成功經驗
為所有功能模組提供統一的配置管理
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseDetectionConfig:
    """基礎檢測配置 - 所有模組的共同配置"""

    # 超時設置
    timeout_base: float = 5.0
    timeout_max: float = 15.0

    # 速率控制
    requests_per_second: float = 0.5

    # 早期停止控制
    max_vulnerabilities: int = 3
    max_consecutive_failures: int = 5

    # 進度報告
    progress_update_interval: float = 2.0

    # 重試策略
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    # 防護機制檢測
    protection_detection_enabled: bool = True


@dataclass
class SSRFConfig(BaseDetectionConfig):
    """SSRF 檢測專用配置"""

    # SSRF 特定設置
    internal_scan_enabled: bool = True
    oast_timeout: float = 30.0
    oast_wait_time: float = 3.0  # OAST 回調等待時間
    dns_servers: list[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])

    # 內網掃描範圍
    internal_ranges: list[str] = field(
        default_factory=lambda: [
            "127.0.0.0/8",  # localhost
            "10.0.0.0/8",  # 私有網路 A
            "172.16.0.0/12",  # 私有網路 B
            "192.168.0.0/16",  # 私有網路 C
            "169.254.0.0/16",  # 鏈路本地
        ]
    )

    # 雲服務 metadata 端點
    cloud_metadata_endpoints: list[str] = field(
        default_factory=lambda: [
            "http://169.254.169.254/",  # AWS
            "http://metadata.google.internal/",  # GCP
            "http://100.64.0.1/",  # Azure
        ]
    )

    # 雲元數據優先檢測
    cloud_metadata_first: bool = True

    # 協議測試
    protocols_enabled: list[str] = field(
        default_factory=lambda: ["http", "https", "file", "gopher", "dict", "ftp"]
    )


@dataclass
class XSSConfig(BaseDetectionConfig):
    """XSS 檢測專用配置"""

    # XSS 檢測類型控制
    reflected_xss_enabled: bool = True
    dom_xss_enabled: bool = True
    stored_xss_enabled: bool = True
    blind_xss_enabled: bool = False

    # 存儲型 XSS 檢測設置
    stored_check_delay: float = 5.0
    stored_check_attempts: int = 3

    # 盲 XSS 設置
    blind_xss_timeout: float = 60.0
    blind_xss_callback_domain: str = ""

    # DOM XSS 設置
    dom_analysis_timeout: float = 10.0
    javascript_execution_enabled: bool = True

    # 載荷設置
    payload_encoding_methods: list[str] = field(
        default_factory=lambda: ["none", "url", "html", "javascript", "base64"]
    )

    # Context 檢測
    context_detection_enabled: bool = True
    supported_contexts: list[str] = field(
        default_factory=lambda: ["html", "attribute", "script", "style", "comment"]
    )


@dataclass
class IDORConfig(BaseDetectionConfig):
    """IDOR 檢測專用配置"""

    # IDOR 檢測類型
    horizontal_testing_enabled: bool = True  # 別名，與智能檢測器兼容
    vertical_testing_enabled: bool = True  # 別名，與智能檢測器兼容
    horizontal_escalation_enabled: bool = True
    vertical_escalation_enabled: bool = True
    cross_tenant_enabled: bool = True

    # ID 生成策略
    id_generation_methods: list[str] = field(
        default_factory=lambda: [
            "increment",
            "decrement",
            "uuid",
            "hash",
            "random",
            "bruteforce",
        ]
    )

    # ID 模式識別
    id_pattern_detection_enabled: bool = True
    supported_id_types: list[str] = field(
        default_factory=lambda: ["numeric", "uuid", "hash", "mixed", "custom"]
    )

    # 測試範圍
    id_test_range: int = 100  # 每種方法測試的ID數量
    bruteforce_range: int = 1000  # 暴力破解範圍

    # 權限提升測試
    privilege_escalation_paths: list[str] = field(
        default_factory=lambda: [
            "/admin/",
            "/api/admin/",
            "/management/",
            "/dashboard/",
        ]
    )

    # 權限級別定義 (用於垂直權限提升測試)
    privilege_levels: list[str] = field(
        default_factory=lambda: ["admin", "user", "guest", "anonymous"]
    )

    # 跨用戶測試
    cross_user_simulation_enabled: bool = True
    test_user_count: int = 3


@dataclass
class DetectionStrategy:
    """檢測策略配置 - 控制檢測的激進程度"""

    # 策略類型
    CONSERVATIVE = "conservative"  # 保守：快速、少載荷
    BALANCED = "balanced"  # 平衡：中等速度和覆蓋
    AGGRESSIVE = "aggressive"  # 激進：完整覆蓋、較慢

    strategy_type: str = BALANCED

    def apply_to_config(self, config: BaseDetectionConfig) -> BaseDetectionConfig:
        """將策略應用到配置"""
        if self.strategy_type == self.CONSERVATIVE:
            config.max_vulnerabilities = 2
            config.requests_per_second = 0.2
            config.timeout_base = 3.0
            config.timeout_max = 8.0
        elif self.strategy_type == self.AGGRESSIVE:
            config.max_vulnerabilities = 10
            config.requests_per_second = 1.0
            config.timeout_base = 10.0
            config.timeout_max = 30.0
        # BALANCED 使用默認值

        return config


# 預設配置實例
DEFAULT_SSRF_CONFIG = SSRFConfig()
DEFAULT_XSS_CONFIG = XSSConfig()
DEFAULT_IDOR_CONFIG = IDORConfig()

# 策略實例
CONSERVATIVE_STRATEGY = DetectionStrategy(DetectionStrategy.CONSERVATIVE)
BALANCED_STRATEGY = DetectionStrategy(DetectionStrategy.BALANCED)
AGGRESSIVE_STRATEGY = DetectionStrategy(DetectionStrategy.AGGRESSIVE)
