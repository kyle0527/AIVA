

from dataclasses import dataclass, field
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CrawlingConfig:
    """爬蟲配置"""

    # 深度控制
    max_depth: int = 3
    max_pages: int = 100
    max_forms: int = 50

    # 超時設置
    request_timeout: float = 10.0
    page_load_timeout: float = 30.0

    # 速率限制
    requests_per_second: float = 2.0
    concurrent_requests: int = 5

    # 跟隨重定向
    follow_redirects: bool = True
    max_redirects: int = 5

    # 資源類型過濾
    skip_static_resources: bool = True
    static_extensions: list[str] = field(
        default_factory=lambda: [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".css",
            ".js",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
        ]
    )


@dataclass
class DynamicScanConfig:
    """動態掃描配置"""

    # 啟用動態掃描
    enabled: bool = False

    # 瀏覽器配置
    browser_pool_size: int = 2
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True

    # JavaScript 互動
    js_interaction_enabled: bool = True
    interaction_timeout: float = 5.0

    # 內容提取
    extract_forms: bool = True
    extract_links: bool = True
    extract_ajax: bool = True
    extract_api_calls: bool = True

    # 等待策略
    wait_for_network_idle: bool = True
    network_idle_timeout_ms: int = 2000


@dataclass
class FingerprintConfig:
    """指紋識別配置"""

    # 啟用指紋識別
    enabled: bool = True

    # 檢測類型
    detect_web_server: bool = True
    detect_framework: bool = True
    detect_language: bool = True
    detect_waf: bool = True
    detect_cms: bool = True

    # 被動指紋
    passive_fingerprinting: bool = True

    # 主動指紋（可能觸發防護）
    active_fingerprinting: bool = False


@dataclass
class SecurityConfig:
    """安全配置"""

    # SSL/TLS 驗證
    verify_ssl: bool = True
    ssl_cert_path: str | None = None

    # 代理設置
    use_proxy: bool = False
    proxy_url: str | None = None

    # 認證重試
    auth_retry_attempts: int = 3

    # Cookie 管理
    persist_cookies: bool = True
    cookie_jar_size: int = 1000

    # 用戶代理輪換
    rotate_user_agent: bool = False
    user_agents: list[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """性能配置"""

    # 連接池
    connection_pool_size: int = 10
    keep_alive_timeout: float = 30.0

    # 緩存
    enable_response_cache: bool = False
    cache_size_mb: int = 100

    # 內存限制
    max_memory_mb: int = 512

    # 並發控制
    max_concurrent_scans: int = 3


@dataclass
class ReportingConfig:
    """報告配置"""

    # 進度報告
    progress_updates_enabled: bool = True
    progress_interval_seconds: float = 5.0

    # 詳細程度
    verbosity: str = "normal"  # minimal, normal, detailed, debug

    # 統計收集
    collect_statistics: bool = True
    collect_timing_info: bool = True

    # 錯誤報告
    report_errors: bool = True
    max_errors_to_report: int = 100


@dataclass
class ScanModuleConfig:
    """掃描模組整體配置"""

    crawling: CrawlingConfig = field(default_factory=CrawlingConfig)
    dynamic: DynamicScanConfig = field(default_factory=DynamicScanConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


class ConfigControlCenter:
    """
    掃描模組的配置和控制中心。

    主要功能:
    - 統一管理所有掃描配置
    - 提供配置驗證和默認值
    - 支持配置覆蓋和合併
    - 提供配置快照和恢復
    - 動態調整配置參數

    使用範例:
        # 使用默認配置
        config_center = ConfigControlCenter()

        # 使用自定義配置
        custom_config = ScanModuleConfig(
            crawling=CrawlingConfig(max_depth=5, max_pages=200)
        )
        config_center = ConfigControlCenter(custom_config)

        # 獲取配置
        crawling_config = config_center.get_crawling_config()

        # 更新配置
        config_center.update_crawling_config(max_depth=10)

        # 應用預設模板
        config_center.apply_preset("aggressive")
    """

    def __init__(self, config: ScanModuleConfig | None = None):
        """
        初始化配置控制中心

        Args:
            config: 自定義配置，如果為 None 則使用默認配置
        """
        self._config = config or ScanModuleConfig()
        self._config_history: list[ScanModuleConfig] = []
        self._validate_config()

        logger.info("ConfigControlCenter initialized with configuration")

    def get_config(self) -> ScanModuleConfig:
        """獲取完整配置"""
        return self._config

    def get_crawling_config(self) -> CrawlingConfig:
        """獲取爬蟲配置"""
        return self._config.crawling

    def get_dynamic_config(self) -> DynamicScanConfig:
        """獲取動態掃描配置"""
        return self._config.dynamic

    def get_fingerprint_config(self) -> FingerprintConfig:
        """獲取指紋識別配置"""
        return self._config.fingerprint

    def get_security_config(self) -> SecurityConfig:
        """獲取安全配置"""
        return self._config.security

    def get_performance_config(self) -> PerformanceConfig:
        """獲取性能配置"""
        return self._config.performance

    def get_reporting_config(self) -> ReportingConfig:
        """獲取報告配置"""
        return self._config.reporting

    def update_crawling_config(self, **kwargs: Any) -> None:
        """
        更新爬蟲配置

        Args:
            **kwargs: 要更新的配置項
        """
        self._save_snapshot()
        for key, value in kwargs.items():
            if hasattr(self._config.crawling, key):
                setattr(self._config.crawling, key, value)
                logger.debug(f"Updated crawling config: {key}={value}")
            else:
                logger.warning(f"Unknown crawling config key: {key}")

    def update_dynamic_config(self, **kwargs: Any) -> None:
        """更新動態掃描配置"""
        self._save_snapshot()
        for key, value in kwargs.items():
            if hasattr(self._config.dynamic, key):
                setattr(self._config.dynamic, key, value)
                logger.debug(f"Updated dynamic config: {key}={value}")
            else:
                logger.warning(f"Unknown dynamic config key: {key}")

    def update_security_config(self, **kwargs: Any) -> None:
        """更新安全配置"""
        self._save_snapshot()
        for key, value in kwargs.items():
            if hasattr(self._config.security, key):
                setattr(self._config.security, key, value)
                logger.debug(f"Updated security config: {key}={value}")
            else:
                logger.warning(f"Unknown security config key: {key}")

    def update_performance_config(self, **kwargs: Any) -> None:
        """更新性能配置"""
        self._save_snapshot()
        for key, value in kwargs.items():
            if hasattr(self._config.performance, key):
                setattr(self._config.performance, key, value)
                logger.debug(f"Updated performance config: {key}={value}")
            else:
                logger.warning(f"Unknown performance config key: {key}")

    def apply_preset(self, preset_name: str) -> bool:
        """
        應用預設配置模板

        Args:
            preset_name: 預設名稱 (conservative, balanced, aggressive, stealth)

        Returns:
            bool: 是否成功應用
        """
        self._save_snapshot()

        presets = {
            "conservative": self._get_conservative_preset,
            "balanced": self._get_balanced_preset,
            "aggressive": self._get_aggressive_preset,
            "stealth": self._get_stealth_preset,
            "fast": self._get_fast_preset,
            "deep": self._get_deep_preset,
        }

        if preset_name not in presets:
            logger.warning(f"Unknown preset: {preset_name}")
            return False

        preset_config = presets[preset_name]()
        self._config = preset_config
        self._validate_config()

        logger.info(f"Applied preset: {preset_name}")
        return True

    def restore_previous_config(self) -> bool:
        """
        恢復到上一個配置快照

        Returns:
            bool: 是否成功恢復
        """
        if not self._config_history:
            logger.warning("No previous configuration to restore")
            return False

        self._config = self._config_history.pop()
        logger.info("Restored previous configuration")
        return True

    def reset_to_defaults(self) -> None:
        """重置為默認配置"""
        self._save_snapshot()
        self._config = ScanModuleConfig()
        logger.info("Reset to default configuration")

    def export_config(self) -> dict[str, Any]:
        """導出配置為字典"""
        return {
            "crawling": self._dataclass_to_dict(self._config.crawling),
            "dynamic": self._dataclass_to_dict(self._config.dynamic),
            "fingerprint": self._dataclass_to_dict(self._config.fingerprint),
            "security": self._dataclass_to_dict(self._config.security),
            "performance": self._dataclass_to_dict(self._config.performance),
            "reporting": self._dataclass_to_dict(self._config.reporting),
        }

    def get_config_summary(self) -> str:
        """獲取配置摘要（用於日誌輸出）"""
        crawl = self._config.crawling
        dynamic = self._config.dynamic
        perf = self._config.performance

        lines = [
            "Configuration Summary:",
            (f"  Crawling: max_depth={crawl.max_depth}, max_pages={crawl.max_pages}"),
            (f"  Dynamic Scan: {'enabled' if dynamic.enabled else 'disabled'}"),
            (
                f"  Performance: concurrent={perf.max_concurrent_scans}, "
                f"pool={perf.connection_pool_size}"
            ),
        ]

        return "\n".join(lines)

    def _save_snapshot(self) -> None:
        """保存當前配置快照"""
        import copy

        snapshot = copy.deepcopy(self._config)
        self._config_history.append(snapshot)

        # 限制歷史記錄數量
        if len(self._config_history) > 10:
            self._config_history.pop(0)

    def _validate_config(self) -> None:
        """驗證配置有效性"""
        crawl = self._config.crawling
        perf = self._config.performance

        # 驗證數值範圍
        if crawl.max_depth < 1:
            logger.warning("max_depth < 1, setting to 1")
            crawl.max_depth = 1

        if crawl.max_pages < 1:
            logger.warning("max_pages < 1, setting to 1")
            crawl.max_pages = 1

        if perf.max_concurrent_scans < 1:
            logger.warning("max_concurrent_scans < 1, setting to 1")
            perf.max_concurrent_scans = 1

        # 驗證邏輯一致性
        if self._config.dynamic.enabled and self._config.dynamic.browser_pool_size < 1:
            logger.warning(
                "Dynamic scan enabled but browser_pool_size < 1, setting to 1"
            )
            self._config.dynamic.browser_pool_size = 1

    def _dataclass_to_dict(self, obj: Any) -> dict[str, Any]:
        """將 dataclass 轉換為字典"""
        if not hasattr(obj, "__dataclass_fields__"):
            return obj

        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if hasattr(value, "__dataclass_fields__"):
                result[field_name] = self._dataclass_to_dict(value)
            else:
                result[field_name] = value

        return result

    # 預設配置模板

    def _get_conservative_preset(self) -> ScanModuleConfig:
        """保守模式：快速、安全、低負載"""
        return ScanModuleConfig(
            crawling=CrawlingConfig(
                max_depth=2,
                max_pages=50,
                requests_per_second=0.5,
                concurrent_requests=2,
            ),
            dynamic=DynamicScanConfig(enabled=False),
            fingerprint=FingerprintConfig(active_fingerprinting=False),
            performance=PerformanceConfig(max_concurrent_scans=1),
        )

    def _get_balanced_preset(self) -> ScanModuleConfig:
        """平衡模式：默認配置"""
        return ScanModuleConfig()

    def _get_aggressive_preset(self) -> ScanModuleConfig:
        """激進模式：深度、全面、高負載"""
        return ScanModuleConfig(
            crawling=CrawlingConfig(
                max_depth=5,
                max_pages=500,
                requests_per_second=5.0,
                concurrent_requests=10,
            ),
            dynamic=DynamicScanConfig(enabled=True, browser_pool_size=3),
            fingerprint=FingerprintConfig(active_fingerprinting=True),
            performance=PerformanceConfig(max_concurrent_scans=5),
        )

    def _get_stealth_preset(self) -> ScanModuleConfig:
        """隱秘模式：低速、避免檢測"""
        return ScanModuleConfig(
            crawling=CrawlingConfig(
                max_depth=3,
                max_pages=100,
                requests_per_second=0.2,
                concurrent_requests=1,
            ),
            dynamic=DynamicScanConfig(enabled=False),
            fingerprint=FingerprintConfig(active_fingerprinting=False),
            security=SecurityConfig(rotate_user_agent=True),
            performance=PerformanceConfig(max_concurrent_scans=1),
        )

    def _get_fast_preset(self) -> ScanModuleConfig:
        """快速模式：高速、淺掃描"""
        return ScanModuleConfig(
            crawling=CrawlingConfig(
                max_depth=1,
                max_pages=50,
                requests_per_second=10.0,
                concurrent_requests=20,
            ),
            dynamic=DynamicScanConfig(enabled=False),
            performance=PerformanceConfig(max_concurrent_scans=10),
        )

    def _get_deep_preset(self) -> ScanModuleConfig:
        """深度模式：深度爬取、全面覆蓋"""
        return ScanModuleConfig(
            crawling=CrawlingConfig(
                max_depth=10,
                max_pages=1000,
                requests_per_second=2.0,
                concurrent_requests=5,
                skip_static_resources=False,
            ),
            dynamic=DynamicScanConfig(
                enabled=True,
                browser_pool_size=2,
                extract_forms=True,
                extract_links=True,
                extract_ajax=True,
                extract_api_calls=True,
            ),
            fingerprint=FingerprintConfig(
                active_fingerprinting=True, detect_cms=True, detect_waf=True
            ),
        )
