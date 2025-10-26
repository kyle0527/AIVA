from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScanStrategyType(Enum):  # 重命名避免與 aiva_common.enums.ScanStrategy 衝突
    """掃描策略類型 - Scan模組專用的詳細策略定義"""

    # 保守策略：快速、淺層、低負載
    CONSERVATIVE = "conservative"

    # 平衡策略：中等深度和速度
    BALANCED = "balanced"

    # 深度策略：深入爬取、全面覆蓋
    DEEP = "deep"

    # 快速策略：快速淺掃、僅基本檢查
    FAST = "fast"

    # 激進策略：完整掃描、高負載
    AGGRESSIVE = "aggressive"

    # 隱秘策略：慢速、低調、避免檢測
    STEALTH = "stealth"

    # 目標化策略：專注於特定目標
    TARGETED = "targeted"


@dataclass
class StrategyParameters:
    """策略參數配置"""

    # 爬蟲深度和廣度
    max_depth: int
    max_pages: int
    max_forms: int

    # 速率控制
    requests_per_second: float
    concurrent_requests: int

    # 超時設置
    request_timeout: float
    page_load_timeout: float

    # 動態掃描
    enable_dynamic_scan: bool
    browser_pool_size: int

    # 指紋識別
    enable_active_fingerprinting: bool

    # 性能設置
    max_concurrent_scans: int
    connection_pool_size: int

    # 其他標誌
    skip_static_resources: bool
    follow_redirects: bool


class StrategyController:
    """
    掃描策略控制器。

    根據選擇的策略調整掃描行為參數，包括：
    - 爬蟲深度和廣度
    - 請求速率和並發數
    - 超時設置
    - 動態掃描啟用
    - 指紋識別級別
    - 性能優化設置

    使用範例:
        # 使用預定義策略
        controller = StrategyController("deep")
        params = controller.get_parameters()

        # 使用自定義策略
        controller = StrategyController("balanced")
        controller.customize(max_depth=5, requests_per_second=1.0)
        params = controller.get_parameters()

        # 應用策略到配置
        config_center = ConfigControlCenter()
        controller.apply_to_config(config_center)
    """

    # 預定義的策略參數
    _STRATEGY_PRESETS: dict[str, StrategyParameters] = {
        ScanStrategyType.CONSERVATIVE.value: StrategyParameters(
            max_depth=2,
            max_pages=50,
            max_forms=20,
            requests_per_second=0.5,
            concurrent_requests=2,
            request_timeout=5.0,
            page_load_timeout=15.0,
            enable_dynamic_scan=False,
            browser_pool_size=1,
            enable_active_fingerprinting=False,
            max_concurrent_scans=1,
            connection_pool_size=5,
            skip_static_resources=True,
            follow_redirects=True,
        ),
        ScanStrategyType.BALANCED.value: StrategyParameters(
            max_depth=3,
            max_pages=100,
            max_forms=50,
            requests_per_second=2.0,
            concurrent_requests=5,
            request_timeout=10.0,
            page_load_timeout=30.0,
            enable_dynamic_scan=False,
            browser_pool_size=2,
            enable_active_fingerprinting=False,
            max_concurrent_scans=3,
            connection_pool_size=10,
            skip_static_resources=True,
            follow_redirects=True,
        ),
        ScanStrategyType.DEEP.value: StrategyParameters(
            max_depth=10,
            max_pages=1000,
            max_forms=200,
            requests_per_second=2.0,
            concurrent_requests=5,
            request_timeout=15.0,
            page_load_timeout=45.0,
            enable_dynamic_scan=True,
            browser_pool_size=2,
            enable_active_fingerprinting=True,
            max_concurrent_scans=3,
            connection_pool_size=10,
            skip_static_resources=False,
            follow_redirects=True,
        ),
        ScanStrategyType.FAST.value: StrategyParameters(
            max_depth=1,
            max_pages=50,
            max_forms=20,
            requests_per_second=10.0,
            concurrent_requests=20,
            request_timeout=5.0,
            page_load_timeout=10.0,
            enable_dynamic_scan=False,
            browser_pool_size=1,
            enable_active_fingerprinting=False,
            max_concurrent_scans=10,
            connection_pool_size=20,
            skip_static_resources=True,
            follow_redirects=False,
        ),
        ScanStrategyType.AGGRESSIVE.value: StrategyParameters(
            max_depth=5,
            max_pages=500,
            max_forms=100,
            requests_per_second=5.0,
            concurrent_requests=10,
            request_timeout=10.0,
            page_load_timeout=30.0,
            enable_dynamic_scan=True,
            browser_pool_size=3,
            enable_active_fingerprinting=True,
            max_concurrent_scans=5,
            connection_pool_size=15,
            skip_static_resources=True,
            follow_redirects=True,
        ),
        ScanStrategyType.STEALTH.value: StrategyParameters(
            max_depth=3,
            max_pages=100,
            max_forms=30,
            requests_per_second=0.2,
            concurrent_requests=1,
            request_timeout=15.0,
            page_load_timeout=30.0,
            enable_dynamic_scan=False,
            browser_pool_size=1,
            enable_active_fingerprinting=False,
            max_concurrent_scans=1,
            connection_pool_size=3,
            skip_static_resources=True,
            follow_redirects=True,
        ),
        ScanStrategyType.TARGETED.value: StrategyParameters(
            max_depth=5,
            max_pages=200,
            max_forms=50,
            requests_per_second=1.0,
            concurrent_requests=3,
            request_timeout=10.0,
            page_load_timeout=30.0,
            enable_dynamic_scan=True,
            browser_pool_size=2,
            enable_active_fingerprinting=True,
            max_concurrent_scans=2,
            connection_pool_size=8,
            skip_static_resources=False,
            follow_redirects=True,
        ),
    }

    def __init__(self, strategy: str) -> None:
        """
        初始化策略控制器

        Args:
            strategy: 策略名稱
                (conservative, balanced, deep, fast,
                aggressive, stealth, targeted)
        """
        self.strategy = strategy.lower()
        self._parameters = self._load_strategy_parameters()
        self._customizations: dict[str, Any] = {}

        logger.info(f"StrategyController initialized with strategy: {self.strategy}")

    def get_parameters(self) -> StrategyParameters:
        """
        獲取當前策略參數

        Returns:
            StrategyParameters: 策略參數
        """
        return self._parameters

    def get_strategy_name(self) -> str:
        """獲取策略名稱"""
        return self.strategy

    def get_max_depth(self) -> int:
        """獲取最大爬蟲深度"""
        return self._parameters.max_depth

    def get_max_pages(self) -> int:
        """獲取最大頁面數"""
        return self._parameters.max_pages

    def get_requests_per_second(self) -> float:
        """獲取每秒請求數"""
        return self._parameters.requests_per_second

    def get_concurrent_requests(self) -> int:
        """獲取並發請求數"""
        return self._parameters.concurrent_requests

    def is_dynamic_scan_enabled(self) -> bool:
        """是否啟用動態掃描"""
        return self._parameters.enable_dynamic_scan

    def is_aggressive(self) -> bool:
        """是否為激進策略"""
        return self.strategy in [
            ScanStrategyType.AGGRESSIVE.value,
            ScanStrategyType.DEEP.value,
        ]

    def is_stealth(self) -> bool:
        """是否為隱秘策略"""
        return self.strategy == ScanStrategyType.STEALTH.value

    def customize(self, **kwargs) -> None:
        """
        自定義策略參數

        Args:
            **kwargs: 要覆蓋的參數
        """
        for key, value in kwargs.items():
            if hasattr(self._parameters, key):
                setattr(self._parameters, key, value)
                self._customizations[key] = value
                logger.debug(f"Customized parameter: {key}={value}")
            else:
                logger.warning(f"Unknown parameter: {key}")

    def apply_to_config(self, config_center) -> None:
        """
        將策略應用到配置控制中心

        Args:
            config_center: ConfigControlCenter 實例
        """
        params = self._parameters

        # 更新爬蟲配置
        config_center.update_crawling_config(
            max_depth=params.max_depth,
            max_pages=params.max_pages,
            max_forms=params.max_forms,
            requests_per_second=params.requests_per_second,
            concurrent_requests=params.concurrent_requests,
            request_timeout=params.request_timeout,
            page_load_timeout=params.page_load_timeout,
            skip_static_resources=params.skip_static_resources,
            follow_redirects=params.follow_redirects,
        )

        # 更新動態掃描配置
        config_center.update_dynamic_config(
            enabled=params.enable_dynamic_scan,
            browser_pool_size=params.browser_pool_size,
        )

        # 更新性能配置
        config_center.update_performance_config(
            max_concurrent_scans=params.max_concurrent_scans,
            connection_pool_size=params.connection_pool_size,
        )

        logger.info(f"Applied strategy '{self.strategy}' to ConfigControlCenter")

    def get_strategy_summary(self) -> str:
        """
        獲取策略摘要

        Returns:
            str: 策略描述
        """
        params = self._parameters
        lines = [
            f"Strategy: {self.strategy.upper()}",
            f"  Max Depth: {params.max_depth}",
            f"  Max Pages: {params.max_pages}",
            f"  Request Rate: {params.requests_per_second} req/s",
            f"  Concurrent Requests: {params.concurrent_requests}",
            (
                f"  Dynamic Scan: "
                f"{'Enabled' if params.enable_dynamic_scan else 'Disabled'}"
            ),
            (
                f"  Active Fingerprinting: "
                f"{'Enabled' if params.enable_active_fingerprinting else 'Disabled'}"
            ),
        ]

        if self._customizations:
            lines.append("\n  Customizations:")
            for key, value in self._customizations.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def compare_with(self, other_strategy: str) -> dict[str, tuple[Any, Any]]:
        """
        比較與另一個策略的差異

        Args:
            other_strategy: 另一個策略名稱

        Returns:
            dict: 差異字典 {參數名: (當前值, 其他值)}
        """
        if other_strategy not in self._STRATEGY_PRESETS:
            logger.warning(f"Unknown strategy for comparison: {other_strategy}")
            return {}

        other_params = self._STRATEGY_PRESETS[other_strategy]
        current_params = self._parameters

        differences = {}
        for field in current_params.__dataclass_fields__:
            current_value = getattr(current_params, field)
            other_value = getattr(other_params, field)

            if current_value != other_value:
                differences[field] = (current_value, other_value)

        return differences

    def estimate_scan_time(self, estimated_pages: int) -> float:
        """
        估算掃描時間（秒）

        Args:
            estimated_pages: 預估頁面數

        Returns:
            float: 預估時間（秒）
        """
        params = self._parameters

        # 限制頁面數
        pages = min(estimated_pages, params.max_pages)

        # 基於速率計算時間
        time_by_rate = pages / params.requests_per_second

        # 基於超時計算時間（考慮並發）
        time_by_timeout = (pages * params.request_timeout) / params.concurrent_requests

        # 取較大值，並加上緩衝
        estimated_time = max(time_by_rate, time_by_timeout) * 1.2

        # 如果啟用動態掃描，增加額外時間
        if params.enable_dynamic_scan:
            estimated_time *= 1.5

        return estimated_time

    def get_recommended_strategy_for_target(
        self, target_size: str = "medium", target_type: str = "web_app"
    ) -> str:
        """
        根據目標特徵推薦策略

        Args:
            target_size: 目標大小 (small, medium, large)
            target_type: 目標類型 (web_app, api, static_site, spa)

        Returns:
            str: 推薦的策略名稱
        """
        recommendations = {
            ("small", "web_app"): ScanStrategyType.BALANCED.value,
            ("small", "api"): ScanStrategyType.FAST.value,
            ("small", "static_site"): ScanStrategyType.CONSERVATIVE.value,
            ("small", "spa"): ScanStrategyType.TARGETED.value,
            ("medium", "web_app"): ScanStrategyType.DEEP.value,
            ("medium", "api"): ScanStrategyType.BALANCED.value,
            ("medium", "static_site"): ScanStrategyType.BALANCED.value,
            ("medium", "spa"): ScanStrategyType.AGGRESSIVE.value,
            ("large", "web_app"): ScanStrategyType.AGGRESSIVE.value,
            ("large", "api"): ScanStrategyType.AGGRESSIVE.value,
            ("large", "static_site"): ScanStrategyType.DEEP.value,
            ("large", "spa"): ScanStrategyType.AGGRESSIVE.value,
        }

        key = (target_size.lower(), target_type.lower())
        recommended = recommendations.get(key, ScanStrategyType.BALANCED.value)

        logger.info(
            f"Recommended strategy for {target_type} ({target_size}): {recommended}"
        )
        return recommended

    def _load_strategy_parameters(self) -> StrategyParameters:
        """
        加載策略參數

        Returns:
            StrategyParameters: 策略參數
        """
        if self.strategy in self._STRATEGY_PRESETS:
            return self._STRATEGY_PRESETS[self.strategy]

        # 未知策略，使用默認（balanced）
        logger.warning(
            f"Unknown strategy '{self.strategy}', using 'balanced' as default"
        )
        return self._STRATEGY_PRESETS[ScanStrategyType.BALANCED.value]

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """獲取所有可用的策略名稱"""
        return list(cls._STRATEGY_PRESETS.keys())

    @classmethod
    def get_strategy_description(cls, strategy: str) -> str:
        """
        獲取策略描述

        Args:
            strategy: 策略名稱

        Returns:
            str: 策略描述
        """
        descriptions = {
            ScanStrategyType.CONSERVATIVE.value: "快速、淺層、低負載 - 適合初步探索",
            ScanStrategyType.BALANCED.value: "中等深度和速度 - 適合大多數情況",
            ScanStrategyType.DEEP.value: "深入爬取、全面覆蓋 - 適合徹底掃描",
            ScanStrategyType.FAST.value: "快速淺掃、僅基本檢查 - 適合快速評估",
            ScanStrategyType.AGGRESSIVE.value: "完整掃描、高負載 - 適合專業滲透測試",
            ScanStrategyType.STEALTH.value: "慢速、低調、避免檢測 - 適合規避 WAF",
            ScanStrategyType.TARGETED.value: "專注於特定目標 - 適合已知漏洞驗證",
        }
        return descriptions.get(strategy, "未知策略")

    def clone(self) -> StrategyController:
        """
        克隆當前策略控制器

        Returns:
            StrategyController: 新的策略控制器實例
        """
        new_controller = StrategyController(self.strategy)
        new_controller._parameters = StrategyParameters(**self._parameters.__dict__)
        new_controller._customizations = self._customizations.copy()
        return new_controller

    def reset_customizations(self) -> None:
        """重置所有自定義參數"""
        self._parameters = self._load_strategy_parameters()
        self._customizations.clear()
        logger.info(f"Reset customizations for strategy '{self.strategy}'")
