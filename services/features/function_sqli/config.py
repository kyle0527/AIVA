"""
SQLi 檢測配置管理
統一的配置類別和驗證邏輯
"""



from dataclasses import dataclass, field
from typing import Any


@dataclass
class SqliConfig:
    """SQLi 檢測統一配置類別"""

    # 引擎開關配置
    engines: dict[str, bool] = field(
        default_factory=lambda: {
            "error": True,  # 錯誤檢測引擎
            "boolean": True,  # 布林檢測引擎
            "time": True,  # 時間檢測引擎
            "union": True,  # 聯合檢測引擎
            "oob": False,  # 外帶檢測引擎（預設關閉）
        }
    )

    # 效能參數
    timeout_seconds: float = 30.0
    max_retries: int = 3
    concurrent_limit: int = 5

    # 檢測參數
    time_delay_threshold: float = 3.0  # 時間檢測閾值
    boolean_diff_threshold: float = 0.1  # 布林檢測差異閾值
    max_payload_length: int = 1000  # 最大載荷長度

    # 安全參數
    max_detection_attempts: int = 10  # 最大檢測嘗試次數
    rate_limit_delay: float = 1.0  # 請求間隔延遲

    # 結果配置
    min_confidence_score: float = 0.7  # 最小置信度分數
    include_debug_info: bool = False  # 是否包含調試信息

    def validate(self) -> None:
        """配置驗證"""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.concurrent_limit <= 0:
            raise ValueError("concurrent_limit must be positive")

        if self.time_delay_threshold <= 0:
            raise ValueError("time_delay_threshold must be positive")

        if not (0 <= self.boolean_diff_threshold <= 1):
            raise ValueError("boolean_diff_threshold must be between 0 and 1")

        if not (0 <= self.min_confidence_score <= 1):
            raise ValueError("min_confidence_score must be between 0 and 1")

        if self.max_payload_length <= 0:
            raise ValueError("max_payload_length must be positive")

        # 驗證引擎配置
        valid_engines = {"error", "boolean", "time", "union", "oob"}
        for engine in self.engines:
            if engine not in valid_engines:
                raise ValueError(f"Unknown engine: {engine}")

    @classmethod
    def create_safe_config(cls) -> 'SqliConfig':
        """創建安全的預設配置（用於生產環境）"""
        return cls(
            engines={
                "error": True,
                "boolean": True,
                "time": False,  # 時間檢測可能造成延遲
                "union": True,
                "oob": False,  # OOB檢測需要外部服務
            },
            timeout_seconds=15.0,  # 較短的超時時間
            max_retries=2,  # 較少的重試次數
            concurrent_limit=3,  # 較低的併發限制
            rate_limit_delay=2.0,  # 較長的延遲間隔
            max_detection_attempts=5,  # 較少的檢測嘗試
            include_debug_info=False,
        )

    @classmethod
    def create_aggressive_config(cls) -> 'SqliConfig':
        """創建積極的檢測配置（用於專業測試）"""
        return cls(
            engines={
                "error": True,
                "boolean": True,
                "time": True,
                "union": True,
                "oob": True,  # 啟用所有引擎
            },
            timeout_seconds=60.0,  # 較長的超時時間
            max_retries=5,  # 較多的重試次數
            concurrent_limit=10,  # 較高的併發限制
            time_delay_threshold=5.0,  # 較長的時間閾值
            max_detection_attempts=20,  # 較多的檢測嘗試
            rate_limit_delay=0.5,  # 較短的延遲間隔
            include_debug_info=True,
        )

    def is_engine_enabled(self, engine_name: str) -> bool:
        """檢查指定引擎是否啟用"""
        return self.engines.get(engine_name, False)

    def get_enabled_engines(self) -> list[str]:
        """取得所有啟用的引擎列表"""
        return [engine for engine, enabled in self.engines.items() if enabled]

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "engines": self.engines.copy(),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "concurrent_limit": self.concurrent_limit,
            "time_delay_threshold": self.time_delay_threshold,
            "boolean_diff_threshold": self.boolean_diff_threshold,
            "max_payload_length": self.max_payload_length,
            "max_detection_attempts": self.max_detection_attempts,
            "rate_limit_delay": self.rate_limit_delay,
            "min_confidence_score": self.min_confidence_score,
            "include_debug_info": self.include_debug_info,
        }


# 向後兼容的別名
SqliEngineConfig = SqliConfig
SqliDetectionConfig = SqliConfig  # 舊名稱的兼容性別名
