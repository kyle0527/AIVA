"""
AIVA 統一配置管理
整合所有服務的配置項目
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

# 導入預設值
from .defaults import (
    BIO_NEURON_HIDDEN_SIZE,
    BIO_NEURON_INPUT_SIZE,
    ENABLE_PROMETHEUS,
    MODEL_CACHE_SIZE,
    RATE_LIMIT_BURST,
    RATE_LIMIT_RPS,
    SCAN_CONCURRENT,
    SCAN_MAX_RETRIES,
    SCAN_TIMEOUT,
    get_integration_paths,
)

# ================================
# ✅ 已移除資料庫配置 (v2.0)
# ================================
# v2.0 架構使用本地檔案系統，無需外部資料庫


# ================================
# ✅ 已移除消息隊列配置 (v2.0)
# ================================
# v2.0 架構使用數據合約（AICommand/AICommandResult）
# 通過命令中心直接調用，無需 RabbitMQ


# ================================
# ✅ 已移除服務配置
# ================================
# Redis: 未實際使用，已移除 (2025-11-18)
# Neo4j: 已遷移至 NetworkX，已移除 (2025-11-16)


class SecurityConfig(BaseModel):
    """安全配置"""

    jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("JWT_ALG", "HS256")


class PerformanceConfig(BaseModel):
    """性能配置"""

    req_per_sec_default: int = int(os.getenv("RATE_LIMIT_RPS", str(RATE_LIMIT_RPS)))
    req_per_sec_burst: int = int(os.getenv("RATE_LIMIT_BURST", str(RATE_LIMIT_BURST)))
    data_root: Path = Path(os.getenv("DATA_DIR", "/workspaces/AIVA/data"))


class AIConfig(BaseModel):
    """AI 引擎配置"""

    model_cache_size: int = int(os.getenv("MODEL_CACHE_SIZE", str(MODEL_CACHE_SIZE)))
    bio_neuron_input_size: int = int(
        os.getenv("BIO_INPUT_SIZE", str(BIO_NEURON_INPUT_SIZE))
    )
    bio_neuron_hidden_size: int = int(
        os.getenv("BIO_HIDDEN_SIZE", str(BIO_NEURON_HIDDEN_SIZE))
    )


class ScanConfig(BaseModel):
    """掃描配置"""

    timeout_seconds: float = float(os.getenv("SCAN_TIMEOUT", str(SCAN_TIMEOUT)))
    max_retries: int = int(os.getenv("SCAN_MAX_RETRIES", str(SCAN_MAX_RETRIES)))
    concurrent_limit: int = int(os.getenv("SCAN_CONCURRENT", str(SCAN_CONCURRENT)))


class IntegrationConfig(BaseModel):
    """整合模組配置"""

    data_dir: Path = Path(
        os.getenv(
            "AIVA_INTEGRATION_DATA_DIR", "C:/D/fold7/AIVA-git/data/integration"
        )
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自動推導所有子路徑
        self._paths = get_integration_paths(self.data_dir)

    @property
    def attack_graph_file(self) -> Path:
        return self._paths["attack_graph_file"]

    @property
    def experience_db_url(self) -> str:
        return self._paths["experience_db_url"]

    @property
    def training_dataset_dir(self) -> Path:
        return self._paths["training_dataset_dir"]

    @property
    def model_checkpoint_dir(self) -> Path:
        return self._paths["model_checkpoint_dir"]

    @property
    def raw_data_dir(self) -> Path:
        return self._paths["raw_data_dir"]

    @property
    def processed_data_dir(self) -> Path:
        return self._paths["processed_data_dir"]


class UnifiedSettings(BaseModel):
    """統一配置設定 - v2.0 簡化版"""

    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    ai: AIConfig = AIConfig()
    scan: ScanConfig = ScanConfig()
    integration: IntegrationConfig = IntegrationConfig()

    # 核心監控配置
    core_monitor_interval: int = int(os.getenv("CORE_MONITOR_INTERVAL", "30"))

    # MQ 配置（可選，預設為空表示禁用）
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "")
    exchange_name: str = os.getenv("RABBITMQ_EXCHANGE", "aiva_exchange")

    # 功能開關
    enable_strategy_generator: bool = (
        os.getenv("ENABLE_STRATEGY_GEN", "true").lower() == "true"
    )
    enable_prometheus: bool = (
        os.getenv("ENABLE_PROMETHEUS", str(ENABLE_PROMETHEUS)).lower() == "true"
    )


@lru_cache
def get_settings() -> UnifiedSettings:
    """獲取統一配置（快取）"""
    return UnifiedSettings()


# ================================
# ✅ 已移除舊版配置 (v2.0)
# ================================
# v2.0 架構已完全遷移至新配置系統
# 舊版 Settings 和 get_legacy_settings() 已移除
