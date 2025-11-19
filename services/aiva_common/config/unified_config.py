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
    DB_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_POOL_TIMEOUT,
    DB_POOL_RECYCLE,
    RABBITMQ_PORT,
    RABBITMQ_VHOST,
    MQ_EXCHANGE,
    MQ_DLX,
    ENABLE_PROMETHEUS,
    RATE_LIMIT_RPS,
    RATE_LIMIT_BURST,
    SCAN_TIMEOUT,
    SCAN_MAX_RETRIES,
    SCAN_CONCURRENT,
    MODEL_CACHE_SIZE,
    BIO_NEURON_INPUT_SIZE,
    BIO_NEURON_HIDDEN_SIZE,
    get_integration_paths,
)


class DatabaseConfig(BaseModel):
    """資料庫配置"""

    type: str = os.getenv("DB_TYPE", "hybrid")
    url: str = "postgresql://postgres:postgres@localhost:5432/aiva_db"

    # PostgreSQL 配置 - 使用簡潔的命名
    # 研發階段直接使用預設連接字串
    database_url: str = "postgresql://postgres:postgres@localhost:5432/aiva_db"
    postgres_user: str = os.getenv("POSTGRES_USER", "aiva")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "aiva_secure_password")


class MessageQueueConfig(BaseModel):
    """訊息佇列配置 - 遵循 12-factor app 原則"""

    def _get_rabbitmq_url(self):
        """獲取 RabbitMQ URL"""
        # 檢查是否為離線模式
        environment = os.getenv("ENVIRONMENT", "development")
        if environment == "offline":
            return "memory://localhost"

        url = os.getenv("RABBITMQ_URL")
        if url:
            return url

        # 組合式配置
        host = os.getenv("RABBITMQ_HOST", "localhost")
        port = os.getenv("RABBITMQ_PORT", str(RABBITMQ_PORT))
        user = os.getenv("RABBITMQ_USER")
        password = os.getenv("RABBITMQ_PASSWORD")
        vhost = os.getenv("RABBITMQ_VHOST", RABBITMQ_VHOST)

        if not user or not password:
            # 開發環境提供默認值
            if environment == "development":
                return "memory://localhost"
            return f"amqp://guest:guest@{host}:{port}{vhost}"

        return f"amqp://{user}:{password}@{host}:{port}{vhost}"

    rabbitmq_url: str = ""  # 將在 __post_init__ 中設置
    exchange_name: str = os.getenv("MQ_EXCHANGE", MQ_EXCHANGE)
    dlx_name: str = os.getenv("MQ_DLX", MQ_DLX)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.rabbitmq_url:
            self.rabbitmq_url = self._get_rabbitmq_url()


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
    """統一配置設定"""

    database: DatabaseConfig = DatabaseConfig()
    message_queue: MessageQueueConfig = MessageQueueConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    ai: AIConfig = AIConfig()
    scan: ScanConfig = ScanConfig()
    integration: IntegrationConfig = IntegrationConfig()

    # 核心監控配置
    core_monitor_interval: int = int(os.getenv("CORE_MONITOR_INTERVAL", "30"))

    # 功能開關
    enable_strategy_generator: bool = (
        os.getenv("ENABLE_STRATEGY_GEN", "true").lower() == "true"
    )
    enable_prometheus: bool = (
        os.getenv("ENABLE_PROMETHEUS", str(ENABLE_PROMETHEUS)).lower() == "true"
    )

    # 向後相容屬性
    @property
    def rabbitmq_url(self) -> str:
        return self.message_queue.rabbitmq_url

    @property
    def postgres_dsn(self) -> str:
        db = self.database
        return f"postgresql+asyncpg://{db.postgres_user}:{db.postgres_password}@{db.postgres_host}:{db.postgres_port}/{db.postgres_db}"


@lru_cache
def get_settings() -> UnifiedSettings:
    """獲取統一配置（快取）"""
    return UnifiedSettings()


# 向後相容的設定
class Settings(BaseModel):
    """Runtime configuration for AIVA platform."""

    def _get_rabbitmq_url_legacy(self) -> str:
        """獲取 RabbitMQ URL (向後相容版本)"""
        url = os.getenv("RABBITMQ_URL")
        if url:
            return url

        user = os.getenv("RABBITMQ_USER")
        password = os.getenv("RABBITMQ_PASSWORD")
        if user and password:
            host = os.getenv("RABBITMQ_HOST", "localhost")
            port = os.getenv("RABBITMQ_PORT", "5672")
            vhost = os.getenv("RABBITMQ_VHOST", "/")
            return f"amqp://{user}:{password}@{host}:{port}{vhost}"

        raise ValueError("RABBITMQ_URL or RABBITMQ_USER/RABBITMQ_PASSWORD must be set")

    rabbitmq_url: str = ""  # 將在初始化時設置
    exchange_name: str = os.getenv("MQ_EXCHANGE", MQ_EXCHANGE)
    dlx_name: str = os.getenv("MQ_DLX", MQ_DLX)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.rabbitmq_url:
            self.rabbitmq_url = self._get_rabbitmq_url_legacy()

    postgres_dsn: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://aiva:aiva_secure_password@localhost:5432/aiva",
    )

    jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("JWT_ALG", "HS256")

    req_per_sec_default: int = int(os.getenv("RATE_LIMIT_RPS", str(RATE_LIMIT_RPS)))


@lru_cache
def get_legacy_settings() -> Settings:
    """獲取舊版配置（向後相容）"""
    return Settings()
