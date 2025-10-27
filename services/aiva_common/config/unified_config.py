"""
AIVA 統一配置管理
整合所有服務的配置項目
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """資料庫配置"""
    type: str = os.getenv("AIVA_DB_TYPE", "hybrid")
    url: str = os.getenv("AIVA_DB_URL", "sqlite:///data/aiva.db")
    
    # PostgreSQL 配置
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "aiva")
    postgres_user: str = os.getenv("POSTGRES_USER", "aiva")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "aiva")


class MessageQueueConfig(BaseModel):
    """訊息佇列配置 - 遵循 12-factor app 原則"""
    def _get_rabbitmq_url(self) -> str:
        """獲取 RabbitMQ URL，優先完整 URL，其次組合配置"""
        url = os.getenv("AIVA_RABBITMQ_URL")
        if url:
            return url
            
        # 組合式配置
        host = os.getenv("AIVA_RABBITMQ_HOST", "localhost")
        port = os.getenv("AIVA_RABBITMQ_PORT", "5672")
        user = os.getenv("AIVA_RABBITMQ_USER")
        password = os.getenv("AIVA_RABBITMQ_PASSWORD")
        vhost = os.getenv("AIVA_RABBITMQ_VHOST", "/")
        
        if not user or not password:
            raise ValueError("AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set")
            
        return f"amqp://{user}:{password}@{host}:{port}{vhost}"
    
    rabbitmq_url: str = ""  # 將在 __post_init__ 中設置
    exchange_name: str = os.getenv("AIVA_MQ_EXCHANGE", "aiva.topic")
    dlx_name: str = os.getenv("AIVA_MQ_DLX", "aiva.dlx")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.rabbitmq_url:
            self.rabbitmq_url = self._get_rabbitmq_url()


class CacheConfig(BaseModel):
    """快取配置"""
    redis_url: str = os.getenv("AIVA_REDIS_URL", "redis://localhost:6379/0")
    
    
class GraphDatabaseConfig(BaseModel):
    """圖資料庫配置"""
    neo4j_url: str = os.getenv("AIVA_NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("AIVA_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("AIVA_NEO4J_PASSWORD", "password")


class SecurityConfig(BaseModel):
    """安全配置"""
    jwt_secret: str = os.getenv("AIVA_JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("AIVA_JWT_ALG", "HS256")


class PerformanceConfig(BaseModel):
    """性能配置"""
    req_per_sec_default: int = int(os.getenv("AIVA_RATE_LIMIT_RPS", "25"))
    data_root: Path = Path(os.getenv("AIVA_DATA_DIR", "/workspaces/AIVA/data"))


class AIConfig(BaseModel):
    """AI 引擎配置"""
    model_cache_size: int = int(os.getenv("AIVA_MODEL_CACHE_SIZE", "1000"))
    bio_neuron_input_size: int = int(os.getenv("AIVA_BIO_INPUT_SIZE", "1024"))
    bio_neuron_hidden_size: int = int(os.getenv("AIVA_BIO_HIDDEN_SIZE", "2048"))


class ScanConfig(BaseModel):
    """掃描配置"""
    timeout_seconds: float = float(os.getenv("AIVA_SCAN_TIMEOUT", "30.0"))
    max_retries: int = int(os.getenv("AIVA_SCAN_MAX_RETRIES", "3"))
    concurrent_limit: int = int(os.getenv("AIVA_SCAN_CONCURRENT", "5"))


class UnifiedSettings(BaseModel):
    """統一配置設定"""
    database: DatabaseConfig = DatabaseConfig()
    message_queue: MessageQueueConfig = MessageQueueConfig()
    cache: CacheConfig = CacheConfig()
    graph_db: GraphDatabaseConfig = GraphDatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    ai: AIConfig = AIConfig()
    scan: ScanConfig = ScanConfig()
    
    # 核心監控配置
    core_monitor_interval: int = int(os.getenv("AIVA_CORE_MONITOR_INTERVAL", "30"))
    
    # 功能開關
    enable_strategy_generator: bool = os.getenv("AIVA_ENABLE_STRATEGY_GEN", "true").lower() == "true"
    
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
        url = os.getenv("AIVA_RABBITMQ_URL")
        if url:
            return url
            
        user = os.getenv("AIVA_RABBITMQ_USER")
        password = os.getenv("AIVA_RABBITMQ_PASSWORD")
        if user and password:
            host = os.getenv("AIVA_RABBITMQ_HOST", "localhost")
            port = os.getenv("AIVA_RABBITMQ_PORT", "5672")
            vhost = os.getenv("AIVA_RABBITMQ_VHOST", "/")
            return f"amqp://{user}:{password}@{host}:{port}{vhost}"
            
        raise ValueError("AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set")
    
    rabbitmq_url: str = ""  # 將在初始化時設置
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.rabbitmq_url:
            self.rabbitmq_url = self._get_rabbitmq_url_legacy()
    exchange_name: str = os.getenv("AIVA_MQ_EXCHANGE", "aiva.topic")
    dlx_name: str = os.getenv("AIVA_MQ_DLX", "aiva.dlx")

    postgres_dsn: str = os.getenv(
        "AIVA_POSTGRES_DSN",
        "postgresql+asyncpg://aiva:aiva@localhost:5432/aiva",
    )
    redis_url: str = os.getenv("AIVA_REDIS_URL", "redis://localhost:6379/0")
    neo4j_url: str = os.getenv("AIVA_NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("AIVA_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("AIVA_NEO4J_PASSWORD", "password")

    jwt_secret: str = os.getenv("AIVA_JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("AIVA_JWT_ALG", "HS256")

    req_per_sec_default: int = int(os.getenv("AIVA_RATE_LIMIT_RPS", "25"))


@lru_cache
def get_legacy_settings() -> Settings:
    """獲取舊版配置（向後相容）"""
    return Settings()