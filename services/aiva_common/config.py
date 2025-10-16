from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    """Runtime configuration for AIVA platform."""

    rabbitmq_url: str = os.getenv(
        "AIVA_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"
    )
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
    burst_default: int = int(os.getenv("AIVA_RATE_LIMIT_BURST", "50"))

    # Core Engine 配置
    core_monitor_interval: int = int(
        os.getenv("AIVA_CORE_MONITOR_INTERVAL", "30"))
    enable_strategy_generator: bool = (
        os.getenv("AIVA_ENABLE_STRATEGY_GEN", "false").lower() == "true"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
