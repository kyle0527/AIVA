from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class IntegrationSettings:
    """Integration 模組配置設定"""

    # 資料庫配置
    db_url: str | None
    auto_migrate: bool
    db_pool_size: int
    db_max_overflow: int
    db_pool_timeout: int
    db_pool_recycle: int

    # 認證和安全
    api_token: str | None
    cors_origins: list[str]

    # 速率限制
    rate_limit_rps: float
    rate_limit_burst: int

    # 監控
    enable_prometheus: bool

    # 存儲模式
    storage_mode: str

    @classmethod
    def load(cls) -> IntegrationSettings:
        """從環境變數加載配置"""

        # CORS 來源處理
        cors_env = os.getenv("AIVA_CORS_ORIGINS", "*")
        cors_origins = (
            [s.strip() for s in cors_env.split(",") if s.strip()]
            if cors_env != "*"
            else ["*"]
        )

        return cls(
            # 資料庫配置
            db_url=os.getenv("AIVA_DATABASE_URL"),
            auto_migrate=os.getenv("AUTO_MIGRATE", "0") == "1",
            db_pool_size=int(os.getenv("AIVA_DB_POOL_SIZE", "10")),
            db_max_overflow=int(os.getenv("AIVA_DB_MAX_OVERFLOW", "20")),
            db_pool_timeout=int(os.getenv("AIVA_DB_POOL_TIMEOUT", "30")),
            db_pool_recycle=int(os.getenv("AIVA_DB_POOL_RECYCLE", "1800")),
            # 認證和安全
            api_token=os.getenv("AIVA_INTEGRATION_TOKEN"),
            cors_origins=cors_origins,
            # 速率限制
            rate_limit_rps=float(os.getenv("AIVA_RATE_LIMIT_RPS", "20.0")),
            rate_limit_burst=int(os.getenv("AIVA_RATE_LIMIT_BURST", "60")),
            # 監控
            enable_prometheus=os.getenv("AIVA_ENABLE_PROM", "1") != "0",
            # 存儲模式
            storage_mode=os.getenv("AIVA_STORE", "auto").lower(),
        )

    @property
    def use_sql_storage(self) -> bool:
        """是否使用 SQL 存儲"""
        if self.storage_mode == "memory":
            return False
        elif self.storage_mode == "sql":
            return True
        else:  # auto
            return self.db_url is not None

    def validate(self) -> None:
        """驗證配置有效性"""
        if self.use_sql_storage and not self.db_url:
            raise ValueError("SQL storage mode requires AIVA_DATABASE_URL")

        if self.rate_limit_rps <= 0:
            raise ValueError("Rate limit RPS must be positive")

        if self.rate_limit_burst <= 0:
            raise ValueError("Rate limit burst must be positive")

        if self.db_pool_size <= 0:
            raise ValueError("Database pool size must be positive")
