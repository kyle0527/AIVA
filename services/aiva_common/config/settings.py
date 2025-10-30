"""
現代化配置管理
基於 Pydantic Settings v2 的最佳實踐
"""

from functools import lru_cache
from typing import Optional, List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAIVASettings(BaseSettings):
    """AIVA 基礎配置類別"""
    
    model_config = SettingsConfigDict(
        env_prefix="AIVA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略額外欄位，向前相容
        validate_assignment=True,  # 賦值時驗證
        use_enum_values=True,  # 使用枚舉值
    )


class DatabaseSettings(BaseAIVASettings):
    """資料庫配置"""
    
    url: str = Field(
        default="sqlite:///aiva.db",
        description="資料庫連接 URL"
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="連接池大小"
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="最大溢出連接數"
    )
    echo: bool = Field(
        default=False,
        description="是否輸出 SQL 語句"
    )

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """驗證資料庫 URL 格式"""
        if not v or not isinstance(v, str):
            raise ValueError("Database URL is required")
        
        # 基本格式檢查
        if "://" not in v:
            raise ValueError("Invalid database URL format")
        
        return v


class RedisSettings(BaseAIVASettings):
    """Redis 配置"""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis 連接 URL"
    )
    timeout: float = Field(
        default=5.0,
        gt=0,
        description="連接超時時間（秒）"
    )
    retry_on_timeout: bool = Field(
        default=True,
        description="超時時是否重試"
    )
    max_connections: int = Field(
        default=50,
        gt=0,
        description="最大連接數"
    )


class MessageQueueSettings(BaseAIVASettings):
    """消息隊列配置"""
    
    broker_url: str = Field(
        default="amqp://guest:guest@localhost:5672//",
        description="消息代理 URL"
    )
    result_backend: Optional[str] = Field(
        default=None,
        description="結果後端 URL"
    )
    task_serializer: str = Field(
        default="json",
        description="任務序列化格式"
    )
    result_serializer: str = Field(
        default="json",
        description="結果序列化格式"
    )
    timezone: str = Field(
        default="UTC",
        description="時區設定"
    )
    enable_utc: bool = Field(
        default=True,
        description="啟用 UTC 時間"
    )


class SecuritySettings(BaseAIVASettings):
    """安全配置"""
    
    secret_key: str = Field(
        description="應用程式密鑰"
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT 演算法"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        gt=0,
        description="訪問令牌過期時間（分鐘）"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        gt=0,
        description="刷新令牌過期時間（天）"
    )
    password_min_length: int = Field(
        default=8,
        ge=6,
        description="密碼最小長度"
    )
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="允許的主機列表"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS 允許的來源"
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """驗證密鑰強度"""
        if not v:
            raise ValueError("Secret key is required")
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class LoggingSettings(BaseAIVASettings):
    """日誌配置"""
    
    level: str = Field(
        default="INFO",
        description="日誌級別"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日誌格式"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="日誌文件路徑"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        gt=0,
        description="最大文件大小（字節）"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="備份文件數量"
    )
    json_format: bool = Field(
        default=False,
        description="使用 JSON 格式"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """驗證日誌級別"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v_upper


class AISettings(BaseAIVASettings):
    """AI 模型配置"""
    
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="預設 AI 模型名稱"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="AI 服務 API 密鑰"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="AI 服務 API 基礎 URL"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="溫度參數"
    )
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="最大令牌數"
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="請求超時時間（秒）"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="重試次數"
    )


class PerformanceSettings(BaseAIVASettings):
    """性能調優配置"""
    
    max_workers: int = Field(
        default=4,
        ge=1,
        description="最大工作線程數"
    )
    request_timeout: float = Field(
        default=30.0,
        gt=0,
        description="請求超時時間（秒）"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="每分鐘請求限制"
    )
    cache_ttl: int = Field(
        default=3600,  # 1 hour
        ge=0,
        description="快取存活時間（秒）"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="批處理大小"
    )


class AIVASettings(BaseAIVASettings):
    """AIVA 主配置類別"""
    
    # 應用基本設定
    app_name: str = Field(
        default="AIVA Platform",
        description="應用程式名稱"
    )
    version: str = Field(
        default="1.0.0",
        description="版本號"
    )
    debug: bool = Field(
        default=False,
        description="偵錯模式"
    )
    environment: str = Field(
        default="production",
        description="環境設定"
    )
    
    # 子配置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mq: MessageQueueSettings = Field(default_factory=MessageQueueSettings)
    security: SecuritySettings = SecuritySettings(secret_key="change-me-in-production")
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ai: AISettings = Field(default_factory=AISettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

    @model_validator(mode="after")
    def validate_environment_consistency(self) -> "AIVASettings":
        """驗證環境一致性"""
        if self.environment == "production" and self.debug:
            raise ValueError("Debug mode should not be enabled in production")
        
        if self.environment == "development" and not self.debug:
            # 開發環境建議開啟偵錯模式
            self.debug = True
        
        return self

    def get_database_url(self) -> str:
        """獲取資料庫連接 URL"""
        return self.database.url

    def get_redis_url(self) -> str:
        """獲取 Redis 連接 URL"""
        return self.redis.url

    def is_production(self) -> bool:
        """是否為生產環境"""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """是否為開發環境"""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> AIVASettings:
    """獲取全域配置實例（帶快取）"""
    return AIVASettings()


# 導出常用配置
__all__ = [
    "AIVASettings",
    "BaseAIVASettings", 
    "DatabaseSettings",
    "RedisSettings",
    "MessageQueueSettings",
    "SecuritySettings",
    "LoggingSettings",
    "AISettings",
    "PerformanceSettings",
    "get_settings",
]