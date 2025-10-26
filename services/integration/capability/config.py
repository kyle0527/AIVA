"""
AIVA 能力註冊中心配置檔案
定義系統的核心配置參數和預設值

此配置遵循 aiva_common 的配置管理標準
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from aiva_common.enums import ProgrammingLanguage

class DatabaseConfig(BaseModel):
    """資料庫配置"""
    path: str = Field(default="capability_registry.db", description="SQLite 資料庫路徑")
    backup_enabled: bool = Field(default=True, description="是否啟用備份")
    backup_interval_hours: int = Field(default=24, description="備份間隔(小時)")
    max_backups: int = Field(default=7, description="最大備份數量")


class DiscoveryConfig(BaseModel):
    """能力發現配置"""
    auto_discovery_enabled: bool = Field(default=True, description="是否啟用自動發現")
    discovery_interval_minutes: int = Field(default=60, description="發現間隔(分鐘)")
    scan_directories: List[str] = Field(
        default_factory=lambda: [
            "services/features",
            "services/scan", 
            "services/analysis",
            "tools"
        ],
        description="掃描目錄列表"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "*.pyc",
            ".git",
            "node_modules",
            "target/debug"
        ],
        description="排除模式"
    )


class MonitoringConfig(BaseModel):
    """監控配置"""
    health_check_enabled: bool = Field(default=True, description="是否啟用健康檢查")
    health_check_interval_minutes: int = Field(default=15, description="健康檢查間隔(分鐘)")
    performance_monitoring_enabled: bool = Field(default=True, description="是否啟用性能監控")
    alert_thresholds: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_latency_ms": 5000,
            "min_success_rate": 95.0,
            "max_error_rate": 5.0
        },
        description="警報閾值"
    )


class APIConfig(BaseModel):
    """API 配置"""
    host: str = Field(default="0.0.0.0", description="綁定主機")
    port: int = Field(default=8000, description="綁定端口")
    debug: bool = Field(default=False, description="調試模式")
    docs_enabled: bool = Field(default=True, description="是否啟用 API 文件")
    cors_enabled: bool = Field(default=True, description="是否啟用 CORS")
    rate_limit_enabled: bool = Field(default=True, description="是否啟用速率限制")
    max_requests_per_minute: int = Field(default=100, description="每分鐘最大請求數")


class LoggingConfig(BaseModel):
    """日誌配置"""
    level: str = Field(default="INFO", description="日誌級別")
    format: str = Field(default="structured", description="日誌格式")
    output_file: Optional[str] = Field(None, description="日誌輸出檔案")
    rotate_enabled: bool = Field(default=True, description="是否啟用日誌輪轉")
    max_file_size_mb: int = Field(default=100, description="最大檔案大小(MB)")
    max_files: int = Field(default=10, description="最大檔案數量")


class SecurityConfig(BaseModel):
    """安全配置"""
    authentication_enabled: bool = Field(default=False, description="是否啟用認證")
    api_key_required: bool = Field(default=False, description="是否需要 API 金鑰")
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="允許的來源"
    )
    ssl_enabled: bool = Field(default=False, description="是否啟用 SSL")
    ssl_cert_path: Optional[str] = Field(None, description="SSL 證書路徑")
    ssl_key_path: Optional[str] = Field(None, description="SSL 私鑰路徑")


class CapabilityRegistryConfig(BaseModel):
    """能力註冊中心主配置"""
    
    # 基本資訊
    name: str = Field(default="AIVA Capability Registry", description="系統名稱")
    version: str = Field(default="1.0.0", description="系統版本")
    environment: str = Field(default="development", description="運行環境")
    
    # 各子模組配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # 整合配置
    aiva_common_path: str = Field(
        default="services/aiva_common",
        description="aiva_common 模組路徑"
    )
    tools_enabled: List[str] = Field(
        default_factory=lambda: [
            "schema_validator",
            "schema_codegen_tool", 
            "module_connectivity_tester"
        ],
        description="啟用的工具列表"
    )
    
    # 語言支援配置
    supported_languages: List[ProgrammingLanguage] = Field(
        default_factory=lambda: [
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.GO,
            ProgrammingLanguage.RUST,
            ProgrammingLanguage.TYPESCRIPT
        ],
        description="支援的程式語言"
    )
    
    @classmethod
    def load_from_file(cls, config_path: str) -> "CapabilityRegistryConfig":
        """從檔案載入配置"""
        import json
        import yaml
        
        path = Path(config_path)
        
        if not path.exists():
            # 如果檔案不存在，使用預設配置並創建檔案
            config = cls()
            config.save_to_file(config_path)
            return config
        
        # 根據檔案擴展名選擇解析器
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"不支援的配置檔案格式: {path.suffix}")
        
        return cls.model_validate(data)
    
    def save_to_file(self, config_path: str) -> None:
        """將配置儲存到檔案"""
        import json
        import yaml
        
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump()
        
        # 根據檔案擴展名選擇格式
        if path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支援的配置檔案格式: {path.suffix}")
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """獲取環境變數覆寫"""
        overrides = {}
        
        # 常見的環境變數對映
        env_mappings = {
            "AIVA_DB_PATH": "database.path",
            "AIVA_API_HOST": "api.host", 
            "AIVA_API_PORT": "api.port",
            "AIVA_DEBUG": "api.debug",
            "AIVA_LOG_LEVEL": "logging.level",
            "AIVA_DISCOVERY_ENABLED": "discovery.auto_discovery_enabled",
            "AIVA_MONITORING_ENABLED": "monitoring.health_check_enabled"
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 處理布林值
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                # 處理整數
                elif value.isdigit():
                    value = int(value)
                
                overrides[config_path] = value
        
        return overrides
    
    def apply_environment_overrides(self) -> None:
        """應用環境變數覆寫"""
        overrides = self.get_environment_overrides()
        
        for config_path, value in overrides.items():
            # 使用點記法設定配置值
            parts = config_path.split('.')
            obj = self
            
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            setattr(obj, parts[-1], value)


# 預設配置實例
default_config = CapabilityRegistryConfig()

# 配置載入函數
def load_config(config_path: Optional[str] = None) -> CapabilityRegistryConfig:
    """載入配置"""
    
    if config_path is None:
        # 查看常見的配置檔案位置
        possible_paths = [
            "capability_registry.yaml",
            "capability_registry.yml", 
            "capability_registry.json",
            "config/capability_registry.yaml",
            "config/capability_registry.yml",
            "config/capability_registry.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                config_path = path
                break
        else:
            # 如果沒有找到配置檔案，使用預設配置
            return default_config
    
    # 載入配置檔案
    config = CapabilityRegistryConfig.load_from_file(config_path)
    
    # 應用環境變數覆寫
    config.apply_environment_overrides()
    
    return config


# 配置驗證函數
def validate_config(config: CapabilityRegistryConfig) -> List[str]:
    """驗證配置有效性"""
    
    errors = []
    
    # 檢查必要路徑
    if not Path(config.aiva_common_path).exists():
        errors.append(f"aiva_common 路徑不存在: {config.aiva_common_path}")
    
    # 檢查發現目錄
    for directory in config.discovery.scan_directories:
        if not Path(directory).exists():
            errors.append(f"掃描目錄不存在: {directory}")
    
    # 檢查資料庫路徑
    db_parent = Path(config.database.path).parent
    if not db_parent.exists():
        try:
            db_parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"無法創建資料庫目錄: {str(e)}")
    
    # 檢查端口範圍
    if not (1 <= config.api.port <= 65535):
        errors.append(f"API 端口超出有效範圍: {config.api.port}")
    
    # 檢查 SSL 配置
    if config.security.ssl_enabled:
        if not config.security.ssl_cert_path or not Path(config.security.ssl_cert_path).exists():
            errors.append("SSL 已啟用但證書檔案不存在")
        if not config.security.ssl_key_path or not Path(config.security.ssl_key_path).exists():
            errors.append("SSL 已啟用但私鑰檔案不存在")
    
    return errors