"""Dashboard 配置管理

此模組負責載入和管理 Dashboard 配置。
"""

from pathlib import Path
from typing import Any

from aiva_common.utils import get_logger
import yaml

logger = get_logger(__name__)


class DashboardConfig:
    """Dashboard 配置管理器"""
    
    _instance = None
    _config: dict[str, Any] = {}
    
    def __new__(cls):
        """單例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """載入配置檔"""
        # 配置檔路徑（從 dashboard 目錄往上找）
        current_dir = Path(__file__).parent
        config_file = current_dir.parent.parent / "config" / "dashboard_config.yaml"
        
        if not config_file.exists():
            logger.warning(f"⚠️  配置檔不存在: {config_file}，使用預設值")
            self._config = self._get_default_config()
            return
        
        try:
            with open(config_file, encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"✅ 配置檔載入成功: {config_file}")
        except Exception as e:
            logger.error(f"❌ 配置檔載入失敗: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> dict[str, Any]:
        """獲取預設配置"""
        return {
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30,
                "retry_attempts": 3
            },
            "dashboard": {
                "title": "AIVA Security Dashboard",
                "port": 8501,
                "host": "0.0.0.0",
                "theme": "dark"
            },
            "scan_defaults": {
                "scan_type": "comprehensive",
                "max_depth": 3,
                "timeout": 1800
            },
            "visualization": {
                "colors": {
                    "primary": "#00D9FF",
                    "accent": "#FF00FF",
                    "success": "#00FF00",
                    "warning": "#FFAA00",
                    "error": "#FF0000"
                }
            },
            "performance": {
                "cache_ttl": 300,
                "status_refresh_interval": 2,
                "auto_refresh": True
            }
        }
    
    # ==================== 屬性訪問方法 ====================
    
    @property
    def api_base_url(self) -> str:
        """API 基礎 URL"""
        return self._config.get("api", {}).get("base_url", "http://localhost:8000")
    
    @property
    def api_timeout(self) -> int:
        """API 超時"""
        return self._config.get("api", {}).get("timeout", 30)
    
    @property
    def api_retry_attempts(self) -> int:
        """API 重試次數"""
        return self._config.get("api", {}).get("retry_attempts", 3)
    
    @property
    def dashboard_title(self) -> str:
        """Dashboard 標題"""
        return self._config.get("dashboard", {}).get("title", "AIVA Security Dashboard")
    
    @property
    def dashboard_theme(self) -> str:
        """Dashboard 主題"""
        return self._config.get("dashboard", {}).get("theme", "dark")
    
    @property
    def scan_type_default(self) -> str:
        """預設掃描類型"""
        return self._config.get("scan_defaults", {}).get("scan_type", "comprehensive")
    
    @property
    def scan_max_depth_default(self) -> int:
        """預設掃描深度"""
        return self._config.get("scan_defaults", {}).get("max_depth", 3)
    
    @property
    def scan_timeout_default(self) -> int:
        """預設掃描超時"""
        return self._config.get("scan_defaults", {}).get("timeout", 1800)
    
    @property
    def colors(self) -> dict[str, str]:
        """主題配色"""
        return self._config.get("visualization", {}).get("colors", {})
    
    @property
    def cache_ttl(self) -> int:
        """快取過期時間"""
        return self._config.get("performance", {}).get("cache_ttl", 300)
    
    @property
    def status_refresh_interval(self) -> int:
        """狀態刷新間隔"""
        return self._config.get("performance", {}).get("status_refresh_interval", 2)
    
    @property
    def auto_refresh(self) -> bool:
        """是否自動刷新"""
        return self._config.get("performance", {}).get("auto_refresh", True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """獲取配置值
        
        Args:
            key: 配置鍵（支援點號分隔，如 'api.base_url'）
            default: 預設值
        
        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default


# 全局配置實例
config = DashboardConfig()
