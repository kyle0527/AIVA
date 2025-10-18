# -*- coding: utf-8 -*-
"""
AIVA 系統配置管理

提供統一的配置管理，支援環境變數、配置文件和安全密鑰管理。
適用於開發、測試和生產環境的不同需求。
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class AIVAConfig:
    """
    AIVA 統一配置管理器
    
    支援多種配置來源的優先級：
    1. 環境變數 (最高優先級)
    2. 配置文件 (config.json)
    3. 預設值 (最低優先級)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路徑，預設為 config/config.json
        """
        self.config_file = config_file or "config/config.json"
        self.config_data = {}
        self._load_config()
    
    def _load_config(self):
        """載入配置數據"""
        # 預設配置
        self.config_data = self._get_default_config()
        
        # 載入文件配置
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._merge_config(self.config_data, file_config)
            except Exception as e:
                logging.warning(f"無法載入配置文件 {self.config_file}: {e}")
        
        # 環境變數覆蓋
        self._load_env_overrides()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """取得預設配置"""
        return {
            # 高價值功能模組配置
            "features": {
                "allowlist_domains": [],
                "timeout_seconds": 30,
                "max_concurrent_requests": 5,
                "retry_attempts": 3,
                "user_agents": [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                ]
            },
            
            # Bug Bounty 平台配置
            "bug_bounty": {
                "platforms": {
                    "hackerone": {
                        "enabled": False,
                        "api_token": "",
                        "username": ""
                    },
                    "bugcrowd": {
                        "enabled": False,
                        "api_token": "",
                        "email": ""
                    }
                },
                "report_templates": {
                    "auto_submit": False,
                    "include_poc": True,
                    "severity_threshold": "high"
                }
            },
            
            # OOB 檢測配置
            "oob": {
                "http_callback_urls": [],
                "dns_callback_domains": [],
                "timeout_seconds": 10,
                "polling_interval": 5
            },
            
            # 掃描引擎配置
            "scan": {
                "max_depth": 5,
                "max_pages": 1000,
                "concurrent_workers": 10,
                "request_delay": 1.0,
                "follow_redirects": True,
                "verify_ssl": True
            },
            
            # AI 引擎配置
            "ai_engine": {
                "model_name": "bio_neuron_v1",
                "decision_threshold": 0.7,
                "learning_rate": 0.001,
                "training_batch_size": 32,
                "max_training_epochs": 100
            },
            
            # 日誌配置
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/aiva.log",
                "max_file_size": "10MB",
                "backup_count": 5
            },
            
            # 安全配置
            "security": {
                "enable_rate_limiting": True,
                "max_requests_per_minute": 60,
                "enable_auth": False,
                "jwt_secret": "",
                "session_timeout": 3600
            },
            
            # 數據庫配置
            "database": {
                "type": "sqlite",
                "host": "localhost",
                "port": 5432,
                "database": "aiva.db",
                "username": "",
                "password": ""
            },
            
            # API 配置
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "cors_origins": ["*"],
                "docs_url": "/docs",
                "redoc_url": "/redoc"
            }
        }
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """遞歸合併配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_env_overrides(self):
        """載入環境變數覆蓋"""
        env_mappings = {
            # 高價值功能模組
            "ALLOWLIST_DOMAINS": ("features", "allowlist_domains"),
            "FEATURE_TIMEOUT": ("features", "timeout_seconds"),
            
            # Bug Bounty 平台
            "HACKERONE_API_TOKEN": ("bug_bounty", "platforms", "hackerone", "api_token"),
            "HACKERONE_USERNAME": ("bug_bounty", "platforms", "hackerone", "username"),
            "BUGCROWD_API_TOKEN": ("bug_bounty", "platforms", "bugcrowd", "api_token"),
            "BUGCROWD_EMAIL": ("bug_bounty", "platforms", "bugcrowd", "email"),
            
            # OOB 檢測
            "OOB_HTTP_CALLBACKS": ("oob", "http_callback_urls"),
            "OOB_DNS_DOMAINS": ("oob", "dns_callback_domains"),
            
            # API 配置
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port"),
            "API_DEBUG": ("api", "debug"),
            
            # 數據庫
            "DB_TYPE": ("database", "type"),
            "DB_HOST": ("database", "host"),
            "DB_PORT": ("database", "port"),
            "DB_NAME": ("database", "database"),
            "DB_USER": ("database", "username"),
            "DB_PASSWORD": ("database", "password"),
            
            # 安全
            "JWT_SECRET": ("security", "jwt_secret"),
            "ENABLE_AUTH": ("security", "enable_auth"),
            
            # 日誌
            "LOG_LEVEL": ("logging", "level"),
            "LOG_FILE": ("logging", "file")
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(self.config_data, config_path, env_value)
    
    def _set_nested_value(self, data: Dict[str, Any], path: tuple, value: str):
        """設置嵌套配置值"""
        # 處理特殊類型轉換
        if path[-1] in ["port", "timeout_seconds", "max_concurrent_requests", "retry_attempts"]:
            try:
                value = int(value)
            except ValueError:
                return
        elif path[-1] in ["debug", "enabled", "auto_submit", "include_poc", "follow_redirects", "verify_ssl", "enable_rate_limiting", "enable_auth"]:
            value = value.lower() in ["true", "1", "yes", "on"]
        elif path[-1] in ["allowlist_domains", "http_callback_urls", "dns_callback_domains", "cors_origins"]:
            value = [domain.strip() for domain in value.split(",") if domain.strip()]
        
        # 設置值
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        取得配置值
        
        Args:
            section: 配置節名稱
            key: 配置鍵名稱，如果為 None 則返回整個節
            default: 預設值
            
        Returns:
            配置值
        """
        if section not in self.config_data:
            return default
        
        if key is None:
            return self.config_data[section]
        
        return self.config_data[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """
        設置配置值
        
        Args:
            section: 配置節名稱
            key: 配置鍵名稱
            value: 配置值
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section][key] = value
    
    def save(self, config_file: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路徑，預設使用初始化時的路徑
        """
        file_path = config_file or self.config_file
        
        # 確保目錄存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"無法保存配置文件 {file_path}: {e}")
            raise
    
    def get_allowlist_domains(self) -> List[str]:
        """取得允許的域名清單"""
        domains = self.get("features", "allowlist_domains", [])
        if isinstance(domains, str):
            domains = [d.strip() for d in domains.split(",") if d.strip()]
        return domains
    
    def get_oob_config(self) -> Dict[str, Any]:
        """取得 OOB 檢測配置"""
        return self.get("oob", default={})
    
    def get_bug_bounty_config(self, platform: str) -> Dict[str, Any]:
        """取得 Bug Bounty 平台配置"""
        return self.get("bug_bounty", default={}).get("platforms", {}).get(platform, {})
    
    def is_development(self) -> bool:
        """檢查是否為開發模式"""
        return self.get("api", "debug", False)
    
    def get_database_url(self) -> str:
        """取得數據庫連接字符串"""
        db_config = self.get("database")
        db_type = db_config.get("type", "sqlite")
        
        if db_type == "sqlite":
            return f"sqlite:///{db_config.get('database', 'aiva.db')}"
        elif db_type == "postgresql":
            user = db_config.get("username", "")
            password = db_config.get("password", "")
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 5432)
            database = db_config.get("database", "aiva")
            
            if user and password:
                return f"postgresql://{user}:{password}@{host}:{port}/{database}"
            else:
                return f"postgresql://{host}:{port}/{database}"
        else:
            raise ValueError(f"不支援的數據庫類型: {db_type}")


# 全局配置實例
config = AIVAConfig()


def get_config() -> AIVAConfig:
    """取得全局配置實例"""
    return config


def reload_config():
    """重新載入配置"""
    global config
    config = AIVAConfig()
    return config