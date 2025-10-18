# -*- coding: utf-8 -*-
"""
AIVA API 密鑰和敏感配置管理

管理各種第三方服務的 API 密鑰和敏感信息。
支援環境變數和加密存儲。
"""

import os
import json
import base64
from typing import Dict, Optional
from cryptography.fernet import Fernet
from pathlib import Path
import logging

class APIKeyManager:
    """
    API 密鑰管理器
    
    提供安全的密鑰存儲和檢索功能，支援加密存儲。
    """
    
    def __init__(self, key_file: str = "config/api_keys.json", encryption_key: Optional[str] = None):
        """
        初始化密鑰管理器
        
        Args:
            key_file: 密鑰文件路徑
            encryption_key: 加密金鑰，如果為 None 則從環境變數載入
        """
        self.key_file = key_file
        self.encryption_key = encryption_key or os.getenv("AIVA_ENCRYPTION_KEY")
        self.cipher = None
        
        if self.encryption_key:
            try:
                self.cipher = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
            except Exception as e:
                logging.warning(f"無法初始化加密: {e}")
        
        self.keys = {}
        self._load_keys()
    
    def _load_keys(self):
        """載入密鑰數據"""
        # 從環境變數載入
        self._load_from_env()
        
        # 從文件載入
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'r', encoding='utf-8') as f:
                    file_keys = json.load(f)
                    self.keys.update(file_keys)
            except Exception as e:
                logging.warning(f"無法載入密鑰文件 {self.key_file}: {e}")
    
    def _load_from_env(self):
        """從環境變數載入密鑰"""
        env_keys = {
            # Bug Bounty 平台
            "hackerone_api_token": "HACKERONE_API_TOKEN",
            "hackerone_username": "HACKERONE_USERNAME",
            "bugcrowd_api_token": "BUGCROWD_API_TOKEN",
            "bugcrowd_email": "BUGCROWD_EMAIL",
            
            # OOB 服務
            "burp_collaborator_token": "BURP_COLLABORATOR_TOKEN",
            "interactsh_server": "INTERACTSH_SERVER",
            "webhook_site_token": "WEBHOOK_SITE_TOKEN",
            
            # 雲服務
            "aws_access_key": "AWS_ACCESS_KEY_ID",
            "aws_secret_key": "AWS_SECRET_ACCESS_KEY",
            "azure_client_id": "AZURE_CLIENT_ID",
            "azure_client_secret": "AZURE_CLIENT_SECRET",
            "gcp_service_account": "GCP_SERVICE_ACCOUNT",
            
            # 通知服務
            "slack_webhook": "SLACK_WEBHOOK_URL",
            "discord_webhook": "DISCORD_WEBHOOK_URL",
            "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
            "telegram_chat_id": "TELEGRAM_CHAT_ID",
            
            # 數據庫
            "database_password": "DB_PASSWORD",
            "redis_password": "REDIS_PASSWORD",
            
            # 安全
            "jwt_secret": "JWT_SECRET",
            "api_secret_key": "API_SECRET_KEY",
            "webhook_secret": "WEBHOOK_SECRET",
            
            # 第三方安全工具
            "shodan_api_key": "SHODAN_API_KEY",
            "virustotal_api_key": "VIRUSTOTAL_API_KEY",
            "censys_api_id": "CENSYS_API_ID",
            "censys_api_secret": "CENSYS_API_SECRET"
        }
        
        for key_name, env_var in env_keys.items():
            env_value = os.getenv(env_var)
            if env_value:
                self.keys[key_name] = env_value
    
    def get(self, key_name: str, default: Optional[str] = None, decrypt: bool = True) -> Optional[str]:
        """
        取得 API 密鑰
        
        Args:
            key_name: 密鑰名稱
            default: 預設值
            decrypt: 是否解密（如果密鑰已加密）
            
        Returns:
            API 密鑰值
        """
        value = self.keys.get(key_name, default)
        
        if value and decrypt and self.cipher:
            try:
                # 嘗試解密（如果是加密的）
                if isinstance(value, str) and value.startswith("encrypted:"):
                    encrypted_data = base64.b64decode(value[10:])
                    value = self.cipher.decrypt(encrypted_data).decode()
            except Exception:
                # 如果解密失敗，返回原值
                pass
        
        return value
    
    def set(self, key_name: str, value: str, encrypt: bool = False):
        """
        設置 API 密鑰
        
        Args:
            key_name: 密鑰名稱
            value: 密鑰值
            encrypt: 是否加密存儲
        """
        if encrypt and self.cipher:
            try:
                encrypted_data = self.cipher.encrypt(value.encode())
                value = "encrypted:" + base64.b64encode(encrypted_data).decode()
            except Exception as e:
                logging.warning(f"無法加密密鑰 {key_name}: {e}")
        
        self.keys[key_name] = value
    
    def save(self, key_file: Optional[str] = None):
        """
        保存密鑰到文件
        
        Args:
            key_file: 密鑰文件路徑
        """
        file_path = key_file or self.key_file
        
        # 確保目錄存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            logging.error(f"無法保存密鑰文件 {file_path}: {e}")
            raise
    
    def delete(self, key_name: str):
        """刪除密鑰"""
        if key_name in self.keys:
            del self.keys[key_name]
    
    def list_keys(self) -> list:
        """列出所有密鑰名稱"""
        return list(self.keys.keys())
    
    def has_key(self, key_name: str) -> bool:
        """檢查密鑰是否存在"""
        return key_name in self.keys
    
    def get_bug_bounty_credentials(self, platform: str) -> Dict[str, str]:
        """取得 Bug Bounty 平台認證信息"""
        if platform.lower() == "hackerone":
            return {
                "api_token": self.get("hackerone_api_token", ""),
                "username": self.get("hackerone_username", "")
            }
        elif platform.lower() == "bugcrowd":
            return {
                "api_token": self.get("bugcrowd_api_token", ""),
                "email": self.get("bugcrowd_email", "")
            }
        else:
            return {}
    
    def get_oob_credentials(self) -> Dict[str, str]:
        """取得 OOB 檢測服務認證信息"""
        return {
            "burp_collaborator_token": self.get("burp_collaborator_token", ""),
            "interactsh_server": self.get("interactsh_server", ""),
            "webhook_site_token": self.get("webhook_site_token", "")
        }
    
    def get_notification_credentials(self) -> Dict[str, str]:
        """取得通知服務認證信息"""
        return {
            "slack_webhook": self.get("slack_webhook", ""),
            "discord_webhook": self.get("discord_webhook", ""),
            "telegram_bot_token": self.get("telegram_bot_token", ""),
            "telegram_chat_id": self.get("telegram_chat_id", "")
        }
    
    def generate_encryption_key(self) -> str:
        """生成新的加密金鑰"""
        key = Fernet.generate_key()
        return key.decode()
    
    def validate_keys(self) -> Dict[str, bool]:
        """驗證密鑰的有效性"""
        validation_results = {}
        
        # 檢查必要的密鑰
        required_keys = [
            "jwt_secret",
            "api_secret_key"
        ]
        
        for key_name in required_keys:
            validation_results[key_name] = self.has_key(key_name) and bool(self.get(key_name))
        
        # 檢查 Bug Bounty 平台密鑰
        for platform in ["hackerone", "bugcrowd"]:
            creds = self.get_bug_bounty_credentials(platform)
            validation_results[f"{platform}_credentials"] = all(creds.values())
        
        return validation_results


# 全局密鑰管理器實例
api_keys = APIKeyManager()


def get_api_key(key_name: str, default: Optional[str] = None) -> Optional[str]:
    """快速取得 API 密鑰的便利函數"""
    return api_keys.get(key_name, default)


def set_api_key(key_name: str, value: str, encrypt: bool = False):
    """快速設置 API 密鑰的便利函數"""
    api_keys.set(key_name, value, encrypt)


def has_api_key(key_name: str) -> bool:
    """檢查 API 密鑰是否存在的便利函數"""
    return api_keys.has_key(key_name)


def validate_api_keys() -> Dict[str, bool]:
    """驗證所有 API 密鑰的便利函數"""
    return api_keys.validate_keys()