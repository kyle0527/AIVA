"""
AIVA Security Configuration Example
AIVA 安全配置示例

提供完整的安全配置示例，包括認證、授權、加密等設置。
"""

import os
from pathlib import Path
from typing import Any

# 安全配置示例
SECURITY_CONFIG_EXAMPLE = {
    # 基本安全設置
    "security": {
        "enabled": True,
        "debug_mode": False,
        "log_security_events": True,
        # 加密設置
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_size": 256,
            "use_hardware_security": True,
            "rotate_keys_interval": 86400,  # 24小時
            # RSA 設置
            "rsa": {"key_size": 2048, "use_oaep_padding": True},
        },
        # JWT 令牌設置
        "jwt": {
            "algorithm": "HS256",
            "expiration": 3600,  # 1小時
            "refresh_token_expiration": 604800,  # 7天
            "issuer": "aiva-system",
            "audience": "aiva-services",
            "allow_refresh": True,
            "revocation_check": True,
        },
        # API 密鑰設置
        "api_keys": {
            "length": 32,
            "prefix": "aiva_",
            "expiration": 2592000,  # 30天
            "rate_limit_per_key": 1000,  # 每小時1000次
            "allow_regeneration": True,
        },
        # 認證設置
        "authentication": {
            "methods": ["jwt", "api_key", "certificate"],
            "require_mfa": False,
            "session_timeout": 3600,
            "max_failed_attempts": 5,
            "lockout_duration": 900,  # 15分鐘
            "password_policy": {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True,
            },
        },
        # 授權設置
        "authorization": {
            "model": "rbac",  # role-based access control
            "cache_duration": 300,  # 5分鐘
            "inherit_permissions": True,
            "allow_dynamic_roles": True,
            # 默認角色和權限
            "default_roles": {
                "admin": {"permissions": ["*"], "description": "系統管理員"},
                "user": {
                    "permissions": [
                        "core.command.execute",
                        "core.status.read",
                        "monitoring.metrics.read",
                    ],
                    "description": "普通用戶",
                },
                "guest": {
                    "permissions": ["core.status.read"],
                    "description": "訪客用戶",
                },
            },
        },
        # 審計設置
        "audit": {
            "enabled": True,
            "log_all_requests": True,
            "log_authentication": True,
            "log_authorization": True,
            "log_failed_attempts": True,
            "retention_days": 90,
            "export_format": "json",
        },
    },
    # 網絡安全設置
    "network": {
        # CORS 設置
        "cors": {
            "enabled": True,
            "allowed_origins": [
                "http://localhost:3000",
                "https://aiva.app",
                "https://*.aiva.app",
            ],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allowed_headers": [
                "Content-Type",
                "Authorization",
                "X-API-Key",
                "X-Trace-ID",
                "X-Request-ID",
            ],
            "allow_credentials": True,
            "max_age": 86400,
        },
        # 速率限制設置
        "rate_limiting": {
            "enabled": True,
            "global_limit": {
                "requests_per_minute": 1000,
                "requests_per_hour": 10000,
                "requests_per_day": 100000,
            },
            "per_ip_limit": {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
            },
            "per_user_limit": {
                "requests_per_minute": 200,
                "requests_per_hour": 2000,
                "requests_per_day": 20000,
            },
            "per_api_key_limit": {
                "requests_per_minute": 500,
                "requests_per_hour": 5000,
                "requests_per_day": 50000,
            },
        },
        # TLS 設置
        "tls": {
            "enabled": True,
            "min_version": "1.2",
            "require_client_cert": False,
            "verify_hostname": True,
            "cipher_suites": [
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
                "TLS_AES_128_GCM_SHA256",
            ],
        },
    },
    # 服務間通信安全
    "inter_service": {
        "require_authentication": True,
        "mutual_tls": True,
        "encrypt_communication": True,
        "verify_service_identity": True,
        "allowed_services": ["aiva-core", "aiva-ai", "aiva-scanner", "aiva-reporter"],
    },
}

# 環境變量配置映射
SECURITY_ENV_CONFIG = {
    # 基本配置
    "AIVA_SECURITY_ENABLED": "security.enabled",
    "AIVA_SECURITY_DEBUG": "security.debug_mode",
    # 加密配置
    "AIVA_ENCRYPTION_KEY": "security.encryption.key",
    "AIVA_RSA_PRIVATE_KEY": "security.encryption.rsa.private_key",
    "AIVA_RSA_PUBLIC_KEY": "security.encryption.rsa.public_key",
    # JWT 配置
    "AIVA_JWT_SECRET": "security.jwt.secret",
    "AIVA_JWT_ALGORITHM": "security.jwt.algorithm",
    "AIVA_JWT_EXPIRATION": "security.jwt.expiration",
    # 數據庫配置
    "AIVA_DB_ENCRYPTION_KEY": "database.encryption_key",
    # 服務發現配置
    "AIVA_SERVICE_DISCOVERY_TOKEN": "service_discovery.auth_token",
    # 監控配置
    "AIVA_MONITORING_API_KEY": "monitoring.api_key",
}

# 安全默認路徑
SECURITY_PATHS = {
    "keys_directory": Path.home() / ".aiva" / "keys",
    "certificates_directory": Path.home() / ".aiva" / "certs",
    "audit_logs_directory": Path.home() / ".aiva" / "logs" / "audit",
    "config_file": Path.home() / ".aiva" / "security.yaml",
}


def get_security_config() -> dict[str, Any]:
    """獲取安全配置"""
    config = SECURITY_CONFIG_EXAMPLE.copy()

    # 從環境變量覆蓋配置
    for env_key, config_path in SECURITY_ENV_CONFIG.items():
        env_value = os.environ.get(env_key)
        if env_value:
            # 解析配置路徑並設置值
            keys = config_path.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # 轉換類型
            if env_value.lower() in ("true", "false"):
                env_value = env_value.lower() == "true"
            elif env_value.isdigit():
                env_value = int(env_value)

            current[keys[-1]] = env_value

    return config


def ensure_security_directories():
    """確保安全相關目錄存在"""
    for path in SECURITY_PATHS.values():
        if isinstance(path, Path):
            if path.suffix:  # 是文件
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # 是目錄
                path.mkdir(parents=True, exist_ok=True)


def get_default_permissions() -> list[str]:
    """獲取默認權限列表"""
    return [
        # 核心服務權限
        "core.command.execute",
        "core.command.read",
        "core.status.read",
        "core.config.read",
        "core.config.write",
        # 安全權限
        "security.authenticate",
        "security.authorize",
        "security.manage_tokens",
        "security.manage_api_keys",
        "security.audit",
        # 監控權限
        "monitoring.metrics.read",
        "monitoring.metrics.write",
        "monitoring.logs.read",
        "monitoring.traces.read",
        "monitoring.alerts.read",
        "monitoring.alerts.manage",
        # AI 服務權限
        "ai.task.create",
        "ai.task.read",
        "ai.model.read",
        "ai.inference.execute",
        # 掃描服務權限
        "scanner.scan.create",
        "scanner.scan.read",
        "scanner.results.read",
        # 報告服務權限
        "reporter.report.create",
        "reporter.report.read",
        "reporter.template.read",
        # 系統管理權限
        "system.admin.all",
        "system.user.manage",
        "system.service.manage",
        "system.config.manage",
    ]


# 安全檢查清單
SECURITY_CHECKLIST = {
    "encryption": {
        "description": "加密設置檢查",
        "checks": [
            "確認 AES-256-GCM 加密已啟用",
            "確認 RSA 密鑰長度至少 2048 位",
            "確認密鑰輪換機制已配置",
            "確認敏感數據已加密存儲",
        ],
    },
    "authentication": {
        "description": "認證設置檢查",
        "checks": [
            "確認多種認證方式已配置",
            "確認會話超時設置合理",
            "確認失敗嘗試鎖定機制已啟用",
            "確認密碼策略已實施",
        ],
    },
    "authorization": {
        "description": "授權設置檢查",
        "checks": [
            "確認 RBAC 模型已實施",
            "確認最小權限原則",
            "確認權限繼承機制正確",
            "確認動態角色管理已配置",
        ],
    },
    "network": {
        "description": "網絡安全檢查",
        "checks": [
            "確認 CORS 設置正確",
            "確認速率限制已啟用",
            "確認 TLS 1.2+ 已配置",
            "確認服務間通信已加密",
        ],
    },
    "audit": {
        "description": "審計日誌檢查",
        "checks": [
            "確認安全事件日誌已啟用",
            "確認日誌保留期已設置",
            "確認敏感操作已記錄",
            "確認日誌完整性保護",
        ],
    },
}


def validate_security_config(config: dict[str, Any]) -> list[str]:
    """驗證安全配置"""
    issues = []

    # 檢查必需的配置項
    required_keys = [
        "security.enabled",
        "security.encryption.algorithm",
        "security.jwt.algorithm",
        "security.authentication.methods",
        "security.authorization.model",
    ]

    for key_path in required_keys:
        keys = key_path.split(".")
        current = config

        try:
            for key in keys:
                current = current[key]
        except KeyError:
            issues.append(f"缺少必需配置: {key_path}")

    # 檢查安全性要求
    try:
        if config["security"]["encryption"]["algorithm"] not in [
            "AES-256-GCM",
            "AES-256-CBC",
        ]:
            issues.append("加密算法應使用 AES-256")

        if config["security"]["encryption"]["rsa"]["key_size"] < 2048:
            issues.append("RSA 密鑰長度應至少為 2048 位")

        if config["security"]["jwt"]["expiration"] > 7200:  # 2小時
            issues.append("JWT 過期時間不應超過 2 小時")

        if not config["security"]["audit"]["enabled"]:
            issues.append("建議啟用安全審計日誌")

    except KeyError as e:
        issues.append(f"配置項不完整: {e}")

    return issues


if __name__ == "__main__":
    # 示例使用
    import json

    # 獲取配置
    config = get_security_config()
    print("安全配置示例:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # 驗證配置
    issues = validate_security_config(config)
    if issues:
        print("\n配置問題:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\n配置驗證通過!")

    # 確保目錄存在
    ensure_security_directories()
    print(f"\n安全目錄已創建: {list(SECURITY_PATHS.values())}")
