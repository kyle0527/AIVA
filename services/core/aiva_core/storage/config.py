"""
AIVA 數據存儲配置

配置存儲後端、路徑和其他選項
"""

import os
from pathlib import Path
from typing import Any

# 數據根目錄
DATA_ROOT = Path(os.getenv("AIVA_DATA_DIR", "/workspaces/AIVA/data"))

# 數據庫配置
DATABASE_TYPE = os.getenv(
    "AIVA_DB_TYPE", "hybrid"
)  # sqlite / postgres / jsonl / hybrid
DATABASE_URL = os.getenv("AIVA_DB_URL", f"sqlite:///{DATA_ROOT}/database/aiva.db")

# PostgreSQL 配置（生產環境）
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "aiva"),
    "user": os.getenv("POSTGRES_USER", "aiva"),
    "password": os.getenv("POSTGRES_PASSWORD", "aiva"),
}

# 存儲策略
EXPERIENCE_STORAGE = os.getenv(
    "AIVA_EXPERIENCE_STORAGE", "both"
)  # database / jsonl / both

# 數據保留策略
DATA_RETENTION = {
    # 經驗樣本保留時間（天）
    "high_quality_samples": -1,  # -1 表示永久保留
    "medium_quality_samples": 180,  # 6 個月
    "low_quality_samples": 30,  # 1 個月
    # 訓練會話保留時間（天）
    "successful_sessions": 90,  # 3 個月
    "failed_sessions": 30,  # 1 個月
    # 模型檢查點保留策略
    "best_checkpoints": -1,  # 永久保留
    "recent_checkpoints": 10,  # 保留最近 10 個
}

# 自動備份配置
AUTO_BACKUP = {
    "enabled": os.getenv("AIVA_AUTO_BACKUP", "false").lower() == "true",
    "interval_hours": int(os.getenv("AIVA_BACKUP_INTERVAL", "24")),
    "backup_dir": Path(os.getenv("AIVA_BACKUP_DIR", "/workspaces/AIVA/backups")),
    "keep_days": int(os.getenv("AIVA_BACKUP_KEEP_DAYS", "30")),
}

# 目錄路徑
PATHS: dict[str, Any] = {
    "training": {
        "root": DATA_ROOT / "training",
        "experiences": DATA_ROOT / "training/experiences",
        "sessions": DATA_ROOT / "training/sessions",
        "traces": DATA_ROOT / "training/traces",
        "metrics": DATA_ROOT / "training/metrics",
    },
    "models": {
        "root": DATA_ROOT / "models",
        "checkpoints": DATA_ROOT / "models/checkpoints",
        "production": DATA_ROOT / "models/production",
        "metadata": DATA_ROOT / "models/metadata",
    },
    "knowledge": {
        "root": DATA_ROOT / "knowledge",
        "vectors": DATA_ROOT / "knowledge/vectors",
        "entries": DATA_ROOT / "knowledge/entries.json",
        "payloads": DATA_ROOT / "knowledge/payloads",
    },
    "scenarios": {
        "root": DATA_ROOT / "scenarios",
        "owasp": DATA_ROOT / "scenarios/owasp",
        "custom": DATA_ROOT / "scenarios/custom",
    },
    "database": DATA_ROOT / "database",
    "logs": DATA_ROOT / "logs",
}

# 質量閾值
QUALITY_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.5,
    "low": 0.0,
}

# 批量操作配置
BATCH_CONFIG = {
    "save_batch_size": 100,  # 批量保存大小
    "export_chunk_size": 1000,  # 導出分塊大小
    "query_page_size": 100,  # 查詢分頁大小
}


def get_storage_config() -> dict:
    """獲取存儲配置"""
    return {
        "data_root": str(DATA_ROOT),
        "db_type": DATABASE_TYPE,
        "db_config": {
            "db_path": str(PATHS["database"] / "aiva.db"),
            **POSTGRES_CONFIG,
        },
        "auto_create_dirs": True,
    }
