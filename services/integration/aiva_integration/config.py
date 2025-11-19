"""
Integration Module Configuration

整合模組配置檔案,統一管理所有資料儲存路徑與資料庫連線
"""

import os
from pathlib import Path

# ============================================================================
# 基礎路徑配置
# ============================================================================

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 整合模組資料儲存根目錄 (研發階段直接使用預設路徑)
INTEGRATION_DATA_DIR = DATA_ROOT / "integration"

# 確保目錄存在
INTEGRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 攻擊路徑分析配置
# ============================================================================

# 攻擊路徑圖檔案目錄
ATTACK_PATHS_DIR = INTEGRATION_DATA_DIR / "attack_paths"
ATTACK_PATHS_DIR.mkdir(parents=True, exist_ok=True)

# NetworkX 圖持久化檔案 (自動推導，不需要環境變數)
ATTACK_GRAPH_FILE = ATTACK_PATHS_DIR / "attack_graph.pkl"

# 攻擊路徑匯出目錄 (HTML, Mermaid 等)
ATTACK_PATHS_EXPORT_DIR = ATTACK_PATHS_DIR / "exports"
ATTACK_PATHS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 經驗資料庫配置
# ============================================================================

# 經驗資料庫目錄
EXPERIENCES_DIR = INTEGRATION_DATA_DIR / "experiences"
EXPERIENCES_DIR.mkdir(parents=True, exist_ok=True)

# 經驗資料庫檔案 (SQLite)
EXPERIENCE_DB_FILE = EXPERIENCES_DIR / "experience.db"

# 經驗資料庫 URL (自動推導)
EXPERIENCE_DB_URL = f"sqlite:///{EXPERIENCE_DB_FILE}"

# 經驗資料匯出目錄
EXPERIENCES_EXPORT_DIR = EXPERIENCES_DIR / "exports"
EXPERIENCES_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 訓練資料集配置
# ============================================================================

# 訓練資料集目錄 (自動推導)
TRAINING_DATASET_DIR = INTEGRATION_DATA_DIR / "training_datasets"
TRAINING_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 資料集元資料目錄
DATASET_METADATA_DIR = TRAINING_DATASET_DIR / "metadata"
DATASET_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 模型檢查點配置
# ============================================================================

# 模型檢查點目錄 (自動推導)
MODEL_CHECKPOINT_DIR = INTEGRATION_DATA_DIR / "models"
MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 模型檢查點子目錄
MODEL_CHECKPOINTS_SUBDIR = MODEL_CHECKPOINT_DIR / "checkpoints"
MODEL_CHECKPOINTS_SUBDIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PostgreSQL 配置 (生產環境)
# ============================================================================

# PostgreSQL 連線配置 (研發階段直接使用預設值)
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/aiva_db"
POSTGRES_DSN = DATABASE_URL

# ============================================================================
# 備份配置
# ============================================================================

# 備份目錄
BACKUP_DIR = INTEGRATION_DATA_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# 備份保留策略 (天數)
BACKUP_RETENTION_DAYS = {
    "attack_graph": 7,      # 攻擊路徑圖保留 7 天
    "experience_db": 30,    # 經驗資料庫保留 30 天
    "training_dataset": 90, # 訓練資料集保留 90 天
    "model": 180,           # 模型檢查點保留 180 天
}

# ============================================================================
# 日誌配置
# ============================================================================

# 日誌目錄
LOG_DIR = DATA_ROOT / "logs" / "integration"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日誌檔案
ATTACK_PATH_LOG = LOG_DIR / "attack_path_analyzer.log"
EXPERIENCE_LOG = LOG_DIR / "experience_repository.log"
TRAINING_LOG = LOG_DIR / "training.log"

# ============================================================================
# 輔助函數
# ============================================================================


def get_config_summary() -> dict[str, str]:
    """取得配置摘要

    Returns:
        配置摘要字典
    """
    return {
        "INTEGRATION_DATA_DIR": str(INTEGRATION_DATA_DIR),
        "ATTACK_GRAPH_FILE": str(ATTACK_GRAPH_FILE),
        "EXPERIENCE_DB_URL": EXPERIENCE_DB_URL,
        "TRAINING_DATASET_DIR": str(TRAINING_DATASET_DIR),
        "MODEL_CHECKPOINT_DIR": str(MODEL_CHECKPOINT_DIR),
        "DATABASE_URL": DATABASE_URL.split('@')[0].split('//')[1].split(':')[0] + ":***@" + DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL,  # 隱藏密碼
        "BACKUP_DIR": str(BACKUP_DIR),
    }


def print_config():
    """列印當前配置"""
    print("\n=== Integration Module Configuration ===\n")
    for key, value in get_config_summary().items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    print_config()
