"""
整合模組備份腳本

定期備份攻擊路徑圖和經驗資料庫
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

from services.integration.aiva_integration.config import (
    ATTACK_GRAPH_FILE,
    BACKUP_DIR,
    BACKUP_RETENTION_DAYS,
    EXPERIENCE_DB_FILE,
    MODEL_CHECKPOINT_DIR,
    TRAINING_DATASET_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def backup_file(source: Path, backup_dir: Path, prefix: str) -> Path | None:
    """備份單個檔案

    Args:
        source: 來源檔案路徑
        backup_dir: 備份目錄
        prefix: 備份檔案前綴

    Returns:
        備份檔案路徑,如果失敗則返回 None
    """
    if not source.exists():
        logger.warning(f"Source file not found: {source}")
        return None

    # 建立備份目錄
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 生成備份檔名 (包含時間戳)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{prefix}_{timestamp}{source.suffix}"
    backup_path = backup_dir / backup_filename

    try:
        # 複製檔案
        shutil.copy2(source, backup_path)
        logger.info(f"✅ Backed up: {source.name} -> {backup_path.name}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Failed to backup {source}: {e}")
        return None


def backup_directory(source: Path, backup_dir: Path, prefix: str) -> Path | None:
    """備份整個目錄

    Args:
        source: 來源目錄路徑
        backup_dir: 備份目錄
        prefix: 備份目錄前綴

    Returns:
        備份目錄路徑,如果失敗則返回 None
    """
    if not source.exists() or not source.is_dir():
        logger.warning(f"Source directory not found: {source}")
        return None

    # 建立備份目錄
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 生成備份目錄名 (包含時間戳)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dirname = f"{prefix}_{timestamp}"
    backup_path = backup_dir / backup_dirname

    try:
        # 複製目錄
        shutil.copytree(source, backup_path)
        logger.info(f"✅ Backed up directory: {source.name} -> {backup_path.name}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Failed to backup directory {source}: {e}")
        return None


def cleanup_old_backups(backup_dir: Path, retention_days: int):
    """清理舊備份

    Args:
        backup_dir: 備份目錄
        retention_days: 保留天數
    """
    if not backup_dir.exists():
        return

    now = datetime.now()
    deleted_count = 0

    for backup_item in backup_dir.iterdir():
        # 計算檔案年齡
        file_age_days = (now - datetime.fromtimestamp(backup_item.stat().st_mtime)).days

        if file_age_days > retention_days:
            try:
                if backup_item.is_file():
                    backup_item.unlink()
                elif backup_item.is_dir():
                    shutil.rmtree(backup_item)
                logger.info(f"🗑️  Deleted old backup: {backup_item.name} (age: {file_age_days} days)")
                deleted_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to delete {backup_item}: {e}")

    if deleted_count > 0:
        logger.info(f"✅ Cleaned up {deleted_count} old backup(s)")


def main():
    parser = argparse.ArgumentParser(description="整合模組備份腳本")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="不清理舊備份",
    )
    parser.add_argument(
        "--attack-graph-only",
        action="store_true",
        help="僅備份攻擊路徑圖",
    )
    parser.add_argument(
        "--experience-only",
        action="store_true",
        help="僅備份經驗資料庫",
    )
    args = parser.parse_args()

    logger.info("=== 開始備份整合模組資料 ===\n")

    # 攻擊路徑圖備份
    if not args.experience_only:
        attack_graph_backup_dir = BACKUP_DIR / "attack_paths"
        backup_file(ATTACK_GRAPH_FILE, attack_graph_backup_dir, "attack_graph")

        if not args.no_cleanup:
            cleanup_old_backups(
                attack_graph_backup_dir, BACKUP_RETENTION_DAYS["attack_graph"]
            )

    # 經驗資料庫備份
    if not args.attack_graph_only:
        experience_backup_dir = BACKUP_DIR / "experiences"
        backup_file(EXPERIENCE_DB_FILE, experience_backup_dir, "experience")

        if not args.no_cleanup:
            cleanup_old_backups(
                experience_backup_dir, BACKUP_RETENTION_DAYS["experience_db"]
            )

    # 訓練資料集備份 (可選)
    if not args.attack_graph_only and not args.experience_only:
        dataset_backup_dir = BACKUP_DIR / "training_datasets"
        backup_directory(TRAINING_DATASET_DIR, dataset_backup_dir, "datasets")

        if not args.no_cleanup:
            cleanup_old_backups(
                dataset_backup_dir, BACKUP_RETENTION_DAYS["training_dataset"]
            )

    # 模型檢查點備份 (可選)
    if not args.attack_graph_only and not args.experience_only:
        model_backup_dir = BACKUP_DIR / "models"
        backup_directory(MODEL_CHECKPOINT_DIR, model_backup_dir, "models")

        if not args.no_cleanup:
            cleanup_old_backups(model_backup_dir, BACKUP_RETENTION_DAYS["model"])

    logger.info("\n=== 備份完成 ===")


if __name__ == "__main__":
    main()
