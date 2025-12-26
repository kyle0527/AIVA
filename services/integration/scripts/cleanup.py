"""
整合模組清理腳本

清理舊資料和備份
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from services.integration.aiva_integration.config import (
    ATTACK_PATHS_EXPORT_DIR,
    BACKUP_DIR,
    EXPERIENCES_EXPORT_DIR,
    LOG_DIR,
    ANALYSIS_HISTORY_DIR,  # ✨ 新增：分析歷史目錄
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cleanup_directory(directory: Path, days: int, pattern: str = "*"):
    """清理目錄中的舊檔案

    Args:
        directory: 目錄路徑
        days: 保留天數
        pattern: 檔案模式 (glob)
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return

    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = 0
    total_size = 0

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime < cutoff_date:
                file_size = file_path.stat().st_size
                try:
                    file_path.unlink()
                    deleted_count += 1
                    total_size += file_size
                    logger.info(
                        f"🗑️  Deleted: {file_path.name} (age: {(datetime.now() - file_mtime).days} days)"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to delete {file_path}: {e}")

    if deleted_count > 0:
        logger.info(
            f"✅ Cleaned up {deleted_count} file(s), freed {total_size / 1024 / 1024:.2f} MB"
        )
    else:
        logger.info(f"✅ No files to clean in {directory.name}")


def cleanup_versioned_history(keep_versions: int = 5):
    """清理分析歷史版本，僅保留最新的 N 個版本
    
    ✨ 新功能：依據 INTEGRATION_DATA_ANALYSIS_REPORT.md 建議實現
    
    Args:
        keep_versions: 保留版本數量 (預設: 5)
    """
    if not ANALYSIS_HISTORY_DIR.exists():
        logger.warning(f"分析歷史目錄不存在: {ANALYSIS_HISTORY_DIR}")
        return
        
    # 收集所有版本目錄 (例如: v1, v2, v3, v4, v5)
    version_dirs = sorted(
        [d for d in ANALYSIS_HISTORY_DIR.iterdir() if d.is_dir() and d.name.startswith('v')],
        key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0,
        reverse=True  # 由新到舊排序
    )
    
    if len(version_dirs) <= keep_versions:
        logger.info(f"✅ 版本數量 ({len(version_dirs)}) 未超過保留數量 ({keep_versions})，無需清理")
        return
    
    # 需要刪除的舊版本
    to_delete = version_dirs[keep_versions:]
    deleted_count = 0
    total_size = 0
    
    for version_dir in to_delete:
        try:
            # 計算目錄大小
            dir_size = sum(f.stat().st_size for f in version_dir.rglob('*') if f.is_file())
            
            # 刪除整個版本目錄
            import shutil
            shutil.rmtree(version_dir)
            
            deleted_count += 1
            total_size += dir_size
            logger.info(f"🗑️  Deleted version: {version_dir.name} ({dir_size / 1024 / 1024:.2f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete {version_dir}: {e}")
    
    if deleted_count > 0:
        logger.info(
            f"✅ 清理了 {deleted_count} 個舊版本，釋放 {total_size / 1024 / 1024:.2f} MB 空間"
        )
        logger.info(f"📌 保留版本: {[d.name for d in version_dirs[:keep_versions]]}")
    else:
        logger.info("✅ 無版本需要清理")


def main():
    parser = argparse.ArgumentParser(description="整合模組清理腳本")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="清理多少天前的檔案 (預設: 30)",
    )
    parser.add_argument(
        "--keep-versions",
        type=int,
        default=5,
        help="保留分析歷史版本數量 (預設: 5)",
    )
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="僅清理備份檔案",
    )
    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="僅清理日誌檔案",
    )
    parser.add_argument(
        "--exports-only",
        action="store_true",
        help="僅清理匯出檔案",
    )
    parser.add_argument(
        "--versions-only",
        action="store_true",
        help="僅清理分析歷史舊版本",
    )
    args = parser.parse_args()

    logger.info(f"=== 開始清理整合模組資料 (保留 {args.days} 天) ===\n")

    # 清理分析歷史版本
    if args.versions_only:
        logger.info("清理分析歷史舊版本...")
        cleanup_versioned_history(args.keep_versions)
        logger.info("\n=== 清理完成 ===")
        return

    # 清理備份
    if not args.logs_only and not args.exports_only:
        logger.info("清理備份檔案...")
        for backup_subdir in BACKUP_DIR.glob("*"):
            if backup_subdir.is_dir():
                cleanup_directory(backup_subdir, args.days)

    # 清理匯出檔案
    if not args.backup_only and not args.logs_only:
        logger.info("\n清理匯出檔案...")
        cleanup_directory(ATTACK_PATHS_EXPORT_DIR, args.days, "*.html")
        cleanup_directory(ATTACK_PATHS_EXPORT_DIR, args.days, "*.mmd")
        cleanup_directory(EXPERIENCES_EXPORT_DIR, args.days, "*.jsonl")
        cleanup_directory(EXPERIENCES_EXPORT_DIR, args.days, "*.csv")

    # 清理日誌檔案
    if not args.backup_only and not args.exports_only:
        logger.info("\n清理日誌檔案...")
        cleanup_directory(LOG_DIR, args.days, "*.log")
    
    # 自動清理分析歷史版本（不單獨執行時）
    if not (args.backup_only or args.logs_only or args.exports_only):
        logger.info("\n清理分析歷史舊版本...")
        cleanup_versioned_history(args.keep_versions)

    logger.info("\n=== 清理完成 ===")


if __name__ == "__main__":
    main()
