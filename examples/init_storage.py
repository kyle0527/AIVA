#!/usr/bin/env python3
"""
初始化 AIVA 數據存儲

創建所有必要的目錄和數據庫
"""

import asyncio
import logging
from pathlib import Path
import sys

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.core.aiva_core.storage import StorageManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def main():
    """主函數"""
    # 配置
    data_root = Path("/workspaces/AIVA/data")
    db_type = "hybrid"  # 推薦：hybrid (SQLite + JSONL)

    logger.info(f"Initializing AIVA storage: {data_root}")
    logger.info(f"Database type: {db_type}")

    # 創建存儲管理器
    storage = StorageManager(
        data_root=data_root, db_type=db_type, auto_create_dirs=True
    )

    logger.info("[OK] Storage initialized successfully!")

    # 顯示目錄結構
    print("\n[U+1F4C1] Data Directory Structure:")
    print(f"Root: {data_root}")
    for category, paths in storage.dirs.items():
        print(f"\n{category.upper()}:")
        if isinstance(paths, dict):
            for name, path in paths.items():
                exists = "[OK]" if path.exists() else "[FAIL]"
                print(f"  {exists} {name}: {path}")
        else:
            exists = "[OK]" if paths.exists() else "[FAIL]"
            print(f"  {exists} {paths}")

    # 獲取統計
    stats = await storage.get_statistics()
    print("\n[STATS] Storage Statistics:")
    for key, value in stats.items():
        if isinstance(value, int) and key.endswith("_size"):
            # 格式化大小
            size_mb = value / (1024 * 1024)
            print(f"  {key}: {size_mb:.2f} MB")
        else:
            print(f"  {key}: {value}")

    logger.info("Initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())
