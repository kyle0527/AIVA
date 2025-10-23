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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.core.aiva_core.storage import StorageManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def initialize_storage():
    """初始化 AIVA 數據存儲"""
    # 配置
    data_root = project_root / "data"  # 使用相對路徑
    db_type = "hybrid"  # 推薦：hybrid (SQLite + JSONL)

    logger.info(f"Initializing AIVA storage: {data_root}")
    logger.info(f"Database type: {db_type}")

    # 創建存儲管理器
    storage = StorageManager(
        data_root=data_root, db_type=db_type, auto_create_dirs=True
    )

    logger.info("✅ Storage initialized successfully!")
    
    # 獲取統計資訊
    stats = await storage.get_statistics()
    logger.info(f"Storage statistics: {stats.get('backend')}, Total size: {stats.get('total_size', 0) / (1024*1024):.2f} MB")

    return storage


if __name__ == "__main__":
    logger.info("Running storage initialization directly...")
    asyncio.run(initialize_storage())
