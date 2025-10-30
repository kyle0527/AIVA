#!/usr/bin/env python3
"""
向量存儲統一遷移腳本

將現有的文件式向量存儲（FAISS/numpy）遷移到 PostgreSQL + pgvector
解決數據孤島問題，實現統一的向量存儲管理
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.core.aiva_core.rag import UnifiedVectorStore, VectorStore

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStorageMigration:
    """向量存儲遷移管理器"""

    def __init__(
        self,
        database_url: str = "postgresql://postgres:aiva123@localhost:5432/aiva_db",
        legacy_data_dirs: list[Path] | None = None,
    ):
        self.database_url = database_url
        self.legacy_data_dirs = legacy_data_dirs or [
            Path("./data/knowledge/vectors"),
            Path("./data/vectors"),
            Path("./services/core/aiva_core/rag/data"),
        ]

    async def scan_legacy_data(self) -> list[tuple[Path, dict[str, Any]]]:
        """掃描舊的向量存儲數據"""
        logger.info("🔍 掃描現有的向量存儲數據...")
        
        found_stores = []
        
        for data_dir in self.legacy_data_dirs:
            if not data_dir.exists():
                logger.debug(f"目錄不存在: {data_dir}")
                continue
                
            # 檢查是否有向量存儲文件
            vectors_file = data_dir / "vectors.npy"
            data_file = data_dir / "data.json"
            
            if vectors_file.exists() and data_file.exists():
                # 加載並檢查數據
                try:
                    legacy_store = VectorStore(
                        backend="memory",
                        persist_directory=data_dir,
                    )
                    legacy_store.load()
                    
                    stats = legacy_store.get_statistics()
                    logger.info(f"📁 發現向量存儲: {data_dir}")
                    logger.info(f"   - 文檔數量: {stats['total_documents']}")
                    logger.info(f"   - 後端類型: {stats['backend']}")
                    
                    found_stores.append((data_dir, stats))
                    
                except Exception as e:
                    logger.warning(f"載入向量存儲失敗 {data_dir}: {str(e)}")
        
        logger.info(f"✅ 掃描完成，找到 {len(found_stores)} 個向量存儲")
        return found_stores

    async def migrate_single_store(
        self,
        source_dir: Path,
        target_table: str = "knowledge_vectors",
    ) -> int:
        """遷移單個向量存儲"""
        logger.info(f"🚚 開始遷移向量存儲: {source_dir} -> {target_table}")
        
        try:
            # 創建統一向量存儲
            unified_store = UnifiedVectorStore(
                database_url=self.database_url,
                table_name=target_table,
                legacy_persist_directory=source_dir,
            )
            
            await unified_store.initialize()
            
            # 獲取遷移後的統計信息
            stats = await unified_store.get_statistics()
            migrated_count = stats["total_documents"]
            
            logger.info(f"✅ 遷移完成: {migrated_count} 個文檔")
            
            await unified_store.close()
            return migrated_count
            
        except Exception as e:
            logger.error(f"❌ 遷移失敗 {source_dir}: {str(e)}")
            raise

    async def create_backup(self, source_dir: Path) -> Path:
        """創建遷移前的備份"""
        backup_dir = source_dir.parent / f"{source_dir.name}_backup"
        
        logger.info(f"💾 創建備份: {source_dir} -> {backup_dir}")
        
        import shutil
        shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
        
        logger.info(f"✅ 備份完成: {backup_dir}")
        return backup_dir

    async def verify_migration(self, target_table: str = "knowledge_vectors") -> bool:
        """驗證遷移結果"""
        logger.info(f"🔍 驗證遷移結果: {target_table}")
        
        try:
            unified_store = UnifiedVectorStore(
                database_url=self.database_url,
                table_name=target_table,
            )
            
            await unified_store.initialize()
            
            # 獲取統計信息
            stats = await unified_store.get_statistics()
            logger.info(f"📊 統計信息:")
            logger.info(f"   - 總文檔數: {stats['total_documents']}")
            logger.info(f"   - 後端類型: {stats['backend']}")
            logger.info(f"   - 表名稱: {stats['table_name']}")
            
            # 測試搜索功能
            if stats['total_documents'] > 0:
                logger.info("🔍 測試搜索功能...")
                results = await unified_store.search("test query", top_k=3)
                logger.info(f"   - 搜索結果數: {len(results)}")
                
                if results:
                    logger.info(f"   - 首個結果分數: {results[0]['score']:.4f}")
            
            await unified_store.close()
            
            logger.info("✅ 遷移驗證通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ 遷移驗證失敗: {str(e)}")
            return False

    async def full_migration(
        self,
        create_backups: bool = True,
        target_table: str = "knowledge_vectors",
    ) -> dict[str, Any]:
        """執行完整的向量存儲遷移"""
        logger.info("🚀 開始完整的向量存儲遷移...")
        
        migration_summary = {
            "total_stores": 0,
            "migrated_stores": 0,
            "total_documents": 0,
            "failed_stores": [],
            "backups_created": [],
        }
        
        try:
            # 1. 掃描現有數據
            found_stores = await self.scan_legacy_data()
            migration_summary["total_stores"] = len(found_stores)
            
            if not found_stores:
                logger.info("❌ 沒有找到需要遷移的向量存儲數據")
                return migration_summary
            
            # 2. 遷移每個向量存儲
            for source_dir, _stats in found_stores:
                try:
                    # 創建備份
                    if create_backups:
                        backup_dir = await self.create_backup(source_dir)
                        migration_summary["backups_created"].append(str(backup_dir))
                    
                    # 執行遷移
                    migrated_count = await self.migrate_single_store(
                        source_dir, target_table
                    )
                    
                    migration_summary["migrated_stores"] += 1
                    migration_summary["total_documents"] += migrated_count
                    
                except Exception as e:
                    logger.error(f"遷移失敗 {source_dir}: {str(e)}")
                    migration_summary["failed_stores"].append({
                        "path": str(source_dir),
                        "error": str(e),
                    })
            
            # 3. 驗證遷移結果
            verification_passed = await self.verify_migration(target_table)
            migration_summary["verification_passed"] = verification_passed
            
            # 4. 輸出總結
            logger.info("\n" + "="*60)
            logger.info("📋 向量存儲遷移總結")
            logger.info("="*60)
            logger.info(f"找到的存儲: {migration_summary['total_stores']}")
            logger.info(f"成功遷移: {migration_summary['migrated_stores']}")
            logger.info(f"總文檔數: {migration_summary['total_documents']}")
            logger.info(f"失敗數量: {len(migration_summary['failed_stores'])}")
            logger.info(f"驗證結果: {'通過' if verification_passed else '失敗'}")
            
            if migration_summary["failed_stores"]:
                logger.error("失敗的存儲:")
                for failed in migration_summary["failed_stores"]:
                    logger.error(f"  - {failed['path']}: {failed['error']}")
            
            if migration_summary["migrated_stores"] > 0:
                logger.info("🎉 向量存儲統一遷移完成！")
            else:
                logger.warning("⚠️  沒有成功遷移任何向量存儲")
                
            return migration_summary
            
        except Exception as e:
            logger.error(f"❌ 向量存儲遷移過程中發生錯誤: {str(e)}")
            migration_summary["global_error"] = str(e)
            return migration_summary


async def main():
    """主函數"""
    logger.info("🚀 啟動向量存儲統一遷移...")
    
    # 創建遷移管理器
    migrator = VectorStorageMigration(
        database_url="postgresql://postgres:aiva123@localhost:5432/aiva_db",
        legacy_data_dirs=[
            Path("./data/knowledge/vectors"),
            Path("./data/vectors"),
            Path("./services/core/aiva_core/rag/data"),
            Path("./data/training/vectors"),  # 可能的其他位置
        ],
    )
    
    # 執行完整遷移
    result = await migrator.full_migration(
        create_backups=True,
        target_table="knowledge_vectors",
    )
    
    # 根據結果返回退出碼
    if result.get("verification_passed", False) and result["migrated_stores"] > 0:
        logger.info("✅ 遷移成功完成")
        return 0
    else:
        logger.error("❌ 遷移未能成功完成")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)