"""能力元數據回填腳本 (Day 5)

從 ChromaDB 讀取現有能力數據，轉換並批量寫入 PostgreSQL

遵循 aiva_common v2.0 標準：
✅ 使用統一日誌記錄 (get_logger)
✅ 使用 Pydantic 模型進行數據驗證
✅ 使用統一錯誤處理
✅ 完整類型註解

執行方式：
    python scripts/migration/backfill_capabilities.py

環境要求：
    - PostgreSQL 正在運行 (localhost:5432)
    - ChromaDB 數據存在 (data/vector_db/chroma)
"""

import sys
from datetime import datetime, UTC
from pathlib import Path

# 添加 services 路徑到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
services_dir = project_root / "services"
core_dir = services_dir / "core"

# 添加必要的路徑
sys.path.insert(0, str(services_dir))
sys.path.insert(0, str(core_dir))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aiva_common.utils.logging import get_logger
from aiva_common.schemas.dual_loop import (
    ModuleCapability,
    CapabilityCategory,
    CapabilitySubCategory,
    CapabilityComplexity
)
from aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context
)

from aiva_core.internal_exploration.capability_registry import CapabilityRegistry
from aiva_core.internal_exploration.models import Base

logger = get_logger(__name__)

# 資料庫配置（開發環境）
# TODO: 正式發佈前改用環境變數 os.getenv("AIVA_CAPABILITY_DB_URL")
DB_URL = "postgresql://aiva_user:aiva_password@localhost:5432/aiva_capabilities"
CHROMA_PERSIST_DIR = project_root / "data" / "vector_db" / "chroma"


def get_all_capabilities_from_chromadb() -> list[dict]:
    """從 ChromaDB 讀取所有能力文檔
    
    Returns:
        能力文檔列表
    """
    try:
        import chromadb
        
        logger.info(f"Connecting to ChromaDB at {CHROMA_PERSIST_DIR}...")
        
        # 連接到 ChromaDB
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        
        # 獲取 collection（通常名稱是 "aiva_capabilities"）
        collections = client.list_collections()
        logger.info(f"Found {len(collections)} collections: {[c.name for c in collections]}")
        
        all_docs = []
        
        for collection_info in collections:
            collection = client.get_collection(name=collection_info.name)
            count = collection.count()
            logger.info(f"Collection '{collection_info.name}' has {count} documents")
            
            if count == 0:
                continue
            
            # 獲取所有文檔（批次查詢）
            result = collection.get(
                include=["documents", "metadatas"]  # 不包含 embeddings，避免陣列比較問題
            )
            
            # 轉換為統一格式
            ids = result.get("ids", [])
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])
            
            # 確保不是 None
            if documents is None:
                documents = []
            if metadatas is None:
                metadatas = []
            
            for i, doc_id in enumerate(ids):
                doc_data = {
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "collection": collection_info.name
                }
                all_docs.append(doc_data)
        
        logger.info(f"✅ Retrieved {len(all_docs)} documents from ChromaDB")
        return all_docs
        
    except ImportError:
        logger.error("❌ ChromaDB not installed. Install with: pip install chromadb")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to read from ChromaDB: {e}")
        raise


def convert_chroma_doc_to_capability(doc: dict) -> ModuleCapability | None:
    """將 ChromaDB 文檔轉換為 ModuleCapability
    
    Args:
        doc: ChromaDB 文檔數據
        
    Returns:
        ModuleCapability 實例，轉換失敗則返回 None
    """
    try:
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")
        
        # 從 metadata 提取能力信息
        # 根據 InternalLoopConnector 的格式推斷
        capability_id = metadata.get("capability_id", doc["id"])
        module_name = metadata.get("module", metadata.get("module_name", "unknown"))
        capability_name = metadata.get("name", metadata.get("capability_name", "unknown"))
        function_name = metadata.get("function", capability_name)
        
        # 分類信息
        category_str = metadata.get("category", "other")
        sub_category_str = metadata.get("sub_category")
        complexity_str = metadata.get("complexity", "medium")
        
        # 轉換枚舉值
        try:
            category = CapabilityCategory(category_str.lower())
        except (ValueError, AttributeError):
            category = CapabilityCategory.UTILITY  # 預設為 UTILITY（輔助能力）
        
        try:
            sub_category = CapabilitySubCategory(sub_category_str.lower()) if sub_category_str else None
        except (ValueError, AttributeError):
            sub_category = None
        
        try:
            complexity = CapabilityComplexity(int(complexity_str)) if isinstance(complexity_str, (int, str)) and str(complexity_str).isdigit() else CapabilityComplexity.MODERATE
        except (ValueError, AttributeError, TypeError):
            complexity = CapabilityComplexity.MODERATE
        
        # 構建 ModuleCapability
        capability = ModuleCapability(
            capability_id=capability_id,
            module=module_name,
            name=capability_name,
            function=function_name,
            description=metadata.get("description", content[:200]),  # 使用 content 前 200 字元
            category=category,
            sub_category=sub_category,
            complexity=complexity,
            parameters=metadata.get("parameters", []),
            return_type=metadata.get("return_type", "Any"),
            tags=metadata.get("tags", []),
            language=metadata.get("language", "python"),
            file_path=metadata.get("file_path"),
            health_score=float(metadata.get("health_score", 1.0)),
            availability=float(metadata.get("availability", 1.0)),
            error_rate=float(metadata.get("error_rate", 0.0)),
            invocation_info=None  # ChromaDB 通常不包含調用信息
        )
        
        return capability
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to convert document {doc.get('id')}: {e}")
        return None


def backfill_to_postgresql(capabilities: list[ModuleCapability]) -> dict:
    """批量寫入能力到 PostgreSQL
    
    Args:
        capabilities: 能力列表
        
    Returns:
        統計信息
    """
    logger.info(f"Connecting to PostgreSQL at {DB_URL}...")
    
    # 創建資料庫引擎和 Session
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)  # 確保表存在
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 創建 CapabilityRegistry
        registry = CapabilityRegistry(
            pg_session=session,
            chroma_collection=None  # 不需要寫入 ChromaDB
        )
        
        logger.info(f"Registering {len(capabilities)} capabilities...")
        
        # 批量註冊
        result = registry.register_capabilities(capabilities)
        
        # 統計信息（register_capabilities 返回字典）
        stats = {
            "total_input": len(capabilities),
            "added": result.get("added", 0),
            "updated": result.get("modified", 0),  # 'modified' 對應 'updated'
            "deleted": result.get("deleted", 0),
            "unchanged": result.get("unchanged", 0),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        logger.info("✅ Backfill completed:")
        logger.info(f"   Total input: {stats['total_input']}")
        logger.info(f"   Added: {stats['added']}")
        logger.info(f"   Updated: {stats['updated']}")
        logger.info(f"   Deleted: {stats['deleted']}")
        logger.info(f"   Unchanged: {stats['unchanged']}")
        
        return stats
        
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Backfill failed: {e}")
        raise
    finally:
        session.close()


def verify_backfill(expected_count: int) -> bool:
    """驗證回填結果
    
    Args:
        expected_count: 預期的能力數量
        
    Returns:
        驗證是否通過
    """
    logger.info("Verifying backfill results...")
    
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        from aiva_core.internal_exploration.models import CapabilityRecord
        
        # 查詢記錄數
        actual_count = session.query(CapabilityRecord).count()
        
        logger.info(f"Expected: {expected_count}, Actual: {actual_count}")
        
        if actual_count >= expected_count:
            logger.info("✅ Verification passed")
            return True
        else:
            logger.warning(f"⚠️  Verification failed: missing {expected_count - actual_count} records")
            return False
            
    finally:
        session.close()


def main():
    """主函數"""
    logger.info("=" * 60)
    logger.info("Day 5: 能力元數據回填 (ChromaDB → PostgreSQL)")
    logger.info("=" * 60)
    
    try:
        # 步驟 1: 從 ChromaDB 讀取所有能力
        logger.info("\n[1/4] Reading capabilities from ChromaDB...")
        chroma_docs = get_all_capabilities_from_chromadb()
        
        if not chroma_docs:
            logger.warning("⚠️  No documents found in ChromaDB. Exiting.")
            return
        
        # 步驟 2: 轉換為 ModuleCapability
        logger.info("\n[2/4] Converting to ModuleCapability models...")
        capabilities = []
        conversion_errors = 0
        
        for doc in chroma_docs:
            cap = convert_chroma_doc_to_capability(doc)
            if cap:
                capabilities.append(cap)
            else:
                conversion_errors += 1
        
        logger.info(f"   Converted: {len(capabilities)}/{len(chroma_docs)}")
        if conversion_errors > 0:
            logger.warning(f"   Conversion errors: {conversion_errors}")
        
        if not capabilities:
            logger.error("❌ No valid capabilities to backfill. Exiting.")
            return
        
        # 步驟 3: 批量寫入 PostgreSQL
        logger.info("\n[3/4] Backfilling to PostgreSQL...")
        stats = backfill_to_postgresql(capabilities)
        
        # 步驟 4: 驗證
        logger.info("\n[4/4] Verifying backfill results...")
        verify_backfill(len(capabilities))
        
        # 總結
        logger.info("\n" + "=" * 60)
        logger.info("Backfill Summary")
        logger.info("=" * 60)
        logger.info(f"ChromaDB documents: {len(chroma_docs)}")
        logger.info(f"Valid capabilities: {len(capabilities)}")
        logger.info(f"PostgreSQL added: {stats['added']}")
        logger.info(f"PostgreSQL updated: {stats['updated']}")
        logger.info(f"PostgreSQL unchanged: {stats['unchanged']}")
        logger.info("=" * 60)
        logger.info("✅ Day 5 completed successfully!")
        
    except Exception as e:
        error_context = create_error_context(
            error_type=ErrorType.DATABASE,
            severity=ErrorSeverity.HIGH,
            message="Backfill failed",
            details={"error": str(e)},
            exception=e
        )
        logger.error(f"❌ Backfill script failed: {error_context}")
        sys.exit(1)


if __name__ == "__main__":
    main()
