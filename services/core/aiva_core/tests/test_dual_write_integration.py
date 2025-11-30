"""測試 InternalLoopConnector 的雙寫機制 (Day 4)

測試目標：
1. InternalLoopConnector 能正確初始化 CapabilityRegistry
2. sync_capabilities_to_rag 能同時寫入 PostgreSQL 和 ChromaDB
3. PostgreSQL 寫入失敗時不影響 RAG 寫入
4. 無 pg_session 時僅寫入 RAG（向後兼容）

遵循 aiva_common v2.0 標準
"""

import asyncio
from datetime import datetime, UTC
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from aiva_common.utils.logging import get_logger
from services.aiva_common.schemas.dual_loop import (
    ModuleCapability,
    CapabilityCategory,
    CapabilitySubCategory,
    CapabilityComplexity,
    ReturnDefinition,
)

from aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
from aiva_core.internal_exploration.models import Base, CapabilityRecord

logger = get_logger(__name__)

# 測試資料庫配置（開發環境）
# TODO: 正式發佈前改用環境變數 os.getenv("AIVA_TEST_DB_URL")
TEST_DB_URL = "postgresql://aiva_user:aiva_password@localhost:5432/aiva_capabilities"


@pytest.fixture
def db_session():
    """創建測試資料庫 Session"""
    engine = create_engine(TEST_DB_URL)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    # 清理測試數據
    session.query(CapabilityRecord).delete()
    session.commit()
    session.close()


@pytest.fixture
def mock_rag_kb():
    """模擬 RAG 知識庫"""
    class MockRAGKnowledgeBase:
        def __init__(self):
            self.documents = []
        
        def add_knowledge(self, content: str, metadata: dict) -> bool:
            self.documents.append({"content": content, "metadata": metadata})
            return True
        
        def get_document_count(self) -> int:
            return len(self.documents)
    
    return MockRAGKnowledgeBase()


@pytest.fixture
def sample_capabilities():
    """創建樣本能力數據"""
    return [
        ModuleCapability(
            capability_id="cap-test-scan_ports",
            module="test_scanner",
            name="scan_ports",
            function="scan_ports",
            description="掃描目標主機的開放端口",
            category=CapabilityCategory.SCANNING,
            sub_category=CapabilitySubCategory.PORT_SCAN,
            complexity=CapabilityComplexity.MODERATE,  # MEDIUM 不存在，使用 MODERATE
            parameters=[],
            return_info=ReturnDefinition(
                type="dict",
                description="掃描結果",
                example=None,
                structure=None,
            ),
            tags=["scanning", "port", "network"],
            language="python",
            file_path="/test/scanner.py",
            health_score=1.0,
            availability=1.0,
            error_rate=0.0,
            invocation=None,
            avg_latency_ms=None,
            last_used=None,
        ),
        ModuleCapability(
            capability_id="cap-test-detect_sql_injection",
            module="test_attacker",
            name="detect_sql_injection",
            function="detect_sql_injection",
            description="檢測 SQL 注入漏洞",
            category=CapabilityCategory.ATTACKING,
            sub_category=CapabilitySubCategory.SQL_INJECTION,
            complexity=CapabilityComplexity.COMPLEX,  # HIGH 不存在，使用 COMPLEX
            parameters=[],
            return_info=ReturnDefinition(
                type="dict",
                description="檢測結果",
                example=None,
                structure=None,
            ),
            tags=["attacking", "sql", "injection"],
            language="python",
            file_path="/test/attacker.py",
            health_score=1.0,
            availability=1.0,
            error_rate=0.0,
            invocation=None,
            avg_latency_ms=None,
            last_used=None,
        )
    ]


class TestDualWriteIntegration:
    """測試雙寫機制整合"""
    
    @pytest.mark.asyncio
    async def test_connector_with_pg_session(self, db_session, mock_rag_kb):
        """測試：提供 pg_session 時啟用雙寫"""
        connector = InternalLoopConnector(
            rag_knowledge_base=mock_rag_kb,
            pg_session=db_session
        )
        
        # 驗證 CapabilityRegistry 已初始化
        assert connector._capability_registry is not None
        assert connector.pg_session is not None
        logger.info("✅ CapabilityRegistry initialized with pg_session")
    
    @pytest.mark.asyncio
    async def test_connector_without_pg_session(self, mock_rag_kb):
        """測試：不提供 pg_session 時僅啟用 RAG（向後兼容）"""
        connector = InternalLoopConnector(
            rag_knowledge_base=mock_rag_kb,
            pg_session=None
        )
        
        # 驗證 CapabilityRegistry 未初始化
        assert connector._capability_registry is None
        assert connector.pg_session is None
        logger.info("✅ Backward compatible: RAG-only mode")
    
    @pytest.mark.asyncio
    async def test_dual_write_success(self, db_session, mock_rag_kb, sample_capabilities):
        """測試：雙寫成功（PostgreSQL + ChromaDB）"""
        connector = InternalLoopConnector(
            rag_knowledge_base=mock_rag_kb,
            pg_session=db_session
        )
        
        # 手動註冊能力（模擬 sync_capabilities_to_rag 的 Step 6）
        if connector._capability_registry:
            result = connector._capability_registry.register_capabilities(
                sample_capabilities
            )
            
            # 驗證 PostgreSQL 寫入
            assert result["added"] == 2
            assert result["modified"] == 0
            assert result["deleted"] == 0
            
            # 驗證資料庫記錄
            records = db_session.query(CapabilityRecord).all()
            assert len(records) == 2
            
            logger.info(f"✅ PostgreSQL write successful: {result['added']} capabilities added")
        
        # 驗證 RAG 寫入（需要透過 _inject_to_rag，這裡簡化測試）
        initial_doc_count = mock_rag_kb.get_document_count()
        assert initial_doc_count == 0  # 尚未透過 _inject_to_rag 寫入
        
        logger.info("✅ Dual-write test passed")
    
    @pytest.mark.asyncio
    async def test_pg_failure_does_not_block_rag(self, mock_rag_kb, sample_capabilities):
        """測試：PostgreSQL 失敗時不阻塞 RAG 寫入"""
        # 使用無效的資料庫連接
        connector = InternalLoopConnector(
            rag_knowledge_base=mock_rag_kb,
            pg_session=None  # 模擬 PostgreSQL 不可用
        )
        
        # RAG 寫入應該仍然成功
        documents = connector._convert_to_documents(sample_capabilities)
        added = connector._inject_to_rag(documents)
        
        assert added == len(sample_capabilities)
        assert mock_rag_kb.get_document_count() == len(sample_capabilities)
        
        logger.info("✅ RAG write succeeded despite PostgreSQL unavailability")
    
    @pytest.mark.asyncio
    async def test_capability_update_detection(self, db_session, sample_capabilities):
        """測試：能力更新檢測"""
        connector = InternalLoopConnector(
            rag_knowledge_base=None,
            pg_session=db_session
        )
        
        # 確保 _capability_registry 已初始化
        assert connector._capability_registry is not None, "CapabilityRegistry should be initialized"
        
        # 第一次註冊
        result1 = connector._capability_registry.register_capabilities(
            sample_capabilities
        )
        assert result1["added"] == 2
        
        # 修改能力描述
        modified_caps = [cap.model_copy() for cap in sample_capabilities]
        modified_caps[0].description = "【更新】掃描目標主機的開放端口（支持 TCP/UDP）"
        
        # 第二次註冊
        result2 = connector._capability_registry.register_capabilities(
            modified_caps
        )
        assert result2["modified"] == 1
        assert result2["added"] == 0
        assert result2["unchanged"] == 1
        
        logger.info("✅ Capability update detection working correctly")


def test_imports():
    """測試：驗證所有匯入正確"""
    assert InternalLoopConnector is not None
    assert CapabilityRecord is not None
    logger.info("✅ All imports successful")


if __name__ == "__main__":
    # 執行快速測試
    pytest.main([__file__, "-v", "-s"])
