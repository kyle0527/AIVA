#!/usr/bin/env python3
"""
統一存儲架構配置驗證測試

驗證 Integration Service 的新統一存儲配置是否正常工作
包括 PostgreSQL 連接、數據庫表創建、數據讀寫功能
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas import FindingPayload
from services.aiva_common.schemas.findings import Vulnerability, Target
from services.aiva_common.enums import VulnerabilityType, Severity, Confidence
from services.integration.aiva_integration.reception.unified_storage_adapter import UnifiedStorageAdapter

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_unified_storage():
    """測試統一存儲適配器"""
    
    logger.info("🔧 開始統一存儲架構配置驗證...")
    
    try:
        # 1. 初始化統一存儲適配器
        logger.info("📦 初始化 UnifiedStorageAdapter...")
        storage_adapter = UnifiedStorageAdapter(
            data_root="./data/integration_test",
            db_config={
                "host": "localhost",  # 使用 localhost 因為容器映射到本地端口
                "port": 5432,
                "database": "aiva_db",
                "user": "postgres",
                "password": "aiva123",
            }
        )
        logger.info("✅ UnifiedStorageAdapter 初始化成功")
        
        # 2. 創建測試數據
        logger.info("📝 創建測試 FindingPayload...")
        test_finding = FindingPayload(
            finding_id="finding_test_001",
            scan_id="scan_test_001",
            task_id="task_test_001",
            vulnerability=Vulnerability(
                name=VulnerabilityType.SQLI,
                severity=Severity.HIGH,
                confidence=Confidence.CERTAIN,
                cwe="CWE-89",
            ),
            target=Target(
                url="https://example.com/login",
                method="POST",
                parameter="username",
            ),
            status="confirmed",
            evidence=None,
        )
        logger.info("✅ 測試數據創建成功")
        
        # 3. 測試保存功能
        logger.info("💾 測試保存功能...")
        await storage_adapter.save_finding(test_finding)
        logger.info("✅ 漏洞發現保存成功")
        
        # 4. 測試檢索功能
        logger.info("🔍 測試檢索功能...")
        retrieved_finding = await storage_adapter.get_finding("finding_test_001")
        if retrieved_finding:
            logger.info("✅ 漏洞發現檢索成功")
            logger.info(f"   - 漏洞類型: {retrieved_finding.vulnerability.name}")
            logger.info(f"   - 嚴重程度: {retrieved_finding.vulnerability.severity}")
            logger.info(f"   - 狀態: {retrieved_finding.status}")
        else:
            logger.warning("⚠️  漏洞發現檢索失敗")
        
        # 5. 測試列表功能
        logger.info("📋 測試列表功能...")
        findings_list = await storage_adapter.list_findings(
            scan_id="scan_test_001",
            limit=10
        )
        logger.info(f"✅ 檢索到 {len(findings_list)} 個漏洞發現")
        
        # 6. 測試統計功能
        logger.info("📊 測試統計功能...")
        count = await storage_adapter.count_findings(scan_id="scan_test_001")
        logger.info(f"✅ 統計結果: {count} 個漏洞發現")
        
        # 7. 測試掃描摘要
        logger.info("📈 測試掃描摘要功能...")
        summary = await storage_adapter.get_scan_summary("scan_test_001")
        logger.info("✅ 掃描摘要生成成功")
        logger.info(f"   - 總計: {summary['total_findings']}")
        logger.info(f"   - 按嚴重程度: {summary['by_severity']}")
        logger.info(f"   - 按漏洞類型: {summary['by_vulnerability_type']}")
        
        logger.info("🎉 統一存儲架構配置驗證完成！所有測試通過")
        return True
        
    except Exception as e:
        logger.error(f"❌ 統一存儲架構配置驗證失敗: {str(e)}")
        logger.exception("詳細錯誤信息:")
        return False


async def test_storage_manager_direct():
    """直接測試 StorageManager PostgreSQL 後端"""
    
    logger.info("🔧 開始 StorageManager PostgreSQL 後端測試...")
    
    try:
        from services.core.aiva_core.storage import StorageManager
        
        # 初始化 StorageManager
        storage_manager = StorageManager(
            data_root="./data/storage_test",
            db_type="postgres",
            db_config={
                "host": "localhost",
                "port": 5432,
                "database": "aiva_db",
                "user": "postgres",
                "password": "aiva123",
            },
            auto_create_dirs=True,
        )
        
        logger.info("✅ StorageManager PostgreSQL 後端初始化成功")
        
        # 獲取統計信息
        stats = await storage_manager.get_statistics()
        logger.info("📊 StorageManager 統計信息:")
        logger.info(f"   - 後端類型: {stats.get('backend')}")
        logger.info(f"   - 數據根目錄: {stats.get('data_root')}")
        logger.info(f"   - 總大小: {stats.get('total_size', 0) / (1024*1024):.2f} MB")
        
        logger.info("🎉 StorageManager PostgreSQL 後端測試完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ StorageManager PostgreSQL 後端測試失敗: {str(e)}")
        logger.exception("詳細錯誤信息:")
        return False


async def main():
    """主測試函數"""
    
    logger.info("🚀 開始統一存儲架構完整驗證...")
    
    # 測試 1: StorageManager 直接測試
    logger.info("\n" + "="*60)
    logger.info("測試 1: StorageManager PostgreSQL 後端")
    logger.info("="*60)
    
    success1 = await test_storage_manager_direct()
    
    # 測試 2: UnifiedStorageAdapter 測試
    logger.info("\n" + "="*60)
    logger.info("測試 2: UnifiedStorageAdapter 適配器")
    logger.info("="*60)
    
    success2 = await test_unified_storage()
    
    # 總結
    logger.info("\n" + "="*60)
    logger.info("驗證結果總結")
    logger.info("="*60)
    
    if success1 and success2:
        logger.info("🎉 所有測試通過！統一存儲架構配置成功")
        logger.info("✅ PostgreSQL 後端正常工作")
        logger.info("✅ UnifiedStorageAdapter 適配器正常工作") 
        logger.info("✅ Integration Service 可以使用新的統一存儲架構")
        return 0
    else:
        logger.error("❌ 部分測試失敗，需要檢查配置")
        if not success1:
            logger.error("   - StorageManager PostgreSQL 後端測試失敗")
        if not success2:
            logger.error("   - UnifiedStorageAdapter 適配器測試失敗")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)