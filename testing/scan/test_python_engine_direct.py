"""
直接測試 Python 引擎（避開重度導入）
"""
import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import HttpUrl
from services.aiva_common.schemas.tasks import ScanStartPayload, ScanScope, Authentication
from services.aiva_common.utils import get_logger, new_id
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator

logger = get_logger(__name__)


async def test_direct_scan():
    """直接測試掃描"""
    logger.info("="*60)
    logger.info("🚀 直接測試 Python 引擎")
    logger.info("="*60)
    
    # 創建掃描請求
    scan_id = new_id("scan").replace("-", "_")
    target_url = "http://localhost:3000"
    
    request = ScanStartPayload(
        scan_id=scan_id,
        targets=[HttpUrl(target_url)],
        strategy="deep",
        scope=ScanScope(),
        authentication=Authentication()
    )
    
    logger.info(f"📋 掃描 ID: {scan_id}")
    logger.info(f"🎯 目標: {target_url}")
    
    # 執行掃描
    try:
        orchestrator = ScanOrchestrator()
        logger.info("✅ ScanOrchestrator 初始化成功")
        
        result = await orchestrator.execute_scan(request)
        
        if result and result.status == "completed":
            logger.info(f"✅ 掃描完成！")
            logger.info(f"   - 發現 {len(result.findings)} 個發現")
            logger.info(f"   - 掃描 {len(result.tested_assets)} 個資產")
            return True
        else:
            logger.error(f"❌ 掃描失敗: {result.status if result else 'No result'}")
            return False
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_direct_scan())
    sys.exit(0 if success else 1)
