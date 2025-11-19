"""
TypeScript 引擎測試腳本
測試對靶場的實際掃描功能
"""
import asyncio
import json
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas import Phase1StartPayload
from services.scan.engines.typescript_engine.worker import (
    _execute_typescript_scan,
    _check_typescript_scanner_availability,
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


async def test_scanner_availability():
    """測試 TypeScript 掃描器可用性"""
    logger.info("=" * 60)
    logger.info("測試 1: TypeScript 掃描器可用性檢查")
    logger.info("=" * 60)
    
    available = await _check_typescript_scanner_availability()
    
    if available:
        logger.info("✅ TypeScript 掃描器可用")
        return True
    else:
        logger.error("❌ TypeScript 掃描器不可用")
        return False


async def test_juice_shop_scan():
    """測試掃描 Juice Shop 靶場"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 2: 掃描 Juice Shop 靶場")
    logger.info("=" * 60)
    
    # 構建測試請求
    from pydantic import HttpUrl
    
    req = Phase1StartPayload(
        scan_id="test-juice-shop-001",
        targets=[HttpUrl("http://localhost:3000")],
        selected_engines=["typescript"],
        max_depth=2,
        timeout=120,
    )
    
    logger.info(f"目標: {req.targets[0]}")
    logger.info(f"最大深度: {req.max_depth}")
    logger.info(f"超時: {req.timeout}秒")
    
    try:
        result = await _execute_typescript_scan(req)
        
        logger.info("\n" + "-" * 60)
        logger.info("掃描結果:")
        logger.info("-" * 60)
        logger.info(f"狀態: {result.status}")
        logger.info(f"執行時間: {result.execution_time:.2f}秒")
        logger.info(f"發現資產數: {len(result.assets)}")
        
        if result.summary:
            logger.info(f"URLs 發現: {result.summary.urls_found}")
            logger.info(f"表單發現: {result.summary.forms_found}")
            logger.info(f"APIs 發現: {result.summary.apis_found}")
        
        # 顯示部分資產
        if result.assets:
            logger.info("\n前 10 個資產:")
            for i, asset in enumerate(result.assets[:10], 1):
                logger.info(f"  {i}. {asset.type}: {asset.value[:80]}")
        
        logger.info("\n引擎結果:")
        logger.info(json.dumps(result.engine_results, indent=2))
        
        return result.status == "completed"
        
    except Exception as exc:
        logger.exception(f"掃描失敗: {exc}")
        return False


async def test_webgoat_scan():
    """測試掃描 WebGoat 靶場"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 3: 掃描 WebGoat 靶場")
    logger.info("=" * 60)
    
    from pydantic import HttpUrl
    
    req = Phase1StartPayload(
        scan_id="test-webgoat-001",
        targets=[HttpUrl("http://localhost:8080/WebGoat")],
        selected_engines=["typescript"],
        max_depth=1,
        timeout=60,
    )
    
    logger.info(f"目標: {req.targets[0]}")
    logger.info(f"最大深度: {req.max_depth}")
    
    try:
        result = await _execute_typescript_scan(req)
        
        logger.info("\n" + "-" * 60)
        logger.info("掃描結果:")
        logger.info("-" * 60)
        logger.info(f"狀態: {result.status}")
        logger.info(f"執行時間: {result.execution_time:.2f}秒")
        logger.info(f"發現資產數: {len(result.assets)}")
        
        return result.status == "completed"
        
    except Exception as exc:
        logger.exception(f"掃描失敗: {exc}")
        return False


async def main():
    """主測試函數"""
    logger.info("🚀 開始 TypeScript 引擎完整測試")
    logger.info(f"測試時間: {asyncio.get_event_loop().time()}")
    
    results = {}
    
    # 測試 1: 可用性檢查
    results["availability"] = await test_scanner_availability()
    
    if not results["availability"]:
        logger.error("❌ 掃描器不可用，停止測試")
        return
    
    # 測試 2: Juice Shop
    results["juice_shop"] = await test_juice_shop_scan()
    
    # 測試 3: WebGoat
    results["webgoat"] = await test_webgoat_scan()
    
    # 總結
    logger.info("\n" + "=" * 60)
    logger.info("測試總結")
    logger.info("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        logger.info(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 所有測試通過！")
    else:
        logger.error("\n❌ 部分測試失敗")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ 測試被用戶中斷")
        sys.exit(130)
    except Exception as exc:
        logger.exception(f"測試執行失敗: {exc}")
        sys.exit(1)
