#!/usr/bin/env python3
"""測試動態掃描引擎（Playwright）對 SPA 的爬取能力"""
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import HttpUrl
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


async def test_dynamic_scan(target_url: str):
    """使用動態掃描測試 SPA 靶場"""
    print("=" * 80)
    print(f"🎯 動態掃描測試（Playwright）")
    print("=" * 80)
    print(f"目標: {target_url}")
    print(f"策略: deep (啟用動態掃描)")
    print("=" * 80)
    
    try:
        orchestrator = ScanOrchestrator()
        
        # 使用 "deep" 策略啟用動態掃描
        scan_request = ScanStartPayload(
            scan_id="scan_dynamic_test",
            targets=[HttpUrl(target_url)],
            strategy="deep",  # ← deep 策略會啟用 enable_dynamic_scan=True
            scope={"include_subdomains": False},
        )
        
        start_time = datetime.now()
        print(f"\n⏳ 開始掃描...")
        result = await orchestrator.execute_scan(scan_request)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ 掃描完成")
        print(f"   狀態: {result.status}")
        print(f"   發現 URL: {len([a for a in result.assets if a.type == 'URL'])}")
        print(f"   發現表單: {len([a for a in result.assets if a.has_form])}")
        print(f"   資產總數: {len(result.assets)}")
        print(f"   掃描時長: {duration:.1f}s")
        
        if result.assets:
            print(f"\n   📄 資產樣本 (前 10 個):")
            for i, asset in enumerate(result.assets[:10], 1):
                print(f"     [{i}] {asset.type}: {asset.value}")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_dynamic_scan("http://localhost:3000"))
