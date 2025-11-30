#!/usr/bin/env python3
"""測試動態掃描 - 僅限 localhost 範圍"""
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


async def test_localhost_only():
    """僅掃描 localhost，不跟隨外部鏈接"""
    print("=" * 80)
    print("[TEST] 動態掃描測試 - 僅 localhost")
    print("=" * 80)
    print("目標: http://localhost:3000")
    print("策略: normal (啟用動態掃描)")
    print("範圍: 僅 localhost，不跟隨外部鏈接")
    print("=" * 80)
    
    try:
        orchestrator = ScanOrchestrator()
        
        # 使用 full 策略（會啟用動態掃描）
        scan_request = ScanStartPayload(
            scan_id="scan_localhost_only",
            targets=[HttpUrl("http://localhost:3000")],
            strategy="full",  # full = aggressive (enable_dynamic_scan=True)
            scope={
                "include_subdomains": False,
                "allowed_domains": ["localhost", "127.0.0.1"],  # 限制範圍
            },
        )
        
        start_time = datetime.now()
        print("\n[START] 開始掃描...")
        result = await orchestrator.execute_scan(scan_request)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n[DONE] 掃描完成")
        print(f"   狀態: {result.status}")
        print(f"   發現 URL: {len([a for a in result.assets if a.type == 'URL'])}")
        print(f"   發現表單: {len([a for a in result.assets if a.has_form])}")
        print(f"   資產總數: {len(result.assets)}")
        print(f"   掃描時長: {duration:.1f}s")
        
        # 統計 localhost vs 外部 URL
        localhost_urls = [a for a in result.assets if a.type == "URL" and ("localhost" in a.value or "127.0.0.1" in a.value)]
        external_urls = [a for a in result.assets if a.type == "URL" and "localhost" not in a.value and "127.0.0.1" not in a.value]
        
        print(f"\n   [SCOPE] localhost URL: {len(localhost_urls)}")
        print(f"   [SCOPE] 外部 URL: {len(external_urls)}")
        
        if localhost_urls:
            print(f"\n   [LOCALHOST] 前 20 個 localhost URL:")
            for i, asset in enumerate(localhost_urls[:20], 1):
                print(f"     [{i}] {asset.value}")
        
        if external_urls:
            print(f"\n   [EXTERNAL] 前 10 個外部 URL:")
            for i, asset in enumerate(external_urls[:10], 1):
                print(f"     [{i}] {asset.value}")
        
    except Exception as e:
        print(f"\n[ERROR] 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_localhost_only())
