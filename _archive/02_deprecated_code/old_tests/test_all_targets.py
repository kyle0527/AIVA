#!/usr/bin/env python3
"""
全面測試：測試所有四個靶場 + 驗證多語言引擎
目的：驗證 scan_orchestrator.py 對真實靶場的完整掃描能力
"""
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


async def test_target(target_url: str, strategy: str = "normal"):
    """測試單個靶場"""
    print(f"\n{'='*80}")
    print(f"🎯 測試目標: {target_url}")
    print(f"📋 策略: {strategy}")
    print(f"{'='*80}")
    
    try:
        # 初始化
        orchestrator = ScanOrchestrator()
        
        # 創建掃描請求
        scan_request = ScanStartPayload(
            scan_id=f"scan_test_{target_url.split(':')[-1].replace('/', '')}",
            targets=[HttpUrl(target_url)],
            strategy=strategy,
            scope={"include_subdomains": False},
        )
        
        # 執行掃描
        start_time = datetime.now()
        result = await orchestrator.execute_scan(scan_request)
        duration = (datetime.now() - start_time).total_seconds()
        
        # 顯示結果
        print(f"\n✅ 掃描完成")
        print(f"   狀態: {result.status}")
        print(f"   發現 URL: {len([a for a in result.assets if a.type == 'URL'])}")
        print(f"   發現表單: {len([a for a in result.assets if a.has_form])}")
        print(f"   資產總數: {len(result.assets)}")
        print(f"   掃描時長: {duration:.1f}s")
        
        # 顯示資產樣本
        if result.assets:
            print(f"\n   📄 資產樣本 (前 5 個):")
            for i, asset in enumerate(result.assets[:5], 1):
                print(f"     [{i}] {asset.type}: {asset.value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """測試所有靶場"""
    print("=" * 80)
    print("[TEST] 全面測試：所有靶場 + 多語言引擎驗證")
    print("=" * 80)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 定義所有測試目標
    targets = [
        ("http://localhost:3000", "Juice Shop (Angular SPA)"),
        ("http://localhost:3001", "靶場 #2"),
        ("http://localhost:3003", "靶場 #3"),
        ("http://localhost:8080", "靶場 #4"),
    ]
    
    results = {}
    
    for url, name in targets:
        print(f"\n\n{'#'*80}")
        print(f"# 測試 {name}")
        print(f"{'#'*80}")
        
        success = await test_target(url, strategy="normal")
        results[name] = success
    
    # 總結報告
    print(f"\n\n{'='*80}")
    print("📊 測試總結")
    print(f"{'='*80}")
    
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"   {status}: {name}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\n   總計: {passed}/{total} 通過")
    
    print(f"\n⏰ 測試結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
