#!/usr/bin/env python3
"""
最小化測試：直接測試 ScanOrchestrator 的 HTTP 請求能力
目的：驗證 scan_orchestrator.py 是否能發送真實的 HTTP 請求
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


async def test_orchestrator_http():
    """測試 ScanOrchestrator 是否能發送 HTTP 請求"""
    
    print("=" * 80)
    print("🧪 ScanOrchestrator HTTP 請求測試")
    print("=" * 80)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 測試目標
    target_url = "http://example.com"  # ← 測試有靜態鏈接的網站
    print(f"\n🎯 目標 URL: {target_url}")
    
    try:
        # 步驟 1: 初始化 ScanOrchestrator
        print("\n📍 步驟 1: 初始化 ScanOrchestrator")
        orchestrator = ScanOrchestrator()
        print("   ✅ ScanOrchestrator 初始化完成")
        
        # 步驟 2: 創建掃描請求（使用 normal 策略，會爬更多）
        print("\n📍 步驟 2: 創建掃描請求")
        scan_request = ScanStartPayload(
            scan_id="scan_test_http_001",  # ← 修正：必須以 'scan_' 開頭
            targets=[HttpUrl(target_url)],
            strategy="normal",  # ← 使用 'normal' 策略 (BALANCED, max_depth=3)
            scope={"include_subdomains": False},
        )
        print(f"   策略: {scan_request.strategy}")
        print(f"   目標: {scan_request.targets}")
        
        # 步驟 3: 執行掃描
        print("\n📍 步驟 3: 執行掃描")
        print("   ⏳ 開始掃描...")
        
        result = await orchestrator.execute_scan(scan_request)
        
        print(f"   ✅ 掃描完成")
        print(f"\n📊 掃描結果:")
        print(f"   狀態: {result.status}")
        print(f"   發現 URL: {result.summary.urls_found}")
        print(f"   發現表單: {result.summary.forms_found}")
        print(f"   資產總數: {len(result.assets)}")
        print(f"   掃描時長: {result.summary.scan_duration_seconds}s")
        
        if result.assets:
            print(f"\n   📄 資產樣本 (前 5 個):")
            for i, asset in enumerate(result.assets[:5], 1):
                print(f"     [{i}] {asset.type}: {asset.value[:80]}")
        else:
            print("\n   ❌ 未發現任何資產！")
        
        # 驗證是否真的發送了請求
        print("\n" + "=" * 80)
        if result.summary.urls_found > 0 or len(result.assets) > 0:
            print("✅ 測試通過：ScanOrchestrator 成功發送 HTTP 請求")
            print("   已經發現了資產，證明網路通訊正常")
            return True
        else:
            print("❌ 測試失敗：未發現任何資產")
            print("   可能原因：")
            print("   1. HTTP 請求未真實發送")
            print("   2. 靶場服務未啟動")
            print("   3. URL 隊列管理有問題")
            return False
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 測試異常")
        print("=" * 80)
        logger.exception(f"錯誤: {e}")
        print(f"\n錯誤詳情: {e}")
        
        import traceback
        print("\n堆疊追蹤:")
        traceback.print_exc()
        
        return False


async def main():
    """主函數"""
    print("\n⏰ 測試開始時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print("\nℹ️  請確保 Docker 靶場正在運行...")
    print("   執行: docker ps | grep juice")
    
    try:
        passed = await test_orchestrator_http()
        
        print("\n⏰ 測試結束時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80)
        
        return 0 if passed else 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 用戶中斷測試")
        return 130


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
