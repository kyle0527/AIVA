"""
TypeScript 引擎測試腳本
測試掃描 Juice Shop 靶場
"""

import asyncio
import sys
from pathlib import Path

# 設置 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from services.aiva_common.schemas import Phase1StartPayload
from services.scan.engines.typescript_engine.worker import _execute_typescript_scan
from pydantic import HttpUrl


async def test_juice_shop_scan():
    """測試掃描 Juice Shop"""
    print("\n" + "="*60)
    print("🚀 開始測試 TypeScript 引擎掃描 Juice Shop")
    print("="*60 + "\n")
    
    # 構建測試請求
    payload = Phase1StartPayload(
        scan_id="test-scan-001",
        targets=[HttpUrl("http://localhost:3000")],
        selected_engines=["typescript"],
        max_depth=2,
        timeout=120,
        scan_options={}
    )
    
    print(f"📋 掃描配置:")
    print(f"   目標: {payload.targets[0]}")
    print(f"   深度: {payload.max_depth}")
    print(f"   超時: {payload.timeout}秒")
    print(f"\n⏳ 執行掃描中...\n")
    
    try:
        # 執行掃描
        result = await _execute_typescript_scan(payload)
        
        print("\n" + "="*60)
        print("✅ 掃描完成！")
        print("="*60)
        
        print(f"\n📊 掃描結果統計:")
        print(f"   狀態: {result.status}")
        print(f"   執行時間: {result.execution_time:.2f}秒")
        print(f"   發現資產數: {len(result.assets)}")
        print(f"\n📈 詳細統計:")
        print(f"   URL 發現: {result.summary.urls_found}")
        print(f"   表單發現: {result.summary.forms_found}")
        print(f"   API 發現: {result.summary.apis_found}")
        
        # 顯示前 10 個資產
        if result.assets:
            print(f"\n🔍 資產樣本 (前10個):")
            for i, asset in enumerate(result.assets[:10], 1):
                print(f"   {i}. [{asset.type}] {asset.value[:80]}")
        
        # 顯示引擎結果
        print(f"\n🔧 引擎詳情:")
        for engine, info in result.engine_results.items():
            print(f"   {engine}: {info}")
        
        print("\n" + "="*60)
        
        return result
        
    except Exception as exc:
        print(f"\n❌ 掃描失敗: {exc}")
        import traceback
        traceback.print_exc()
        return None


async def test_webgoat_scan():
    """測試掃描 WebGoat"""
    print("\n" + "="*60)
    print("🚀 開始測試 TypeScript 引擎掃描 WebGoat")
    print("="*60 + "\n")
    
    payload = Phase1StartPayload(
        scan_id="test-scan-002",
        targets=[HttpUrl("http://localhost:8080/WebGoat")],
        selected_engines=["typescript"],
        max_depth=1,
        timeout=60,
        scan_options={}
    )
    
    print(f"📋 掃描配置:")
    print(f"   目標: {payload.targets[0]}")
    print(f"   深度: {payload.max_depth}")
    print(f"\n⏳ 執行掃描中...\n")
    
    try:
        result = await _execute_typescript_scan(payload)
        
        print("\n✅ WebGoat 掃描完成！")
        print(f"   發現資產: {len(result.assets)}")
        print(f"   執行時間: {result.execution_time:.2f}秒")
        
        return result
        
    except Exception as exc:
        print(f"\n❌ WebGoat 掃描失敗: {exc}")
        return None


async def main():
    """主函數"""
    print("\n" + "="*60)
    print("🎯 AIVA TypeScript 引擎實戰測試")
    print("   測試日期: 2025-11-18")
    print("="*60)
    
    # 測試 1: Juice Shop
    result1 = await test_juice_shop_scan()
    
    # 等待一下
    await asyncio.sleep(2)
    
    # 測試 2: WebGoat (簡短測試)
    # result2 = await test_webgoat_scan()
    
    print("\n" + "="*60)
    print("🏁 測試完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
