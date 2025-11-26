"""
完整的四引擎驗證腳本

驗證所有引擎是否能實際對 Juice Shop 產生效果。
"""

import asyncio
import logging
from services.scan.coordinators import MultiEngineCoordinator
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


async def test_single_engine(engine_name: str):
    """測試單一引擎"""
    print(f"\n{'='*60}")
    print(f"🧪 測試 {engine_name.upper()} 引擎")
    print(f"{'='*60}")
    
    coordinator = MultiEngineCoordinator()
    
    try:
        if engine_name == "rust":
            # Rust 測試 Phase 0
            result = await coordinator.execute_phase0(
                scan_id=f"{engine_name}_test",
                targets=["http://localhost:3000"]
            )
        else:
            # 其他引擎測試 Phase 1
            result = await coordinator.execute_phase1(
                scan_id=f"{engine_name}_test",
                targets=["http://localhost:3000"],
                selected_engines=[engine_name],
                max_depth=5,
                max_urls=1000
            )
        
        # 輸出結果
        print(f"\n✅ {engine_name.upper()} 引擎測試成功!")
        print(f"  📦 資產數: {len(result.assets)}")
        print(f"  ⏱️  執行時間: {result.execution_time:.2f}s")
        
        if hasattr(result, 'summary') and result.summary:
            print(f"  🔗 URLs: {result.summary.urls_found}")
            if hasattr(result.summary, 'forms_found'):
                print(f"  📝 表單: {result.summary.forms_found}")
        
        # 顯示前 5 個資產
        if result.assets:
            print(f"\n  📋 前 5 個資產:")
            for i, asset in enumerate(result.assets[:5], 1):
                print(f"    {i}. [{asset.type}] {asset.value[:80]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ {engine_name.upper()} 引擎測試失敗!")
        print(f"  錯誤: {str(e)}")
        import traceback
        print(f"\n  詳細錯誤:")
        print(traceback.format_exc())
        return False


async def test_multi_engine():
    """測試多引擎協同"""
    print(f"\n{'='*60}")
    print(f"🧪 測試多引擎協同 (Python + Rust)")
    print(f"{'='*60}")
    
    coordinator = MultiEngineCoordinator()
    
    try:
        result = await coordinator.execute_phase1(
            scan_id="multi_engine_test",
            targets=["http://localhost:3000"],
            selected_engines=["python", "rust"],
            max_depth=5,
            max_urls=1000
        )
        
        print(f"\n✅ 多引擎測試成功!")
        print(f"  📦 總資產數: {len(result.assets)}")
        print(f"  ⏱️  執行時間: {result.execution_time:.2f}s")
        
        # 分析各引擎貢獻
        if hasattr(result, 'engine_results'):
            print(f"\n  🔧 各引擎貢獻:")
            for engine_name, engine_data in result.engine_results.items():
                print(f"    - {engine_name}: {engine_data.get('asset_count', 0)} 個資產")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 多引擎測試失敗!")
        print(f"  錯誤: {str(e)}")
        return False


async def main():
    """主測試流程"""
    print(f"\n{'#'*60}")
    print(f"# AIVA Scan 引擎完整驗證")
    print(f"# 目標: http://localhost:3000 (Juice Shop)")
    print(f"{'#'*60}")
    
    # 檢查 Juice Shop 是否運行
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000", timeout=5)
            print(f"\n✅ Juice Shop 運行中 (HTTP {response.status_code})")
    except Exception as e:
        print(f"\n❌ 無法連接 Juice Shop: {e}")
        print(f"   請確認 Juice Shop 正在運行: docker run -p 3000:3000 bkimminich/juice-shop")
        return
    
    # 測試各引擎
    results = {}
    
    # 1. Rust 引擎 (Phase 0)
    results['rust'] = await test_single_engine('rust')
    await asyncio.sleep(2)
    
    # 2. Python 引擎
    results['python'] = await test_single_engine('python')
    await asyncio.sleep(2)
    
    # 3. TypeScript 引擎
    results['typescript'] = await test_single_engine('typescript')
    await asyncio.sleep(2)
    
    # 4. Go 引擎
    results['go'] = await test_single_engine('go')
    await asyncio.sleep(2)
    
    # 5. 多引擎協同
    results['multi'] = await test_multi_engine()
    
    # 總結
    print(f"\n{'='*60}")
    print(f"📊 測試總結")
    print(f"{'='*60}")
    
    for engine_name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {engine_name.upper():<15} {status}")
    
    total_pass = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\n  總計: {total_pass}/{total_tests} 個測試通過")
    
    if total_pass == total_tests:
        print(f"\n🎉 恭喜！所有引擎驗證通過！")
    else:
        print(f"\n⚠️  部分引擎需要修復，請查看上方錯誤信息")


if __name__ == "__main__":
    asyncio.run(main())
