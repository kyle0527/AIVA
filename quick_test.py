#!/usr/bin/env python3
"""
AIVA 快速測試

一鍵執行所有關鍵測試

用法:
    python quick_test.py              # 運行所有測試
    python quick_test.py --skip-scan  # 跳過掃描測試 (更快)
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def run_quick_tests(skip_scan=False):
    """運行快速測試套件"""
    
    print("\n" + "=" * 80)
    print("  🚀 AIVA 快速測試")
    print("=" * 80 + "\n")
    
    tests = []
    results = {}
    
    # 測試 1: 引擎檢查
    print("1️⃣  檢查引擎可用性...")
    try:
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        coordinator = MultiEngineCoordinator()
        await coordinator.initialize()
        available = [e.value for e in coordinator.available_engines]
        
        if len(available) > 0:
            print(f"   ✅ {len(available)}/4 引擎可用: {', '.join(available)}")
            results["Engines"] = True
        else:
            print(f"   ❌ 沒有可用引擎")
            results["Engines"] = False
    except Exception as e:
        print(f"   ❌ 失敗: {str(e)[:60]}")
        results["Engines"] = False
    
    # 測試 2: HTTP 請求
    print("\n2️⃣  測試 HTTP 連接...")
    try:
        from services.scan.engines.python_engine.core_crawling_engine.http_client_hi import HiHttpClient
        from services.scan.engines.python_engine.authentication_manager import AuthenticationManager
        from services.scan.engines.python_engine.header_configuration import HeaderConfiguration
        
        auth = AuthenticationManager(None)
        headers = HeaderConfiguration(None)
        client = HiHttpClient(auth, headers, requests_per_second=10.0)
        
        response = await client.get("http://example.com")
        if response and response.status_code == 200:
            print(f"   ✅ HTTP 客戶端正常工作")
            results["HTTP"] = True
        else:
            print(f"   ⚠️  HTTP 請求異常")
            results["HTTP"] = False
        
        await client.close()
        
    except Exception as e:
        print(f"   ❌ 失敗: {str(e)[:60]}")
        results["HTTP"] = False
    
    # 測試 3: 命令處理器
    print("\n3️⃣  測試命令處理器...")
    try:
        from services.scan.command_handler import ScanCommandHandler
        handler = ScanCommandHandler()
        
        # 只測試初始化
        print(f"   ✅ 命令處理器創建成功")
        results["Handler"] = True
        
    except Exception as e:
        print(f"   ❌ 失敗: {str(e)[:60]}")
        results["Handler"] = False
    
    # 測試 4: 快速掃描 (可選)
    if not skip_scan:
        print("\n4️⃣  執行快速掃描測試...")
        try:
            from services.scan.command_handler import ScanCommandHandler
            handler = ScanCommandHandler()
            
            test_target = "http://localhost:3000"
            print(f"   目標: {test_target}")
            
            result = await handler.scan(
                target_url=test_target,
                scan_type="quick",
                max_depth=1,
                max_urls=5
            )
            
            if result and result.get("status") != "error":
                print(f"   ✅ 掃描完成")
                results["Scan"] = True
            else:
                print(f"   ⚠️  掃描異常: {result.get('message', 'Unknown')}")
                results["Scan"] = False
                
        except Exception as e:
            print(f"   ❌ 失敗: {str(e)[:60]}")
            results["Scan"] = False
    else:
        print("\n4️⃣  跳過掃描測試 (--skip-scan)")
        results["Scan"] = None
    
    # 總結
    print("\n" + "=" * 80)
    print("  📊 測試結果")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  已跳過"
        elif result:
            status = "✅ 通過"
        else:
            status = "❌ 失敗"
        print(f"  {test_name:15} {status}")
    
    # 計算通過率
    completed = [v for v in results.values() if v is not None]
    if completed:
        passed = sum(1 for v in completed if v)
        total = len(completed)
        percentage = (passed / total) * 100
        
        print(f"\n  通過率: {passed}/{total} ({percentage:.0f}%)")
        
        if percentage == 100:
            print("\n  🎉 所有測試通過！")
        elif percentage >= 75:
            print("\n  ⚠️  部分測試失敗")
        else:
            print("\n  ❌ 多項測試失敗，請檢查配置")
    
    print()
    
    return all(v for v in completed if v is not None)


async def main():
    skip_scan = "--skip-scan" in sys.argv
    
    try:
        success = await run_quick_tests(skip_scan=skip_scan)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 測試過程發生錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
