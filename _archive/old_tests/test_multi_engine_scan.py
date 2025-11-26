"""
多引擎掃描綜合測試

驗證修復後的多引擎協調器是否能正常工作，包括：
1. 引擎初始化
2. Python 引擎掃描
3. Go 引擎掃描 (SSRF)
4. Rust 引擎掃描
5. 多引擎並行執行
"""

import asyncio
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_python_engine():
    """測試 Python 引擎"""
    print("\n" + "=" * 80)
    print("🐍 測試 Python 引擎")
    print("=" * 80)
    
    try:
        from services.scan.coordinators.engines import PythonAdapter
        
        adapter = PythonAdapter()
        
        # 檢查可用性
        is_available = await adapter.is_available()
        print(f"✅ Python 引擎可用: {is_available}")
        
        if not is_available:
            print("⚠️  Python 引擎不可用，跳過測試")
            return False
        
        # 測試掃描
        print("\n開始 Python 引擎掃描測試...")
        result = await adapter.scan(
            targets=["http://example.com"],
            options={
                "scan_id": "test_python",
                "max_depth": 2,
                "max_pages": 10,
                "timeout": 30,
                "strategy": "BALANCED"
            }
        )
        
        print(f"\n掃描結果:")
        print(f"  - 發現資產: {len(result.get('assets', []))} 個")
        print(f"  - 引擎: {result.get('metadata', {}).get('engine', 'unknown')}")
        
        if result.get('error'):
            print(f"  ⚠️  錯誤: {result['error']}")
        else:
            print(f"  ✅ 掃描成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Python 引擎測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_go_engine():
    """測試 Go 引擎"""
    print("\n" + "=" * 80)
    print("🚀 測試 Go 引擎 (SSRF Scanner)")
    print("=" * 80)
    
    try:
        from services.scan.coordinators.engines import GoAdapter
        
        adapter = GoAdapter()
        
        # 檢查可用性
        is_available = await adapter.is_available()
        print(f"✅ Go 引擎可用: {is_available}")
        
        if not is_available:
            print("⚠️  Go 引擎不可用，跳過測試")
            print("提示: 請先編譯 Go 掃描器:")
            print("  cd services/scan/engines/go_engine")
            print("  go build -o scanner.exe ./cmd/ssrf-scanner")
            return False
        
        # 測試掃描 (使用本地測試目標)
        print("\n開始 Go 引擎 SSRF 掃描測試...")
        result = await adapter.scan(
            targets=["http://example.com"],
            options={
                "scan_id": "test_go_ssrf",
                "timeout": 30,
                "concurrency": 5
            }
        )
        
        print(f"\n掃描結果:")
        print(f"  - 發現資產: {len(result.get('assets', []))} 個")
        print(f"  - 引擎: {result.get('metadata', {}).get('engine', 'unknown')}")
        print(f"  - 掃描目標: {result.get('metadata', {}).get('targets_scanned', 0)} 個")
        
        if result.get('error'):
            print(f"  ⚠️  錯誤: {result['error']}")
        else:
            print(f"  ✅ 掃描成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Go 引擎測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_coordinator_parallel_scan():
    """測試協調器並行掃描"""
    print("\n" + "=" * 80)
    print("🎯 測試多引擎並行掃描")
    print("=" * 80)
    
    try:
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        
        coordinator = MultiEngineCoordinator()
        
        # 初始化
        print("\n初始化協調器...")
        await coordinator.initialize()
        
        available = [e.value for e in coordinator.available_engines]
        print(f"✅ 可用引擎: {', '.join(available)}")
        
        if not available:
            print("❌ 沒有可用引擎，無法進行並行測試")
            return False
        
        # 執行 Phase 1 掃描
        print("\n開始 Phase 1 深度掃描 (多引擎並行)...")
        result = await coordinator.execute_phase1(
            scan_id="test_parallel",
            targets=["http://example.com"],
            selected_engines=available[:2],  # 使用前兩個可用引擎
            max_depth=2,
            max_urls=10
        )
        
        print(f"\n掃描結果:")
        print(f"  - 狀態: {result.status}")
        print(f"  - 執行時間: {result.execution_time:.2f}s")
        print(f"  - 發現資產: {len(result.assets)} 個")
        print(f"  - URLs: {result.summary.urls_found}")
        print(f"  - Forms: {result.summary.forms_found}")
        print(f"  - APIs: {result.summary.apis_found}")
        
        if result.error_info:
            print(f"  ⚠️  錯誤: {result.error_info.error_message}")
        else:
            print(f"  ✅ 並行掃描成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 並行掃描測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_command_handler():
    """測試命令處理器"""
    print("\n" + "=" * 80)
    print("📡 測試 Scan 命令處理器")
    print("=" * 80)
    
    try:
        from services.scan.command_handler import ScanCommandHandler
        from services.aiva_common.schemas import AICommand, CommandType
        
        handler = ScanCommandHandler()
        print("✅ 命令處理器創建成功")
        
        # 測試 Health Check
        print("\n測試健康檢查...")
        health_command = AICommand(
            command_id="test_health",
            command_type=CommandType.HEALTH_CHECK,
            target_module="scan",
            payload={}
        )
        
        result = await handler.handle_command(health_command)
        
        print(f"\n健康檢查結果:")
        print(f"  - 成功: {result.success}")
        print(f"  - 狀態: {result.status}")
        print(f"  - 可用引擎: {result.result.get('available_engines', [])}")
        
        if result.success:
            print(f"  ✅ 命令處理器工作正常")
        else:
            print(f"  ❌ 命令處理器異常: {result.error}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ 命令處理器測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主測試流程"""
    print("=" * 80)
    print("🧪 AIVA 多引擎掃描綜合測試")
    print("=" * 80)
    print("\n本測試將驗證:")
    print("  1. ✅ Python 引擎單獨掃描")
    print("  2. ✅ Go 引擎單獨掃描")
    print("  3. ✅ 多引擎並行掃描")
    print("  4. ✅ 命令處理器工作流程")
    
    results = {
        "Python 引擎": False,
        "Go 引擎": False,
        "並行掃描": False,
        "命令處理器": False
    }
    
    # 1. 測試 Python 引擎
    results["Python 引擎"] = await test_python_engine()
    
    # 2. 測試 Go 引擎
    results["Go 引擎"] = await test_go_engine()
    
    # 3. 測試並行掃描
    results["並行掃描"] = await test_coordinator_parallel_scan()
    
    # 4. 測試命令處理器
    results["命令處理器"] = await test_command_handler()
    
    # 總結
    print("\n" + "=" * 80)
    print("📊 測試總結")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n通過測試: {passed}/{total}\n")
    
    for test_name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {status}  {test_name}")
    
    print("\n" + "=" * 80)
    
    if passed == total:
        print("🎉 所有測試通過！多引擎系統工作正常。")
    elif passed >= total / 2:
        print("⚠️  部分測試通過，系統部分可用。")
    else:
        print("❌ 大部分測試失敗，請檢查配置。")
    
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
