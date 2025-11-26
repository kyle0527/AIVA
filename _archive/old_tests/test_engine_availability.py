"""
引擎可用性診斷腳本

測試所有四個掃描引擎的可用性，並驗證 MultiEngineCoordinator 的初始化。

修復驗證 (2025-11-22):
- Go 引擎已編譯為 scanner.exe
- MultiEngineCoordinator 已添加 initialize() 方法
- command_handler 已確保調用初始化
"""

import asyncio
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_engine_availability():
    """測試所有引擎的可用性"""
    
    print("=" * 80)
    print("🔍 AIVA 引擎可用性診斷")
    print("=" * 80)
    print()
    
    # 1. 測試直接導入適配器
    print("[步驟 1] 測試適配器導入...")
    try:
        from services.scan.coordinators.engines import (
            PythonAdapter, TypeScriptAdapter, RustAdapter, GoAdapter
        )
        print("  ✅ 成功導入所有適配器")
    except Exception as e:
        print(f"  ❌ 適配器導入失敗: {e}")
        return
    
    print()
    
    # 2. 測試每個適配器的可用性
    print("[步驟 2] 測試各引擎可用性...")
    print()
    
    adapters = {
        "Python": PythonAdapter(),
        "TypeScript": TypeScriptAdapter(),
        "Rust": RustAdapter(),
        "Go": GoAdapter(),
    }
    
    results = {}
    
    for engine_name, adapter in adapters.items():
        try:
            is_available = await adapter.is_available()
            results[engine_name] = is_available
            
            status = "✅ 可用" if is_available else "❌ 不可用"
            print(f"  [{engine_name:12}] {status}")
            
            # 如果是 Go 引擎，顯示詳細資訊
            if engine_name == "Go" and is_available:
                print(f"    └─ 掃描器路徑: {adapter.go_scanner_path}")
            
        except Exception as e:
            results[engine_name] = False
            print(f"  [{engine_name:12}] ❌ 檢查失敗: {e}")
    
    print()
    
    # 3. 測試 MultiEngineCoordinator 初始化
    print("[步驟 3] 測試 MultiEngineCoordinator 初始化...")
    try:
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        
        coordinator = MultiEngineCoordinator()
        print("  ✅ 協調器創建成功")
        
        # 測試初始化
        await coordinator.initialize()
        print("  ✅ 協調器初始化成功")
        
        # 顯示可用引擎
        available = [e.value for e in coordinator.available_engines]
        print(f"  📊 可用引擎: {', '.join(available) if available else '無'}")
        
    except Exception as e:
        print(f"  ❌ 協調器測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 4. 測試 ScanCommandHandler
    print("[步驟 4] 測試 ScanCommandHandler 初始化...")
    try:
        from services.scan.command_handler import ScanCommandHandler
        
        handler = ScanCommandHandler()
        print("  ✅ 命令處理器創建成功")
        
        # 測試惰性初始化
        await handler._ensure_coordinator_initialized()
        print("  ✅ 命令處理器協調器初始化成功")
        
        available = [e.value for e in handler.coordinator.available_engines]
        print(f"  📊 可用引擎: {', '.join(available) if available else '無'}")
        
    except Exception as e:
        print(f"  ❌ 命令處理器測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 5. 總結
    print("=" * 80)
    print("📊 診斷總結")
    print("=" * 80)
    
    available_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n引擎可用性: {available_count}/{total_count}")
    print()
    
    for engine, available in results.items():
        status = "✅" if available else "❌"
        print(f"  {status} {engine}")
    
    print()
    
    if available_count == 0:
        print("⚠️  警告：沒有任何引擎可用！")
        print()
        print("可能的原因：")
        print("  1. Go 引擎：scanner.exe 未編譯")
        print("     解決方法: cd services/scan/engines/go_engine && go build -o scanner.exe ./cmd/ssrf-scanner")
        print()
        print("  2. Python 引擎：掃描編排器模組缺失")
        print("     解決方法: 檢查 services/scan/engines/python_engine/scan_orchestrator.py")
        print()
        print("  3. Rust/TypeScript 引擎：二進制文件或腳本缺失")
        print()
    elif available_count == total_count:
        print("🎉 所有引擎都已就緒！")
        print()
        print("後續步驟：")
        print("  1. 運行完整掃描測試: python test_all_targets.py")
        print("  2. 測試動態引擎: python test_dynamic_scan.py")
        print("  3. 驗證 HTTP 請求: python diagnose_http.py")
    else:
        print(f"⚠️  部分引擎可用 ({available_count}/{total_count})")
        print()
        print("建議：")
        print("  - 已可用的引擎可以正常使用")
        print("  - 不可用的引擎需要編譯或配置")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_engine_availability())
