#!/usr/bin/env python3
"""
AIVA 統一測試工具

整合所有掃描測試功能，提供簡單的命令行界面。

用法:
    python aiva_test.py engines              # 檢查引擎可用性
    python aiva_test.py http                 # 測試 HTTP 請求功能
    python aiva_test.py scan <target>        # 執行快速掃描測試
    python aiva_test.py dynamic <target>     # 測試動態掃描 (Playwright)
    python aiva_test.py all-targets          # 測試所有 Docker 靶場
    python aiva_test.py full                 # 完整測試套件

範例:
    python aiva_test.py engines
    python aiva_test.py scan http://localhost:3000
    python aiva_test.py dynamic http://localhost:3000
    python aiva_test.py all-targets
"""

import asyncio
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """打印美化的標題"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """打印章節標題"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print('─' * 80 + "\n")


# ==================== 測試 1: 引擎可用性 ====================
async def test_engines():
    """測試所有掃描引擎的可用性"""
    print_header("🔍 引擎可用性診斷")
    
    try:
        from services.scan.coordinators.engines import (
            PythonAdapter, TypeScriptAdapter, RustAdapter, GoAdapter
        )
        print("✅ 成功導入所有適配器\n")
    except Exception as e:
        print(f"❌ 適配器導入失敗: {e}\n")
        return False
    
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
            
            if engine_name == "Go" and is_available:
                print(f"    └─ 路徑: {adapter.go_scanner_path}")
            
        except Exception as e:
            results[engine_name] = False
            print(f"  [{engine_name:12}] ❌ 檢查失敗: {e}")
    
    print()
    
    # 測試 MultiEngineCoordinator
    print_section("📊 協調器初始化測試")
    
    try:
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        
        coordinator = MultiEngineCoordinator()
        await coordinator.initialize()
        
        available = [e.value for e in coordinator.available_engines]
        print("✅ 協調器初始化成功")
        print(f"📊 可用引擎: {', '.join(available) if available else '無'}\n")
        
        return len(available) > 0
        
    except Exception as e:
        print(f"❌ 協調器測試失敗: {e}\n")
        return False


# ==================== 測試 2: HTTP 請求 ====================
# 定義常量避免重複字串
LOCALHOST_3000 = "http://localhost:3000"

async def test_http():
    """測試 HTTP 客戶端功能"""
    print_header("🌐 HTTP 請求功能測試")
    
    try:
        from services.scan.engines.python_engine.core_crawling_engine.http_client_hi import HiHttpClient
        from services.scan.engines.python_engine.authentication_manager import AuthenticationManager
        from services.scan.engines.python_engine.header_configuration import HeaderConfiguration
        from services.aiva_common.schemas import Authentication
        
        # 創建預設的 Authentication 對象
        default_auth = Authentication(method="none", credentials=None)
        auth = AuthenticationManager(default_auth)
        headers = HeaderConfiguration(None)
        
        client = HiHttpClient(
            auth, headers,
            requests_per_second=10.0,
            per_host_rps=5.0,
            retries=3,
            timeout=10.0
        )
        
        print("✅ HTTP 客戶端初始化完成\n")
        
        test_urls = [
            (LOCALHOST_3000, "Juice Shop 靶場"),
            ("http://example.com", "測試域名"),
            ("http://httpbin.org/get", "HTTP 測試服務")
        ]
        
        success_count = 0
        
        for url, description in test_urls:
            print(f"測試: {description}")
            print(f"  URL: {url}")
            
            try:
                response = await client.get(url)
                
                if response is None:
                    print("  ❌ 返回 None\n")
                else:
                    success_count += 1
                    print(f"  ✅ 成功 (Status: {response.status_code}, Size: {len(response.text)} bytes)\n")
                    
            except Exception as e:
                print(f"  ❌ 異常: {e}\n")
        
        await client.close()
        
        print(f"結果: {success_count}/{len(test_urls)} 個請求成功\n")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ HTTP 測試失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ==================== 測試 3: 快速掃描 ====================
async def test_scan(target: str):
    """測試快速掃描功能"""
    print_header(f"🚀 快速掃描測試: {target}")
    
    try:
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        
        coordinator = MultiEngineCoordinator()
        await coordinator.initialize()
        
        print(f"開始掃描: {target}\n")
        
        result = await coordinator.execute_strategy_fast(
            scan_id=f"test_scan_{int(asyncio.get_event_loop().time())}",
            targets=[target],
            max_depth=2
        )
        
        print("✅ 掃描完成")
        print(f"  - 發現資產: {len(result.assets)} 個")
        print(f"  - URLs: {result.summary.urls_found}")
        print(f"  - 表單: {result.summary.forms_found}")
        print(f"  - APIs: {result.summary.apis_found}")
        print(f"  - 執行時間: {result.summary.scan_duration_seconds}s\n")
        
        if result.assets:
            print("前 5 個資產:")
            for i, asset in enumerate(result.assets[:5], 1):
                print(f"  {i}. [{asset.type}] {asset.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 掃描失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ==================== 測試 4: 動態掃描 ====================
async def test_dynamic(target: str):
    """測試動態掃描 (Playwright)"""
    print_header(f"🎭 動態掃描測試: {target}")
    
    try:
        from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
        from services.aiva_common.schemas import Phase1StartPayload
        from pydantic import HttpUrl
        
        orchestrator = ScanOrchestrator()
        
        payload = Phase1StartPayload(
            scan_id=f"dynamic_test_{int(asyncio.get_event_loop().time())}",
            targets=[HttpUrl(target)],
            strategy="DYNAMIC",  # 強制使用動態策略
            phase0_result=None  # 添加必要的參數
        )
        
        print("開始動態掃描 (使用 Playwright)...\n")
        
        result = await orchestrator.execute_phase1(payload)
        
        print("✅ 掃描完成")
        print(f"  - 發現資產: {len(result.assets)} 個")
        print(f"  - URLs: {result.summary.urls_found}")
        print(f"  - 表單: {result.summary.forms_found}")
        print(f"  - 執行時間: {result.summary.scan_duration_seconds}s\n")
        
        if result.assets:
            print("前 10 個資產:")
            for i, asset in enumerate(result.assets[:10], 1):
                print(f"  {i}. [{asset.type}] {asset.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 動態掃描失敗: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ==================== 測試 5: 所有靶場 ====================
async def test_all_targets():
    """測試所有 Docker 靶場"""
    print_header("🎯 測試所有 Docker 靶場")
    
    targets = [
        (LOCALHOST_3000, "Juice Shop (主要)"),
        ("http://localhost:3001", "Juice Shop (副本1)"),
        ("http://localhost:3003", "Juice Shop (副本2)"),
        ("http://localhost:8080", "WebGoat"),
    ]
    
    results = {}
    
    for target, description in targets:
        print_section(f"測試: {description}")
        print(f"URL: {target}\n")
        
        try:
            from services.scan.engines.python_engine.core_crawling_engine.http_client_hi import HiHttpClient
            from services.scan.engines.python_engine.authentication_manager import AuthenticationManager
            from services.scan.engines.python_engine.header_configuration import HeaderConfiguration
            from services.aiva_common.schemas import Authentication
            
            # 創建預設的 Authentication 對象
            default_auth = Authentication(method="none", credentials=None)
            auth = AuthenticationManager(default_auth)
            headers = HeaderConfiguration(None)
            client = HiHttpClient(auth, headers)
            
            response = await client.get(target)
            
            if response:
                print(f"  ✅ 連接成功 (Status: {response.status_code})")
                print(f"  - 響應大小: {len(response.text)} bytes")
                print(f"  - Content-Type: {response.headers.get('content-type', 'N/A')}")
                results[target] = True
            else:
                print("  ❌ 連接失敗")
                results[target] = False
            
            await client.close()
            
        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
            results[target] = False
    
    print_section("📊 測試總結")
    success = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"成功: {success}/{total} 個靶場\n")
    
    for (target, description), status in zip(targets, results.values()):
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {description}: {target}")
    
    return success > 0


# ==================== 測試 6: 完整測試套件 ====================
async def test_full():
    """執行完整測試套件"""
    print_header("🧪 AIVA 完整測試套件")
    
    tests = [
        ("引擎可用性", test_engines),
        ("HTTP 請求", test_http),
        ("快速掃描", lambda: test_scan(LOCALHOST_3000)),
        ("所有靶場", test_all_targets),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print_section(f"執行測試: {test_name}")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ 測試失敗: {e}\n")
            results[test_name] = False
    
    print_header("📊 測試總結")
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name:20} {status}")
    
    success = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n總計: {success}/{total} 個測試通過\n")
    
    return success == total


# ==================== 主程序 ====================
def show_usage():
    """顯示使用說明"""
    print(__doc__)


async def main():
    """主函數"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "engines":
        await test_engines()
    
    elif command == "http":
        await test_http()
    
    elif command == "scan":
        if len(sys.argv) < 3:
            print("❌ 錯誤: 請提供目標 URL")
            print("用法: python aiva_test.py scan <target>")
            return
        target = sys.argv[2]
        await test_scan(target)
    
    elif command == "dynamic":
        if len(sys.argv) < 3:
            print("❌ 錯誤: 請提供目標 URL")
            print("用法: python aiva_test.py dynamic <target>")
            return
        target = sys.argv[2]
        await test_dynamic(target)
    
    elif command == "all-targets":
        await test_all_targets()
    
    elif command == "full":
        await test_full()
    
    else:
        print(f"❌ 未知命令: {command}\n")
        show_usage()


if __name__ == "__main__":
    asyncio.run(main())
