#!/usr/bin/env python3
"""
AIVA 系統診斷工具

位置: service_backbone/performance/diagnose.py
版本: v1.0 (2026-01-21 移入 aiva_core)

功能:
- 引擎可用性檢查 (Python/TypeScript/Rust/Go)
- Docker 靶場狀態檢查
- HTTP 連接測試

使用方式:
    from services.core.aiva_core.service_backbone.performance.diagnose import run_diagnosis
    await run_diagnosis()
"""

import asyncio
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def check_engines():
    """檢查引擎可用性"""
    print_header("🔍 引擎可用性檢查")
    
    try:
        from services.scan.coordinators.multi_engine_coordinator import (
            MultiEngineCoordinator,
        )
        
        coordinator = MultiEngineCoordinator()
        await coordinator.initialize()
        
        available = [e.value for e in coordinator.available_engines]
        
        print(f"✅ 可用引擎 ({len(available)}/4):")
        for engine in available:
            print(f"  - {engine}")
        
        if len(available) < 4:
            print("\n⚠️  不可用引擎:")
            all_engines = {"python", "typescript", "rust", "go"}
            unavailable = all_engines - set(available)
            for engine in unavailable:
                print(f"  - {engine}")
                if engine == "go":
                    print("    提示: 需要編譯 Go 掃描器")
                    print("    命令: cd services/scan/engines/go_engine && go build -o scanner.exe ./cmd/ssrf-scanner")
                elif engine == "typescript":
                    print("    提示: 需要編譯 TypeScript 引擎")
                    print("    命令: cd services/scan/engines/typescript_engine && npm install && npm run build")
        
        return len(available) > 0
        
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return False


def check_docker():
    """檢查 Docker 靶場狀態"""
    print_header("🐳 Docker 靶場檢查")
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print("❌ Docker 未運行或無法訪問")
            return False
        
        lines = result.stdout.strip().split('\n')
        
        # 檢查靶場
        targets = {
            "juice-shop-live": "3000",
            "ecstatic_ritchie": "3001",
            "vigilant_shockley": "3003",
            "laughing_jang": "8080"
        }
        
        found = {}
        for line in lines:
            for name, port in targets.items():
                if name in line and port in line:
                    found[name] = port
        
        print(f"✅ 找到 {len(found)}/4 個靶場:")
        for name, port in found.items():
            print(f"  - {name} (localhost:{port})")
        
        if len(found) < 4:
            print("\n⚠️  缺少的靶場:")
            for name, port in targets.items():
                if name not in found:
                    print(f"  - {name} (應在 localhost:{port})")
        
        return len(found) > 0
        
    except subprocess.TimeoutExpired:
        print("❌ Docker 命令超時")
        return False
    except FileNotFoundError:
        print("❌ Docker 未安裝")
        return False
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return False


async def check_http():
    """測試 HTTP 連接"""
    print_header("🌐 HTTP 連接測試")
    
    try:
        from services.scan.engines.python_engine.authentication_manager import (
            AuthenticationManager,
        )
        from services.scan.engines.python_engine.core_crawling_engine.http_client_hi import (
            HiHttpClient,
        )
        from services.scan.engines.python_engine.header_configuration import (
            HeaderConfiguration,
        )
        
        auth = AuthenticationManager(None)
        headers = HeaderConfiguration(None)
        client = HiHttpClient(auth, headers, requests_per_second=10.0)
        
        targets = [
            "http://localhost:3000",
            "http://example.com",
        ]
        
        success = 0
        for target in targets:
            try:
                response = await client.get(target)
                if response and response.status_code == 200:
                    print(f"✅ {target} (200 OK)")
                    success += 1
                else:
                    print(f"❌ {target} (連接失敗)")
            except Exception as e:
                print(f"❌ {target} ({str(e)[:50]})")
        
        await client.close()
        
        return success > 0
        
    except Exception as e:
        print(f"❌ HTTP 測試失敗: {e}")
        return False


async def full_diagnosis():
    """完整診斷"""
    print_header("🔬 AIVA 系統診斷")
    
    tests = [
        ("Docker 靶場", lambda: asyncio.create_task(asyncio.to_thread(check_docker))),
        ("引擎可用性", check_engines),
        ("HTTP 連接", check_http),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n檢查: {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                results[test_name] = await test_func()
            else:
                task = test_func()
                if isinstance(task, asyncio.Task):
                    results[test_name] = await task
                else:
                    results[test_name] = task
        except Exception as e:
            print(f"❌ 檢查失敗: {e}")
            results[test_name] = False
    
    print_header("📊 診斷總結")
    
    for test_name, result in results.items():
        status = "✅ 正常" if result else "❌ 異常"
        print(f"  {test_name:20} {status}")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n🎉 所有檢查通過！系統就緒。")
    else:
        print("\n⚠️  發現問題，請查看上方詳細信息。")
    
    return all_ok


async def main():
    if len(sys.argv) < 2:
        await full_diagnosis()
    else:
        command = sys.argv[1].lower()
        
        if command == "engines":
            await check_engines()
        elif command == "docker":
            check_docker()
        elif command == "http":
            await check_http()
        else:
            print(f"未知命令: {command}")
            print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
