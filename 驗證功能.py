#!/usr/bin/env python3
"""
AIVA 驗證腳本
用於驗證各種功能是否正常工作
"""

import subprocess
import sys
from pathlib import Path

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_internal_modules():
    """測試內部模組執行"""
    print_section("測試 1: 內部模組執行")
    
    tools_dir = Path(__file__).parent / "services" / "core" / "aiva_core" / "internal_exploration" / "python_tools"
    
    print("\n✅ 測試內部模組列表...")
    result = subprocess.run(
        [sys.executable, "aiva_cli_implementation.py", "--list"],
        cwd=tools_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        flow_count = sum(1 for line in lines if line.strip() and line[0].isdigit())
        print(f"   ✅ 成功！找到 {flow_count} 個 Flow")
        return True
    else:
        print(f"   ❌ 失敗: {result.stderr}")
        return False

def test_module_analysis():
    """測試模組分析結果"""
    print_section("測試 2: 外部模組分析結果")
    
    module_analysis_dir = Path(__file__).parent / "module_analysis"
    
    if not module_analysis_dir.exists():
        print("   ⚠️  module_analysis 目錄不存在")
        return False
    
    analysis_files = list(module_analysis_dir.glob("*/analysis_results.json"))
    print(f"\n   ✅ 找到 {len(analysis_files)} 個分析結果:")
    
    for file in analysis_files:
        module_name = file.parent.name
        size = file.stat().st_size
        print(f"      - {module_name}: {size:,} bytes")
    
    return len(analysis_files) > 0

def test_docker_services():
    """測試 Docker 服務"""
    print_section("測試 3: Docker 服務狀態")
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            containers = [line for line in result.stdout.split('\n') if line.strip()]
            print(f"\n   ✅ 找到 {len(containers)} 個運行中的容器:")
            for container in containers:
                print(f"      {container}")
            return True
        else:
            print("   ⚠️  無法獲取 Docker 狀態")
            return False
    except FileNotFoundError:
        print("   ⚠️  Docker 未安裝或未啟動")
        return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  Docker 響應超時")
        return False

def test_juice_shop():
    """測試 Juice Shop 靶場"""
    print_section("測試 4: Juice Shop 靶場連接")
    
    try:
        import requests
        response = requests.get("http://localhost:3000", timeout=3)
        if response.status_code == 200:
            print("\n   ✅ Juice Shop 靶場正常運行 (http://localhost:3000)")
            return True
        else:
            print(f"\n   ⚠️  Juice Shop 返回狀態碼: {response.status_code}")
            return False
    except ImportError:
        print("\n   ⚠️  requests 庫未安裝")
        return False
    except Exception as e:
        print(f"\n   ❌ 無法連接 Juice Shop: {e}")
        return False

def test_classification():
    """測試分類器"""
    print_section("測試 5: 外部模組分類器")
    
    tools_dir = Path(__file__).parent / "services" / "core" / "aiva_core" / "internal_exploration" / "python_tools"
    
    print("\n✅ 測試批次分類器...")
    result = subprocess.run(
        [sys.executable, "aiva_external_module_batch_classifier.py", 
         "-w", "../../..", "-o", "./test_output", "-v"],
        cwd=tools_dir,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if "發現" in result.stdout:
        print(f"   結果: {result.stdout}")
        return True
    else:
        print(f"   ⚠️  {result.stdout}")
        return False

def main():
    print("="*70)
    print("  AIVA 功能驗證腳本")
    print("="*70)
    print("\n正在執行各項測試...")
    
    results = {
        "內部模組執行": test_internal_modules(),
        "外部模組分析": test_module_analysis(),
        "Docker 服務": test_docker_services(),
        "Juice Shop 靶場": test_juice_shop(),
        "分類器功能": test_classification()
    }
    
    # 總結
    print_section("測試總結")
    print()
    for name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {name:20s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n   總計: {passed}/{total} 項通過")
    
    # 建議
    print_section("建議")
    print()
    if results["內部模組執行"]:
        print("   ✅ 可以使用: 執行Flow.bat, 啟動能力選單.bat")
    
    if results["外部模組分析"]:
        print("   ✅ 可以分析外部模組結構")
    
    if not results["Docker 服務"]:
        print("   ⚠️  外部功能模組（function_sqli 等）需要 Docker 運行")
    
    if results["Juice Shop 靶場"]:
        print("   ✅ 可以對 Juice Shop 進行測試")
    else:
        print("   ⚠️  Juice Shop 未運行，某些測試無法執行")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
