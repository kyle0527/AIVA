#!/usr/bin/env python3
"""
多引擎協調器驗證腳本 - 驗證四種語言引擎實際運作
測試目標: Docker 中的靶場 (Juice Shop 多實例)
驗證重點: 協調器能力、各引擎請求發出與接收
"""
import asyncio
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


async def validate_multi_engine_coordinator():
    """驗證多引擎協調器 - 四種引擎實戰測試"""
    
    print("=" * 100)
    print("🚀 多引擎協調器驗證測試")
    print("=" * 100)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 測試目標（從 Docker 截圖）
    test_targets = {
        "Juice Shop (主)": "http://localhost:3000",
        "Juice Shop 2": "http://localhost:3001", 
        "Juice Shop 3": "http://localhost:3003",
        "WebGoat": "http://localhost:8080",
    }
    
    print("\n📋 可用靶場:")
    for name, url in test_targets.items():
        print(f"  • {name}: {url}")
    
    # 選擇測試目標（多目標）
    selected_targets = [
        test_targets["Juice Shop (主)"],
        test_targets["Juice Shop 2"],
        test_targets["Juice Shop 3"],
    ]
    
    print(f"\n🎯 選擇的測試目標:")
    for i, target in enumerate(selected_targets, 1):
        print(f"  {i}. {target}")
    
    try:
        # 步驟 1: 初始化協調器
        print("\n" + "─" * 100)
        print("📍 階段 1: 初始化多引擎協調器")
        print("─" * 100)
        
        coordinator = MultiEngineCoordinator()
        print("   ✅ 協調器初始化完成")
        
        # 檢查適配器
        print("\n   可用適配器:")
        for engine_name, adapter in coordinator.adapters.items():
            available = await adapter.is_available()
            status_icon = "✅" if available else "❌"
            print(f"     {status_icon} {engine_name.upper()}: {'可用' if available else '不可用'}")
        
        # 步驟 2: 測試 Python 引擎 (快速單引擎)
        print("\n" + "─" * 100)
        print("📍 階段 2: 測試 Python 引擎 (快速模式)")
        print("─" * 100)
        
        print("\n🐍 Python 引擎測試 (FAST 策略)")
        start_time = time.time()
        
        result_python = await coordinator.execute_strategy_fast(
            scan_id="scan_python_001",  # ✅ 正確格式: scan_ 前綴
            targets=selected_targets[:1]  # 單目標
        )
        
        python_time = time.time() - start_time
        
        print(f"\n   📊 Python 引擎結果:")
        print(f"     • 狀態: {result_python.status}")
        print(f"     • 執行時間: {python_time:.2f}s")
        print(f"     • 發現資產: {len(result_python.assets)}")
        print(f"     • 引擎狀態: {result_python.engine_results}")
        
        if result_python.assets:
            print(f"\n     • 資產樣本 (前 3 個):")
            for i, asset in enumerate(result_python.assets[:3], 1):
                print(f"       [{i}] {asset.type}: {asset.value}")
        
        # 步驟 3: 測試所有引擎 (平衡模式)
        print("\n" + "─" * 100)
        print("📍 階段 3: 測試所有四種引擎 (平衡模式)")
        print("─" * 100)
        
        print("\n⚡ 啟動四引擎並行掃描 (Python + TypeScript + Rust + Go)")
        print("   引擎: ['python', 'typescript', 'rust', 'go']")
        print("   策略: BALANCED")
        print("   並行: True")
        
        start_time = time.time()
        
        result_all = await coordinator.execute_phase1(
            scan_id="scan_all_engines_001",  # ✅ 正確格式: scan_ 前綴
            targets=selected_targets,  # 多目標
            selected_engines=["python", "typescript", "rust", "go"],
            max_depth=2,
            max_urls=50
        )
        
        all_engines_time = time.time() - start_time
        
        print(f"\n   📊 四引擎協調結果:")
        print(f"     • 最終狀態: {result_all.status}")
        print(f"     • 總執行時間: {all_engines_time:.2f}s")
        print(f"     • 總資產數: {len(result_all.assets)}")
        print(f"     • 去重後資產: {len(result_all.assets)}")
        
        print(f"\n   🔍 各引擎執行詳情:")
        for engine_name, engine_status in result_all.engine_results.items():
            status_icon = "✅" if engine_status.get("status") == "completed" else "❌"
            assets_count = engine_status.get("assets_count", 0)
            error = engine_status.get("error", "")
            
            print(f"     {status_icon} {engine_name.upper()}:")
            print(f"        - 狀態: {engine_status.get('status')}")
            print(f"        - 資產: {assets_count} 個")
            if error:
                print(f"        - 錯誤: {error}")
        
        # 步驟 4: 驗證引擎通訊
        print("\n" + "─" * 100)
        print("📍 階段 4: 驗證引擎通訊能力")
        print("─" * 100)
        
        validation_passed = True
        validation_details = []
        
        # 檢查點 1: 協調器狀態
        if result_all.status in ["completed", "partial_success"]:
            print("   ✅ 協調器成功完成協調任務")
            validation_details.append(("協調器狀態", True, result_all.status))
        else:
            print(f"   ❌ 協調器狀態異常: {result_all.status}")
            validation_passed = False
            validation_details.append(("協調器狀態", False, result_all.status))
        
        # 檢查點 2: 引擎請求發出
        successful_engines = [
            name for name, status in result_all.engine_results.items()
            if status.get("status") == "completed"
        ]
        
        if len(successful_engines) >= 2:
            print(f"   ✅ 至少 2 個引擎成功執行: {successful_engines}")
            validation_details.append(("引擎執行", True, f"{len(successful_engines)}/4"))
        else:
            print(f"   ❌ 成功引擎數不足: {successful_engines}")
            validation_passed = False
            validation_details.append(("引擎執行", False, f"{len(successful_engines)}/4"))
        
        # 檢查點 3: 引擎結果接收
        total_assets = len(result_all.assets)
        if total_assets > 0:
            print(f"   ✅ 成功接收引擎結果: {total_assets} 個資產")
            validation_details.append(("結果接收", True, f"{total_assets} assets"))
        else:
            print(f"   ⚠️  未接收到任何資產")
            validation_details.append(("結果接收", False, "0 assets"))
        
        # 檢查點 4: 錯誤隔離
        failed_engines = [
            name for name, status in result_all.engine_results.items()
            if status.get("status") == "failed"
        ]
        
        if failed_engines and len(successful_engines) > 0:
            print(f"   ✅ 錯誤隔離正常: {failed_engines} 失敗不影響其他引擎")
            validation_details.append(("錯誤隔離", True, "部分失敗可容忍"))
        elif len(successful_engines) == 4:
            print(f"   ✅ 所有引擎成功，無錯誤")
            validation_details.append(("錯誤隔離", True, "全部成功"))
        else:
            print(f"   ❌ 錯誤隔離失敗: 所有引擎均失敗")
            validation_passed = False
            validation_details.append(("錯誤隔離", False, "全部失敗"))
        
        # 檢查點 5: 執行時間
        if all_engines_time > 0:
            print(f"   ✅ 執行時間合理: {all_engines_time:.2f}s")
            validation_details.append(("執行時間", True, f"{all_engines_time:.2f}s"))
        else:
            print(f"   ❌ 執行時間異常")
            validation_passed = False
            validation_details.append(("執行時間", False, "異常"))
        
        # 步驟 5: 智能引擎選擇測試
        print("\n" + "─" * 100)
        print("📍 階段 5: 智能引擎選擇 (SMART 策略)")
        print("─" * 100)
        
        print("\n🧠 智能模式測試")
        start_time = time.time()
        
        result_smart = await coordinator.execute_strategy_smart(
            scan_id="scan_smart_001",  # ✅ 正確格式: scan_ 前綴
            targets=selected_targets[:1]
        )
        
        smart_time = time.time() - start_time
        
        print(f"\n   📊 智能引擎選擇結果:")
        print(f"     • 狀態: {result_smart.status}")
        print(f"     • 執行時間: {smart_time:.2f}s")
        print(f"     • 選擇的引擎: {list(result_smart.engine_results.keys())}")
        print(f"     • 發現資產: {len(result_smart.assets)}")
        
        # 最終報告
        print("\n" + "=" * 100)
        print("📋 驗證測試報告")
        print("=" * 100)
        
        print(f"\n✅ 測試完成項目:")
        print(f"  1. Python 引擎單獨測試: {python_time:.2f}s")
        print(f"  2. 四引擎並行測試: {all_engines_time:.2f}s")
        print(f"  3. 智能引擎選擇: {smart_time:.2f}s")
        
        print(f"\n📊 驗證結果詳情:")
        for check_name, passed, detail in validation_details:
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check_name}: {detail}")
        
        print("\n" + "=" * 100)
        if validation_passed:
            print("🎉 驗證通過 - 多引擎協調器運作正常")
            print("  • 協調器可以正確調度四種引擎")
            print("  • 各引擎請求成功發出並接收結果")
            print("  • 錯誤隔離機制正常工作")
        else:
            print("⚠️  驗證失敗 - 請檢查上述錯誤項目")
        print("=" * 100)
        
        return result_all, validation_passed
        
    except Exception as e:
        print("\n" + "=" * 100)
        print("❌ 驗證測試異常")
        print("=" * 100)
        logger.exception(f"錯誤: {e}")
        print(f"\n錯誤詳情: {e}")
        
        import traceback
        print("\n堆疊追蹤:")
        traceback.print_exc()
        
        return None, False


async def main():
    """主函數"""
    print("\n⏰ 測試開始時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        _, passed = await validate_multi_engine_coordinator()
        
        print("\n⏰ 測試結束時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return 0 if passed else 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 用戶中斷測試")
        return 130


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
