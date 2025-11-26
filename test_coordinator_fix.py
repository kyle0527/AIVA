"""
測試協調器修復 - 驗證各引擎收到正確參數

測試目標:
1. 協調器正確轉換 AI 簡化指令
2. 各引擎只收到使用指南要求的參數
3. 靶場能收到真實請求
"""

import asyncio
import json
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator


async def test_coordinator_parameter_translation():
    """測試協調器參數轉換功能"""
    
    coordinator = MultiEngineCoordinator()
    
    # AI 簡化指令
    ai_command = {
        "scan_id": "test_scan_001",
        "targets": ["http://localhost:3000"],
        "depth": 3,
        "timeout": 10,
    }
    
    print("=" * 60)
    print("AI 簡化指令:")
    print(json.dumps(ai_command, indent=2, ensure_ascii=False))
    print()
    
    # 測試各引擎轉換
    engines = ["go", "rust", "python", "typescript"]
    
    for engine in engines:
        print(f"\n{'='*60}")
        print(f"引擎: {engine.upper()}")
        print("-" * 60)
        
        engine_options = coordinator._translate_command_for_engine(
            engine, ai_command, max_urls=100
        )
        
        print(f"轉換後參數:")
        print(json.dumps(engine_options, indent=2, ensure_ascii=False))
        
        # 檢查參數數量
        param_count = len(engine_options)
        print(f"\n參數數量: {param_count}")
        
        # 檢查是否符合使用指南
        if engine == "go":
            required = ["scan_id", "concurrency", "timeout"]
            print(f"使用指南要求: {required}")
            missing = [k for k in required if k not in engine_options]
            extra = [k for k in engine_options if k not in required + ["targets"]]
            
        elif engine == "rust":
            required = ["mode", "timeout"]
            print(f"使用指南要求: {required}")
            missing = [k for k in required if k not in engine_options]
            extra = [k for k in engine_options if k not in required + ["scan_id"]]
            
        elif engine == "python":
            required = ["strategy", "max_depth", "max_pages", "timeout"]
            print(f"使用指南要求: {required}")
            missing = [k for k in required if k not in engine_options]
            extra = [k for k in engine_options if k not in required + ["scan_id"]]
            
        elif engine == "typescript":
            required = ["max_depth", "timeout"]
            print(f"使用指南要求: {required}")
            missing = [k for k in required if k not in engine_options]
            extra = [k for k in engine_options if k not in required + ["scan_id"]]
        
        if missing:
            print(f"❌ 缺少參數: {missing}")
        if extra:
            print(f"❌ 多餘參數: {extra}")
        if not missing and not extra:
            print("✅ 參數完全符合使用指南")
    
    print("\n" + "=" * 60)


async def test_go_engine_with_juiceshop():
    """測試 Go 引擎攻擊 Juice Shop"""
    
    print("\n" + "=" * 60)
    print("測試 Go 引擎攻擊 Juice Shop")
    print("=" * 60)
    
    coordinator = MultiEngineCoordinator()
    
    result = await coordinator.execute_phase1(
        scan_id="test_go_juiceshop",
        targets=["http://localhost:3000"],
        selected_engines=["go"],
        max_depth=2,
        max_urls=50
    )
    
    print(f"\n掃描結果:")
    print(f"  狀態: {result.status}")
    print(f"  資產數: {len(result.assets)}")
    print(f"  執行時間: {result.execution_time:.2f}s")
    
    if result.engine_results:
        for engine, status in result.engine_results.items():
            print(f"  {engine}: {status}")
    
    if result.error_info:
        print(f"  錯誤: {result.error_info}")
    
    print("\n⚠️ 請檢查 Juice Shop 日誌,確認收到 Go 引擎的請求!")


async def main():
    """主測試流程"""
    
    # 測試 1: 參數轉換驗證
    await test_coordinator_parameter_translation()
    
    # 測試 2: Go 引擎實戰測試
    await test_go_engine_with_juiceshop()


if __name__ == "__main__":
    asyncio.run(main())
