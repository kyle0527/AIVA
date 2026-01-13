#!/usr/bin/env python3
"""
演示：直接使用檢測器類別（無需 CommandHandler 或 CLI）
這是最底層的使用方式，適合：
1. 自定義測試腳本
2. 快速原型開發
3. 深度定制化需求
"""
import asyncio
import sys
from pathlib import Path

# 添加路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("🧪 測試：直接使用檢測器類別（無 CommandHandler / CLI）")
print("="*70)
print()


async def test_xss_detector_direct():
    """直接使用 XSS 檢測器"""
    print("【測試 1】直接使用 TraditionalXssDetector")
    print("-"*70)
    
    from services.features.function_xss.traditional_detector import TraditionalXssDetector
    from services.features.function_xss.payload_generator import XssPayloadGenerator
    from services.aiva_common.schemas.tasks import (
        FunctionTaskPayload, 
        FunctionTaskTarget,
        FunctionTaskTestConfig
    )
    import httpx
    
    # 創建任務
    task = FunctionTaskPayload(
        task_id="direct_test_001",
        scan_id="direct_scan_001",
        target=FunctionTaskTarget(
            url="http://localhost:3000/search",
            parameter="q",
            method="GET",
            parameter_location="query"
        ),
        test_config=FunctionTaskTestConfig(timeout=30.0)
    )
    
    print(f"📝 目標: {task.target.url}")
    print(f"   參數: {task.target.parameter}")
    print()
    
    # 生成 payloads
    generator = XssPayloadGenerator()
    payloads = generator.generate_basic_payloads()[:5]  # 只測試前5個
    
    print(f"🔧 生成 {len(payloads)} 個測試 payloads")
    print()
    
    # 直接使用檢測器
    try:
        detector = TraditionalXssDetector(task, timeout=30)
        results = await detector.execute(payloads)
        
        print(f"✅ 檢測完成")
        print(f"   發現漏洞數: {len(results)}")
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n   漏洞 #{i}:")
            print(f"      Payload: {result.payload[:50]}...")
            print(f"      狀態碼: {result.response_status}")
            
    except Exception as e:
        print(f"❌ 檢測失敗: {e}")
    
    print()


async def test_sqli_detector_direct():
    """直接使用 SQLi 檢測器"""
    print("【測試 2】直接使用 SqliDetector")
    print("-"*70)
    
    from services.features.function_sqli.detector.sqli_detector import SqliDetector
    from services.aiva_common.schemas.tasks import (
        FunctionTaskPayload,
        FunctionTaskTarget
    )
    import httpx
    
    # 創建任務
    task = FunctionTaskPayload(
        task_id="direct_sqli_001",
        scan_id="direct_scan_002",
        target=FunctionTaskTarget(
            url="http://localhost:3000/api/user",
            parameter="id",
            method="GET",
            parameter_location="query"
        )
    )
    
    print(f"📝 目標: {task.target.url}")
    print(f"   參數: {task.target.parameter}")
    print()
    
    try:
        async with httpx.AsyncClient() as client:
            # 創建檢測器
            detector = SqliDetector(
                task=task,
                client=client,
                config=None  # 使用默認配置
            )
            
            # 執行檢測（會自動使用所有引擎）
            results = await detector.detect()
            
            print(f"✅ 檢測完成")
            print(f"   發現漏洞數: {len(results)}")
            
            for i, result in enumerate(results[:3], 1):
                print(f"\n   漏洞 #{i}:")
                print(f"      引擎: {result.engine_name}")
                print(f"      Payload: {result.payload[:50]}...")
                
    except Exception as e:
        print(f"❌ 檢測失敗: {e}")
    
    print()


async def test_bizlogic_tester_direct():
    """直接使用業務邏輯測試器"""
    print("【測試 3】直接使用 PriceManipulationTester")
    print("-"*70)
    
    from services.features.function_bizlogic.price_manipulation_tester import PriceManipulationTester
    
    print(f"📝 目標: http://localhost:3000/api/basket/1")
    print(f"   測試: 價格操控檢測")
    print()
    
    try:
        # 直接創建測試器
        tester = PriceManipulationTester(
            base_url="http://localhost:3000",
            auth_token="your_token_here",
            user_role="customer"
        )
        
        # 執行測試
        results = await tester.run_all_tests(
            endpoint="/api/basket/1"
        )
        
        print(f"✅ 檢測完成")
        print(f"   發現問題數: {len(results)}")
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n   問題 #{i}:")
            print(f"      類型: {result.get('type')}")
            print(f"      嚴重性: {result.get('severity')}")
            
    except Exception as e:
        print(f"❌ 檢測失敗: {e}")
    
    print()


async def main():
    """主測試"""
    
    # 測試 1：XSS
    await test_xss_detector_direct()
    
    # 測試 2：SQLi
    await test_sqli_detector_direct()
    
    # 測試 3：業務邏輯
    await test_bizlogic_tester_direct()
    
    print("="*70)
    print("🎉 測試完成")
    print()
    print("💡 總結：即使沒有 CommandHandler 和 CLI，也能直接使用檢測器！")
    print()
    print("📊 三種執行方式對比：")
    print()
    print("   1️⃣  CommandHandler  - AI 核心統一調度（推薦）")
    print("       優點：統一接口、異步管理、錯誤處理")
    print("       缺點：需要註冊處理器")
    print()
    print("   2️⃣  CLI 接口        - 命令行執行（部分支持）")
    print("       優點：簡單、獨立")
    print("       缺點：需要 subprocess、性能較差")
    print()
    print("   3️⃣  直接導入類別     - 最底層方式（總是可用）")
    print("       優點：完全控制、靈活")
    print("       缺點：需要手動管理異步、錯誤處理")
    print()
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
