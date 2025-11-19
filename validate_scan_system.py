#!/usr/bin/env python3
"""
掃描驗證腳本 - 根據使用者手冊進行實際測試
測試目標: Docker 中的靶場 (Juice Shop, WebGoat)
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import TwoPhaseScanOrchestrator
from services.aiva_common.mq import RabbitBroker
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


async def validate_scan_system():
    """驗證掃描系統 - 按照使用者手冊流程"""
    
    print("=" * 80)
    print("🧪 AIVA Scan 模組驗證測試")
    print("=" * 80)
    
    # 測試目標（從 Docker 截圖）
    test_targets = {
        "Juice Shop (主)": "http://localhost:3000",
        "WebGoat": "http://localhost:8080",
        "Juice Shop 2": "http://localhost:3001",
        "Juice Shop 3": "http://localhost:3003",
    }
    
    print("\n📋 測試目標:")
    for name, url in test_targets.items():
        print(f"  - {name}: {url}")
    
    # 選擇測試目標
    target_name = "Juice Shop (主)"
    target_url = test_targets[target_name]
    
    print(f"\n🎯 選擇測試目標: {target_name}")
    print(f"   URL: {target_url}")
    
    broker = None
    try:
        # 步驟 1: 連接 RabbitMQ
        print("\n" + "─" * 80)
        print("📍 步驟 1: 連接 RabbitMQ")
        print("─" * 80)
        
        rabbitmq_url = "amqp://guest:guest@localhost:5672/"
        print(f"   連接 URL: {rabbitmq_url}")
        
        broker = RabbitBroker(rabbitmq_url)
        await broker.connect()
        print("   ✅ RabbitMQ 連接成功")
        
        # 步驟 2: 創建兩階段掃描器
        print("\n" + "─" * 80)
        print("📍 步驟 2: 初始化兩階段掃描器")
        print("─" * 80)
        
        orchestrator = TwoPhaseScanOrchestrator(broker)
        print(f"   Phase0 超時: {orchestrator.phase0_timeout}秒 ({orchestrator.phase0_timeout/60:.1f}分鐘)")
        print(f"   Phase1 超時: {orchestrator.phase1_timeout}秒 ({orchestrator.phase1_timeout/60:.1f}分鐘)")
        print("   ✅ 掃描器初始化完成")
        
        # 步驟 3: 執行兩階段掃描
        print("\n" + "─" * 80)
        print("📍 步驟 3: 執行兩階段掃描")
        print("─" * 80)
        
        print("\n🚀 開始掃描流程...")
        print("   Phase0: Rust 快速偵察 (5-10 分鐘)")
        print("   AI 決策: 是否需要 Phase1")
        print("   Phase1: 多引擎深度掃描 (10-30 分鐘，按需)")
        
        result = await orchestrator.execute_two_phase_scan(
            targets=[target_url],
            trace_id="validation-test-001",
            max_depth=3,
            max_urls=1000
        )
        
        # 步驟 4: 分析結果
        print("\n" + "=" * 80)
        print("✅ 掃描完成 - 結果分析")
        print("=" * 80)
        
        print(f"\n📊 基本資訊:")
        print(f"   掃描 ID: {result.scan_id}")
        print(f"   最終狀態: {result.status}")
        print(f"   總執行時間: {result.total_execution_time:.2f} 秒 ({result.total_execution_time/60:.1f} 分鐘)")
        
        # Phase0 結果
        if result.phase0_result:
            print(f"\n📋 Phase0 結果分析:")
            print(f"   狀態: {result.phase0_result.status}")
            print(f"   執行時間: {result.phase0_result.execution_time:.2f} 秒")
            print(f"   發現資產: {len(result.phase0_result.assets)} 個")
            
            summary = result.phase0_result.summary
            print(f"\n   掃描摘要:")
            print(f"     - URLs: {summary.urls_found}")
            print(f"     - 表單: {summary.forms_found}")
            print(f"     - APIs: {summary.apis_found}")
            print(f"     - 掃描時長: {summary.scan_duration_seconds:.0f} 秒")
            
            if result.phase0_result.fingerprints:
                fp = result.phase0_result.fingerprints
                print(f"\n   技術棧指紋:")
                if fp.web_server:
                    print(f"     - Web Server: {fp.web_server}")
                if fp.frameworks:
                    print(f"     - Frameworks: {fp.frameworks}")
                if fp.cms:
                    print(f"     - CMS: {fp.cms}")
                if fp.technologies:
                    print(f"     - Technologies: {fp.technologies}")
            
            # 顯示前 5 個資產
            if result.phase0_result.assets:
                print(f"\n   資產清單 (前 5 個):")
                for i, asset in enumerate(result.phase0_result.assets[:5], 1):
                    params_str = f", 參數: {asset.parameters}" if asset.parameters else ""
                    form_str = " [表單]" if asset.has_form else ""
                    print(f"     [{i}] {asset.type}: {asset.value}{form_str}{params_str}")
        
        # AI 決策
        print(f"\n🤖 AI 決策:")
        print(f"   需要 Phase1: {'是' if result.need_phase1 else '否'}")
        if result.decision_reasoning:
            print(f"   決策理由: {result.decision_reasoning}")
        
        # Phase1 結果
        if result.phase1_result:
            print(f"\n📋 Phase1 結果分析:")
            print(f"   狀態: {result.phase1_result.status}")
            print(f"   執行時間: {result.phase1_result.execution_time:.2f} 秒")
            print(f"   發現資產: {len(result.phase1_result.assets)} 個")
            
            summary = result.phase1_result.summary
            print(f"\n   掃描摘要:")
            print(f"     - URLs: {summary.urls_found}")
            print(f"     - 表單: {summary.forms_found}")
            print(f"     - APIs: {summary.apis_found}")
            
            print(f"\n   引擎執行結果:")
            for engine, engine_result in result.phase1_result.engine_results.items():
                status_icon = "✅" if engine_result.get("status") == "completed" else "❌"
                print(f"     {status_icon} {engine}: {engine_result.get('status')} ({engine_result.get('findings', 0)} 發現)")
            
            # 顯示前 10 個資產
            if result.phase1_result.assets:
                print(f"\n   資產清單 (前 10 個):")
                for i, asset in enumerate(result.phase1_result.assets[:10], 1):
                    params_str = f", 參數: {asset.parameters}" if asset.parameters else ""
                    form_str = " [表單]" if asset.has_form else ""
                    print(f"     [{i}] {asset.type}: {asset.value}{form_str}{params_str}")
        
        # 步驟 5: 驗證結果
        print("\n" + "─" * 80)
        print("📍 步驟 5: 驗證測試結果")
        print("─" * 80)
        
        validation_passed = True
        
        # 檢查點 1: 掃描成功完成
        if result.status == "success":
            print("   ✅ 掃描成功完成")
        else:
            print(f"   ❌ 掃描狀態異常: {result.status}")
            validation_passed = False
        
        # 檢查點 2: Phase0 執行
        if result.phase0_result and result.phase0_result.status == "success":
            print("   ✅ Phase0 執行成功")
        else:
            print("   ❌ Phase0 執行失敗")
            validation_passed = False
        
        # 檢查點 3: 發現資產
        total_assets = len(result.phase0_result.assets if result.phase0_result else [])
        if result.phase1_result:
            total_assets += len(result.phase1_result.assets)
        
        if total_assets > 0:
            print(f"   ✅ 發現資產: {total_assets} 個")
        else:
            print("   ⚠️  未發現任何資產")
            validation_passed = False
        
        # 檢查點 4: 執行時間合理
        if result.total_execution_time > 0:
            print(f"   ✅ 執行時間合理: {result.total_execution_time:.2f} 秒")
        else:
            print("   ❌ 執行時間異常")
            validation_passed = False
        
        # 最終結論
        print("\n" + "=" * 80)
        if validation_passed:
            print("🎉 驗證通過 - 掃描系統運作正常")
        else:
            print("⚠️  驗證失敗 - 請檢查上述錯誤")
        print("=" * 80)
        
        return result, validation_passed
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 驗證測試失敗")
        print("=" * 80)
        logger.exception(f"錯誤: {e}")
        print(f"\n錯誤詳情: {e}")
        
        import traceback
        print("\n堆疊追蹤:")
        traceback.print_exc()
        
        return None, False
        
    finally:
        if broker:
            await broker.close()
            print("\n🔌 已關閉 MQ 連接")


async def main():
    """主函數"""
    print("\n⏰ 測試開始時間:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        result, passed = await validate_scan_system()
        
        print("\n⏰ 測試結束時間:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return 0 if passed else 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 用戶中斷測試")
        return 130


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
