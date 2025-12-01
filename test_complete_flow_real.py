"""
測試完整流程 - 從 AI 命令到 Rust 引擎的真實掃描

目的：驗證整個流程是否真的對靶場發送請求，而不是使用模擬數據
"""

import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.aiva_common.command_center import AICommandCenter
from services.aiva_common.schemas import AICommand, CommandType, CommandPriority
from services.scan import ScanCommandHandler


async def test_real_scan_flow():
    """測試真實掃描流程"""
    print("=" * 80)
    print("測試 AI → Scan → Rust 引擎 完整流程")
    print("=" * 80)
    
    # 1. 初始化命令中心
    command_center = AICommandCenter()
    print("✅ 1. AI 命令中心已初始化")
    
    # 2. 註冊 Scan 模組
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    print("✅ 2. Scan 命令處理器已註冊")
    
    # 3. 檢查 Rust 引擎狀態
    print("\n" + "=" * 80)
    print("檢查 Rust 引擎狀態")
    print("=" * 80)
    
    try:
        from services.scan.engines.rust_engine.python_bridge import rust_info_gatherer
        
        if rust_info_gatherer.is_available():
            print(f"✅ Rust 引擎可用")
            print(f"   二進制路徑: {rust_info_gatherer.rust_binary_path}")
            
            # 檢查是否是模擬版本
            if rust_info_gatherer.__class__.__name__ == "MockRustInfoGatherer":
                print("⚠️  警告: 正在使用模擬的 Rust 引擎！")
                print("   原因: Rust 二進制文件未找到或不可用")
                return False
            else:
                print("✅ 使用真實的 Rust 引擎")
        else:
            print("❌ Rust 引擎不可用")
            return False
    except Exception as e:
        print(f"❌ 無法載入 Rust 引擎: {e}")
        return False
    
    # 4. 創建 Phase 0 掃描命令（靶場地址）
    print("\n" + "=" * 80)
    print("發送 Phase 0 掃描命令到靶場")
    print("=" * 80)
    
    # 使用本地 Juice Shop 靶場
    target_url = "http://localhost:3000"
    
    command = AICommand(
        command_id="scan_real_scan_001",  # 必須以 scan_ 開頭
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={
            "scan_id": "scan_real_scan_001",  # 必須以 scan_ 開頭
            "targets": [target_url],
            "max_depth": 3,
            "timeout": 300  # 必須 >= 60 秒
        },
        priority=CommandPriority.HIGH,
        trace_id="real_scan_trace_001"
    )
    
    print(f"目標: {target_url}")
    print(f"命令 ID: {command.command_id}")
    print(f"掃描深度: {command.payload['max_depth']}")
    
    # 5. 執行命令
    print("\n執行中...")
    print("-" * 80)
    
    result = await command_center.execute(command)
    
    # 6. 分析結果
    print("\n" + "=" * 80)
    print("執行結果分析")
    print("=" * 80)
    
    print(f"狀態: {result.status.value}")
    print(f"成功: {'是' if result.success else '否'}")
    print(f"執行時間: {result.execution_time:.2f} 秒")
    
    if result.success and result.result:
        phase0_result = result.result
        
        # 檢查是否有真實的掃描結果
        assets = phase0_result.get('assets', [])
        summary = phase0_result.get('summary', {})
        
        print(f"\n掃描統計:")
        print(f"   資產數: {len(assets)}")
        print(f"   URL: {summary.get('urls_found', 0)}")
        print(f"   表單: {summary.get('forms_found', 0)}")
        print(f"   API: {summary.get('apis_found', 0)}")
        
        # 檢查關鍵指標
        if len(assets) > 0:
            print("\n✅ 發現了真實資產！")
            print("\n前 5 個資產:")
            for i, asset in enumerate(assets[:5], 1):
                print(f"   {i}. {asset.get('type', 'unknown')}: {asset.get('value', 'N/A')}")
            
            # 驗證是否是真實數據
            print("\n驗證數據真實性:")
            
            # 檢查 1: 資產 URL 是否包含真實靶場地址
            real_assets = [a for a in assets if target_url in str(a.get('value', ''))]
            if real_assets:
                print(f"   ✅ 發現 {len(real_assets)} 個包含靶場地址的資產")
            else:
                print(f"   ⚠️  資產未包含靶場地址，可能是模擬數據")
            
            # 檢查 2: 是否有典型的模擬數據特徵
            mock_keywords = ['DEMO_KEY', 'mock', 'test', 'fake']
            mock_data = any(
                any(keyword in str(asset.get('value', '')).upper() for keyword in mock_keywords)
                for asset in assets
            )
            
            if mock_data:
                print(f"   ⚠️  發現模擬數據特徵（DEMO_KEY/mock/test）")
                return False
            else:
                print(f"   ✅ 無模擬數據特徵")
            
            return True
        else:
            print("\n⚠️  未發現任何資產")
            print("   可能原因:")
            print("   1. 靶場未啟動（請確認 http://localhost:3000 可訪問）")
            print("   2. Rust 引擎連接失敗")
            print("   3. 使用了模擬數據")
            return False
    
    if result.error:
        print(f"\n❌ 錯誤: {result.error}")
        return False
    
    return False


async def check_juiceshop():
    """檢查 Juice Shop 是否運行"""
    print("\n" + "=" * 80)
    print("檢查 Juice Shop 靶場狀態")
    print("=" * 80)
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000", timeout=5)
            print(f"✅ Juice Shop 運行中 (HTTP {response.status_code})")
            return True
    except Exception as e:
        print(f"❌ Juice Shop 未運行: {e}")
        print("\n請先啟動靶場:")
        print("   docker run -p 3000:3000 bkimminich/juice-shop")
        return False


async def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("AIVA 完整流程真實性測試")
    print("=" * 80)
    print("\n目標: 驗證 AI 命令是否真的對靶場發送請求")
    print("方法: 檢查整個流程是否使用真實 Rust 引擎還是模擬數據\n")
    
    # 1. 檢查靶場
    juiceshop_running = await check_juiceshop()
    
    if not juiceshop_running:
        print("\n⚠️  跳過測試（靶場未運行）")
        return
    
    # 2. 測試完整流程
    result = await test_real_scan_flow()
    
    # 3. 最終結論
    print("\n" + "=" * 80)
    print("最終結論")
    print("=" * 80)
    
    if result:
        print("✅ 確認: 完整流程使用真實 Rust 引擎對靶場發送了真實請求")
        print("   - Rust 二進制文件存在且可執行")
        print("   - 掃描結果包含真實靶場數據")
        print("   - 無模擬數據特徵")
    else:
        print("❌ 警告: 完整流程可能使用了模擬數據")
        print("   - 請檢查 Rust 引擎是否正確編譯")
        print("   - 請確認靶場可訪問")
        print("   - 查看上方詳細日誌確定問題")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
