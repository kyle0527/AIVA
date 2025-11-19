"""
測試 Phase 0→1→2 完整閉環
"""
import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas import ScanStartPayload
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator


async def test_mvp_loop():
    """測試完整的 MVP 閉環"""
    print("=" * 80)
    print("🚀 開始測試 Phase 0→1→2 MVP 閉環")
    print("=" * 80)
    
    # 創建掃描請求
    request = ScanStartPayload(
        scan_id="scan_test_mvp_001",  # 必須以 scan_ 開頭
        targets=["http://localhost:3000"],
        strategy="deep",  # 啟用動態掃描 (Playwright)
    )
    
    print(f"\n📋 掃描配置:")
    print(f"   目標: {request.targets}")
    print(f"   策略: {request.strategy}")
    print(f"   掃描ID: {request.scan_id}")
    print()
    
    # 創建編排器並執行掃描
    orchestrator = ScanOrchestrator()
    
    try:
        print("⏳ 開始執行掃描...\n")
        result = await orchestrator.execute_scan(request)
        
        print("\n" + "=" * 80)
        print("✅ 掃描完成！")
        print("=" * 80)
        
        # 顯示結果摘要
        print(f"\n📊 掃描結果摘要:")
        print(f"   掃描ID: {result.scan_id}")
        print(f"   狀態: {result.status}")
        print(f"   持續時間: {result.summary.scan_duration_seconds:.2f} 秒")
        print(f"   發現 URL: {result.summary.urls_found}")
        print(f"   發現表單: {result.summary.forms_found}")
        print(f"   發現 API: {result.summary.apis_found}")
        print(f"   總資產數: {len(result.assets)}")
        
        # 顯示部分資產
        if result.assets:
            print(f"\n📦 資產樣本 (前 5 個):")
            for i, asset in enumerate(result.assets[:5], 1):
                print(f"   {i}. [{asset.type}] {asset.value[:80]}")
        
        # 顯示指紋
        if result.fingerprints:
            print(f"\n🔍 技術棧指紋:")
            for tech, confidence in result.fingerprints.items():
                print(f"   - {tech}: {confidence}")
        
        print("\n" + "=" * 80)
        print("🎉 測試完成！請檢查日誌中的 Phase 2 漏洞掃描輸出")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 掃描失敗: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 運行測試
    result = asyncio.run(test_mvp_loop())
    
    # 返回適當的退出碼
    sys.exit(0 if result else 1)
