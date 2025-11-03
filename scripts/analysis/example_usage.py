#!/usr/bin/env python3
"""
AIVA 重複定義修復工具 - 使用示例
展示如何程式化調用修復工具

作者: AIVA 架構團隊
版本: 1.0.0
"""

import asyncio
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def print_header():
    """顯示標題"""
    print("🔧 AIVA 重複定義修復工具 - 使用示例")
    print("=" * 50)

def print_usage_suggestions():
    """顯示使用建議"""
    print("\n🎯 使用建議:")
    print("1. 先運行試運行模式查看修復計劃:")
    print("   python scripts/analysis/duplication_fix_tool.py --phase 1 --dry-run")
    print("   或")
    print("   .\\fix-duplicates.ps1 -Phase 1 -DryRun")
    
    print("\n2. 確認無誤後執行實際修復:")
    print("   python scripts/analysis/duplication_fix_tool.py --phase 1")
    print("   或")
    print("   .\\fix-duplicates.ps1 -Phase 1")
    
    print("\n3. 驗證修復結果:")
    print("   python scripts/analysis/duplication_fix_tool.py --verify")
    print("   或")
    print("   .\\fix-duplicates.ps1 -Verify")

def handle_import_error(e):
    """處理導入錯誤"""
    print(f"❌ 導入錯誤: {e}")
    print("\n💡 解決方案:")
    print("1. 確保在 AIVA 專案根目錄執行")
    print("2. 檢查 aiva_common 模組是否完整")
    print("3. 運行 'pip install -e .' 重新安裝依賴")

async def demo_fix_preview(tool):
    """演示修復預覽"""
    print("\n2. 執行階段一修復預覽")
    result = await tool.execute_phase_1_fixes()
    
    if result.success:
        print("✅ 修復預覽成功")
        print(f"📊 修復項目數量: {result.data.get('total_fixes', 0)}")
        
        fixes_by_type = result.data.get('fixes_by_type', {})
        for fix_type, count in fixes_by_type.items():
            print(f"  - {fix_type}: {count} 項")
    else:
        print("❌ 修復預覽失敗")
        for error in result.errors or []:
            print(f"  錯誤: {error}")

async def demo_verification(tool):
    """演示驗證測試"""
    print("\n3. 執行驗證測試")
    verify_result = await tool.verify_fixes()
    
    if verify_result.success:
        print("✅ 驗證測試通過")
        
        data = verify_result.data or {}
        for test_type, test_result in data.items():
            if isinstance(test_result, dict):
                status = "通過" if test_result.get("success") else "失敗"
                print(f"  - {test_type}: {status}")
    else:
        print("❌ 驗證測試失敗")
        for error in verify_result.errors or []:
            print(f"  錯誤: {error}")

async def main():
    """修復工具使用示例"""
    print_header()
    
    try:
        from scripts.analysis.duplication_fix_tool import AIVADuplicationFixTool
        
        # 創建工具實例
        print("\n1. 創建修復工具實例 (試運行模式)")
        tool = AIVADuplicationFixTool(dry_run=True)
        print("✅ 工具實例創建成功")
        
        # 演示修復預覽
        await demo_fix_preview(tool)
        
        # 演示驗證測試
        await demo_verification(tool)
        
        # 顯示使用建議
        print_usage_suggestions()
        
    except ImportError as e:
        handle_import_error(e)
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        print("\n請檢查錯誤詳情並確保環境配置正確")


if __name__ == "__main__":
    asyncio.run(main())