"""
測試增強分類處理器
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_classifier_processor import EnhancedClassifierProcessor

def test_enhanced_processor():
    """測試增強處理器的模組分組功能"""
    
    print("🚀 測試增強分類處理器...")
    
    # 創建處理器實例
    processor = EnhancedClassifierProcessor(
        work_dir="test_enhanced_output",
        verbose=True
    )
    
    # 使用現有的分類數據
    classification_data_path = "../../../../features_classification/classification_data.json"
    
    # 運行完整管道
    success = processor.run_full_pipeline(
        classification_data_path=classification_data_path
    )
    
    if success:
        print("\n✅ 測試成功！")
        print("📁 檢查輸出目錄: test_enhanced_output")
        print("📄 重點查看:")
        print("  - 4_standardization/module_grouped_capabilities.json")
        print("  - 4_standardization/ai_friendly_capabilities.md")
    else:
        print("\n❌ 測試失敗")
    
    return success

if __name__ == "__main__":
    test_enhanced_processor()