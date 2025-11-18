#!/usr/bin/env python3
"""
AIVA 圖表產出清理工具
自動清理 diagram_auto_composer.py 產生的冗餘檔案
"""

import os
import sys
import glob
import shutil
from pathlib import Path
from typing import List

class DiagramOutputCleaner:
    """圖表產出清理器"""
    
    def __init__(self, output_dir: str = "_out"):
        self.output_dir = Path(output_dir)
        self.architecture_dir = self.output_dir / "architecture_diagrams"
        
    def cleanup_auto_generated_diagrams(self, module_name: str = None) -> dict:
        """清理自動產生的個別組件圖"""
        
        if not self.architecture_dir.exists():
            return {"error": "Architecture diagrams directory not found"}
        
        # 定義要清理的檔案模式
        cleanup_patterns = [
            "aiva_*_Function_*.mmd",
            "aiva_*_Module.mmd",
        ]
        
        if module_name:
            # 只清理特定模組的檔案
            cleanup_patterns = [f"aiva_{module_name}*Function*.mmd", 
                             f"aiva_{module_name}*Module.mmd"]
        
        cleaned_files = []
        preserved_files = []
        
        # 掃描和清理檔案
        for pattern in cleanup_patterns:
            files_to_clean = list(self.architecture_dir.glob(pattern))
            for file_path in files_to_clean:
                try:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
                except Exception as e:
                    print(f"⚠️ 無法刪除 {file_path}: {e}")
        
        # 列出保留的重要檔案
        important_patterns = [
            "*_INTEGRATED_ARCHITECTURE.mmd",
            "*_AUTO_INTEGRATED.mmd", 
            "*.json",
            "*_ANALYSIS.md"
        ]
        
        for pattern in important_patterns:
            preserved_files.extend([str(f) for f in self.output_dir.rglob(pattern)])
        
        return {
            "cleaned_count": len(cleaned_files),
            "cleaned_files": cleaned_files,
            "preserved_count": len(preserved_files),
            "preserved_files": preserved_files
        }
    
    def backup_important_files(self, backup_dir: str = "backup") -> List[str]:
        """備份重要的整合圖檔"""
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        important_files = [
            "*_INTEGRATED_ARCHITECTURE.mmd",
            "*_ARCHITECTURE_ANALYSIS.md",
            "*_diagram_classification.json"
        ]
        
        backed_up = []
        for pattern in important_files:
            files = list(self.output_dir.rglob(pattern))
            for file_path in files:
                backup_file = backup_path / file_path.name
                try:
                    shutil.copy2(file_path, backup_file)
                    backed_up.append(str(backup_file))
                except Exception as e:
                    print(f"⚠️ 備份失敗 {file_path}: {e}")
        
        return backed_up
    
    def get_cleanup_summary(self) -> dict:
        """獲取清理前的統計摘要"""
        
        if not self.architecture_dir.exists():
            return {"error": "Directory not found"}
        
        total_files = len(list(self.architecture_dir.glob("*.mmd")))
        auto_generated = len(list(self.architecture_dir.glob("aiva_*.mmd"))) 
        important_files = len(list(self.output_dir.rglob("*INTEGRATED*.mmd")))
        
        return {
            "total_diagram_files": total_files,
            "auto_generated_files": auto_generated,
            "important_integrated_files": important_files,
            "cleanup_recommendation": auto_generated > 50
        }

def main():
    """主要執行邏輯"""
    
    cleaner = DiagramOutputCleaner()
    
    # 檢查當前狀況
    print("🔍 檢查圖表檔案狀況...")
    summary = cleaner.get_cleanup_summary()
    
    if "error" in summary:
        print(f"❌ {summary['error']}")
        return
    
    print("📊 統計資訊：")
    print(f"   總圖表檔案: {summary['total_diagram_files']}")
    print(f"   自動產生檔案: {summary['auto_generated_files']}")
    print(f"   重要整合檔案: {summary['important_integrated_files']}")
    
    if not summary['cleanup_recommendation']:
        print("✅ 檔案數量合理，無需清理")
        return
    
    # 執行備份
    print("\n📋 備份重要檔案...")
    backed_up = cleaner.backup_important_files()
    print(f"✅ 已備份 {len(backed_up)} 個重要檔案")
    
    # 詢問是否執行清理
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        confirm = "y"
    else:
        confirm = input(f"\n🗑️ 發現 {summary['auto_generated_files']} 個自動產生檔案，是否清理？(y/n): ")
    
    if confirm.lower() == 'y':
        # 執行清理
        print("\n🧹 執行清理...")
        result = cleaner.cleanup_auto_generated_diagrams()
        
        print("✅ 清理完成！")
        print(f"   已刪除: {result['cleaned_count']} 個檔案")
        print(f"   保留: {result['preserved_count']} 個重要檔案")
        
        # 顯示保留的檔案
        if result['preserved_files']:
            print("\n📋 保留的重要檔案：")
            for file_path in result['preserved_files'][:5]:  # 只顯示前5個
                print(f"   ✅ {Path(file_path).name}")
            if len(result['preserved_files']) > 5:
                print(f"   ... 還有 {len(result['preserved_files'])-5} 個檔案")
        
    else:
        print("❌ 清理作業已取消")

if __name__ == "__main__":
    main()