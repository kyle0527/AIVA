"""
AIVA 分析路徑驗證工具
用途: 驗證新的 analysis_data 路徑結構是否正確設置
作者: AIVA Team
日期: 2025-12-15
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any

class AnalysisPathValidator:
    """分析路徑結構驗證器"""
    
    def __init__(self, project_root=None):
        if project_root is None:
            # 從當前文件位置推導專案根目錄
            self.root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        else:
            self.root = Path(project_root)
        
        self.analysis_data_root = self.root / "services" / "integration" / "analysis_data"
        self.modules = ["core", "features", "scan", "integration"]
        self.categories = ["capabilities", "flows", "classifications"]
    
    def validate_structure(self):
        """驗證目錄結構"""
        print("=" * 60)
        print("AIVA 分析路徑結構驗證")
        print("=" * 60)
        print()
        
        results: dict[str, Any] = {
            "root_exists": False,
            "modules_valid": {},
            "files_found": {},
            "errors": []
        }
        
        # 檢查根目錄
        if self.analysis_data_root.exists():
            print(f"✓ 根目錄存在: {self.analysis_data_root}")
            results["root_exists"] = True
        else:
            print(f"✗ 根目錄不存在: {self.analysis_data_root}")
            results["errors"].append("根目錄不存在")
            return results
        
        print()
        
        # 檢查各模組
        for module in self.modules:
            print(f"檢查模組: {module}")
            module_dir = self.analysis_data_root / module
            
            if not module_dir.exists():
                print("  ✗ 模組目錄不存在")
                results["modules_valid"][module] = False
                continue
            
            print("  ✓ 模組目錄存在")
            results["modules_valid"][module] = True
            results["files_found"][module] = {}
            
            # 檢查各類別
            for category in self.categories:
                category_dir = module_dir / category
                
                if not category_dir.exists():
                    print(f"    ✗ {category}/ 目錄不存在")
                    results["files_found"][module][category] = []
                    continue
                
                # 列出文件
                files = list(category_dir.glob("*.json"))
                results["files_found"][module][category] = [f.name for f in files]
                
                if files:
                    print(f"    ✓ {category}/ 包含 {len(files)} 個文件")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        print(f"      - {f.name} ({size_kb:.2f} KB)")
                else:
                    print(f"    ⚠ {category}/ 目錄為空")
            
            print()
        
        return results
    
    def validate_latest_files(self):
        """驗證 latest_* 文件"""
        print("=" * 60)
        print("Latest 文件驗證")
        print("=" * 60)
        print()
        
        results: dict[str, dict[str, Any]] = {}
        
        for module in self.modules:
            print(f"模組: {module}")
            results[module] = {}
            
            for category in self.categories:
                latest_file = (
                    self.analysis_data_root 
                    / module 
                    / category 
                    / f"latest_{module}_{category}.json"
                )
                
                if latest_file.exists():
                    try:
                        with open(latest_file, encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 驗證基本結構
                        if "metadata" in data:
                            gen_time = data["metadata"].get("generated_at", "N/A")
                            total = data["metadata"].get("total_flows", 0)
                            print(f"  ✓ {category}: {total} 個流程 (生成時間: {gen_time})")
                            results[module][category] = {"exists": True, "valid": True, "total_flows": total}
                        else:
                            print(f"  ⚠ {category}: 文件格式異常")
                            results[module][category] = {"exists": True, "valid": False}
                    
                    except json.JSONDecodeError:
                        print(f"  ✗ {category}: JSON 解析失敗")
                        results[module][category] = {"exists": True, "valid": False, "error": "JSON 解析失敗"}
                    except Exception as e:
                        print(f"  ✗ {category}: {str(e)}")
                        results[module][category] = {"exists": True, "valid": False, "error": str(e)}
                else:
                    print(f"  - {category}: 文件不存在（模組可能尚未分析）")
                    results[module][category] = {"exists": False}
            
            print()
        
        return results
    
    def generate_report(self):
        """生成完整驗證報告"""
        structure_results = self.validate_structure()
        latest_results = self.validate_latest_files()
        
        print("=" * 60)
        print("驗證摘要")
        print("=" * 60)
        print()
        
        # 統計
        total_modules = len(self.modules)
        valid_modules = sum(1 for v in structure_results["modules_valid"].values() if v)
        
        total_files = 0
        valid_files = 0
        
        for module in self.modules:
            if module in latest_results:
                for category, info in latest_results[module].items():
                    total_files += 1
                    if info.get("exists") and info.get("valid"):
                        valid_files += 1
        
        print(f"模組狀態: {valid_modules}/{total_modules} 有效")
        print(f"Latest 文件: {valid_files}/{total_files} 有效")
        print()
        
        if structure_results["errors"]:
            print("錯誤:")
            for error in structure_results["errors"]:
                print(f"  ✗ {error}")
        else:
            print("✓ 所有檢查通過")
        
        print()
        print("=" * 60)
        
        return {
            "structure": structure_results,
            "latest_files": latest_results,
            "summary": {
                "valid_modules": valid_modules,
                "total_modules": total_modules,
                "valid_files": valid_files,
                "total_files": total_files
            }
        }


def main():
    """主程式"""
    validator = AnalysisPathValidator()
    results = validator.generate_report()
    
    # 保存報告
    report_file = validator.analysis_data_root / "validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"詳細報告已保存到: {report_file}")


if __name__ == "__main__":
    main()
