"""
🧹 自動文件整理腳本 - Auto Cleanup Script
根據 CLEANUP_PLAN.md 自動整理項目文件

功能:
1. 歸檔完成的文檔到 _archive/
2. 刪除臨時和重複文件
3. 清理輸出目錄
4. 生成整理報告
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class ProjectCleaner:
    """項目清理器"""
    
    def __init__(self, dry_run: bool = True):
        """初始化清理器"""
        self.dry_run = dry_run
        self.root = Path.cwd()
        self.archive_dir = self.root / "_archive" / f"cleanup_{datetime.now().strftime('%Y%m%d')}"
        self.stats = {
            'archived': 0,
            'deleted': 0,
            'moved': 0,
            'kept': 0
        }
        self.actions = []
        
    def archive_documents(self):
        """歸檔完成的文檔"""
        print("\n📦 歸檔文檔...")
        
        docs_to_archive = [
            "AI_ARCHITECTURE_ANALYSIS.md",
            "AI_ARRAY_ANALYSIS_CONCLUSION.md",
            "AI_COMPETITIVE_ANALYSIS.md",
            "ARCHITECTURE_CONTRACT_COMPLIANCE_REPORT.md",
            "CLI_AND_AI_TRAINING_GUIDE.md",
            "CLI_COMMAND_REFERENCE.md",
            "CLI_CORE_MODULE_FLOWS.md",
            "CLI_CROSS_MODULE_GUIDE.md",
            "CLI_IMPLEMENTATION_COMPLETE.md",
            "CLI_QUICK_REFERENCE.md",
            "CLI_UNIFIED_SETUP_GUIDE.md",
            "FILE_ORGANIZATION_REPORT.md",
            "PROJECT_ORGANIZATION_COMPLETE.md",
            "SCHEMA_DEFINITION_CLEANUP_REPORT.md",
            "SERVICES_ARCHITECTURE_COMPLIANCE_REPORT.md",
            "SERVICES_ORGANIZATION_SUMMARY.md",
            "SPECIALIZED_AI_CORE_DESIGN.md",
            "SPECIALIZED_AI_IMPLEMENTATION_PLAN.md",
        ]
        
        for doc in docs_to_archive:
            src = self.root / doc
            if src.exists():
                dst = self.archive_dir / "docs" / doc
                self._archive_file(src, dst, "文檔")
    
    def cleanup_temp_scripts(self):
        """清理臨時腳本"""
        print("\n🗑️  清理臨時腳本...")
        
        scripts_to_delete = [
            "temp_generate_stats.py",
            "train_ai_with_cli.py",
            "train_cli_matching.py",
            "final_report.py",
            "benchmark_performance.py",
        ]
        
        for script in scripts_to_delete:
            src = self.root / script
            if src.exists():
                self._delete_file(src, "臨時腳本")
    
    def move_tests(self):
        """移動測試文件到 tests/ 目錄"""
        print("\n📁 整理測試文件...")
        
        test_files = [
            "test_ai_core.py",
            "test_ai_real_data.py",
            "test_internal_communication.py",
            "test_message_system.py",
            "test_simple_matcher.py",
        ]
        
        tests_dir = self.root / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        for test_file in test_files:
            src = self.root / test_file
            if src.exists():
                dst = tests_dir / test_file
                self._move_file(src, dst, "測試文件")
    
    def cleanup_output_dirs(self):
        """清理輸出目錄"""
        print("\n🧹 清理輸出目錄...")
        
        # 清理舊的 tree 輸出
        out_dir = self.root / "_out"
        if out_dir.exists():
            for file in out_dir.glob("tree_*.txt"):
                if "tree_ultimate_chinese_20251017" not in file.name:
                    self._delete_file(file, "舊 tree 輸出")
        
        # 清理舊備份目錄
        old_dirs = [
            "_out1101016",
            "emoji_backups_cp950",
            "emoji_backups2",
        ]
        
        for dir_name in old_dirs:
            dir_path = self.root / dir_name
            if dir_path.exists():
                dst = self.archive_dir / "old_outputs" / dir_name
                self._archive_file(dir_path, dst, "舊輸出目錄")
    
    def _archive_file(self, src: Path, dst: Path, file_type: str):
        """歸檔文件"""
        action = f"歸檔 {file_type}: {src.name} → {dst.relative_to(self.root)}"
        
        if self.dry_run:
            print(f"  [預演] {action}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                shutil.rmtree(src)
            else:
                shutil.copy2(src, dst)
                src.unlink()
            print(f"  ✓ {action}")
        
        self.actions.append(action)
        self.stats['archived'] += 1
    
    def _delete_file(self, src: Path, file_type: str):
        """刪除文件"""
        action = f"刪除 {file_type}: {src.name}"
        
        if self.dry_run:
            print(f"  [預演] {action}")
        else:
            if src.is_dir():
                shutil.rmtree(src)
            else:
                src.unlink()
            print(f"  ✓ {action}")
        
        self.actions.append(action)
        self.stats['deleted'] += 1
    
    def _move_file(self, src: Path, dst: Path, file_type: str):
        """移動文件"""
        action = f"移動 {file_type}: {src.name} → {dst.relative_to(self.root)}"
        
        if self.dry_run:
            print(f"  [預演] {action}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  ✓ {action}")
        
        self.actions.append(action)
        self.stats['moved'] += 1
    
    def generate_report(self):
        """生成整理報告"""
        report_file = self.root / "CLEANUP_REPORT.md"
        
        total_actions = sum(self.stats.values())
        
        report = f"""# 🧹 文件整理報告

**執行時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**模式**: {'預演模式 (未實際執行)' if self.dry_run else '實際執行'}

## 📊 統計摘要

- **歸檔文件**: {self.stats['archived']}
- **刪除文件**: {self.stats['deleted']}
- **移動文件**: {self.stats['moved']}
- **總操作數**: {total_actions}

## 📝 詳細操作

"""
        
        for i, action in enumerate(self.actions, 1):
            report += f"{i}. {action}\n"
        
        report += f"""
---

## ✅ 結果

{'**注意**: 這是預演報告,未實際執行。請使用 --execute 執行實際操作。' if self.dry_run else '✓ 所有操作已成功執行'}

"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 報告已生成: {report_file}")
        return report_file
    
    def run(self):
        """執行清理"""
        print("=" * 70)
        print("🧹 項目文件整理")
        print("=" * 70)
        print(f"模式: {'預演 (Dry Run)' if self.dry_run else '實際執行'}")
        print("=" * 70)
        
        self.archive_documents()
        self.cleanup_temp_scripts()
        self.move_tests()
        self.cleanup_output_dirs()
        
        print("\n" + "=" * 70)
        print("📊 整理統計")
        print("=" * 70)
        for key, value in self.stats.items():
            print(f"  {key.capitalize()}: {value}")
        
        report_file = self.generate_report()
        
        print("\n" + "=" * 70)
        print(f"{'✓ 預演完成!' if self.dry_run else '✓ 整理完成!'}")
        print("=" * 70)
        
        if self.dry_run:
            print("\n💡 提示: 使用 --execute 參數執行實際操作")

def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🧹 自動文件整理腳本')
    parser.add_argument('--execute', action='store_true',
                       help='實際執行 (預設為預演模式)')
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(dry_run=not args.execute)
    cleaner.run()

if __name__ == "__main__":
    main()
