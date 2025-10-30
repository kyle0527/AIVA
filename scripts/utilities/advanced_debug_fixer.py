#!/usr/bin/env python3
"""
AIVA 高級偵錯錯誤修復工具
專門修復類型檢查和導入問题的進階修復器
"""

import re
from pathlib import Path

class AdvancedDebugFixer:
    """進階偵錯錯誤修復器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.fixed_files = []
        
    def fix_numpy_type_issues(self, file_path: Path) -> bool:
        """修復 numpy 類型問題"""
        print(f"🔧 修復 numpy 類型問題: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修復 numpy floating[Any] -> float 問題
            # 添加 .item() 方法轉換
            content = re.sub(
                r'return np\.mean\(([^)]+)\)',
                r'return float(np.mean(\1))',
                content
            )
            
            # 修復 numpy intp -> int 問題
            content = re.sub(
                r'return np\.argmax\(([^)]+)\)',
                r'return int(np.argmax(\1))',
                content
            )
            
            # 修復其他 numpy 返回類型問題
            content = re.sub(
                r'return -np\.mean\(([^)]+)\)',
                r'return float(-np.mean(\1))',
                content
            )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def fix_import_issues(self, file_path: Path) -> bool:
        """修復導入問題"""
        print(f"🔧 修復導入問題: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 移除未使用的導入
            if 'Union' in content and 'from typing import' in content:
                # 檢查是否真的使用了 Union
                if not re.search(r'\bUnion\[', content):
                    content = re.sub(
                        r'(from typing import [^)]+), Union',
                        r'\1',
                        content
                    )
                    content = re.sub(
                        r'Union, (from typing import [^)]+)',
                        r'\1',
                        content
                    )
            
            # 移除未使用的 time 導入
            if 'import time' in content and not re.search(r'\btime\.', content):
                content = re.sub(r'import time\n', '', content)
            
            # 修復未使用變數的問題
            # 添加 _ 前綴或刪除賦值
            content = re.sub(
                r'(\s+)val_loss = ([^)]+)\n',
                r'\1_ = \2  # 未使用的變數\n',
                content
            )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def fix_plugin_issues(self, file_path: Path) -> bool:
        """修復插件相關問題"""
        print(f"🔧 修復插件問題: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修復 PluginMetadata 缺少參數問題
            if 'return PluginMetadata(' in content:
                content = re.sub(
                    r'return PluginMetadata\(\s*name=entry_point\.name,\s*version="1\.0\.0",\s*entry_point=f"{plugin_class\.__module__}:{plugin_class\.__name__}"\s*\)',
                    '''return PluginMetadata(
                name=entry_point.name,
                version="1.0.0",
                description="Auto-generated plugin",
                author="AIVA",
                license="MIT",
                category="general",
                min_aiva_version="1.0.0",
                max_aiva_version="2.0.0",
                enabled=True,
                priority=0,
                entry_point=f"{plugin_class.__module__}:{plugin_class.__name__}"
            )''',
                    content,
                    flags=re.MULTILINE | re.DOTALL
                )
            
            # 修復 __self__ 屬性存取問題
            content = re.sub(
                r'key=lambda h: getattr\(h\.__self__\.metadata, \'priority\', 0\)',
                r'key=lambda h: getattr(getattr(h, "metadata", None), "priority", 0)',
                content
            )
            
            # 添加條件導入以避免導入錯誤
            if 'import pkg_resources' in content:
                content = re.sub(
                    r'import pkg_resources',
                    '''try:
    import pkg_resources
except ImportError:
    pkg_resources = None''',
                    content
                )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def fix_missing_imports(self, file_path: Path) -> bool:
        """修復缺少的導入"""
        print(f"🔧 修復缺少的導入: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 檢查並添加缺少的導入
            imports_to_add = []
            
            # 檢查是否使用了 subprocess 但沒有導入
            if 'subprocess.Popen' in content and 'import subprocess' not in content:
                imports_to_add.append('import subprocess')
            
            # 檢查是否使用了 asyncio 但沒有導入
            if 'asyncio.Task' in content and 'import asyncio' not in content:
                imports_to_add.append('import asyncio')
            
            # 檢查是否使用了 json 但沒有導入
            if 'json.loads' in content and 'import json' not in content:
                imports_to_add.append('import json')
            
            if imports_to_add:
                # 找到現有導入的位置
                import_section_end = 0
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_section_end = i + 1
                
                # 添加新的導入
                for import_stmt in imports_to_add:
                    lines.insert(import_section_end, import_stmt)
                    import_section_end += 1
                
                content = '\n'.join(lines)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def run_targeted_fixes(self) -> None:
        """運行針對性修復"""
        print("🎯 運行針對性修復...")
        
        # 特定文件的特定修復
        specific_fixes = [
            ("services/core/aiva_core/ai_engine/learning_engine.py", [
                self.fix_numpy_type_issues,
                self.fix_import_issues
            ]),
            ("services/aiva_common/plugins/__init__.py", [
                self.fix_plugin_issues,
                self.fix_import_issues
            ]),
            ("services/aiva_common/tools/schema_codegen_tool.py", [
                self.fix_import_issues
            ]),
            ("services/aiva_common/ai/cross_language_bridge.py", [
                self.fix_missing_imports,
                self.fix_import_issues
            ])
        ]
        
        fixed_count = 0
        
        for file_path, fix_functions in specific_fixes:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                print(f"⏭️ 跳過不存在的文件: {file_path}")
                continue
            
            print(f"\n📝 針對性修復: {file_path}")
            
            file_fixed = False
            
            for fix_func in fix_functions:
                if fix_func(full_path):
                    file_fixed = True
            
            if file_fixed:
                fixed_count += 1
                self.fixed_files.append(str(full_path))
                print(f"  ✅ 已修復")
            else:
                print(f"  ⏭️ 無需修復")
        
        print(f"\n🎉 針對性修復完成！共修復 {fixed_count} 個檔案")
    
    def generate_advanced_report(self) -> None:
        """生成進階修復報告"""
        report_path = self.project_root / "reports" / "debugging" / "advanced_debug_fix_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# AIVA 進階偵錯錯誤修復報告

**修復時間**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**修復工具**: AIVA Advanced Debug Fixer v1.0

## 進階修復統計

- **針對性修復檔案**: {len(self.fixed_files)}
- **修復類型**: numpy 類型問題、導入問題、插件問題

## 修復的檔案

{chr(10).join(f"- {Path(f).relative_to(self.project_root)}" for f in self.fixed_files)}

## 進階修復類型

### 1. Numpy 類型修復
- `np.mean()` 返回值類型轉換為 `float()`
- `np.argmax()` 返回值類型轉換為 `int()`
- 修復 numpy floating[Any] 與 Python float 的兼容性

### 2. 導入問題修復
- 移除未使用的 `Union` 導入
- 移除未使用的 `time` 導入
- 修復未使用變數問題

### 3. 插件系統修復
- 修復 `PluginMetadata` 缺少參數問題
- 修復 `__self__` 屬性存取問題
- 添加條件導入以避免導入錯誤

### 4. 缺少導入修復
- 自動檢測並添加缺少的標準庫導入
- 修復 subprocess、asyncio、json 等模組導入

## 修復效果

此進階修復工具專門針對：
1. ✅ 類型檢查錯誤
2. ✅ 導入解析問題
3. ✅ 插件系統兼容性
4. ✅ Numpy/Python 類型轉換

## 建議

1. 將此工具整合到 CI/CD 流程中
2. 定期運行以保持代碼質量
3. 考慮添加更多特定錯誤模式的修復邏輯

---
*由 AIVA Advanced Debug Fixer 自動生成*
""")
        
        print(f"📄 進階修復報告已生成: {report_path}")

def main():
    """主函數"""
    print("🔧 AIVA 進階偵錯錯誤修復工具")
    print("=" * 50)
    
    # 初始化修復器
    fixer = AdvancedDebugFixer()
    
    # 運行針對性修復
    fixer.run_targeted_fixes()
    
    # 生成報告
    fixer.generate_advanced_report()
    
    print("\n🎯 進階修復完成！建議重新運行類型檢查以確認結果。")

if __name__ == "__main__":
    main()