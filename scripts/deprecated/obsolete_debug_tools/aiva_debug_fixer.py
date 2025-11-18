#!/usr/bin/env python3
"""
AIVA 偵錯錯誤修復工具
依照規範自動修復常見的 Python 類型檢查和語法錯誤
"""

import re
import ast
from pathlib import Path
from typing import List, Dict
import subprocess

class AIVADebugFixer:
    """AIVA 專案偵錯錯誤修復器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.fixed_files = []
        self.errors_found = []
        
    def analyze_current_errors(self) -> Dict[str, List[str]]:
        """分析當前錯誤"""
        print("🔍 分析當前偵錯錯誤...")
        
        error_categories = {
            "type_annotations": [],
            "unused_imports": [],
            "missing_imports": [],
            "generic_types": [],
            "encoding_issues": [],
            "attribute_access": []
        }
        
        # 運行 Pylance/Pyright 檢查
        try:
            result = subprocess.run(
                ["python", "-m", "pyright", "--outputformat", "json"], 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print("✅ 沒有發現 Pyright 錯誤")
            else:
                print(f"⚠️ Pyright 發現 {result.stderr.count('error')} 個錯誤")
                
        except FileNotFoundError:
            print("⚠️ 未找到 Pyright，使用內建檢查")
            
        return error_categories
    
    def fix_type_annotation_errors(self, file_path: Path) -> bool:
        """修復類型註解錯誤"""
        print(f"🔧 修復類型註解: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修復 1: 添加泛型類型參數
            # subprocess.Popen -> subprocess.Popen[bytes]
            content = re.sub(
                r'subprocess\.Popen(?!\[)',
                'subprocess.Popen[bytes]',
                content
            )
            
            # 修復 2: asyncio.Task -> asyncio.Task[Any]
            content = re.sub(
                r'asyncio\.Task(?!\[)',
                'asyncio.Task[Any]',
                content
            )
            
            # 修復 3: 默認參數類型修復
            # dict = None -> Optional[Dict[str, Any]] = None
            content = re.sub(
                r'(\w+): dict = None',
                r'\1: Optional[Dict[str, Any]] = None',
                content
            )
            
            # 修復 4: 添加 Optional 導入如果需要
            if 'Optional[' in content and 'from typing import' in content:
                if 'Optional' not in content.split('from typing import')[1].split('\n')[0]:
                    content = re.sub(
                        r'from typing import ([^)]+)',
                        lambda m: f"from typing import {m.group(1)}, Optional" if 'Optional' not in m.group(1) else m.group(0),
                        content
                    )
            
            # 修復 5: 添加 Any 導入如果需要
            if 'Any' in content and 'from typing import' in content:
                if 'Any' not in content.split('from typing import')[1].split('\n')[0]:
                    content = re.sub(
                        r'from typing import ([^)]+)',
                        lambda m: f"from typing import {m.group(1)}, Any" if 'Any' not in m.group(1) else m.group(0),
                        content
                    )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def fix_unused_imports(self, file_path: Path) -> bool:
        """修復未使用的導入"""
        print(f"🔧 修復未使用導入: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 分析 AST 找出真正未使用的導入
            try:
                tree = ast.parse(content)
                
                # 收集所有導入
                imports = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports[alias.name] = node.lineno
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports[alias.name] = node.lineno
                
                # 檢查使用情況
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        used_names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        # 對於 module.attr 形式
                        if isinstance(node.value, ast.Name):
                            used_names.add(node.value.id)
                
                # 移除明顯未使用的導入（但保留特殊情況）
                lines = content.split('\n')
                for name, line_no in imports.items():
                    if name not in used_names:
                        # 特殊情況不移除
                        if name in ['os', 'sys', 'logging', 'datetime']:
                            continue
                        
                        # 檢查是否在字符串中使用
                        if any(name in line for line in lines if 'import' not in line):
                            continue
                            
                        # 移除該行導入
                        import_line = lines[line_no - 1]
                        if f'import {name}' in import_line:
                            lines[line_no - 1] = ''
                
                content = '\n'.join(lines)
                
            except SyntaxError:
                # 如果 AST 解析失敗，使用簡單的正則表達式
                pass
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def fix_encoding_issues(self, file_path: Path) -> bool:
        """修復編碼相關問題"""
        print(f"🔧 修復編碼問題: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修復 1: sys.stdout.reconfigure 問題
            if 'sys.stdout.reconfigure(encoding=' in content:
                content = re.sub(
                    r'sys\.stdout\.reconfigure\(encoding=\'utf-8\'\)',
                    '# sys.stdout.reconfigure(encoding=\'utf-8\')  # 僅在支持的 Python 版本中可用',
                    content
                )
                
            if 'sys.stderr.reconfigure(encoding=' in content:
                content = re.sub(
                    r'sys\.stderr\.reconfigure\(encoding=\'utf-8\'\)',
                    '# sys.stderr.reconfigure(encoding=\'utf-8\')  # 僅在支持的 Python 版本中可用',
                    content
                )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def fix_attribute_access_errors(self, file_path: Path) -> bool:
        """修復屬性存取錯誤"""
        print(f"🔧 修復屬性存取: {file_path.relative_to(self.project_root)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修復常見的屬性存取錯誤
            # 添加適當的類型檢查
            
            # 修復 result["stdout"].strip() 錯誤
            if 'result["stdout"].strip()' in content:
                content = re.sub(
                    r'result\["stdout"\]\.strip\(\)',
                    'str(result["stdout"]).strip() if result["stdout"] else ""',
                    content
                )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path}: {e}")
            
        return False
    
    def scan_and_fix_files(self) -> None:
        """掃描並修復所有檔案"""
        print("🚀 掃描專案檔案...")
        
        # 找出需要修復的 Python 檔案
        python_files = []
        
        # 掃描 services 目錄
        services_dir = self.project_root / "services"
        if services_dir.exists():
            python_files.extend(services_dir.rglob("*.py"))
        
        # 掃描根目錄的 Python 檔案
        python_files.extend(self.project_root.glob("*.py"))
        
        print(f"📁 找到 {len(python_files)} 個 Python 檔案")
        
        fixed_count = 0
        
        for file_path in python_files:
            # 跳過某些目錄
            if any(part in str(file_path) for part in ['__pycache__', '.venv', 'node_modules', '_archive']):
                continue
            
            print(f"\n📝 檢查: {file_path.relative_to(self.project_root)}")
            
            fixed = False
            
            # 修復類型註解錯誤
            if self.fix_type_annotation_errors(file_path):
                fixed = True
                
            # 修復編碼問題
            if self.fix_encoding_issues(file_path):
                fixed = True
                
            # 修復屬性存取錯誤
            if self.fix_attribute_access_errors(file_path):
                fixed = True
            
            # 修復未使用導入（最後執行）
            if self.fix_unused_imports(file_path):
                fixed = True
            
            if fixed:
                fixed_count += 1
                self.fixed_files.append(str(file_path))
                print(f"  ✅ 已修復")
            else:
                print(f"  ⏭️ 無需修復")
        
        print(f"\n🎉 修復完成！共修復 {fixed_count} 個檔案")
        
        if self.fixed_files:
            print("\n📋 已修復的檔案:")
            for file_path in self.fixed_files:
                print(f"  - {Path(file_path).relative_to(self.project_root)}")
    
    def validate_fixes(self) -> bool:
        """驗證修復結果"""
        print("\n🔍 驗證修復結果...")
        
        try:
            # 嘗試編譯所有修復的檔案
            all_valid = True
            
            for file_path in self.fixed_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 嘗試解析 AST
                    ast.parse(content)
                    print(f"  ✅ 語法正確: {Path(file_path).relative_to(self.project_root)}")
                    
                except SyntaxError as e:
                    print(f"  ❌ 語法錯誤: {Path(file_path).relative_to(self.project_root)} - {e}")
                    all_valid = False
                    
            return all_valid
            
        except Exception as e:
            print(f"❌ 驗證失敗: {e}")
            return False
    
    def generate_report(self) -> None:
        """生成修復報告"""
        report_path = self.project_root / "reports" / "debugging" / "debug_fix_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# AIVA 偵錯錯誤修復報告

**修復時間**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**修復工具**: AIVA Debug Fixer v1.0

## 修復統計

- **總檢查檔案**: {len(list(self.project_root.rglob("*.py")))}
- **修復檔案數**: {len(self.fixed_files)}
- **錯誤類型**: 類型註解、未使用導入、編碼問題、屬性存取

## 修復的檔案

{chr(10).join(f"- {Path(f).relative_to(self.project_root)}" for f in self.fixed_files)}

## 修復類型

### 1. 類型註解修復
- `subprocess.Popen` → `subprocess.Popen[bytes]`
- `asyncio.Task` → `asyncio.Task[Any]`
- `dict = None` → `Optional[Dict[str, Any]] = None`

### 2. 編碼問題修復
- 註解掉不支援的 `sys.stdout.reconfigure()` 調用

### 3. 屬性存取修復
- 添加類型安全檢查
- 修復動態屬性存取問題

### 4. 未使用導入清理
- 移除明顯未使用的導入
- 保留特殊用途導入

## 建議

1. 定期運行此修復工具
2. 配置 pre-commit hooks 防止錯誤再次出現
3. 考慮添加更嚴格的類型檢查配置

---
*由 AIVA Debug Fixer 自動生成*
""")
        
        print(f"📄 修復報告已生成: {report_path}")

def main():
    """主函數"""
    print("🔧 AIVA 偵錯錯誤修復工具")
    print("=" * 50)
    
    # 初始化修復器
    fixer = AIVADebugFixer()
    
    # 分析錯誤
    fixer.analyze_current_errors()
    
    # 掃描並修復
    fixer.scan_and_fix_files()
    
    # 驗證修復
    if fixer.validate_fixes():
        print("\n✅ 所有修復都通過驗證")
    else:
        print("\n⚠️ 部分修復可能需要手動檢查")
    
    # 生成報告
    fixer.generate_report()
    
    print("\n🎯 修復完成！建議重新運行類型檢查以確認結果。")

if __name__ == "__main__":
    main()