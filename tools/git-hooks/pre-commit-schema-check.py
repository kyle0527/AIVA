#!/usr/bin/env python3
"""
AIVA Schema 合規性 Pre-commit Hook
================================

此腳本作為 Git pre-commit hook，在提交前檢查 schema 合規性。
防止不合規的程式碼被提交到版本庫。

安裝方法：
    # 複製到 Git hooks 目錄
    cp tools/git-hooks/pre-commit-schema-check.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    
    # 或使用 pre-commit 框架
    pip install pre-commit
    pre-commit install

使用 pre-commit 框架時，請在 .pre-commit-config.yaml 中添加：
repos:
  - repo: local
    hooks:
      - id: schema-compliance
        name: AIVA Schema Compliance Check
        entry: python tools/git-hooks/pre-commit-schema-check.py
        language: python
        files: '\\.(go|rs|ts)$'
        pass_filenames: false
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set

class PreCommitSchemaChecker:
    """Git Pre-commit Schema 合規性檢查器"""
    
    def __init__(self):
        self.repo_root = self._find_repo_root()
        self.changed_files = self._get_changed_files()
        
    def _find_repo_root(self) -> Path:
        """尋找 Git 版本庫根目錄"""
        current = Path.cwd()
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _get_changed_files(self) -> List[str]:
        """取得本次提交的變更檔案"""
        try:
            # 取得暫存區的檔案
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\\n')
                return [f for f in files if f]  # 過濾空字符串
            else:
                print(f"⚠️ 無法取得變更檔案: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"⚠️ 執行 git diff 時出錯: {e}")
            return []
    
    def should_check_compliance(self) -> bool:
        """判斷是否需要執行 schema 合規性檢查"""
        # 檢查變更檔案中是否包含需要檢查的檔案類型
        relevant_extensions = {'.go', '.rs', '.ts'}
        relevant_paths = {'services/', 'schemas/'}
        
        for file_path in self.changed_files:
            file_ext = Path(file_path).suffix
            
            # 檢查副檔名
            if file_ext in relevant_extensions:
                return True
            
            # 檢查路徑
            for path_prefix in relevant_paths:
                if file_path.startswith(path_prefix):
                    return True
        
        return False
    
    def run_quick_check(self) -> bool:
        """執行快速合規性檢查（只檢查變更的模組）"""
        print("🔍 執行 Schema 合規性檢查...")
        
        # 識別受影響的模組
        affected_modules = self._identify_affected_modules()
        
        if not affected_modules:
            print("✅ 沒有需要檢查的模組")
            return True
        
        print(f"📋 檢查 {len(affected_modules)} 個受影響的模組...")
        
        # 對每個模組執行基本檢查
        issues_found = []
        
        for module_path in affected_modules:
            issues = self._check_module_quickly(module_path)
            if issues:
                issues_found.extend(issues)
        
        if issues_found:
            print("\\n❌ 發現 Schema 合規性問題:")
            for issue in issues_found:
                print(f"  • {issue}")
                
            print("\\n💡 修復建議:")
            print("  1. 移除自定義 FindingPayload 定義")
            print("  2. 使用標準 schema 導入")
            print("  3. 運行完整檢查: python tools/schema_compliance_validator.py")
            
            return False
        else:
            print("✅ Schema 合規性檢查通過")
            return True
    
    def _identify_affected_modules(self) -> Set[Path]:
        """識別受變更影響的模組"""
        affected_modules = set()
        
        for file_path in self.changed_files:
            file_path_obj = Path(file_path)
            
            # 向上搜尋直到找到模組根目錄
            current = file_path_obj.parent
            while current != Path('.'):
                # Go 模組 (有 go.mod)
                if (self.repo_root / current / 'go.mod').exists():
                    affected_modules.add(current)
                    break
                
                # Rust 模組 (有 Cargo.toml)
                if (self.repo_root / current / 'Cargo.toml').exists():
                    affected_modules.add(current)
                    break
                
                # TypeScript 模組 (有 package.json 或 tsconfig.json)
                if ((self.repo_root / current / 'package.json').exists() or 
                    (self.repo_root / current / 'tsconfig.json').exists()):
                    affected_modules.add(current)
                    break
                
                if current == Path('.'):
                    break
                current = current.parent
        
        return affected_modules
    
    def _check_module_quickly(self, module_path: Path) -> List[str]:
        """對單一模組執行快速檢查"""
        issues = []
        module_full_path = self.repo_root / module_path
        
        # 檢查模組類型和對應的合規性
        if (module_full_path / 'go.mod').exists():
            issues.extend(self._check_go_module_quick(module_full_path))
        elif (module_full_path / 'Cargo.toml').exists():
            issues.extend(self._check_rust_module_quick(module_full_path))
        elif ((module_full_path / 'package.json').exists() or 
              (module_full_path / 'tsconfig.json').exists()):
            issues.extend(self._check_typescript_module_quick(module_full_path))
        
        return issues
    
    def _check_go_module_quick(self, module_path: Path) -> List[str]:
        """快速檢查 Go 模組"""
        issues = []
        
        # 檢查變更的 Go 檔案
        for file_path in self.changed_files:
            if file_path.endswith('.go') and file_path.startswith(str(module_path.relative_to(self.repo_root))):
                full_file_path = self.repo_root / file_path
                
                if full_file_path.exists():
                    try:
                        content = full_file_path.read_text(encoding='utf-8')
                        
                        # 檢查是否使用標準 schema
                        has_standard_import = 'aiva_common_go/schemas/generated' in content
                        
                        # 檢查是否有自定義 FindingPayload
                        has_custom_finding = 'type FindingPayload struct' in content
                        
                        if has_custom_finding:
                            issues.append(f"{file_path}: 發現自定義 FindingPayload 定義")
                        
                        if not has_standard_import and ('FindingPayload' in content or 'Vulnerability' in content):
                            issues.append(f"{file_path}: 未使用標準 schema 導入")
                            
                    except Exception:
                        continue
        
        return issues
    
    def _check_rust_module_quick(self, module_path: Path) -> List[str]:
        """快速檢查 Rust 模組"""
        issues = []
        
        # 檢查是否有完整的 schema 實現
        schema_mod = module_path / 'src' / 'schemas' / 'generated' / 'mod.rs'
        if schema_mod.exists():
            try:
                content = schema_mod.read_text(encoding='utf-8')
                if not content.strip() or 'TODO' in content:
                    issues.append(f"{schema_mod.relative_to(self.repo_root)}: Schema 實現不完整")
            except Exception:
                pass
        else:
            # 檢查是否在使用 Finding 相關結構但沒有標準 schema
            for file_path in self.changed_files:
                if file_path.endswith('.rs') and file_path.startswith(str(module_path.relative_to(self.repo_root))):
                    full_file_path = self.repo_root / file_path
                    
                    if full_file_path.exists():
                        try:
                            content = full_file_path.read_text(encoding='utf-8')
                            if 'FindingPayload' in content and 'schemas::generated' not in content:
                                issues.append(f"{file_path}: 未使用標準 schema 模組")
                        except Exception:
                            continue
        
        return issues
    
    def _check_typescript_module_quick(self, module_path: Path) -> List[str]:
        """快速檢查 TypeScript 模組"""
        issues = []
        
        for file_path in self.changed_files:
            if file_path.endswith('.ts') and file_path.startswith(str(module_path.relative_to(self.repo_root))):
                full_file_path = self.repo_root / file_path
                
                if full_file_path.exists():
                    try:
                        content = full_file_path.read_text(encoding='utf-8')
                        
                        # 檢查是否使用標準 schema
                        has_standard_import = 'schemas/aiva_schemas' in content
                        
                        # 檢查是否有自定義介面定義
                        has_custom_interface = ('interface' in content and 
                                              ('Finding' in content or 'Payload' in content))
                        
                        if has_custom_interface and not has_standard_import:
                            issues.append(f"{file_path}: 可能使用自定義介面而非標準 schema")
                        
                        if 'FindingPayload' in content and not has_standard_import:
                            issues.append(f"{file_path}: 未使用標準 schema 導入")
                            
                    except Exception:
                        continue
        
        return issues

def main():
    """主要執行函數"""
    print("🔍 AIVA Schema 合規性 Pre-commit 檢查")
    
    checker = PreCommitSchemaChecker()
    
    # 檢查是否需要執行合規性檢查
    if not checker.should_check_compliance():
        print("ℹ️  本次提交未涉及需要檢查的檔案，跳過 schema 合規性檢查")
        return 0
    
    # 執行快速檢查
    if checker.run_quick_check():
        print("✅ Pre-commit schema 檢查通過")
        return 0
    else:
        print("\\n❌ Pre-commit schema 檢查失敗")
        print("請修復上述問題後再次提交，或運行完整檢查取得詳細資訊:")
        print("  python tools/schema_compliance_validator.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())