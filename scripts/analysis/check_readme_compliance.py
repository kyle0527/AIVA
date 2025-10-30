#!/usr/bin/env python3
"""
AIVA README 合規性檢查工具

檢查所有 README.md 檔案是否遵循 aiva_common 修護規範
並自動更新缺失的規範內容
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

class ReadmeComplianceChecker:
    """README 合規性檢查器"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.compliance_template = self._load_compliance_template()
        self.results = []
        
    def _load_compliance_template(self) -> Dict[str, str]:
        """載入 aiva_common 修護規範模板"""
        return {
            "section_title": "## 🔧 開發規範與最佳實踐",
            "aiva_common_header": "### 📐 **aiva_common 修護規範遵循**",
            "importance_note": "> **重要**: 本模組嚴格遵循 [aiva_common 修護規範](../aiva_common/README.md#🔧-開發指南)，確保所有定義、枚舉引用及修復都在同一套標準之下。",
            "standard_import_header": "#### ✅ **標準導入範例**",
            "prohibited_practices_header": "#### 🚨 **嚴格禁止的做法**",
            "module_specific_header": "#### 🔍 **模組特定枚舉判斷標準**",
            "development_checklist_header": "#### 📋 **開發檢查清單**",
            "repair_principles_header": "#### 🛠️ **修復原則**",
            "repair_principle_text": "**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些可能是預留的 API 介面或未來功能的基礎架構。"
        }
    
    def scan_all_readmes(self) -> List[Path]:
        """掃描所有 README.md 檔案（排除第三方庫）"""
        readme_files = []
        
        # 排除的目錄模式
        exclude_patterns = [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            "target",  # Rust build 目錄
            "build",
            "dist",
            "vendor",  # Go vendor 目錄
        ]
        
        def should_exclude(path: Path) -> bool:
            """檢查路徑是否應被排除"""
            path_parts = path.parts
            for pattern in exclude_patterns:
                if pattern in path_parts:
                    return True
            return False
        
        # 主要服務模組
        services_dir = self.workspace_root / "services"
        if services_dir.exists():
            for readme in services_dir.rglob("README.md"):
                if not should_exclude(readme) and "aiva_common" not in str(readme):
                    readme_files.append(readme)
        
        # 工具模組
        tools_dir = self.workspace_root / "tools"
        if tools_dir.exists():
            for readme in tools_dir.rglob("README.md"):
                if not should_exclude(readme):
                    readme_files.append(readme)
        
        # 其他重要模組
        for subdir in ["web", "testing", "utilities"]:
            subdir_path = self.workspace_root / subdir
            if subdir_path.exists():
                readme_path = subdir_path / "README.md"
                if readme_path.exists() and not should_exclude(readme_path):
                    readme_files.append(readme_path)
        
        # 根目錄 README
        root_readme = self.workspace_root / "README.md"
        if root_readme.exists():
            readme_files.append(root_readme)
            
        return sorted(readme_files)
    
    def check_readme_compliance(self, readme_path: Path) -> Dict:
        """檢查單個 README 的合規性"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "file": str(readme_path),
                "error": f"無法讀取檔案: {e}",
                "compliant": False
            }
        
        # 檢查項目
        checks = {}
        
        # 1. 檢查是否有目錄且在最前面
        toc_pattern = r'^## 📋? 目錄|^## 📑 目錄|^## � 目錄'
        has_toc = bool(re.search(toc_pattern, content, re.MULTILINE))
        
        # 檢查目錄位置（應在前 200 行內）
        lines = content.split('\n')
        toc_early = False
        if has_toc:
            for i, line in enumerate(lines[:200]):
                if re.match(toc_pattern, line):
                    toc_early = True
                    break
        
        checks["has_toc"] = has_toc
        checks["toc_early"] = toc_early
        
        # 2. 檢查是否有開發規範章節
        has_dev_standards = "開發規範與最佳實踐" in content
        checks["has_dev_standards"] = has_dev_standards
        
        # 3. 檢查是否有 aiva_common 修護規範
        has_aiva_common_section = "aiva_common 修護規範" in content or "aiva_common" in content
        checks["has_aiva_common_section"] = has_aiva_common_section
        
        # 4. 檢查是否有標準導入範例
        has_import_examples = "標準導入範例" in content or "from.*aiva_common" in content
        checks["has_import_examples"] = has_import_examples
        
        # 5. 檢查是否有禁止做法說明
        has_prohibited_practices = "禁止做法" in content or "嚴格禁止" in content
        checks["has_prohibited_practices"] = has_prohibited_practices
        
        # 6. 檢查是否有修復原則
        has_repair_principles = "修復原則" in content or "保留未使用函數" in content
        checks["has_repair_principles"] = has_repair_principles
        
        # 7. 檢查相對路徑引用的 aiva_common
        relative_path = self._get_aiva_common_relative_path(readme_path)
        correct_aiva_common_path = f"[aiva_common 修護規範]({relative_path}/README.md#🔧-開發指南)" in content
        checks["correct_aiva_common_path"] = correct_aiva_common_path
        
        # 計算合規性分數
        total_checks = len(checks)
        passed_checks = sum(1 for v in checks.values() if v)
        compliance_score = passed_checks / total_checks
        
        # 判斷是否合規（80% 以上通過）
        is_compliant = compliance_score >= 0.8
        
        return {
            "file": str(readme_path),
            "relative_path": str(readme_path.relative_to(self.workspace_root)),
            "checks": checks,
            "compliance_score": compliance_score,
            "compliant": is_compliant,
            "missing_items": [k for k, v in checks.items() if not v],
            "aiva_common_relative_path": relative_path
        }
    
    def _get_aiva_common_relative_path(self, readme_path: Path) -> str:
        """計算到 aiva_common 的相對路徑"""
        try:
            # 從 README 位置到 aiva_common 的相對路徑
            readme_dir = readme_path.parent
            aiva_common_path = self.workspace_root / "services" / "aiva_common"
            relative = os.path.relpath(aiva_common_path, readme_dir)
            return relative.replace("\\", "/")  # 統一使用 Unix 風格路徑
        except Exception:
            return "../aiva_common"  # 預設值
    
    def generate_compliance_section(self, readme_path: Path, module_type: str = "service") -> str:
        """生成合規性章節內容"""
        relative_path = self._get_aiva_common_relative_path(readme_path)
        
        # 根據模組類型調整內容
        if "integration" in str(readme_path).lower():
            module_name = "Integration"
            import_example = """# ✅ 正確 - Integration 模組的標準導入
from ..aiva_common.enums import (
    Severity,                # 風險評級 (CRITICAL, HIGH, MEDIUM, LOW)
    Confidence,              # 信心度 (CERTAIN, FIRM, POSSIBLE)
    TaskStatus,              # 任務狀態 (PENDING, RUNNING, COMPLETED)
    AssetType,               # 資產類型 (URL, HOST, REPOSITORY)
    VulnerabilityStatus,     # 漏洞狀態 (NEW, OPEN, IN_PROGRESS)
)
from ..aiva_common.schemas import (
    FindingPayload,          # 發現結果標準格式
    CVSSv3Metrics,           # CVSS v3.1 標準評分
    SARIFResult,             # SARIF v2.1.0 報告格式
    AivaMessage,             # 統一訊息格式
)"""
        elif "core" in str(readme_path).lower():
            module_name = "Core"
            import_example = """# ✅ 正確 - Core 模組的標準導入
from ..aiva_common.enums import (
    Severity,                # 風險評級 (CRITICAL, HIGH, MEDIUM, LOW)
    Confidence,              # 信心度 (CERTAIN, FIRM, POSSIBLE)
    TaskStatus,              # 任務狀態 (PENDING, RUNNING, COMPLETED)
    RiskLevel,               # 風險等級 (CRITICAL, HIGH, MEDIUM, LOW)
    ThreatLevel,             # 威脅等級
)
from ..aiva_common.schemas import (
    TaskUpdatePayload,       # 任務更新格式
    AivaMessage,             # 統一訊息格式
    FindingPayload,          # 發現結果標準格式
)"""
        elif "scan" in str(readme_path).lower():
            module_name = "Scan"
            import_example = """# ✅ 正確 - Scan 模組的標準導入
from ..aiva_common.enums import (
    Severity,                # 風險評級 (CRITICAL, HIGH, MEDIUM, LOW)
    Confidence,              # 信心度 (CERTAIN, FIRM, POSSIBLE)
    VulnerabilityType,       # 漏洞類型 (SQL_INJECTION, XSS, SSRF)
    AssetType,               # 資產類型 (URL, HOST, REPOSITORY)
)
from ..aiva_common.schemas import (
    ScanStartPayload,        # 掃描啟動格式
    SARIFResult,             # SARIF v2.1.0 報告格式
    CVSSv3Metrics,           # CVSS v3.1 標準評分
    FindingPayload,          # 發現結果標準格式
)"""
        elif "features" in str(readme_path).lower():
            module_name = "Features"
            import_example = """# ✅ 正確 - Features 模組的標準導入
from ..aiva_common.enums import (
    Severity,                # 風險評級 (CRITICAL, HIGH, MEDIUM, LOW)
    Confidence,              # 信心度 (CERTAIN, FIRM, POSSIBLE)
    VulnerabilityType,       # 漏洞類型 (SQL_INJECTION, XSS, SSRF)
    Exploitability,          # 可利用性評估
)
from ..aiva_common.schemas import (
    FunctionTaskPayload,     # 功能任務格式
    FunctionTaskResult,      # 功能結果格式
    FindingPayload,          # 發現結果標準格式
    SARIFResult,             # SARIF v2.1.0 報告格式
)"""
        else:
            module_name = "本模組"
            import_example = """# ✅ 正確 - 標準導入範例
from ..aiva_common.enums import (
    Severity,                # 風險評級 (CRITICAL, HIGH, MEDIUM, LOW)
    Confidence,              # 信心度 (CERTAIN, FIRM, POSSIBLE)
    TaskStatus,              # 任務狀態 (PENDING, RUNNING, COMPLETED)
)
from ..aiva_common.schemas import (
    FindingPayload,          # 發現結果標準格式
    AivaMessage,             # 統一訊息格式
)"""
        
        template = f"""## 🔧 開發規範與最佳實踐

### 📐 **aiva_common 修護規範遵循**

> **重要**: {module_name} 模組嚴格遵循 [aiva_common 修護規範]({relative_path}/README.md#🔧-開發指南)，確保所有定義、枚舉引用及修復都在同一套標準之下。

#### ✅ **標準導入範例**

```python
{import_example}
```

#### 🚨 **嚴格禁止的做法**

```python
# ❌ 禁止 - 重複定義通用枚舉
class Severity(str, Enum):  # 錯誤!使用 aiva_common.Severity
    CRITICAL = "critical"

# ❌ 禁止 - 重複定義標準結構  
class FindingPayload(BaseModel):  # 錯誤!使用 aiva_common.FindingPayload
    finding_id: str

# ❌ 禁止 - 自創評分標準
class CustomCVSS(BaseModel):  # 錯誤!使用 aiva_common.CVSSv3Metrics
    score: float
```

#### 🔍 **模組特定枚舉判斷標準**

只有滿足 **所有** 條件時，才允許在模組內定義專屬枚舉：

1. **完全專屬性**: 該枚舉概念僅用於本模組內部邏輯
2. **非通用概念**: 不是跨模組共享的概念（如 Severity、Confidence）
3. **高度技術專屬**: 與模組特定技術實現緊密相關
4. **不影響互操作性**: 不會破壞跨模組數據交換

#### 📋 **開發檢查清單**

在{module_name}模組開發時，請確認：

- [ ] **國際標準優先**: 優先使用 CVSS、SARIF、CVE、CWE 等官方標準
- [ ] **aiva_common 導入**: 所有通用概念都從 aiva_common 導入
- [ ] **無重複定義**: 確保沒有重複定義已存在的枚舉或 Schema
- [ ] **模組專用性**: 新定義的枚舉確實僅用於本模組內部
- [ ] **文檔完整性**: 所有自定義類型都有完整的 docstring 說明

#### 🛠️ **修復原則**

**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些可能是預留的 API 介面或未來功能的基礎架構。

---"""
        
        return template
    
    def update_readme_with_compliance(self, readme_path: Path) -> bool:
        """更新 README 使其符合合規性要求"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 無法讀取 {readme_path}: {e}")
            return False
        
        # 檢查當前合規性
        compliance_check = self.check_readme_compliance(readme_path)
        
        if compliance_check["compliant"]:
            print(f"✅ {readme_path.name} 已經符合規範")
            return True
        
        modified = False
        
        # 1. 檢查並添加目錄（如果不存在或位置不正確）
        if not compliance_check["checks"]["has_toc"] or not compliance_check["checks"]["toc_early"]:
            content = self._ensure_toc_at_top(content)
            modified = True
        
        # 2. 檢查並添加開發規範章節
        if not compliance_check["checks"]["has_dev_standards"]:
            compliance_section = self.generate_compliance_section(readme_path)
            
            # 在文檔末尾添加合規性章節
            if "---\n\n**維護狀態**" in content:
                # 在維護狀態前插入
                content = content.replace("---\n\n**維護狀態**", f"{compliance_section}\n\n---\n\n**維護狀態**")
            elif content.endswith("---"):
                # 在最後的分隔線前插入
                content = content[:-3] + f"\n{compliance_section}\n\n---"
            else:
                # 直接添加到末尾
                content += f"\n\n{compliance_section}"
            
            modified = True
        
        # 3. 更新 aiva_common 相對路徑引用
        if not compliance_check["checks"]["correct_aiva_common_path"]:
            relative_path = compliance_check["aiva_common_relative_path"]
            
            # 修正 aiva_common 路徑引用
            patterns = [
                (r'\[aiva_common 修護規範\]\([^)]+\)', f'[aiva_common 修護規範]({relative_path}/README.md#🔧-開發指南)'),
                (r'\[aiva_common\]\([^)]+\)', f'[aiva_common]({relative_path}/README.md)'),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
        
        # 儲存修改
        if modified:
            try:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 已更新 {readme_path.name}")
                return True
            except Exception as e:
                print(f"❌ 無法寫入 {readme_path}: {e}")
                return False
        
        return True
    
    def _ensure_toc_at_top(self, content: str) -> str:
        """確保目錄在文檔前面"""
        lines = content.split('\n')
        
        # 尋找現有目錄
        toc_start = -1
        toc_end = -1
        
        for i, line in enumerate(lines):
            if re.match(r'^## 📋? 目錄|^## 📑 目錄|^## � 目錄', line):
                toc_start = i
                # 尋找目錄結束（下一個 ## 或 ---）
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('##') or lines[j].startswith('---'):
                        toc_end = j
                        break
                break
        
        # 如果沒有目錄，創建一個基本的
        if toc_start == -1:
            # 找到標題後插入目錄
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('# ') or line.startswith('## '):
                    if i > 0:  # 不是第一行
                        header_end = i
                        break
            
            # 插入基本目錄
            basic_toc = [
                "",
                "## 📋 目錄",
                "",
                "- [🛠️ 開發工具箱](#️-開發工具箱)",
                "- [📊 概覽](#-概覽)",
                "- [🚀 快速開始](#-快速開始)",
                "- [🔧 開發規範與最佳實踐](#-開發規範與最佳實踐)",
                "- [📚 文檔](#-文檔)",
                "",
                "---",
                ""
            ]
            
            lines = lines[:header_end] + basic_toc + lines[header_end:]
        
        return '\n'.join(lines)
    
    def run_full_check(self) -> Dict:
        """執行完整的合規性檢查"""
        readme_files = self.scan_all_readmes()
        results = {
            "total_files": len(readme_files),
            "compliant_files": 0,
            "non_compliant_files": 0,
            "files": [],
            "summary": {}
        }
        
        print(f"🔍 掃描到 {len(readme_files)} 個 README 檔案")
        print("=" * 60)
        
        for readme_path in readme_files:
            print(f"\n📝 檢查: {readme_path.relative_to(self.workspace_root)}")
            
            compliance_check = self.check_readme_compliance(readme_path)
            results["files"].append(compliance_check)
            
            if compliance_check["compliant"]:
                results["compliant_files"] += 1
                print(f"  ✅ 合規 ({compliance_check['compliance_score']:.1%})")
            else:
                results["non_compliant_files"] += 1
                print(f"  ❌ 不合規 ({compliance_check['compliance_score']:.1%})")
                print(f"     缺失項目: {', '.join(compliance_check['missing_items'])}")
        
        # 生成摘要
        results["summary"] = {
            "compliance_rate": results["compliant_files"] / results["total_files"] if results["total_files"] > 0 else 0,
            "common_issues": self._analyze_common_issues(results["files"])
        }
        
        return results
    
    def _analyze_common_issues(self, files_data: List[Dict]) -> Dict[str, int]:
        """分析常見問題"""
        issues = {}
        for file_data in files_data:
            for missing_item in file_data.get("missing_items", []):
                issues[missing_item] = issues.get(missing_item, 0) + 1
        return dict(sorted(issues.items(), key=lambda x: x[1], reverse=True))
    
    def fix_all_readmes(self) -> Dict:
        """修復所有不合規的 README"""
        readme_files = self.scan_all_readmes()
        results = {
            "total_files": len(readme_files),
            "updated_files": 0,
            "failed_files": 0,
            "already_compliant": 0,
            "details": []
        }
        
        print(f"🔧 開始修復 {len(readme_files)} 個 README 檔案")
        print("=" * 60)
        
        for readme_path in readme_files:
            print(f"\n📝 處理: {readme_path.relative_to(self.workspace_root)}")
            
            # 檢查當前狀態
            before_check = self.check_readme_compliance(readme_path)
            
            if before_check["compliant"]:
                results["already_compliant"] += 1
                results["details"].append({
                    "file": str(readme_path.relative_to(self.workspace_root)),
                    "status": "already_compliant",
                    "before_score": before_check["compliance_score"],
                    "after_score": before_check["compliance_score"]
                })
                continue
            
            # 嘗試修復
            success = self.update_readme_with_compliance(readme_path)
            
            if success:
                # 重新檢查
                after_check = self.check_readme_compliance(readme_path)
                results["updated_files"] += 1
                results["details"].append({
                    "file": str(readme_path.relative_to(self.workspace_root)),
                    "status": "updated",
                    "before_score": before_check["compliance_score"],
                    "after_score": after_check["compliance_score"],
                    "improvements": [item for item in before_check["missing_items"] 
                                  if item not in after_check.get("missing_items", [])]
                })
            else:
                results["failed_files"] += 1
                results["details"].append({
                    "file": str(readme_path.relative_to(self.workspace_root)),
                    "status": "failed",
                    "before_score": before_check["compliance_score"],
                    "error": "更新失敗"
                })
        
        return results


def main():
    """主執行函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA README 合規性檢查工具")
    parser.add_argument("--workspace", default=".", help="工作區根目錄路徑")
    parser.add_argument("--check-only", action="store_true", help="僅檢查，不執行修復")
    parser.add_argument("--fix", action="store_true", help="自動修復不合規的檔案")
    parser.add_argument("--output", help="輸出結果到 JSON 檔案")
    
    args = parser.parse_args()
    
    checker = ReadmeComplianceChecker(args.workspace)
    
    if args.fix:
        print("🚀 執行自動修復模式")
        results = checker.fix_all_readmes()
        
        print("\n" + "=" * 60)
        print("📊 修復結果摘要")
        print("=" * 60)
        print(f"📁 總檔案數: {results['total_files']}")
        print(f"✅ 已合規檔案: {results['already_compliant']}")
        print(f"🔧 已修復檔案: {results['updated_files']}")
        print(f"❌ 修復失敗: {results['failed_files']}")
        
        if results['updated_files'] > 0:
            print(f"\n🎉 成功更新了 {results['updated_files']} 個檔案!")
            print("\n📝 更新詳情:")
            for detail in results['details']:
                if detail['status'] == 'updated':
                    score_improvement = detail['after_score'] - detail['before_score']
                    print(f"  ✅ {detail['file']}")
                    print(f"     合規性: {detail['before_score']:.1%} → {detail['after_score']:.1%} (+{score_improvement:.1%})")
                    if detail.get('improvements'):
                        print(f"     改進項目: {', '.join(detail['improvements'])}")
    
    else:
        print("🔍 執行合規性檢查")
        results = checker.run_full_check()
        
        print("\n" + "=" * 60)
        print("📊 檢查結果摘要")
        print("=" * 60)
        print(f"📁 總檔案數: {results['total_files']}")
        print(f"✅ 合規檔案: {results['compliant_files']}")
        print(f"❌ 不合規檔案: {results['non_compliant_files']}")
        print(f"📈 整體合規率: {results['summary']['compliance_rate']:.1%}")
        
        if results['summary']['common_issues']:
            print(f"\n🔍 常見問題統計:")
            for issue, count in results['summary']['common_issues'].items():
                print(f"  • {issue}: {count} 個檔案")
        
        if results['non_compliant_files'] > 0:
            print(f"\n💡 建議執行: python {__file__} --fix 來自動修復問題")
    
    # 輸出結果到檔案
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 結果已儲存到: {args.output}")


if __name__ == "__main__":
    main()