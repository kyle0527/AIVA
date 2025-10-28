#!/usr/bin/env python3
"""
AIVA 跨語言 Schema 合規性驗證工具
====================================

此工具用於驗證 AIVA 專案中各語言模組是否正確使用標準 schema，
確保遵循單一事實來源原則，防止 schema 漂移。

功能：
- 掃描 Go 模組，檢查是否使用標準 aiva_common_go schema
- 掃描 Rust 模組，檢查是否使用標準 schema
- 掃描 TypeScript 模組，檢查是否使用標準 schema
- 生成合規性報告
- 支持 CI/CD 集成

使用方法：
    python schema_compliance_validator.py --check-all
    python schema_compliance_validator.py --language go
    python schema_compliance_validator.py --ci-mode
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class ComplianceStatus(Enum):
    COMPLIANT = "✅ 合規"
    PARTIAL = "⚠️ 部分合規"  
    NON_COMPLIANT = "❌ 不合規"
    NOT_APPLICABLE = "N/A"

@dataclass
class SchemaIssue:
    file_path: str
    line_number: int
    issue_type: str
    description: str
    suggestion: str

@dataclass  
class ModuleCompliance:
    module_path: str
    language: str
    status: ComplianceStatus
    issues: List[SchemaIssue]
    compliance_score: float
    using_standard_schema: bool
    custom_schemas_found: List[str]

class SchemaComplianceValidator:
    """跨語言 Schema 合規性驗證器"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.results: List[ModuleCompliance] = []
        
        # 標準 schema 導入模式
        self.standard_imports = {
            'go': [
                r'github\.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas',
                r'schemas\s+"github\.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"',
                r'"github\.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"'
            ],
            'rust': [
                r'use\s+.*schemas::generated',
                r'mod\s+schemas',
                r'use\s+schemas::generated::'
            ],
            'typescript': [
                r'from\s+[\'"].*features/common/typescript/aiva_common_ts/schemas/generated/schemas[\'"]',
                r'import.*from\s+[\'"].*features/common/typescript/aiva_common_ts/schemas/generated/schemas[\'"]'
            ]
        }
        
        # 禁止的自定義 schema 模式（排除標準生成文件）
        self.forbidden_patterns = {
            'go': [
                r'type\s+FindingPayload\s+struct',
                r'type\s+Finding\s+struct.*\{[^}]*finding_id',
                r'type\s+Vulnerability\s+struct.*\{[^}]*severity'
            ],
            'rust': [
                r'struct\s+FindingPayload\s*\{',
                r'struct\s+Finding\s*\{[^}]*finding_id',
                r'struct\s+Vulnerability\s*\{[^}]*severity'
            ],
            'typescript': [
                r'interface\s+\w*Finding\w*.*\{[^}]*finding_id',
                r'interface\s+\w*Vulnerability\w*.*\{[^}]*severity',
                r'type\s+\w*Finding\w*\s*='
            ]
        }
        
        # 標準生成文件路徑（這些文件中的 schema 定義是允許的）
        self.standard_schema_paths = {
            'go': [
                'aiva_common_go/schemas/generated',
                'schemas/generated'
            ],
            'rust': [
                'schemas/generated/mod.rs',
                'src/schemas/generated/mod.rs'
            ],
            'typescript': [
                'features/common/typescript/aiva_common_ts/schemas/generated/schemas.d.ts',
                'aiva_common_ts/schemas/generated/schemas.d.ts'
            ]
        }

    def _is_standard_schema_file(self, file_path: Path, language: str) -> bool:
        """檢查文件是否為標準生成的 schema 文件"""
        file_str = str(file_path).replace('\\', '/')
        for standard_path in self.standard_schema_paths[language]:
            if standard_path in file_str:
                return True
        return False

    def scan_all_modules(self) -> List[ModuleCompliance]:
        """掃描所有模組"""
        print("🔍 開始掃描所有模組...")
        
        # 掃描 Go 模組
        go_modules = self._find_go_modules()
        for module in go_modules:
            compliance = self._check_go_module(module)
            self.results.append(compliance)
            
        # 掃描 Rust 模組  
        rust_modules = self._find_rust_modules()
        for module in rust_modules:
            compliance = self._check_rust_module(module)
            self.results.append(compliance)
            
        # 掃描 TypeScript 模組
        ts_modules = self._find_typescript_modules()
        for module in ts_modules:
            compliance = self._check_typescript_module(module)
            self.results.append(compliance)
            
        return self.results

    def _find_go_modules(self) -> List[Path]:
        """尋找 Go 模組"""
        modules = []
        
        # 掃描 services/features 下的 Go 模組
        features_dir = self.workspace_root / "services" / "features"
        if features_dir.exists():
            for item in features_dir.iterdir():
                if item.is_dir() and item.name.endswith("_go"):
                    go_mod = item / "go.mod"
                    if go_mod.exists():
                        modules.append(item)
                        
        return modules

    def _find_rust_modules(self) -> List[Path]:
        """尋找 Rust 模組"""
        modules = []
        
        # 掃描 services 下的 Rust 模組
        services_dir = self.workspace_root / "services"
        if services_dir.exists():
            for root, dirs, files in os.walk(services_dir):
                if "Cargo.toml" in files:
                    root_path = Path(root)
                    # 檢查是否為 Rust 模組（包含 src 目錄）
                    if (root_path / "src").exists():
                        modules.append(root_path)
                        
        return modules

    def _find_typescript_modules(self) -> List[Path]:
        """尋找 TypeScript 模組"""
        modules = []
        
        # 掃描 services 下的 TypeScript/JavaScript 模組
        services_dir = self.workspace_root / "services"
        if services_dir.exists():
            for root, dirs, files in os.walk(services_dir):
                # 排除 node_modules 等目錄
                dirs[:] = [d for d in dirs if d not in ["node_modules", "dist", "build", "target", "__pycache__"]]
                
                # 檢查是否有 TypeScript 配置檔案
                if any(f in files for f in ["tsconfig.json", "package.json"]):
                    root_path = Path(root)
                    # 只包含非依賴模組
                    if "node_modules" not in str(root_path):
                        # 檢查是否有 TypeScript 檔案
                        ts_files = list(root_path.rglob("*.ts"))
                        if ts_files:
                            modules.append(root_path)
                        
        return modules

    def _check_go_module(self, module_path: Path) -> ModuleCompliance:
        """檢查 Go 模組合規性"""
        issues = []
        using_standard = False
        custom_schemas = []
        
        # 掃描所有 Go 檔案
        go_files = list(module_path.rglob("*.go"))
        
        for go_file in go_files:
            try:
                content = go_file.read_text(encoding='utf-8')
                lines = content.split('\\n')
                
                # 檢查標準 schema 導入
                for i, line in enumerate(lines):
                    for pattern in self.standard_imports['go']:
                        if re.search(pattern, line):
                            using_standard = True
                            break
                
                # 檢查禁止的自定義 schema（但排除標準生成文件）
                if not self._is_standard_schema_file(go_file, 'go'):
                    for i, line in enumerate(lines):
                        for pattern in self.forbidden_patterns['go']:
                            if re.search(pattern, line):
                                custom_schemas.append(f"{go_file.name}:{i+1}")
                                issues.append(SchemaIssue(
                                    file_path=str(go_file.relative_to(self.workspace_root)),
                                    line_number=i+1,
                                    issue_type="自定義 Schema",
                                    description=f"發現自定義 schema 定義: {line.strip()}",
                                    suggestion="使用 aiva_common_go/schemas/generated 中的標準定義"
                                ))
            except Exception as e:
                issues.append(SchemaIssue(
                    file_path=str(go_file.relative_to(self.workspace_root)),
                    line_number=0,
                    issue_type="讀取錯誤",
                    description=f"無法讀取檔案: {e}",
                    suggestion="檢查檔案編碼或權限"
                ))
        
        # 計算合規分數
        compliance_score = self._calculate_compliance_score(using_standard, len(issues), len(go_files))
        status = self._determine_status(compliance_score, using_standard, len(issues))
        
        return ModuleCompliance(
            module_path=str(module_path.relative_to(self.workspace_root)),
            language="Go",
            status=status,
            issues=issues,
            compliance_score=compliance_score,
            using_standard_schema=using_standard,
            custom_schemas_found=custom_schemas
        )

    def _check_rust_module(self, module_path: Path) -> ModuleCompliance:
        """檢查 Rust 模組合規性"""
        issues = []
        using_standard = False
        custom_schemas = []
        
        # 掃描所有 Rust 檔案
        rust_files = list(module_path.rglob("*.rs"))
        
        # 特別檢查是否有完整的 schema 實現
        schema_mod = module_path / "src" / "schemas" / "generated" / "mod.rs"
        if schema_mod.exists():
            try:
                content = schema_mod.read_text(encoding='utf-8')
                if "FindingPayload" in content and "impl" in content:
                    using_standard = True
                elif content.strip() == "" or "TODO" in content:
                    issues.append(SchemaIssue(
                        file_path=str(schema_mod.relative_to(self.workspace_root)),
                        line_number=1,
                        issue_type="未完整實現",
                        description="Schema 檔案存在但未完整實現",
                        suggestion="完成 Rust schema 生成實現"
                    ))
            except Exception as e:
                issues.append(SchemaIssue(
                    file_path=str(schema_mod.relative_to(self.workspace_root)),
                    line_number=0,
                    issue_type="讀取錯誤", 
                    description=f"無法讀取 schema 檔案: {e}",
                    suggestion="檢查檔案編碼或權限"
                ))
        
        for rust_file in rust_files:
            try:
                content = rust_file.read_text(encoding='utf-8')
                lines = content.split('\\n')
                
                # 檢查自定義 schema 定義（排除標準 schema 檔案）
                if not self._is_standard_schema_file(rust_file, 'rust'):
                    for i, line in enumerate(lines):
                        for pattern in self.forbidden_patterns['rust']:
                            if re.search(pattern, line):
                                custom_schemas.append(f"{rust_file.name}:{i+1}")
                                issues.append(SchemaIssue(
                                    file_path=str(rust_file.relative_to(self.workspace_root)),
                                    line_number=i+1,
                                    issue_type="自定義 Schema",
                                    description=f"發現自定義 schema 定義: {line.strip()}",
                                    suggestion="使用標準 schema 模組"
                                ))
            except Exception as e:
                continue
        
        compliance_score = self._calculate_compliance_score(using_standard, len(issues), len(rust_files))
        status = self._determine_status(compliance_score, using_standard, len(issues))
        
        return ModuleCompliance(
            module_path=str(module_path.relative_to(self.workspace_root)),
            language="Rust", 
            status=status,
            issues=issues,
            compliance_score=compliance_score,
            using_standard_schema=using_standard,
            custom_schemas_found=custom_schemas
        )

    def _check_typescript_module(self, module_path: Path) -> ModuleCompliance:
        """檢查 TypeScript 模組合規性"""
        issues = []
        using_standard = False
        custom_schemas = []
        
        # 掃描所有 TypeScript 檔案
        ts_files = list(module_path.rglob("*.ts"))
        
        for ts_file in ts_files:
            try:
                content = ts_file.read_text(encoding='utf-8')
                lines = content.split('\\n')
                
                # 檢查標準 schema 導入
                for i, line in enumerate(lines):
                    for pattern in self.standard_imports['typescript']:
                        if re.search(pattern, line):
                            using_standard = True
                            break
                
                # 檢查自定義 schema 定義（排除標準 schema 檔案）
                if not self._is_standard_schema_file(ts_file, 'typescript'):
                    for i, line in enumerate(lines):
                        for pattern in self.forbidden_patterns['typescript']:
                            if re.search(pattern, line):
                                custom_schemas.append(f"{ts_file.name}:{i+1}")
                                issues.append(SchemaIssue(
                                    file_path=str(ts_file.relative_to(self.workspace_root)),
                                    line_number=i+1,
                                    issue_type="自定義 Schema",
                                    description=f"發現自定義 schema 定義: {line.strip()}",
                                    suggestion="使用 aiva_common_ts/schemas/generated/schemas.d.ts 中的標準定義"
                                ))
            except Exception as e:
                continue
        
        compliance_score = self._calculate_compliance_score(using_standard, len(issues), len(ts_files))
        status = self._determine_status(compliance_score, using_standard, len(issues))
        
        return ModuleCompliance(
            module_path=str(module_path.relative_to(self.workspace_root)),
            language="TypeScript",
            status=status,
            issues=issues,
            compliance_score=compliance_score,
            using_standard_schema=using_standard,
            custom_schemas_found=custom_schemas
        )

    def _calculate_compliance_score(self, using_standard: bool, issue_count: int, file_count: int) -> float:
        """計算合規分數 (0-100)"""
        if file_count == 0:
            return 100.0
            
        base_score = 100.0 if using_standard else 0.0
        penalty = min(issue_count * 10, 80)  # 每個問題扣 10 分，最多扣 80 分
        
        return max(base_score - penalty, 0.0)

    def _determine_status(self, score: float, using_standard: bool, issue_count: int) -> ComplianceStatus:
        """確定合規狀態"""
        if score >= 90 and using_standard and issue_count == 0:
            return ComplianceStatus.COMPLIANT
        elif score >= 60 and using_standard:
            return ComplianceStatus.PARTIAL
        else:
            return ComplianceStatus.NON_COMPLIANT

    def generate_report(self, output_format: str = "console") -> str:
        """生成合規性報告"""
        if output_format == "json":
            return self._generate_json_report()
        elif output_format == "markdown":
            return self._generate_markdown_report()
        else:
            return self._generate_console_report()

    def _generate_console_report(self) -> str:
        """生成控制台報告"""
        report = []
        report.append("\\n" + "="*60)
        report.append("🔍 AIVA 跨語言 Schema 合規性報告")
        report.append("="*60)
        
        # 統計摘要
        total = len(self.results)
        compliant = sum(1 for r in self.results if r.status == ComplianceStatus.COMPLIANT)
        partial = sum(1 for r in self.results if r.status == ComplianceStatus.PARTIAL)
        non_compliant = sum(1 for r in self.results if r.status == ComplianceStatus.NON_COMPLIANT)
        
        avg_score = sum(r.compliance_score for r in self.results) / total if total > 0 else 0
        
        report.append(f"\\n📊 總覽統計:")
        report.append(f"  • 總模組數: {total}")
        report.append(f"  • ✅ 完全合規: {compliant} ({compliant/total*100:.1f}%)")
        report.append(f"  • ⚠️ 部分合規: {partial} ({partial/total*100:.1f}%)")
        report.append(f"  • ❌ 不合規: {non_compliant} ({non_compliant/total*100:.1f}%)")
        report.append(f"  • 📈 平均分數: {avg_score:.1f}/100")
        
        # 按語言分組
        by_language = {}
        for result in self.results:
            if result.language not in by_language:
                by_language[result.language] = []
            by_language[result.language].append(result)
        
        for language, modules in by_language.items():
            report.append(f"\\n🔧 {language} 模組 ({len(modules)} 個):")
            for module in modules:
                status_icon = module.status.value
                score = f"{module.compliance_score:.1f}"
                issue_count = len(module.issues)
                
                report.append(f"  {status_icon} {module.module_path}")
                report.append(f"    分數: {score}/100 | 問題: {issue_count} 個 | 標準Schema: {'是' if module.using_standard_schema else '否'}")
                
                if module.issues:
                    report.append("    主要問題:")
                    for issue in module.issues[:3]:  # 只顯示前3個問題
                        report.append(f"      • {issue.issue_type}: {issue.description}")
                    if len(module.issues) > 3:
                        report.append(f"      • ... 還有 {len(module.issues) - 3} 個問題")
        
        # 改進建議
        report.append("\\n💡 改進建議:")
        if non_compliant > 0:
            report.append("  1. 優先修復不合規模組，使用標準 schema 定義")
        if partial > 0:
            report.append("  2. 解決部分合規模組的剩餘問題")
        report.append("  3. 建立 CI/CD 檢查防止 schema 漂移")
        report.append("  4. 更新開發規範，禁止自定義 Finding 相關結構")
        
        return "\\n".join(report)

    def _generate_json_report(self) -> str:
        """生成 JSON 格式報告"""
        data = {
            "timestamp": "2025-10-26T00:00:00Z",
            "summary": {
                "total_modules": len(self.results),
                "compliant": sum(1 for r in self.results if r.status == ComplianceStatus.COMPLIANT),
                "partial": sum(1 for r in self.results if r.status == ComplianceStatus.PARTIAL),
                "non_compliant": sum(1 for r in self.results if r.status == ComplianceStatus.NON_COMPLIANT),
                "average_score": sum(r.compliance_score for r in self.results) / len(self.results) if self.results else 0
            },
            "modules": []
        }
        
        for result in self.results:
            module_data = {
                "path": result.module_path,
                "language": result.language,
                "status": result.status.name,
                "compliance_score": result.compliance_score,
                "using_standard_schema": result.using_standard_schema,
                "issues": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "description": issue.description,
                        "suggestion": issue.suggestion
                    }
                    for issue in result.issues
                ]
            }
            data["modules"].append(module_data)
        
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _generate_markdown_report(self) -> str:
        """生成 Markdown 格式報告"""
        report = []
        report.append("# AIVA 跨語言 Schema 合規性報告\\n")
        report.append(f"**生成時間**: 2025年10月26日\\n")
        
        # 統計摘要
        total = len(self.results)
        compliant = sum(1 for r in self.results if r.status == ComplianceStatus.COMPLIANT)
        partial = sum(1 for r in self.results if r.status == ComplianceStatus.PARTIAL)
        non_compliant = sum(1 for r in self.results if r.status == ComplianceStatus.NON_COMPLIANT)
        avg_score = sum(r.compliance_score for r in self.results) / total if total > 0 else 0
        
        report.append("## 📊 合規性統計\\n")
        report.append(f"| 指標 | 數量 | 比例 |")
        report.append(f"|------|------|------|")
        report.append(f"| 總模組數 | {total} | 100% |")
        report.append(f"| ✅ 完全合規 | {compliant} | {compliant/total*100:.1f}% |")
        report.append(f"| ⚠️ 部分合規 | {partial} | {partial/total*100:.1f}% |")
        report.append(f"| ❌ 不合規 | {non_compliant} | {non_compliant/total*100:.1f}% |")
        report.append(f"| 📈 平均分數 | {avg_score:.1f}/100 | - |\\n")
        
        # 詳細結果
        report.append("## 🔍 詳細結果\\n")
        
        by_language = {}
        for result in self.results:
            if result.language not in by_language:
                by_language[result.language] = []
            by_language[result.language].append(result)
        
        for language, modules in by_language.items():
            report.append(f"### {language} 模組\\n")
            
            for module in modules:
                status_emoji = "✅" if module.status == ComplianceStatus.COMPLIANT else "⚠️" if module.status == ComplianceStatus.PARTIAL else "❌"
                report.append(f"#### {status_emoji} `{module.module_path}`\\n")
                report.append(f"- **分數**: {module.compliance_score:.1f}/100")
                report.append(f"- **使用標準Schema**: {'是' if module.using_standard_schema else '否'}")
                report.append(f"- **問題數量**: {len(module.issues)}\\n")
                
                if module.issues:
                    report.append("**問題列表**:\\n")
                    for issue in module.issues:
                        report.append(f"- 📁 `{issue.file_path}:{issue.line_number}`")
                        report.append(f"  - **類型**: {issue.issue_type}")
                        report.append(f"  - **描述**: {issue.description}")
                        report.append(f"  - **建議**: {issue.suggestion}\\n")
        
        return "\\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="AIVA 跨語言 Schema 合規性驗證工具")
    parser.add_argument("--workspace", default=".", help="工作區根目錄路徑")
    parser.add_argument("--language", choices=["go", "rust", "typescript", "all"], default="all", help="指定檢查的語言")
    parser.add_argument("--format", choices=["console", "json", "markdown"], default="console", help="輸出格式")
    parser.add_argument("--output", help="輸出檔案路徑")
    parser.add_argument("--ci-mode", action="store_true", help="CI/CD 模式，非零退出碼表示有問題")
    parser.add_argument("--threshold", type=float, default=80.0, help="CI 模式的最低合規分數閾值")
    
    args = parser.parse_args()
    
    # 建立驗證器
    validator = SchemaComplianceValidator(args.workspace)
    
    # 執行掃描
    results = validator.scan_all_modules()
    
    # 生成報告
    report = validator.generate_report(args.format)
    
    # 輸出報告
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"報告已保存到: {args.output}")
    else:
        print(report)
    
    # CI 模式檢查
    if args.ci_mode:
        avg_score = sum(r.compliance_score for r in results) / len(results) if results else 0
        non_compliant_count = sum(1 for r in results if r.status == ComplianceStatus.NON_COMPLIANT)
        
        if avg_score < args.threshold or non_compliant_count > 0:
            print(f"\\n❌ CI 檢查失敗: 平均分數 {avg_score:.1f} 低於閾值 {args.threshold} 或有不合規模組")
            sys.exit(1)
        else:
            print(f"\\n✅ CI 檢查通過: 平均分數 {avg_score:.1f}")
            sys.exit(0)

if __name__ == "__main__":
    main()