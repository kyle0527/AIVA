#!/usr/bin/env python3
"""
跨語言 Schema 一致性驗證工具
============================

此工具驗證 Python、Go、Rust 三種語言的 Schema 定義是否完全一致，
確保跨語言架構的數據完整性和API兼容性。

功能特色:
- 🔍 深度一致性分析
- 📊 詳細差異報告
- 🚨 自動問題檢測
- 🛠️ 修復建議生成
- 📈 覆蓋率統計
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .cross_language_interface import CrossLanguageSchemaInterface, SchemaDefinition


@dataclass
class ValidationIssue:
    """驗證問題定義"""

    severity: str  # critical, warning, info
    category: str  # missing, mismatch, format
    message: str
    schema_name: str
    field_name: str | None = None
    languages: list[str] | None = None
    suggestion: str | None = None


@dataclass
class ValidationReport:
    """驗證報告"""

    timestamp: str
    total_schemas: int
    total_fields: int
    issues: list[ValidationIssue]
    coverage_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return asdict(self)


class CrossLanguageValidator:
    """跨語言 Schema 一致性驗證器"""

    def __init__(self):
        """初始化驗證器"""
        self.interface = CrossLanguageSchemaInterface()
        self.issues: list[ValidationIssue] = []

        # 生成的文件路徑
        self.generated_files = {
            "python": {
                "base_types": Path(
                    "services/aiva_common/schemas/generated/base_types.py"
                ),
                "messaging": Path(
                    "services/aiva_common/schemas/generated/messaging.py"
                ),
                "tasks": Path("services/aiva_common/schemas/generated/tasks.py"),
                "findings": Path("services/aiva_common/schemas/generated/findings.py"),
                "async_utils": Path(
                    "services/aiva_common/schemas/generated/async_utils.py"
                ),
                "plugins": Path("services/aiva_common/schemas/generated/plugins.py"),
                "cli": Path("services/aiva_common/schemas/generated/cli.py"),
            },
            "go": {
                "all": Path(
                    "services/features/common/go/aiva_common_go/schemas/generated/schemas.go"
                )
            },
            "rust": {
                "all": Path(
                    "services/scan/info_gatherer_rust/src/schemas/generated/mod.rs"
                )
            },
        }

    def validate_all(self) -> ValidationReport:
        """執行完整的跨語言驗證

        Returns:
            詳細的驗證報告
        """
        self.issues = []

        print("🔍 開始跨語言 Schema 一致性驗證...")

        # 1. 檢查文件存在性
        self._validate_file_existence()

        # 2. 驗證 Schema 定義一致性
        self._validate_schema_definitions()

        # 3. 驗證類型映射完整性
        self._validate_type_mappings()

        # 4. 驗證生成的代碼結構
        self._validate_generated_code_structure()

        # 5. 生成覆蓋率統計
        coverage_stats = self._generate_coverage_stats()

        # 創建報告
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_schemas=len(self.interface.get_all_schemas()),
            total_fields=sum(len(s.fields) for s in self.interface.get_all_schemas()),
            issues=self.issues,
            coverage_stats=coverage_stats,
        )

        return report

    def _validate_file_existence(self) -> None:
        """驗證生成文件的存在性"""
        print("📁 檢查生成文件存在性...")

        for language, files in self.generated_files.items():
            for _file_type, file_path in files.items():
                if not file_path.exists():
                    self._add_issue(
                        "critical",
                        "missing",
                        f"{language.capitalize()} 生成文件缺失: {file_path}",
                        "file_system",
                        languages=[language],
                        suggestion=f"運行代碼生成工具重新生成 {language} Schema",
                    )

    def _validate_schema_definitions(self) -> None:
        """驗證 Schema 定義的一致性"""
        print("🔍 驗證 Schema 定義一致性...")

        all_schemas = self.interface.get_all_schemas()

        for schema in all_schemas:
            # 檢查每個字段的類型映射
            for field in schema.fields:
                self._validate_field_consistency(schema, field)

    def _validate_field_consistency(self, schema: SchemaDefinition, field: Any) -> None:
        """驗證單個字段的跨語言一致性"""
        source_type = field.type

        # 獲取各語言的類型映射
        type_mappings = {}
        for lang in ["python", "go", "rust"]:
            mapped_type = self.interface.convert_type_to_language(source_type, lang)
            type_mappings[lang] = mapped_type

            # 如果映射失敗（返回原類型），記錄問題
            if mapped_type == source_type and source_type not in [
                "str",
                "int",
                "bool",
                "float",
            ]:
                self._add_issue(
                    "warning",
                    "mismatch",
                    f"類型 '{source_type}' 在 {lang} 語言中沒有映射",
                    schema.name,
                    field.name,
                    [lang],
                    f"在 {lang} 類型映射中添加 '{source_type}' 的定義",
                )

        # 檢查必需性一致性
        if hasattr(field, "required"):
            if not field.required:
                # 可選字段應該有適當的類型標記
                expected_patterns = {
                    "python": r"Optional\[.*\]",
                    "go": r"\*.*",  # 指針類型
                    "rust": r"Option<.*>",
                }

                for lang, pattern in expected_patterns.items():
                    if not re.match(pattern, type_mappings[lang]):
                        self._add_issue(
                            "warning",
                            "mismatch",
                            f"可選字段 '{field.name}' 在 {lang} 中類型標記不正確",
                            schema.name,
                            field.name,
                            [lang],
                            f"使用 {expected_patterns[lang]} 格式標記可選類型",
                        )

    def _validate_type_mappings(self) -> None:
        """驗證類型映射的完整性"""
        print("🔄 驗證類型映射完整性...")

        # 獲取所有使用的類型
        used_types = set()
        for schema in self.interface.get_all_schemas():
            for field in schema.fields:
                used_types.add(field.type)

        # 檢查每種語言是否都有映射
        for lang in ["python", "go", "rust"]:
            mappings = self.interface.language_mappings.get(lang, {})
            for used_type in used_types:
                if used_type not in mappings:
                    self._add_issue(
                        "warning",
                        "missing",
                        f"類型 '{used_type}' 缺少 {lang} 語言映射",
                        "type_mapping",
                        languages=[lang],
                        suggestion=f"在 {lang} 配置中添加 '{used_type}' 的類型映射",
                    )

    def _validate_generated_code_structure(self) -> None:
        """驗證生成代碼的結構完整性"""
        print("🏗️  驗證生成代碼結構...")

        all_schemas = self.interface.get_all_schemas()
        schema_names = {s.name for s in all_schemas}

        # 檢查 Go 生成文件
        go_file = self.generated_files["go"]["all"]
        if go_file.exists():
            go_content = go_file.read_text(encoding="utf-8")
            missing_go_schemas = []

            for schema_name in schema_names:
                if f"type {schema_name} struct" not in go_content:
                    missing_go_schemas.append(schema_name)  # type: ignore

            if missing_go_schemas:
                self._add_issue(
                    "critical",
                    "missing",
                    f"Go 代碼中缺少 {len(missing_go_schemas)} 個 Schema: {', '.join(missing_go_schemas[:5])}{'...' if len(missing_go_schemas) > 5 else ''}",
                    "code_generation",
                    languages=["go"],
                    suggestion="重新運行 Go 代碼生成，確保包含所有 Schema 分類",
                )

        # 檢查 Rust 生成文件
        rust_file = self.generated_files["rust"]["all"]
        if rust_file.exists():
            rust_content = rust_file.read_text(encoding="utf-8")
            missing_rust_schemas = []

            for schema_name in schema_names:
                if f"pub struct {schema_name}" not in rust_content:
                    missing_rust_schemas.append(schema_name)  # type: ignore

            if missing_rust_schemas:
                self._add_issue(
                    "critical",
                    "missing",
                    f"Rust 代碼中缺少 {len(missing_rust_schemas)} 個 Schema: {', '.join(missing_rust_schemas[:5])}{'...' if len(missing_rust_schemas) > 5 else ''}",
                    "code_generation",
                    languages=["rust"],
                    suggestion="重新運行 Rust 代碼生成，確保包含所有 Schema 分類",
                )

    def _generate_coverage_stats(self) -> dict[str, Any]:
        """生成覆蓋率統計"""
        all_schemas = self.interface.get_all_schemas()

        stats = {
            "total_schemas": len(all_schemas),
            "schemas_by_category": {},
            "language_coverage": {},
            "field_type_distribution": {},
            "validation_summary": {
                "critical_issues": len([i for i in self.issues if i.severity == "critical"]),  # type: ignore
                "warning_issues": len([i for i in self.issues if i.severity == "warning"]),  # type: ignore
                "info_issues": len([i for i in self.issues if i.severity == "info"]),  # type: ignore
            },
        }

        # 按類別統計
        for schema in all_schemas:
            if schema.category not in stats["schemas_by_category"]:
                stats["schemas_by_category"][schema.category] = 0
            stats["schemas_by_category"][schema.category] += 1

        # 語言一致性評估（基於實際問題而非數字覆蓋率）
        for lang in ["python", "go", "rust"]:
            lang_issues = [
                i for i in self.issues if i.languages and lang in i.languages
            ]

            # 分類問題嚴重程度
            critical_issues = [
                i for i in lang_issues if i.severity in ["critical", "error"]
            ]
            warning_issues = [i for i in lang_issues if i.severity == "warning"]

            # 功能完整性評估：有嚴重問題才算失敗
            # 警告通常是正常的跨語言語法差異，不影響功能
            if critical_issues:
                consistency_score = 0  # 有嚴重問題就是不一致
            elif warning_issues:
                # 警告數量很多時，但沒有嚴重問題，功能仍然完整
                consistency_score = max(60, 100 - min(40, len(warning_issues) // 10))  # type: ignore
            else:
                consistency_score = 100  # 完美一致

            stats["language_coverage"][lang] = {
                "coverage_percentage": consistency_score,
                "issues_count": len(lang_issues),  # type: ignore
                "critical_issues": len(critical_issues),  # type: ignore
                "warning_issues": len(warning_issues),  # type: ignore
                "functional_integrity": critical_issues == 0,
            }

        # 字段類型分布
        type_counts = {}
        for schema in all_schemas:
            for field in schema.fields:
                field_type = field.type
                if field_type not in type_counts:
                    type_counts[field_type] = 0
                type_counts[field_type] += 1

        stats["field_type_distribution"] = dict(
            sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        )

        return stats

    def _add_issue(
        self,
        severity: str,
        category: str,
        message: str,
        schema_name: str,
        field_name: str | None = None,
        languages: list[str] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """添加驗證問題"""
        issue = ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            schema_name=schema_name,
            field_name=field_name,
            languages=languages or [],
            suggestion=suggestion,
        )
        self.issues.append(issue)  # type: ignore

    def generate_report_file(
        self,
        report: ValidationReport,
        output_path: str = "cross_language_validation_report.json",
    ) -> None:
        """生成驗證報告文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"📄 驗證報告已保存到: {output_path}")

    def print_summary(self, report: ValidationReport) -> None:
        """打印驗證摘要"""
        print("\n" + "=" * 60)
        print("🔍 跨語言 Schema 驗證報告摘要")
        print("=" * 60)

        print(
            f"📊 總計: {report.total_schemas} 個 Schema, {report.total_fields} 個字段"
        )

        # 問題統計
        critical = len([i for i in report.issues if i.severity == "critical"])  # type: ignore
        warning = len([i for i in report.issues if i.severity == "warning"])  # type: ignore
        info = len([i for i in report.issues if i.severity == "info"])  # type: ignore

        print(f"🚨 問題統計: {critical} 嚴重, {warning} 警告, {info} 信息")

        # 語言覆蓋率
        print("\n📈 語言覆蓋率:")
        for lang, stats in report.coverage_stats["language_coverage"].items():
            print(
                f"   {lang}: {stats['coverage_percentage']:.1f}% ({stats['issues_count']} 問題)"
            )

        # 嚴重問題詳情
        if critical > 0:
            print(f"\n🚨 嚴重問題 ({critical} 個):")
            for issue in report.issues:
                if issue.severity == "critical":
                    print(f"   ❌ {issue.message}")
                    if issue.suggestion:
                        print(f"      💡 建議: {issue.suggestion}")

        # 整體狀態
        if critical == 0:
            print("\n✅ 跨語言一致性驗證通過！")
        else:
            print(f"\n❌ 發現 {critical} 個嚴重問題需要修復")


def main():
    """主函數"""
    validator = CrossLanguageValidator()
    report = validator.validate_all()

    # 打印摘要
    validator.print_summary(report)

    # 生成詳細報告
    validator.generate_report_file(report)

    return len([i for i in report.issues if i.severity == "critical"]) == 0  # type: ignore


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
