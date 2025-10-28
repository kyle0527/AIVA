#!/usr/bin/env python3
"""
AIVA Schema 合規性 CI/CD 集成腳本
================================

此腳本專為 CI/CD 流水線設計，提供：
- 快速合規性檢查
- GitHub Actions 集成
- 自動化報告生成
- 失敗時的詳細診斷

支持的 CI/CD 系統：
- GitHub Actions
- Azure DevOps
- Jenkins
- GitLab CI

使用範例：
    # 基本檢查
    python ci_schema_check.py
    
    # 生成 PR 評論報告
    python ci_schema_check.py --pr-comment
    
    # 嚴格模式（任何問題都失敗）
    python ci_schema_check.py --strict
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CIResult:
    success: bool
    exit_code: int
    summary: str
    detailed_report: str
    metrics: Dict[str, float]
    recommendations: List[str]

class CISchemaChecker:
    """CI/CD Schema 合規性檢查器"""
    
    def __init__(self):
        self.workspace = Path.cwd()
        self.is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
        self.is_azure_devops = os.getenv('TF_BUILD') == 'True'
        self.is_jenkins = os.getenv('JENKINS_URL') is not None
        self.is_gitlab = os.getenv('GITLAB_CI') == 'true'
        
    def run_compliance_check(self, strict_mode: bool = False, threshold: float = 80.0) -> CIResult:
        """執行合規性檢查"""
        try:
            # 執行主要驗證工具
            validator_path = self.workspace / "tools" / "schema_compliance_validator.py"
            
            if not validator_path.exists():
                return CIResult(
                    success=False,
                    exit_code=1,
                    summary="❌ Schema 驗證工具不存在",
                    detailed_report="找不到 schema_compliance_validator.py",
                    metrics={},
                    recommendations=["確保 tools/schema_compliance_validator.py 存在"]
                )
            
            # 構建命令
            cmd = [
                sys.executable, 
                str(validator_path),
                "--workspace", str(self.workspace),
                "--format", "json",
                "--ci-mode",
                "--threshold", str(threshold)
            ]
            
            # 執行檢查
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='ignore'  # 忽略編碼錯誤
            )
            
            # 解析結果
            if result.stdout:
                try:
                    # 嘗試從輸出中找到 JSON 數據
                    lines = result.stdout.strip().split('\n')
                    json_line = None
                    for line in reversed(lines):
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            json_line = line
                            break
                    
                    if not json_line:
                        # 如果沒找到 JSON，嘗試最後一行
                        json_line = lines[-1] if lines else "{}"
                    
                    data = json.loads(json_line)
                    metrics = data.get('summary', {})
                    modules = data.get('modules', [])
                    
                    # 分析結果
                    total_modules = metrics.get('total_modules', 0)
                    compliant = metrics.get('compliant', 0)
                    non_compliant = metrics.get('non_compliant', 0)
                    avg_score = metrics.get('average_score', 0)
                    
                    # 確定成功狀態
                    success = result.returncode == 0
                    if strict_mode:
                        success = success and non_compliant == 0
                    
                    # 生成摘要
                    if success:
                        summary = f"✅ Schema 合規性檢查通過 ({compliant}/{total_modules} 模組合規, 平均分數: {avg_score:.1f})"
                    else:
                        summary = f"❌ Schema 合規性檢查失敗 ({non_compliant} 個不合規模組, 平均分數: {avg_score:.1f})"
                    
                    # 生成詳細報告
                    detailed_report = self._generate_ci_report(data, strict_mode)
                    
                    # 生成建議
                    recommendations = self._generate_recommendations(modules, strict_mode)
                    
                    return CIResult(
                        success=success,
                        exit_code=0 if success else 1,
                        summary=summary,
                        detailed_report=detailed_report,
                        metrics=metrics,
                        recommendations=recommendations
                    )
                    
                except json.JSONDecodeError:
                    # 如果 JSON 解析失敗，使用控制台輸出
                    return CIResult(
                        success=result.returncode == 0,
                        exit_code=result.returncode,
                        summary="Schema 合規性檢查完成" if result.returncode == 0 else "Schema 合規性檢查失敗",
                        detailed_report=result.stdout + result.stderr,
                        metrics={},
                        recommendations=["檢查 schema_compliance_validator.py 輸出格式"]
                    )
            else:
                return CIResult(
                    success=False,
                    exit_code=1,
                    summary="❌ 驗證工具無輸出",
                    detailed_report=result.stderr or "未知錯誤",
                    metrics={},
                    recommendations=["檢查 Python 環境和依賴"]
                )
                
        except Exception as e:
            return CIResult(
                success=False,
                exit_code=1,
                summary=f"❌ 執行過程出錯: {e}",
                detailed_report=str(e),
                metrics={},
                recommendations=["檢查環境配置和權限"]
            )
    
    def _generate_ci_report(self, data: Dict, strict_mode: bool) -> str:
        """生成 CI 適用的詳細報告"""
        report = []
        
        summary = data.get('summary', {})
        modules = data.get('modules', [])
        
        # 標題
        report.append("# AIVA Schema 合規性檢查報告")
        report.append("")
        
        # 統計摘要
        report.append("## 📊 檢查統計")
        report.append(f"- **總模組數**: {summary.get('total_modules', 0)}")
        report.append(f"- **✅ 完全合規**: {summary.get('compliant', 0)}")
        report.append(f"- **⚠️ 部分合規**: {summary.get('partial', 0)}")
        report.append(f"- **❌ 不合規**: {summary.get('non_compliant', 0)}")
        report.append(f"- **📈 平均分數**: {summary.get('average_score', 0):.1f}/100")
        report.append("")
        
        # 問題模組
        problem_modules = [m for m in modules if m['status'] != 'COMPLIANT']
        if problem_modules:
            report.append("## ⚠️ 需要關注的模組")
            report.append("")
            
            for module in problem_modules[:10]:  # 限制顯示數量
                status_emoji = "❌" if module['status'] == 'NON_COMPLIANT' else "⚠️"
                report.append(f"### {status_emoji} `{module['path']}`")
                report.append(f"- **語言**: {module['language']}")
                report.append(f"- **分數**: {module['compliance_score']:.1f}/100")
                report.append(f"- **使用標準Schema**: {'是' if module['using_standard_schema'] else '否'}")
                
                if module['issues']:
                    report.append("- **主要問題**:")
                    for issue in module['issues'][:3]:  # 只顯示前3個問題
                        report.append(f"  - `{issue['file']}:{issue['line']}` - {issue['description']}")
                    if len(module['issues']) > 3:
                        report.append(f"  - ... 還有 {len(module['issues']) - 3} 個問題")
                
                report.append("")
        
        # 成功的模組
        compliant_modules = [m for m in modules if m['status'] == 'COMPLIANT']
        if compliant_modules:
            report.append(f"## ✅ 合規模組 ({len(compliant_modules)} 個)")
            report.append("")
            for module in compliant_modules:
                report.append(f"- `{module['path']}` ({module['language']}) - {module['compliance_score']:.1f}/100")
            report.append("")
        
        return "\\n".join(report)
    
    def _generate_recommendations(self, modules: List[Dict], strict_mode: bool) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        non_compliant = [m for m in modules if m['status'] == 'NON_COMPLIANT']
        partial = [m for m in modules if m['status'] == 'PARTIAL']
        
        if non_compliant:
            recommendations.append(f"優先修復 {len(non_compliant)} 個不合規模組")
            
            # 按語言分組建議
            by_lang = {}
            for module in non_compliant:
                lang = module['language']
                if lang not in by_lang:
                    by_lang[lang] = []
                by_lang[lang].append(module['path'])
            
            for lang, paths in by_lang.items():
                if lang == "Go":
                    recommendations.append("Go 模組請使用: import schemas \"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated\"")
                elif lang == "Rust":
                    recommendations.append("Rust 模組請實現完整的 schemas::generated 模組")
                elif lang == "TypeScript":
                    recommendations.append("TypeScript 模組請使用: import { FindingPayload } from '../../../schemas/aiva_schemas'")
        
        if partial:
            recommendations.append(f"解決 {len(partial)} 個部分合規模組的剩餘問題")
        
        if not non_compliant and not partial:
            recommendations.append("所有模組均已合規，建議建立定期檢查機制")
        
        return recommendations
    
    def post_github_comment(self, result: CIResult) -> bool:
        """發布 GitHub PR 評論"""
        if not self.is_github_actions:
            return False
            
        pr_number = os.getenv('GITHUB_PR_NUMBER')
        if not pr_number:
            return False
            
        # 構建評論內容
        comment = f"""## 🔍 Schema 合規性檢查結果

{result.summary}

<details>
<summary>詳細報告</summary>

{result.detailed_report}

</details>

### 💡 改進建議
"""
        
        for i, rec in enumerate(result.recommendations, 1):
            comment += f"{i}. {rec}\\n"
        
        # TODO: 實際發布評論需要 GitHub API 整合
        print("GitHub 評論內容:")
        print(comment)
        return True
    
    def setup_output_variables(self, result: CIResult):
        """設置 CI 系統輸出變數"""
        if self.is_github_actions:
            # GitHub Actions 輸出
            print(f"::set-output name=success::{result.success}")
            print(f"::set-output name=score::{result.metrics.get('average_score', 0)}")
            print(f"::set-output name=compliant::{result.metrics.get('compliant', 0)}")
            print(f"::set-output name=non_compliant::{result.metrics.get('non_compliant', 0)}")
            
            if not result.success:
                print(f"::error title=Schema Compliance Failed::{result.summary}")
            
        elif self.is_azure_devops:
            # Azure DevOps 輸出
            print(f"##vso[task.setvariable variable=SchemaComplianceSuccess]{result.success}")
            print(f"##vso[task.setvariable variable=SchemaComplianceScore]{result.metrics.get('average_score', 0)}")
            
        elif self.is_jenkins:
            # Jenkins 環境變數
            with open("schema_compliance_result.properties", "w") as f:
                f.write(f"SCHEMA_COMPLIANCE_SUCCESS={result.success}\\n")
                f.write(f"SCHEMA_COMPLIANCE_SCORE={result.metrics.get('average_score', 0)}\\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Schema 合規性 CI/CD 檢查")
    parser.add_argument("--strict", action="store_true", help="嚴格模式：任何不合規都失敗")
    parser.add_argument("--threshold", type=float, default=80.0, help="最低合規分數閾值")
    parser.add_argument("--pr-comment", action="store_true", help="生成 PR 評論")
    parser.add_argument("--quiet", action="store_true", help="靜默模式，只輸出結果")
    
    args = parser.parse_args()
    
    checker = CISchemaChecker()
    result = checker.run_compliance_check(args.strict, args.threshold)
    
    # 設置 CI 系統輸出變數
    checker.setup_output_variables(result)
    
    # 發布 PR 評論
    if args.pr_comment:
        checker.post_github_comment(result)
    
    # 輸出結果
    if not args.quiet:
        print(result.summary)
        if not result.success:
            print("\\n" + result.detailed_report)
            print("\\n💡 建議:")
            for rec in result.recommendations:
                print(f"  • {rec}")
    
    # 退出
    sys.exit(result.exit_code)

if __name__ == "__main__":
    main()