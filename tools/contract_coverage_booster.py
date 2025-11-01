#!/usr/bin/env python3
"""
AIVA 合約覆蓋率提升工具

自動化工具，幫助識別並提升 AIVA 項目中標準合約的使用覆蓋率。
包含分析、建議、自動重構等功能。
"""

import ast
import json
import re
import sys
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import argparse
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ContractUsageAnalysis:
    """合約使用分析結果"""
    total_files: int
    files_with_standard_contracts: int
    files_with_local_models: int
    standard_contract_coverage: float
    local_model_coverage: float
    most_used_contracts: List[Tuple[str, int]]
    local_model_candidates: List[Tuple[str, int]]
    module_usage: Dict[str, Dict[str, Any]]
    improvement_opportunities: List[Dict[str, Any]]


@dataclass
class RefactoringTask:
    """重構任務"""
    file_path: str
    task_type: str  # 'add_standard_import', 'replace_local_model', 'standardize_response'
    description: str
    old_code: Optional[str] = None
    new_code: Optional[str] = None
    confidence: float = 1.0  # 0.0-1.0 置信度


class ContractCoverageAnalyzer:
    """合約覆蓋率分析器"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.exclude_patterns = [
            '__pycache__', '.git', 'venv', '.venv', 
            'node_modules', '.pytest_cache', 'build', 'dist'
        ]
        
        # 標準合約清單 (從 aiva_common.schemas)
        self.standard_contracts = [
            'FindingPayload', 'AivaMessage', 'AttackPlan', 'FunctionTelemetry',
            'ScanStartPayload', 'MessageHeader', 'Authentication', 'APIResponse',
            'ExecutionError', 'Target', 'Asset', 'Experience', 'Task', 'Scan'
        ]
        
        # 常見本地合約模式
        self.local_model_patterns = [
            r'class (\w+)(?:Request|Response|Config|Model|Data|Payload)\(BaseModel\)',
            r'class (\w+)\(BaseModel\)',
        ]
        
        # API響應模式
        self.response_patterns = [
            r'return\s+{\s*["\']success["\']:\s*True',
            r'return\s+{\s*["\']success["\']:\s*False',
            r'return\s+(?:jsonify|Response)\(',
        ]
    
    def scan_python_files(self) -> List[Path]:
        """掃描所有 Python 文件"""
        python_files = []
        
        for py_file in self.project_root.rglob('*.py'):
            # 排除特定目錄
            if any(pattern in str(py_file) for pattern in self.exclude_patterns):
                continue
            python_files.append(py_file)
        
        return python_files
    
    def analyze_file_contracts(self, file_path: Path) -> Dict[str, Any]:
        """分析單個文件的合約使用情況"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            analysis = {
                'has_standard_imports': False,
                'has_local_models': False,
                'standard_contracts': [],
                'local_models': [],
                'response_patterns': [],
                'import_lines': [],
                'api_endpoints': 0,
                'improvement_opportunities': []
            }
            
            # 1. 分析標準合約導入
            import_patterns = [
                r'from services\.aiva_common\.schemas import ([^\\n]+)',
                r'from aiva_common\.schemas import ([^\\n]+)'
            ]
            
            for pattern in import_patterns:
                imports = re.findall(pattern, content)
                if imports:
                    analysis['has_standard_imports'] = True
                    analysis['import_lines'].extend(imports)
                    
                    for import_line in imports:
                        contracts = [c.strip() for c in import_line.replace('(', '').replace(')', '').split(',')]
                        analysis['standard_contracts'].extend([c for c in contracts if c.strip()])
            
            # 2. 分析本地模型定義
            for pattern in self.local_model_patterns:
                models = re.findall(pattern, content)
                if models:
                    analysis['has_local_models'] = True
                    analysis['local_models'].extend(models)
            
            # 3. 分析響應模式
            for pattern in self.response_patterns:
                matches = re.findall(pattern, content)
                analysis['response_patterns'].extend(matches)
            
            # 4. 檢查 API 端點數量
            api_decorators = re.findall(r'@app\.(get|post|put|delete|patch)', content)
            analysis['api_endpoints'] = len(api_decorators)
            
            # 5. 識別改進機會
            analysis['improvement_opportunities'] = self._identify_improvements(file_path, content, analysis)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"分析文件失敗 {file_path}: {e}")
            return {}
    
    def _identify_improvements(self, _file_path: Path, _content: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """識別改進機會"""
        opportunities = []
        
        # 1. API 端點但沒有使用標準響應格式
        if analysis['api_endpoints'] > 0 and not analysis['has_standard_imports']:
            opportunities.append({
                'type': 'add_standard_response',
                'description': 'API 端點建議使用 APIResponse 標準格式',
                'priority': 'high',
                'effort': 'medium'
            })
        
        # 2. 有本地模型但可能可以標準化
        if analysis['local_models']:
            for model in analysis['local_models']:
                if any(keyword in model.lower() for keyword in ['response', 'request', 'payload', 'message']):
                    opportunities.append({
                        'type': 'standardize_local_model',
                        'description': f'本地模型 {model} 可能可以使用標準合約',
                        'model_name': model,
                        'priority': 'medium',
                        'effort': 'high'
                    })
        
        # 3. 使用舊式響應格式
        if analysis['response_patterns'] and 'APIResponse' not in analysis['standard_contracts']:
            opportunities.append({
                'type': 'upgrade_response_format',
                'description': '響應格式可以升級為 APIResponse 標準',
                'priority': 'medium',
                'effort': 'low'
            })
        
        return opportunities
    
    def run_full_analysis(self) -> ContractUsageAnalysis:
        """運行完整分析"""
        logger.info("開始合約覆蓋率全面分析...")
        
        python_files = self.scan_python_files()
        total_files = len(python_files)
        
        # 統計數據
        files_with_standard_contracts = 0
        files_with_local_models = 0
        contract_usage = Counter()
        local_model_usage = Counter()
        module_usage = defaultdict(dict)
        all_opportunities = []
        
        logger.info(f"掃描 {total_files} 個 Python 文件...")
        
        for file_path in python_files:
            analysis = self.analyze_file_contracts(file_path)
            
            if not analysis:
                continue
            
            # 統計標準合約使用
            if analysis['has_standard_imports']:
                files_with_standard_contracts += 1
                for contract in analysis['standard_contracts']:
                    contract_usage[contract.strip()] += 1
            
            # 統計本地模型
            if analysis['has_local_models']:
                files_with_local_models += 1
                for model in analysis['local_models']:
                    local_model_usage[model] += 1
            
            # 模組統計
            module_name = self._get_module_name(file_path)
            module_usage[module_name] = {
                'files': module_usage[module_name].get('files', 0) + 1,
                'standard_contracts': len(analysis['standard_contracts']),
                'local_models': len(analysis['local_models']),
                'api_endpoints': analysis['api_endpoints']
            }
            
            # 收集改進機會
            for opp in analysis['improvement_opportunities']:
                opp['file_path'] = str(file_path)
                all_opportunities.append(opp)
        
        # 計算覆蓋率
        standard_coverage = (files_with_standard_contracts / total_files * 100) if total_files > 0 else 0
        local_coverage = (files_with_local_models / total_files * 100) if total_files > 0 else 0
        
        return ContractUsageAnalysis(
            total_files=total_files,
            files_with_standard_contracts=files_with_standard_contracts,
            files_with_local_models=files_with_local_models,
            standard_contract_coverage=standard_coverage,
            local_model_coverage=local_coverage,
            most_used_contracts=contract_usage.most_common(10),
            local_model_candidates=local_model_usage.most_common(10),
            module_usage=dict(module_usage),
            improvement_opportunities=all_opportunities
        )
    
    def _get_module_name(self, file_path: Path) -> str:
        """獲取模組名稱"""
        relative_path = file_path.relative_to(self.project_root)
        parts = relative_path.parts
        
        if len(parts) > 1:
            return parts[0]
        else:
            return parts[0] if parts else 'root'
    
    def generate_improvement_plan(self, analysis: ContractUsageAnalysis) -> Dict[str, Any]:
        """生成改進計劃"""
        plan = {
            'current_status': {
                'coverage_rate': analysis.standard_contract_coverage,
                'status': self._get_coverage_status(analysis.standard_contract_coverage)
            },
            'quick_wins': [],
            'medium_term_goals': [],
            'long_term_objectives': [],
            'priority_modules': [],
            'automation_opportunities': []
        }
        
        # 根據覆蓋率生成建議
        if analysis.standard_contract_coverage < 10:
            plan['quick_wins'].extend([
                '在 API 模組中推廣 APIResponse 標準格式',
                '創建合約使用範本和快速開始指南',
                '在新功能開發中強制使用標準合約'
            ])
        elif analysis.standard_contract_coverage < 20:
            plan['quick_wins'].extend([
                '重構現有 API 端點使用標準響應格式',
                '標準化錯誤處理使用 ExecutionError',
                '統一訊息格式使用 AivaMessage'
            ])
        else:
            plan['quick_wins'].extend([
                '優化高頻使用的本地合約',
                '建立自動化遷移工具',
                '完善性能監控機制'
            ])
        
        # 中期目標
        plan['medium_term_goals'] = [
            '達成 25%+ 標準合約覆蓋率',
            '清理和標準化本地合約定義',
            '建立 CI/CD 覆蓋率檢查',
            '完善開發者培訓體系'
        ]
        
        # 長期目標
        plan['long_term_objectives'] = [
            '實現 35%+ 標準合約覆蓋率',
            '建立智能合約推薦系統',
            '實現跨語言合約同步',
            '完善合約版本管理體系'
        ]
        
        # 優先模組
        priority_modules = []
        for module, stats in analysis.module_usage.items():
            if stats.get('api_endpoints', 0) > 0 and stats.get('standard_contracts', 0) == 0:
                priority_modules.append({
                    'module': module,
                    'reason': 'API 模組缺乏標準合約',
                    'api_endpoints': stats.get('api_endpoints', 0)
                })
        
        plan['priority_modules'] = sorted(priority_modules, 
                                        key=lambda x: x['api_endpoints'], 
                                        reverse=True)[:5]
        
        return plan
    
    def _get_coverage_status(self, coverage: float) -> str:
        """獲取覆蓋率狀態"""
        if coverage >= 30:
            return 'excellent'
        elif coverage >= 20:
            return 'good'
        elif coverage >= 10:
            return 'fair'
        else:
            return 'needs_improvement'
    
    def generate_refactoring_tasks(self, analysis: ContractUsageAnalysis) -> List[RefactoringTask]:
        """生成重構任務"""
        tasks = []
        
        for opportunity in analysis.improvement_opportunities:
            if opportunity['type'] == 'add_standard_response':
                task = RefactoringTask(
                    file_path=opportunity['file_path'],
                    task_type='add_standard_import',
                    description='添加 APIResponse 標準導入',
                    new_code='from services.aiva_common.schemas import APIResponse',
                    confidence=0.9
                )
                tasks.append(task)
            
            elif opportunity['type'] == 'upgrade_response_format':
                task = RefactoringTask(
                    file_path=opportunity['file_path'],
                    task_type='standardize_response',
                    description='升級響應格式為 APIResponse 標準',
                    confidence=0.7
                )
                tasks.append(task)
        
        return tasks
    
    def export_analysis_report(self, analysis: ContractUsageAnalysis, output_path: Path):
        """導出分析報告"""
        improvement_plan = self.generate_improvement_plan(analysis)
        refactoring_tasks = self.generate_refactoring_tasks(analysis)
        
        report = {
            'analysis_summary': asdict(analysis),
            'improvement_plan': improvement_plan,
            'refactoring_tasks': [asdict(task) for task in refactoring_tasks],
            'generated_at': str(Path().cwd()),
            'recommendations': self._generate_recommendations(analysis)
        }
        
        # 導出 JSON
        json_output = output_path.with_suffix('.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 導出 Markdown
        md_output = output_path.with_suffix('.md')
        with open(md_output, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(analysis, improvement_plan))
        
        logger.info(f"報告已導出: {json_output}, {md_output}")
    
    def _generate_recommendations(self, analysis: ContractUsageAnalysis) -> List[str]:
        """生成具體建議"""
        recommendations = []
        
        if analysis.standard_contract_coverage < 15:
            recommendations.extend([
                '立即在 API 模組中推廣 APIResponse 標準',
                '創建合約使用檢查清單',
                '建立開發者合約培訓計劃'
            ])
        
        if analysis.local_model_candidates:
            top_local = analysis.local_model_candidates[0]
            recommendations.append(
                f'考慮將高頻本地合約 "{top_local[0]}" (使用{top_local[1]}次) 標準化'
            )
        
        return recommendations
    
    def _generate_markdown_report(self, analysis: ContractUsageAnalysis, plan: Dict[str, Any]) -> str:
        """生成 Markdown 格式報告"""
        status_emoji = {
            'excellent': '🟢',
            'good': '🟡', 
            'fair': '🟠',
            'needs_improvement': '🔴'
        }
        
        status = plan['current_status']['status']
        emoji = status_emoji.get(status, '❓')
        
        report = f"""# AIVA 合約覆蓋率分析報告

## 📊 當前狀態

{emoji} **覆蓋率**: {analysis.standard_contract_coverage:.1f}% ({analysis.files_with_standard_contracts}/{analysis.total_files} 文件)  
📈 **狀態**: {status.replace('_', ' ').title()}  
📂 **本地模型**: {analysis.local_model_coverage:.1f}% ({analysis.files_with_local_models} 文件)

## 🔥 最常用標準合約

| 排名 | 合約名稱 | 使用次數 |
|------|----------|----------|
"""
        
        for i, (contract, count) in enumerate(analysis.most_used_contracts[:5], 1):
            report += f"| {i} | `{contract}` | {count} |\n"
        
        report += """
## 🏠 本地合約候選 (標準化機會)

| 排名 | 合約名稱 | 使用次數 |
|------|----------|----------|
"""
        
        for i, (contract, count) in enumerate(analysis.local_model_candidates[:5], 1):
            report += f"| {i} | `{contract}` | {count} |\n"
        
        report += """
## 🎯 立即行動項

"""
        for item in plan['quick_wins']:
            report += f"- {item}\n"
        
        report += """
## 📅 中期目標

"""
        for item in plan['medium_term_goals']:
            report += f"- {item}\n"
        
        report += """
## 🚀 長期規劃

"""
        for item in plan['long_term_objectives']:
            report += f"- {item}\n"
        
        if plan['priority_modules']:
            report += """
## 🎯 優先處理模組

"""
            for module in plan['priority_modules']:
                report += f"- **{module['module']}**: {module['reason']} ({module['api_endpoints']} API 端點)\n"
        
        report += """
---
**生成工具**: AIVA 合約覆蓋率分析器  
**建議**: 定期運行此分析以追蹤改進進度
"""
        
        return report


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="AIVA 合約覆蓋率提升工具")
    parser.add_argument("--project-root", default=".", help="項目根目錄路徑")
    parser.add_argument("--output", default="reports/contract_coverage_analysis", help="輸出文件路徑")
    parser.add_argument("--quick-check", action="store_true", help="快速檢查模式")
    parser.add_argument("--generate-tasks", action="store_true", help="生成重構任務")
    
    args = parser.parse_args()
    
    # 創建分析器
    analyzer = ContractCoverageAnalyzer(Path(args.project_root))
    
    if args.quick_check:
        # 快速檢查模式
        python_files = analyzer.scan_python_files()
        total_files = len(python_files)
        
        files_with_contracts = 0
        for file_path in python_files[:100]:  # 僅檢查前100個文件
            analysis = analyzer.analyze_file_contracts(file_path)
            if analysis.get('has_standard_imports'):
                files_with_contracts += 1
        
        coverage = (files_with_contracts / min(100, total_files)) * 100
        print(f"📊 快速檢查結果: 覆蓋率約 {coverage:.1f}%")
        return
    
    # 完整分析
    analysis = analyzer.run_full_analysis()
    
    # 顯示結果
    print(f"""
📊 AIVA 合約覆蓋率分析結果
{'=' * 40}

📈 標準合約覆蓋率: {analysis.standard_contract_coverage:.1f}%
📂 本地模型覆蓋率: {analysis.local_model_coverage:.1f}%
📁 總文件數: {analysis.total_files}
✅ 使用標準合約: {analysis.files_with_standard_contracts} 文件
🏠 包含本地模型: {analysis.files_with_local_models} 文件

🔥 最常用標準合約:
""")
    
    for i, (contract, count) in enumerate(analysis.most_used_contracts[:5], 1):
        print(f"  {i}. {contract:<20} - {count:3d}次")
    
    print(f"""
💡 改進機會: {len(analysis.improvement_opportunities)} 個
""")
    
    # 導出報告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.export_analysis_report(analysis, output_path)
    
    if args.generate_tasks:
        tasks = analyzer.generate_refactoring_tasks(analysis)
        print(f"\n🔧 生成 {len(tasks)} 個重構任務")
        
        for i, task in enumerate(tasks[:10], 1):
            print(f"  {i}. {task.description} ({task.confidence:.1%} 置信度)")


if __name__ == "__main__":
    main()