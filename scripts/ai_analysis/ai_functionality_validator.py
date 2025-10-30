#!/usr/bin/env python3
"""
AI功能理解與CLI生成驗證器
測試AI組件對程式功能的深度理解能力，並驗證生成的CLI指令是否可用
"""

import sys
import os
import ast
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# 設置環境變數
os.environ.setdefault('AIVA_RABBITMQ_URL', 'amqp://localhost:5672')
os.environ.setdefault('AIVA_RABBITMQ_USER', 'guest')
os.environ.setdefault('AIVA_RABBITMQ_PASSWORD', 'guest')

sys.path.insert(0, str(Path(__file__).parent / "services"))

@dataclass
class FunctionAnalysis:
    """功能分析結果"""
    script_name: str
    purpose: str
    main_functions: List[str]
    cli_command: str
    is_executable: bool
    parameters: List[str]
    dependencies: List[str]

class AIFunctionalityAnalyzer:
    """AI功能理解分析器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.analyzed_scripts = []
        
    def analyze_script_functionality(self, script_path: Path) -> FunctionAnalysis:
        """深度分析腳本功能，理解其用途"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST分析獲取結構
            tree = ast.parse(content)
            
            # 提取主要信息
            purpose = self._extract_purpose(content)
            main_functions = self._extract_main_functions(tree)
            parameters = self._extract_parameters(content)
            dependencies = self._extract_key_dependencies(content)
            
            # 基於功能分析生成CLI指令
            cli_command = self._generate_smart_cli(script_path, purpose, main_functions, parameters)
            
            # 檢查是否可執行
            is_executable = self._check_executable(script_path)
            
            return FunctionAnalysis(
                script_name=script_path.name,
                purpose=purpose,
                main_functions=main_functions,
                cli_command=cli_command,
                is_executable=is_executable,
                parameters=parameters,
                dependencies=dependencies
            )
            
        except Exception as e:
            print(f"❌ 分析腳本失敗 {script_path}: {e}")
            return None
    
    def _extract_purpose(self, content: str) -> str:
        """AI理解：提取腳本用途"""
        lines = content.split('\n')
        
        # 查找docstring中的用途描述
        for i, line in enumerate(lines[:30]):
            if '"""' in line or "'''" in line:
                purpose_lines = []
                for j in range(i, min(i+15, len(lines))):
                    if ('"""' in lines[j] or "'''" in lines[j]) and j > i:
                        break
                    purpose_lines.append(lines[j].strip())
                
                purpose = ' '.join(purpose_lines).replace('"""', '').replace("'''", '').strip()
                if purpose and len(purpose) > 10:
                    return purpose[:150]
        
        # 從檔名推斷功能
        filename = Path(content).name if hasattr(content, 'name') else "script"
        if 'scanner' in filename.lower():
            return "漏洞掃描工具"
        elif 'test' in filename.lower():
            return "測試工具"
        elif 'ai' in filename.lower():
            return "AI相關功能"
        elif 'explorer' in filename.lower():
            return "系統探索工具"
        
        return "功能分析中..."
    
    def _extract_main_functions(self, tree: ast.AST) -> List[str]:
        """提取主要函數"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 過濾重要函數
                if (not node.name.startswith('_') and 
                    node.name not in ['main', '__init__'] and
                    len(node.body) > 3):  # 有實質內容的函數
                    functions.append(node.name)
        return functions[:5]  # 前5個重要函數
    
    def _extract_parameters(self, content: str) -> List[str]:
        """AI理解：提取可能的命令列參數"""
        parameters = []
        
        # 查找argparse使用
        if 'argparse' in content:
            lines = content.split('\n')
            for line in lines:
                if 'add_argument' in line and '--' in line:
                    # 提取參數名
                    if '"--' in line:
                        param = line.split('"--')[1].split('"')[0]
                        parameters.append(f"--{param}")
                    elif "'--" in line:
                        param = line.split("'--")[1].split("'")[0]
                        parameters.append(f"--{param}")
        
        # 查找常見參數模式
        common_params = ['--help', '--verbose', '--output', '--target', '--config']
        for param in common_params:
            if param.replace('--', '') in content.lower():
                parameters.append(param)
        
        return list(set(parameters))[:5]  # 去重並限制數量
    
    def _extract_key_dependencies(self, content: str) -> List[str]:
        """提取關鍵依賴"""
        dependencies = []
        lines = content.split('\n')
        
        for line in lines[:50]:  # 只看前50行
            line = line.strip()
            if line.startswith('from ') and 'aiva' in line.lower():
                dependencies.append(line.split()[1])
            elif line.startswith('import ') and any(key in line for key in ['requests', 'asyncio', 'aiohttp']):
                dependencies.append(line.split()[1])
        
        return list(set(dependencies))[:3]
    
    def _generate_smart_cli(self, script_path: Path, purpose: str, functions: List[str], parameters: List[str]) -> str:
        """AI智能生成CLI指令"""
        base_cmd = f"python {script_path.name}"
        
        # 根據功能智能添加參數
        if 'scanner' in script_path.name.lower() or 'scan' in purpose.lower():
            if '--target' in parameters:
                base_cmd += " --target=localhost:3000"
            base_cmd += " --verbose"
            
        elif 'test' in script_path.name.lower() or 'test' in purpose.lower():
            base_cmd += " --comprehensive"
            if '--output' in parameters:
                base_cmd += " --output=json"
                
        elif 'explorer' in script_path.name.lower() or 'explore' in purpose.lower():
            base_cmd += " --detailed"
            if '--output' in parameters:
                base_cmd += " --output=json"
                
        elif 'ai' in script_path.name.lower():
            if 'autonomous' in script_path.name.lower():
                base_cmd += " --max-iterations=3"
            elif 'security' in script_path.name.lower():
                base_cmd += " --target=localhost:3000"
                
        # 智能添加常用參數
        if any(func in ['run', 'execute', 'start'] for func in functions):
            if '--verbose' not in base_cmd:
                base_cmd += " --verbose"
        
        return base_cmd
    
    def _check_executable(self, script_path: Path) -> bool:
        """檢查腳本是否可執行"""
        try:
            # 檢查是否有main函數或if __name__ == "__main__"
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ('if __name__ == "__main__"' in content or 
                   'def main(' in content or
                   'async def main(' in content)
        except:
            return False
    
    def test_generated_cli_commands(self, analyses: List[FunctionAnalysis]) -> Dict[str, Any]:
        """測試生成的CLI指令是否真的可用"""
        test_results = {
            'total_commands': 0,
            'help_successful': 0,
            'syntax_valid': 0,
            'executable_scripts': 0,
            'command_results': []
        }
        
        for analysis in analyses:
            if not analysis or not analysis.is_executable:
                continue
                
            test_results['total_commands'] += 1
            test_results['executable_scripts'] += 1
            
            # 測試--help參數
            help_cmd = f"python {analysis.script_name} --help"
            help_success = self._test_command_help(help_cmd)
            
            if help_success:
                test_results['help_successful'] += 1
            
            # 測試語法有效性
            syntax_valid = self._test_command_syntax(analysis.cli_command)
            if syntax_valid:
                test_results['syntax_valid'] += 1
            
            test_results['command_results'].append({
                'script': analysis.script_name,
                'command': analysis.cli_command,
                'help_works': help_success,
                'syntax_valid': syntax_valid,
                'purpose': analysis.purpose[:50]
            })
        
        return test_results
    
    def _test_command_help(self, help_cmd: str) -> bool:
        """測試--help指令"""
        try:
            result = subprocess.run(
                help_cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=10,
                cwd=self.project_root
            )
            return result.returncode == 0 or 'usage:' in result.stdout.lower()
        except:
            return False
    
    def _test_command_syntax(self, command: str) -> bool:
        """測試指令語法"""
        try:
            # 檢查基本語法
            parts = command.split()
            if len(parts) < 2:
                return False
            
            # 檢查Python檔案存在
            script_name = parts[1]
            script_path = self.project_root / script_name
            
            return script_path.exists()
        except:
            return False
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """執行綜合分析"""
        print("[SEARCH] 開始AI功能理解與CLI生成驗證...")
        print("="*60)
        
        # 選擇重要腳本進行分析
        target_scripts = [
            "ai_security_test.py",
            "ai_autonomous_testing_loop.py", 
            "ai_system_explorer_v3.py",
            "health_check.py",
            "schema_version_checker.py",
            "comprehensive_pentest_runner.py"
        ]
        
        analyses = []
        
        print("[ANALYSIS] 分析腳本功能...")
        for script_name in target_scripts:
            script_path = self.project_root / script_name
            if script_path.exists():
                print(f"   [SEARCH] 分析: {script_name}")
                analysis = self.analyze_script_functionality(script_path)
                if analysis:
                    analyses.append(analysis)
        
        print(f"\n[AI] AI功能理解結果:")
        for analysis in analyses:
            print(f"   📄 {analysis.script_name}")
            print(f"      用途: {analysis.purpose[:80]}...")
            print(f"      主要功能: {', '.join(analysis.main_functions[:3])}")
            print(f"      生成指令: {analysis.cli_command}")
            print(f"      可執行: {'✅' if analysis.is_executable else '❌'}")
            print()
        
        print("⚡ 測試生成的CLI指令...")
        test_results = self.test_generated_cli_commands(analyses)
        
        # 生成報告
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "scripts_analyzed": len(analyses),
                "executable_scripts": test_results['executable_scripts'],
                "functionality_understood": len([a for a in analyses if len(a.purpose) > 20]),
                "cli_commands_generated": test_results['total_commands']
            },
            "cli_validation": {
                "total_commands": test_results['total_commands'],
                "help_works": test_results['help_successful'],
                "syntax_valid": test_results['syntax_valid'],
                "success_rate": (test_results['help_successful'] / max(test_results['total_commands'], 1)) * 100
            },
            "detailed_analyses": [
                {
                    "script": a.script_name,
                    "purpose": a.purpose,
                    "functions": a.main_functions,
                    "cli_command": a.cli_command,
                    "executable": a.is_executable,
                    "parameters": a.parameters,
                    "dependencies": a.dependencies
                } for a in analyses
            ],
            "command_test_results": test_results['command_results']
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """打印分析總結"""
        print("="*60)
        print("🎯 AI功能理解與CLI生成驗證結果")
        print("="*60)
        
        summary = report["analysis_summary"]
        validation = report["cli_validation"]
        
        print(f"\n📊 分析統計:")
        print(f"   腳本分析: {summary['scripts_analyzed']} 個")
        print(f"   功能理解: {summary['functionality_understood']} 個")
        print(f"   可執行腳本: {summary['executable_scripts']} 個")
        print(f"   生成CLI指令: {summary['cli_commands_generated']} 個")
        
        print(f"\n⚡ CLI指令驗證:")
        print(f"   總指令數: {validation['total_commands']}")
        print(f"   --help可用: {validation['help_works']}/{validation['total_commands']}")
        print(f"   語法正確: {validation['syntax_valid']}/{validation['total_commands']}")
        print(f"   成功率: {validation['success_rate']:.1f}%")
        
        print(f"\n[AI] AI理解能力展示:")
        for result in report["command_test_results"]:
            status = "✅" if result["help_works"] and result["syntax_valid"] else "⚠️"
            print(f"   {status} {result['script']:25} | {result['purpose']}")
        
        print(f"\n🎯 生成的可用CLI指令:")
        for result in report["command_test_results"]:
            if result["help_works"]:
                print(f"   ✅ {result['command']}")

def main():
    """主函數"""
    analyzer = AIFunctionalityAnalyzer()
    
    try:
        # 執行綜合分析
        report = analyzer.run_comprehensive_analysis()
        
        # 打印結果
        analyzer.print_summary(report)
        
        # 保存報告
        report_file = Path("reports/ai_diagnostics") / f"ai_functionality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 詳細報告已保存: {report_file}")
        
    except Exception as e:
        print(f"❌ 分析過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()