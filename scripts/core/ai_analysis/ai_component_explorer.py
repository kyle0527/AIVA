#!/usr/bin/env python3
"""
AI組件探索器 - 專門探索AIVA核心模組內的可插拔AI組件
分析AI與程式的區別：
- AI: 核心模組內可插拔的智能組件
- 程式: AIVA的五大模組架構
"""

import sys
import os
import json
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AIComponent:
    """AI組件資訊"""
    name: str
    module_path: str
    description: str
    ai_type: str  # engine, learning, decision, etc.
    is_pluggable: bool
    dependencies: List[str]
    functions: List[str]
    classes: List[str]

@dataclass
class ModuleInfo:
    """模組資訊"""
    name: str
    path: str
    type: str  # core, scan, integration, features, common
    ai_components: List[AIComponent]
    traditional_components: List[str]

class AIComponentExplorer:
    """AI組件探索器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.five_modules = {
            'aiva_common': 'services/aiva_common',
            'core': 'services/core',
            'scan': 'services/scan', 
            'integration': 'services/integration',
            'features': 'services/features'
        }
        self.ai_components: List[AIComponent] = []
        self.modules_info: Dict[str, ModuleInfo] = {}
        
    def identify_ai_components(self) -> List[AIComponent]:
        """識別AI組件 - 基於文件名和內容特徵"""
        ai_keywords = [
            'ai_', 'neural', 'learning', 'intelligence', 'decision',
            'bio_neuron', 'anti_hallucination', 'commander', 'agent',
            'smart_', 'intelligent_', 'adaptive_', 'auto_'
        ]
        
        ai_components = []
        
        # 遍歷核心模組尋找AI組件
        core_path = self.project_root / 'services/core'
        if core_path.exists():
            for py_file in core_path.rglob('*.py'):
                if any(keyword in py_file.name.lower() for keyword in ai_keywords):
                    component = self._analyze_ai_component(py_file)
                    if component:
                        ai_components.append(component)
        
        return ai_components
    
    def _analyze_ai_component(self, file_path: Path) -> AIComponent:
        """分析單個AI組件文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析AST獲取函數和類
            tree = ast.parse(content)
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            
            # 提取描述（從docstring或註釋）
            description = self._extract_description(content)
            
            # 判斷AI類型
            ai_type = self._determine_ai_type(file_path.name, content)
            
            # 檢查是否可插拔
            is_pluggable = self._check_pluggable(content)
            
            # 提取依賴
            dependencies = self._extract_dependencies(content)
            
            return AIComponent(
                name=file_path.stem,
                module_path=str(file_path.relative_to(self.project_root)),
                description=description,
                ai_type=ai_type,
                is_pluggable=is_pluggable,
                dependencies=dependencies,
                functions=functions,
                classes=classes
            )
            
        except Exception as e:
            print(f"分析AI組件失敗 {file_path}: {e}")
            return None
    
    def _extract_description(self, content: str) -> str:
        """提取組件描述"""
        lines = content.split('\n')
        
        # 查找模組docstring
        for i, line in enumerate(lines):
            if '"""' in line and i < 20:  # 前20行內的docstring
                desc_lines = []
                start = i
                in_docstring = True
                
                for j in range(start, min(start + 10, len(lines))):
                    if '"""' in lines[j] and j > start:
                        break
                    desc_lines.append(lines[j].strip())
                
                description = ' '.join(desc_lines).replace('"""', '').strip()
                return description[:200] if description else "AI組件"
        
        return "AI組件"
    
    def _determine_ai_type(self, filename: str, content: str) -> str:
        """判斷AI組件類型"""
        if 'engine' in filename.lower():
            return 'AI引擎'
        elif 'learning' in filename.lower():
            return '學習系統'
        elif 'decision' in filename.lower():
            return '決策系統'
        elif 'commander' in filename.lower():
            return 'AI指揮官'
        elif 'neuron' in filename.lower():
            return '神經網路'
        elif 'agent' in filename.lower():
            return 'AI代理'
        elif 'smart' in filename.lower():
            return '智能模組'
        else:
            return 'AI組件'
    
    def _check_pluggable(self, content: str) -> bool:
        """檢查是否為可插拔組件"""
        pluggable_indicators = [
            'register', 'plugin', 'interface', 'abstract',
            'factory', 'registry', 'loader', 'manager'
        ]
        
        return any(indicator in content.lower() for indicator in pluggable_indicators)
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """提取依賴關係"""
        dependencies = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                if 'aiva' in line.lower() or 'services' in line:
                    dependencies.append(line)
        
        return dependencies[:5]  # 限制數量
    
    def analyze_five_modules(self) -> Dict[str, ModuleInfo]:
        """分析五大模組架構"""
        modules_info = {}
        
        for module_name, module_path in self.five_modules.items():
            full_path = self.project_root / module_path
            
            if not full_path.exists():
                continue
                
            # 分析該模組
            ai_components = []
            traditional_components = []
            
            # 遍歷模組文件
            for py_file in full_path.rglob('*.py'):
                if self._is_ai_component_file(py_file):
                    component = self._analyze_ai_component(py_file)
                    if component:
                        ai_components.append(component)
                else:
                    traditional_components.append(py_file.stem)
            
            modules_info[module_name] = ModuleInfo(
                name=module_name,
                path=module_path,
                type=self._get_module_type(module_name),
                ai_components=ai_components,
                traditional_components=traditional_components[:10]  # 限制數量
            )
        
        return modules_info
    
    def _is_ai_component_file(self, file_path: Path) -> bool:
        """判斷是否為AI組件文件"""
        ai_keywords = [
            'ai_', 'neural', 'learning', 'intelligence', 'decision',
            'bio_neuron', 'anti_hallucination', 'commander', 'agent',
            'smart_', 'intelligent_', 'adaptive_', 'auto_'
        ]
        
        return any(keyword in file_path.name.lower() for keyword in ai_keywords)
    
    def _get_module_type(self, module_name: str) -> str:
        """獲取模組類型描述"""
        descriptions = {
            'aiva_common': '通用基礎模組',
            'core': '核心業務模組 (AI組件密集)',
            'scan': '掃描發現模組',
            'integration': '整合服務模組',
            'features': '功能檢測模組'
        }
        return descriptions.get(module_name, '未知模組')
    
    def generate_cli_commands(self) -> List[str]:
        """基於探索結果生成CLI指令"""
        commands = []
        
        # 基於AI組件生成啟動命令
        for component in self.ai_components:
            if 'commander' in component.name.lower():
                commands.append(f"python -m services.core.aiva_core.{component.name} --mode=interactive")
            elif 'learning' in component.name.lower(): 
                commands.append(f"python -m services.core.aiva_core.{component.name} --auto-train")
            elif 'engine' in component.name.lower():
                commands.append(f"python -m services.core.aiva_core.{component.name} --initialize")
        
        # 基於模組生成掃描命令
        for module_name, module_info in self.modules_info.items():
            if module_name == 'scan':
                commands.append(f"python -m services.scan.aiva_scan.vulnerability_scanner --target=localhost:3000")
                commands.append(f"python -m services.scan.aiva_scan.network_scanner --range=192.168.1.0/24")
            elif module_name == 'features':
                commands.append(f"python -m services.features.function_sqli --payload-file=payloads.txt")
                commands.append(f"python -m services.features.function_xss --target=http://localhost:3000")
        
        # 基於系統整體生成集成命令
        commands.extend([
            "python ai_security_test.py --comprehensive",
            "python ai_autonomous_testing_loop.py --max-iterations=5",
            "python ai_system_explorer_v3.py --detailed --output=json",
            "python schema_version_checker.py --fix --report"
        ])
        
        return commands
    
    def run_exploration(self) -> Dict[str, Any]:
        """執行完整探索"""
        print("🔍 開始AI組件與五大模組探索...")
        
        # 分析五大模組
        print("📋 分析五大模組架構...")
        self.modules_info = self.analyze_five_modules()
        
        # 收集所有AI組件
        print("🤖 識別AI組件...")
        for module_info in self.modules_info.values():
            self.ai_components.extend(module_info.ai_components)
        
        # 生成CLI命令
        print("⚡ 生成CLI指令...")
        cli_commands = self.generate_cli_commands()
        
        # 生成報告
        report = {
            "exploration_timestamp": datetime.now().isoformat(),
            "five_modules_summary": {
                module_name: {
                    "type": info.type,
                    "ai_components_count": len(info.ai_components),
                    "traditional_components_count": len(info.traditional_components),
                    "ai_components": [
                        {
                            "name": comp.name,
                            "type": comp.ai_type,
                            "pluggable": comp.is_pluggable,
                            "functions_count": len(comp.functions),
                            "classes_count": len(comp.classes)
                        } for comp in info.ai_components
                    ]
                } for module_name, info in self.modules_info.items()
            },
            "ai_components_detailed": [
                {
                    "name": comp.name,
                    "path": comp.module_path,
                    "description": comp.description,
                    "type": comp.ai_type,
                    "pluggable": comp.is_pluggable,
                    "functions": comp.functions[:5],  # 限制數量
                    "classes": comp.classes,
                    "dependencies": comp.dependencies
                } for comp in self.ai_components
            ],
            "cli_commands_generated": cli_commands,
            "statistics": {
                "total_modules": len(self.modules_info),
                "total_ai_components": len(self.ai_components),
                "pluggable_ai_components": sum(1 for comp in self.ai_components if comp.is_pluggable),
                "cli_commands_count": len(cli_commands)
            }
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """打印探索總結"""
        print("\n" + "="*60)
        print("🎯 AIVA AI組件與五大模組探索總結")
        print("="*60)
        
        # 五大模組統計
        print(f"\n📋 五大模組架構:")
        for module_name, module_data in report["five_modules_summary"].items():
            ai_count = module_data["ai_components_count"]
            trad_count = module_data["traditional_components_count"]
            print(f"   {module_name:15} | AI組件: {ai_count:2d} | 傳統組件: {trad_count:2d} | {module_data['type']}")
        
        # AI組件統計
        stats = report["statistics"]
        print(f"\n🤖 AI組件統計:")
        print(f"   總AI組件數: {stats['total_ai_components']}")
        print(f"   可插拔組件: {stats['pluggable_ai_components']}")
        print(f"   生成CLI命令: {stats['cli_commands_count']}")
        
        # AI組件詳情
        print(f"\n🧠 發現的AI組件:")
        for comp in report["ai_components_detailed"]:
            pluggable_mark = "🔌" if comp["pluggable"] else "🔒"
            print(f"   {pluggable_mark} {comp['name']:25} | {comp['type']:12} | 函數: {len(comp['functions']):2d} | 類別: {len(comp['classes']):2d}")
        
        # 生成的CLI命令
        print(f"\n⚡ 生成的CLI指令 (前10個):")
        for i, cmd in enumerate(report["cli_commands_generated"][:10], 1):
            print(f"   {i:2d}. {cmd}")
        
        if len(report["cli_commands_generated"]) > 10:
            print(f"   ... 還有 {len(report['cli_commands_generated']) - 10} 個命令")

def main():
    """主函數"""
    explorer = AIComponentExplorer()
    
    try:
        # 執行探索
        report = explorer.run_exploration()
        
        # 打印結果
        explorer.print_summary(report)
        
        # 保存詳細報告
        report_file = Path("reports/ai_diagnostics") / f"ai_components_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 詳細報告已保存: {report_file}")
        
        return report
        
    except Exception as e:
        print(f"❌ 探索過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()