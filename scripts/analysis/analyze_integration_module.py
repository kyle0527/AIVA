#!/usr/bin/env python3
"""
AIVA 整合模組圖表自動化組合工具
應用「完整產出 + 智能篩選」策略分析整合模組

⚠️ 核心理念：完整產出的智慧
本腳本會產生大量個別組件圖檔，原因：
- 🔍 無法預知哪個組件包含關鍵架構洞察  
- 🎯 最重要的模式往往隱藏在看似次要的組件中
- 🛡️ 寧可產出 300 個圖删除 295 個，也不要遺漏 1 個關鍵發現

推薦工作流程：
1. 完整產出所有圖檔（本腳本）
2. 深度分析分類結果和模式
3. 人工識別真正的價值  
4. 使用 cleanup_diagram_output.py 智能清理
"""

import os
import sys
import re
import json
import ast
import inspect
from dataclasses import dataclass, asdict
from typing import List, Dict, Literal, Optional, Set, Any
from pathlib import Path

@dataclass
class ComponentInfo:
    """組件資訊"""
    name: str
    type: Literal["class", "function", "module", "service", "integration"]
    file_path: str
    layer: str
    dependencies: Optional[List[str]] = None
    complexity_score: int = 0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class IntegrationDiagramClassification:
    """整合模組圖表分類元資料"""
    category: Literal["core", "detail", "integration", "service", "data", "security"]
    priority: int  # 1-10，數字越小優先級越高
    complexity: Literal["low", "medium", "high"]
    abstraction_level: Literal["system", "service", "component", "function"]
    dependencies: List[str]
    file_path: str
    integration_type: str = ""
    
class IntegrationModuleAnalyzer:
    """整合模組分析器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.integration_dir = self.base_dir / "services" / "integration"
        self.output_dir = self.base_dir / "_out" / "architecture_diagrams"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 整合模組的關鍵路徑和組件
        self.integration_patterns = {
            "ai_operation_recorder": {"priority": 1, "type": "core"},
            "reporting": {"priority": 2, "type": "service"}, 
            "perf_feedback": {"priority": 2, "type": "service"},
            "analysis": {"priority": 3, "type": "service"},
            "reception": {"priority": 3, "type": "integration"},
            "remediation": {"priority": 4, "type": "service"},
            "threat_intel": {"priority": 5, "type": "service"},
            "security": {"priority": 3, "type": "security"},
            "observability": {"priority": 4, "type": "service"},
            "middlewares": {"priority": 6, "type": "detail"},
            "examples": {"priority": 10, "type": "detail"}
        }
        
    def scan_integration_components(self) -> List[ComponentInfo]:
        """掃描整合模組的所有組件"""
        components = []
        
        if not self.integration_dir.exists():
            print(f"⚠️ 整合模組目錄不存在: {self.integration_dir}")
            return components
            
        print(f"🔍 掃描整合模組: {self.integration_dir}")
        
        # 掃描 aiva_integration 目錄下的所有 Python 檔案
        aiva_integration_dir = self.integration_dir / "aiva_integration"
        if aiva_integration_dir.exists():
            for py_file in aiva_integration_dir.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                    
                file_components = self._analyze_python_file(py_file)
                components.extend(file_components)
        
        # 掃描其他重要檔案
        for py_file in self.integration_dir.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            if "aiva_integration" in str(py_file):
                continue  # 已經掃描過了
                
            file_components = self._analyze_python_file(py_file)
            components.extend(file_components)
            
        print(f"✅ 發現 {len(components)} 個組件")
        return components
    
    def _analyze_python_file(self, file_path: Path) -> List[ComponentInfo]:
        """分析 Python 檔案中的組件"""
        components = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 使用 AST 分析
            tree = ast.parse(content)
            
            relative_path = file_path.relative_to(self.base_dir)
            layer = self._determine_layer(file_path)
            
            # 分析類別
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    components.append(ComponentInfo(
                        name=node.name,
                        type="class",
                        file_path=str(relative_path),
                        layer=layer,
                        complexity_score=self._calculate_complexity(node)
                    ))
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    components.append(ComponentInfo(
                        name=node.name,
                        type="function", 
                        file_path=str(relative_path),
                        layer=layer,
                        complexity_score=self._calculate_complexity(node)
                    ))
                    
            # 模組級別組件
            module_name = file_path.stem
            components.append(ComponentInfo(
                name=module_name,
                type="module",
                file_path=str(relative_path),
                layer=layer,
                complexity_score=len(components)
            ))
            
        except Exception as e:
            print(f"⚠️ 分析檔案失敗 {file_path}: {e}")
            
        return components
    
    def _determine_layer(self, file_path: Path) -> str:
        """確定組件所屬的層級"""

        
        if "ai_operation_recorder" in str(file_path):
            return "core"
        elif any(x in str(file_path) for x in ["reporting", "analysis", "reception"]):
            return "service"
        elif "security" in str(file_path) or "threat_intel" in str(file_path):
            return "security"
        elif any(x in str(file_path) for x in ["perf_feedback", "observability"]):
            return "monitoring"
        elif "integration" in str(file_path):
            return "integration"
        else:
            return "support"
    
    def _calculate_complexity(self, node) -> int:
        """計算組件複雜度"""
        if isinstance(node, ast.ClassDef):
            return len([n for n in node.body if isinstance(n, ast.FunctionDef)])
        elif isinstance(node, ast.FunctionDef):
            return len(list(ast.walk(node)))
        else:
            return 1
    
    def generate_component_diagrams(self, components: List[ComponentInfo]) -> List[str]:
        """為每個組件生成個別的圖表"""
        generated_files = []
        
        print(f"📊 生成 {len(components)} 個組件圖表...")
        
        for component in components:
            diagram_content = self._create_component_diagram(component)
            
            # 生成檔案名
            safe_file_path = component.file_path.replace("/", "_").replace("\\", "_")
            filename = f"aiva_integration_{safe_file_path}_{component.type}_{component.name}.mmd"
            output_file = self.output_dir / filename
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(diagram_content)
                generated_files.append(str(output_file))
            except Exception as e:
                print(f"⚠️ 生成圖表失敗 {filename}: {e}")
        
        return generated_files
    
    def _create_component_diagram(self, component: ComponentInfo) -> str:
        """為單個組件創建 Mermaid 圖表"""
        
        diagram_lines = [
            "```mermaid",
            f"---",
            f"title: Integration Component - {component.name}",
            f"---",
            "flowchart TD",
            "",
        ]
        
        # 主要組件節點
        node_id = "C1"
        diagram_lines.append(f'    {node_id}["{component.name}"]')
        
        # 根據類型添加相關節點
        if component.type == "class":
            diagram_lines.extend([
                f'    D1[("Data Layer")]',
                f'    S1[("Service Layer")]',
                f'    {node_id} --> D1',
                f'    S1 --> {node_id}'
            ])
        elif component.type == "function":
            diagram_lines.extend([
                f'    I1["Input"]',
                f'    O1["Output"]', 
                f'    I1 --> {node_id}',
                f'    {node_id} --> O1'
            ])
        elif component.type == "module":
            diagram_lines.extend([
                f'    API1["API Interface"]',
                f'    CORE1["Core Logic"]',
                f'    DB1[("Database")]',
                f'    API1 --> {node_id}',
                f'    {node_id} --> CORE1',
                f'    CORE1 --> DB1'
            ])
        
        # 添加樣式
        diagram_lines.extend([
            "",
            "    %% Styling",
            f"    classDef {component.layer} fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            f"    class {node_id} {component.layer}",
        ])
        
        diagram_lines.append("```")
        return "\n".join(diagram_lines)
    
    def classify_components(self, components: List[ComponentInfo]) -> List[IntegrationDiagramClassification]:
        """分類整合模組組件"""
        classifications = []
        
        for component in components:
            classification = self._classify_single_component(component)
            classifications.append(classification)
            
        return classifications
    
    def _classify_single_component(self, component: ComponentInfo) -> IntegrationDiagramClassification:
        """分類單個組件"""
        
        # 根據檔案路徑和名稱確定分類
        category = "detail"  # 預設
        priority = 5
        complexity = "medium"
        integration_type = ""
        
        # 檢查是否是關鍵組件
        for pattern, info in self.integration_patterns.items():
            if pattern in component.file_path or pattern in component.name.lower():
                category = info["type"]
                priority = info["priority"]
                integration_type = pattern
                break
        
        # 根據複雜度調整
        if component.complexity_score > 10:
            complexity = "high"
            priority = max(1, priority - 1)  # 複雜度高的優先級提升
        elif component.complexity_score < 3:
            complexity = "low"
            priority = min(10, priority + 1)  # 複雜度低的優先級降低
        
        # 根據層級調整抽象級別
        abstraction_map = {
            "core": "system",
            "service": "service", 
            "integration": "service",
            "security": "component",
            "monitoring": "component",
            "support": "function"
        }
        abstraction_level = abstraction_map.get(component.layer, "component")
        
        return IntegrationDiagramClassification(
            category=category,
            priority=priority,
            complexity=complexity,
            abstraction_level=abstraction_level,
            dependencies=component.dependencies,
            file_path=component.file_path,
            integration_type=integration_type
        )
    
    def generate_integrated_architecture(self, classifications: List[IntegrationDiagramClassification]) -> str:
        """生成整合模組的整合架構圖"""
        
        # 按優先級和類型分組
        core_components = [c for c in classifications if c.category == "core" and c.priority <= 2]
        service_components = [c for c in classifications if c.category == "service" and c.priority <= 4]
        integration_components = [c for c in classifications if c.category == "integration" and c.priority <= 4]
        security_components = [c for c in classifications if c.category == "security" and c.priority <= 5]
        
        diagram_lines = [
            "```mermaid",
            "---", 
            "title: AIVA Integration Module Architecture",
            "---",
            "flowchart TB",
            "",
            "    %% Core Integration Services",
            "    subgraph CORE[\"Core Integration Layer\"]",
        ]
        
        # 添加核心組件
        for i, comp in enumerate(core_components[:5]):  # 最多5個核心組件
            comp_name = comp.file_path.split('/')[-1].replace('.py', '')
            diagram_lines.append(f'        C{i+1}["{comp_name}"]')
        
        diagram_lines.extend([
            "    end",
            "",
            "    %% Service Integration Layer", 
            "    subgraph SERVICES[\"Service Integration Layer\"]",
        ])
        
        # 添加服務組件
        for i, comp in enumerate(service_components[:6]):  # 最多6個服務組件
            comp_name = comp.file_path.split('/')[-1].replace('.py', '')
            diagram_lines.append(f'        S{i+1}["{comp_name}"]')
        
        diagram_lines.extend([
            "    end",
            "",
            "    %% Security & Monitoring",
            "    subgraph SECURITY[\"Security & Monitoring\"]",
        ])
        
        # 添加安全組件
        for i, comp in enumerate(security_components[:4]):  # 最多4個安全組件
            comp_name = comp.file_path.split('/')[-1].replace('.py', '')
            diagram_lines.append(f'        SEC{i+1}["{comp_name}"]')
        
        diagram_lines.extend([
            "    end",
            "",
            "    %% Integration Points",
            "    subgraph INTEGRATION[\"External Integration\"]",
        ])
        
        # 添加整合點
        for i, comp in enumerate(integration_components[:4]):
            comp_name = comp.file_path.split('/')[-1].replace('.py', '')
            diagram_lines.append(f'        I{i+1}["{comp_name}"]')
        
        diagram_lines.extend([
            "    end",
            "",
            "    %% Data Flow",
            "    CORE --> SERVICES",
            "    SERVICES --> SECURITY", 
            "    SERVICES --> INTEGRATION",
            "    SECURITY --> CORE",
            "",
            "    %% Styling",
            "    classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef service fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef security fill:#fff3e0,stroke:#e65100,stroke-width:2px", 
            "    classDef integration fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
            "",
            "    class C1,C2,C3,C4,C5 core",
            "    class S1,S2,S3,S4,S5,S6 service",
            "    class SEC1,SEC2,SEC3,SEC4 security",
            "    class I1,I2,I3,I4 integration",
            "```"
        ])
        
        return "\n".join(diagram_lines)

def main():
    """主要執行函數"""
    
    print("🚀 AIVA 整合模組架構分析開始...")
    print("💡 應用「完整產出 + 智能篩選」策略\n")
    
    analyzer = IntegrationModuleAnalyzer()
    
    # 第一階段：完整掃描組件
    print("📋 第一階段：完整掃描整合模組組件")
    components = analyzer.scan_integration_components()
    
    if not components:
        print("❌ 未發現任何組件，請檢查整合模組路徑")
        return
    
    # 第二階段：生成所有個別圖表
    print(f"\n📊 第二階段：生成 {len(components)} 個個別組件圖表")
    print("⚠️  提醒：會產生大量圖檔，這是故意的！")
    generated_files = analyzer.generate_component_diagrams(components)
    
    # 第三階段：智能分類
    print(f"\n🧠 第三階段：智能分類和分析")
    classifications = analyzer.classify_components(components)
    
    # 統計分類結果
    stats = {}
    for classification in classifications:
        category = classification.category
        stats[category] = stats.get(category, 0) + 1
    
    print("📊 分類統計:")
    for category, count in stats.items():
        percentage = (count / len(classifications)) * 100
        print(f"   {category}: {count} 個組件 ({percentage:.1f}%)")
    
    # 第四階段：生成整合架構圖
    print(f"\n🏗️ 第四階段：生成整合架構圖")
    integrated_diagram = analyzer.generate_integrated_architecture(classifications)
    
    # 儲存結果
    output_dir = analyzer.base_dir / "_out"
    
    # 整合架構圖
    integrated_file = output_dir / "INTEGRATION_MODULE_AUTO_INTEGRATED.mmd"
    with open(integrated_file, 'w', encoding='utf-8') as f:
        f.write(integrated_diagram)
    
    # 分類資料
    classification_data = {
        "total_components": len(components),
        "generated_files": len(generated_files), 
        "classifications": [asdict(c) for c in classifications],
        "statistics": stats,
        "analysis_timestamp": "2025-10-24"
    }
    
    classification_file = output_dir / "integration_diagram_classification.json"
    with open(classification_file, 'w', encoding='utf-8') as f:
        json.dump(classification_data, f, indent=2, ensure_ascii=False)
    
    # 結果報告
    print(f"\n✅ 整合模組分析完成！")
    print(f"📊 總組件數: {len(components)}")
    print(f"📁 已生成圖檔: {len(generated_files)} 個")
    print(f"🎯 整合架構圖: {integrated_file}")
    print(f"📋 分類資料: {classification_file}")
    
    print(f"\n💡 下一步建議：")
    print(f"1. 檢視整合架構圖: {integrated_file}")
    print(f"2. 分析分類資料: {classification_file}") 
    print(f"3. 人工識別關鍵模式和價值")
    print(f"4. 執行清理: python scripts/cleanup_diagram_output.py")
    
    print(f"\n🧠 記住：笨方法的智慧 - 先完整產出，再智能篩選！")

if __name__ == "__main__":
    main()