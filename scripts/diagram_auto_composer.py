#!/usr/bin/env python3
"""
AIVA 圖表自動化組合工具
實現架構圖表優化框架的核心功能

⚠️ 重要理念：完整產出的智慧
「有時候笨方法才是好方法」

本腳本刻意產生大量個別組件圖檔，原因：
- 🔍 無法預知哪個組件包含關鍵架構洞察
- 🎯 最重要的模式往往隱藏在看似次要的組件中
- 🛡️ 寧可產出 300 個圖删除 295 個，也不要遺漏 1 個關鍵發現

推薦工作流程：
1. 完整產出所有圖檔（本腳本）
2. 深度分析分類結果和模式  
3. 人工識別真正的價值
4. 使用 cleanup_diagram_output.py 智能清理

核心原則：完整性 > 效率，理解 > 刪除

⚠️ 不建議在此腳本中加入 --auto-cleanup，
   因為最佳的清理決策需要人工智慧參與！
"""

import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Literal, Optional, Set
from pathlib import Path
import yaml

@dataclass
class DiagramClassification:
    """圖表分類元資料"""
    category: Literal["core", "detail", "integration", "example"]
    priority: int  # 1-10，數字越小優先級越高
    complexity: Literal["low", "medium", "high"]
    abstraction_level: Literal["system", "module", "component", "function"]
    dependencies: List[str]
    file_path: str
    reference_count: int = 0

@dataclass  
class Component:
    """架構組件"""
    name: str
    type: str
    layer: str
    connections: List[str]
    metadata: Dict[str, str]

class DiagramAnalyzer:
    """自動圖表分析器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.filename_patterns = {
            r".*_Module\.mmd$": ("core", 2, "medium", "module"),
            r".*_Function_.*__init__\.mmd$": ("detail", 8, "low", "function"),
            r".*_integration_.*\.mmd$": ("integration", 4, "medium", "component"),
            r".*_examples?_.*\.mmd$": ("example", 9, "low", "function"),
            r"\d{2}_.*\.mmd$": ("core", 1, "high", "system"),  # 手動核心圖
            r".*_(controller|manager|orchestrator|engine)_.*\.mmd$": ("core", 3, "medium", "component"),
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """載入模組配置"""
        default_config = {
            "core_keywords": ["controller", "manager", "orchestrator", "engine"],
            "integration_keywords": ["integration", "api", "service"],
            "complexity_thresholds": {"high": 20, "medium": 8}
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def classify_diagram(self, diagram_path: str) -> DiagramClassification:
        """基於檔案名稱、內容和依賴關係自動分類"""
        
        # 檔名模式匹配
        filename = os.path.basename(diagram_path)
        category, priority, complexity_hint, abstraction = self._match_filename_pattern(filename)
        
        # 內容分析
        content = self._read_diagram_content(diagram_path)
        actual_complexity = self._analyze_complexity(content)
        dependencies = self._extract_dependencies(content)
        
        # 使用實際複雜度覆蓋提示
        final_complexity = actual_complexity if actual_complexity != "low" else complexity_hint
        
        return DiagramClassification(
            category=category,
            priority=priority,
            complexity=final_complexity,
            abstraction_level=abstraction,
            dependencies=dependencies,
            file_path=diagram_path
        )
    
    def _match_filename_pattern(self, filename: str) -> tuple:
        """匹配檔名模式"""
        for pattern, classification in self.filename_patterns.items():
            if re.match(pattern, filename):
                return classification
        
        # 預設分類
        return ("detail", 7, "low", "function")
    
    def _read_diagram_content(self, file_path: str) -> str:
        """讀取圖表內容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"警告：無法讀取檔案 {file_path}: {e}")
            return ""
    
    def _analyze_complexity(self, content: str) -> Literal["low", "medium", "high"]:
        """分析圖表複雜度"""
        if not content:
            return "low"
        
        # 計算節點和連接數量
        arrow_count = content.count("-->") + content.count("-.->") + content.count("==>")
        subgraph_count = content.count("subgraph")
        node_count = len(re.findall(r'\w+\[[^\]]*\]|\w+\([^\)]*\)|\w+\{[^\}]*\}', content))
        
        total_complexity = arrow_count + (subgraph_count * 3) + (node_count * 0.5)
        
        thresholds = self.config["complexity_thresholds"]
        if total_complexity > thresholds["high"]:
            return "high"
        elif total_complexity > thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """提取依賴關係"""
        dependencies = []
        
        # 查找 import 語句和模組引用
        import_patterns = [
            r'from\s+([.\w]+)\s+import',
            r'import\s+([.\w]+)',
            r'class\s+\w+\([^)]*([A-Z]\w+)[^)]*\)',  # 繼承關係
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return list(set(dependencies))  # 去重

class DiagramComposer:
    """圖表組合引擎"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        
    def create_module_overview(self, classified_diagrams: List[DiagramClassification]) -> str:
        """創建模組概覽圖"""
        
        # 1. 選擇核心組件
        core_diagrams = [d for d in classified_diagrams 
                        if d.category in ["core", "integration"] and d.priority <= 5]
        
        # 2. 分析組件間關係
        components = self._extract_components(core_diagrams)
        
        # 3. 生成分層架構
        layers = self._generate_layers(components)
        
        # 4. 創建 Mermaid 語法
        return self._generate_mermaid_syntax(layers)
    
    def _extract_components(self, diagrams: List[DiagramClassification]) -> List[Component]:
        """從圖表中提取組件資訊"""
        components = []
        
        for diagram in diagrams:
            component_name = self._extract_component_name(diagram.file_path)
            component_type = self._determine_component_type(component_name, diagram.dependencies)
            
            component = Component(
                name=component_name,
                type=component_type,
                layer="",  # 稍後分配
                connections=diagram.dependencies,
                metadata={"file_path": diagram.file_path, "priority": str(diagram.priority)}
            )
            components.append(component)
        
        return components
    
    def _extract_component_name(self, file_path: str) -> str:
        """從檔案路徑提取組件名稱"""
        filename = os.path.basename(file_path)
        # 移除副檔名和前綴，提取主要組件名稱
        name_parts = filename.replace('.mmd', '').split('_')
        
        # 尋找關鍵的組件名稱部分
        key_parts = []
        for part in name_parts[2:]:  # 跳過 aiva_module 前綴
            if part.lower() in ['controller', 'manager', 'engine', 'orchestrator', 'config']:
                key_parts.append(part)
                break  # 找到關鍵字就停止，避免重複
            elif len(part) > 3 and part.isalpha() and len(key_parts) < 2:  # 有意義的名詞，限制數量
                key_parts.append(part)
                
        return " ".join(key_parts).title() if key_parts else filename
    
    def _determine_component_type(self, name: str, dependencies: List[str]) -> str:
        """判斷組件類型"""
        name_lower = name.lower()
        
        if "controller" in name_lower or "orchestrator" in name_lower:
            return "controller"
        elif "manager" in name_lower:
            return "manager"  
        elif "engine" in name_lower:
            return "engine"
        elif "config" in name_lower:
            return "config"
        elif len(dependencies) > 3:
            return "service"
        else:
            return "component"
    
    def _generate_layers(self, components: List[Component]) -> Dict[str, List[Component]]:
        """生成分層架構"""
        layers = {
            "interface": [],
            "control": [],
            "service": [],
            "data": [],
            "integration": []
        }
        
        for component in components:
            layer = self._determine_layer(component)
            component.layer = layer
            layers[layer].append(component)
        
        return layers
    
    def _determine_layer(self, component: Component) -> str:
        """判斷組件應該歸屬的層級"""
        name_lower = component.name.lower()
        comp_type = component.type
        
        if comp_type == "controller" or "orchestrator" in name_lower:
            return "control"
        elif comp_type == "engine" or "scanner" in name_lower:
            return "service"
        elif "config" in name_lower or "setting" in name_lower:
            return "data"
        elif "api" in name_lower or "interface" in name_lower:
            return "interface"
        elif "integration" in name_lower:
            return "integration"
        else:
            return "service"  # 預設
    
    def _generate_mermaid_syntax(self, layers: Dict[str, List[Component]]) -> str:
        """生成 Mermaid 流程圖語法"""
        
        mermaid_lines = [
            "---",
            f"title: {self.module_name.title()} Module Integrated Architecture", 
            "---",
            "flowchart TB",
            ""
        ]
        
        # 為每個層級創建子圖
        layer_names = {
            "interface": "User Interface Layer",
            "control": "Control & Strategy Layer", 
            "service": "Service & Processing Layer",
            "data": "Data & Configuration Layer",
            "integration": "Integration Services"
        }
        
        node_counter = 1
        node_mapping = {}
        
        for layer_key, layer_name in layer_names.items():
            components = layers.get(layer_key, [])
            if not components:
                continue
                
            mermaid_lines.append(f'    subgraph {layer_key.upper()}["{layer_name}"]')
            
            for component in components:
                node_id = f"n{node_counter}"
                node_mapping[component.name] = node_id
                node_counter += 1
                
                # 使用適當的節點形狀
                shape = self._get_node_shape(component.type)
                mermaid_lines.append(f'        {node_id}{shape}["{component.name}"]')
            
            mermaid_lines.append("    end")
            mermaid_lines.append("")
        
        # 添加連接關係（簡化版本）
        mermaid_lines.extend(self._generate_connections(layers, node_mapping))
        
        # 添加樣式
        mermaid_lines.extend(self._generate_styles())
        
        return "\n".join(mermaid_lines)
    
    def _get_node_shape(self, component_type: str) -> str:
        """根據組件類型選擇節點形狀"""
        shapes = {
            "controller": "",      # 矩形
            "manager": "",         # 矩形  
            "engine": "((",        # 圓形
            "config": "[(",        # 資料庫形狀
            "service": "(",        # 圓角矩形
            "component": ""        # 預設矩形
        }
        return shapes.get(component_type, "")
    
    def _generate_connections(self, layers: Dict[str, List[Component]], 
                            node_mapping: Dict[str, str]) -> List[str]:
        """生成層級間的連接"""
        connections = ["    %% Layer Connections"]
        
        # 簡化的層級間連接邏輯
        layer_order = ["interface", "control", "service", "data", "integration"]
        
        for i in range(len(layer_order) - 1):
            current_layer = layers.get(layer_order[i], [])
            next_layer = layers.get(layer_order[i + 1], [])
            
            if current_layer and next_layer:
                # 連接第一個組件作為代表
                current_node = node_mapping.get(current_layer[0].name)
                next_node = node_mapping.get(next_layer[0].name)
                
                if current_node and next_node:
                    connections.append(f"    {current_node} --> {next_node}")
        
        connections.append("")
        return connections
    
    def _generate_styles(self) -> List[str]:
        """生成樣式定義"""
        return [
            "    %% Styling",
            "    classDef control fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef service fill:#f3e5f5,stroke:#4a148c,stroke-width:2px", 
            "    classDef data fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
            "    classDef integration fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "",
            "    class CONTROL control",
            "    class SERVICE service", 
            "    class DATA data",
            "    class INTEGRATION integration"
        ]

def main():
    """主要執行函數"""
    
    # 配置 - 使用絕對路徑
    base_dir = Path(__file__).parent.parent  # 回到專案根目錄
    input_dir = base_dir / "_out" / "architecture_diagrams" 
    output_dir = base_dir / "_out"
    module_name = "scan"  # 可以通過參數傳入
    
    print(f"🔍 分析 {module_name} 模組的架構圖表...")
    
    # 1. 掃描和分類圖表
    analyzer = DiagramAnalyzer()
    classified_diagrams = []
    
    pattern = f"aiva_{module_name}_*.mmd"
    for file_path in input_dir.glob(pattern):
        classification = analyzer.classify_diagram(str(file_path))
        classified_diagrams.append(classification)
    
    print(f"✅ 發現 {len(classified_diagrams)} 個相關圖表")
    
    # 2. 生成統計報告
    stats = {}
    for diagram in classified_diagrams:
        category = diagram.category
        stats[category] = stats.get(category, 0) + 1
    
    print("📊 分類統計:")
    for category, count in stats.items():
        print(f"   {category}: {count} 個圖表")
    
    # 3. 創建組合圖表
    composer = DiagramComposer(module_name)
    integrated_diagram = composer.create_module_overview(classified_diagrams)
    
    # 4. 儲存結果
    output_path = output_dir / f"{module_name.upper()}_MODULE_AUTO_INTEGRATED.mmd"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(integrated_diagram)
    
    print(f"🎯 已生成整合架構圖: {output_path}")
    
    # 5. 儲存分類資訊
    classification_data = [asdict(d) for d in classified_diagrams]
    classification_path = output_dir / f"{module_name}_diagram_classification.json"
    with open(classification_path, 'w', encoding='utf-8') as f:
        json.dump(classification_data, f, indent=2, ensure_ascii=False)
    
    print(f"📋 已儲存分類資訊: {classification_path}")
    print("✨ 自動化組合完成！")

if __name__ == "__main__":
    main()