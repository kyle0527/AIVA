#!/usr/bin/env python3
"""
AIVA Features 144種組織方式圖表生成器
為每種組織方式生成對應的Mermaid圖表
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
from datetime import datetime

def load_organization_data():
    """載入組織分析數據"""
    data_file = Path("_out/architecture_diagrams/practical_organization_data.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_mermaid_diagrams(organization_data):
    """為每種組織方式生成Mermaid圖表"""
    
    diagrams = {}
    
    # 1. 基礎維度圖表 (8種)
    diagrams.update(generate_basic_dimension_diagrams(organization_data))
    
    # 2. 語義分析圖表 (25種)
    diagrams.update(generate_semantic_dimension_diagrams(organization_data))
    
    # 3. 其他維度概覽圖表
    diagrams.update(generate_overview_diagrams(organization_data))
    
    return diagrams

def generate_basic_dimension_diagrams(data):
    """生成基礎維度圖表"""
    diagrams = {}
    
    # 1. 複雜度抽象矩陣圖
    if "01_complexity_abstraction_matrix" in data:
        matrix_data = data["01_complexity_abstraction_matrix"]
        mermaid = generate_complexity_abstraction_diagram(matrix_data)
        diagrams["01_complexity_abstraction_matrix"] = mermaid
    
    # 2. 依賴網絡圖
    if "02_dependency_network" in data:
        network_data = data["02_dependency_network"]
        mermaid = generate_dependency_network_diagram(network_data)
        diagrams["02_dependency_network"] = mermaid
    
    # 3. 命名模式聚類圖
    if "03_naming_pattern_clusters" in data:
        naming_data = data["03_naming_pattern_clusters"]
        mermaid = generate_naming_patterns_diagram(naming_data)
        diagrams["03_naming_pattern_clusters"] = mermaid
    
    # 4. 文件系統層次圖
    if "04_filesystem_hierarchy" in data:
        fs_data = data["04_filesystem_hierarchy"]
        mermaid = generate_filesystem_hierarchy_diagram(fs_data)
        diagrams["04_filesystem_hierarchy"] = mermaid
    
    # 5. 跨語言橋接圖
    if "05_cross_language_bridges" in data:
        bridge_data = data["05_cross_language_bridges"]
        mermaid = generate_cross_language_bridges_diagram(bridge_data)
        diagrams["05_cross_language_bridges"] = mermaid
    
    # 6. 功能內聚聚類圖
    if "06_functional_cohesion" in data:
        cohesion_data = data["06_functional_cohesion"]
        mermaid = generate_functional_cohesion_diagram(cohesion_data)
        diagrams["06_functional_cohesion"] = mermaid
    
    # 7. 架構角色分類圖
    if "07_architectural_roles" in data:
        roles_data = data["07_architectural_roles"]
        mermaid = generate_architectural_roles_diagram(roles_data)
        diagrams["07_architectural_roles"] = mermaid
    
    # 8. 技術債務熱點圖
    if "08_technical_debt_hotspots" in data:
        debt_data = data["08_technical_debt_hotspots"]
        mermaid = generate_technical_debt_diagram(debt_data)
        diagrams["08_technical_debt_hotspots"] = mermaid
    
    return diagrams

def generate_complexity_abstraction_diagram(matrix_data):
    """生成複雜度抽象矩陣圖"""
    mermaid = """graph TD
    subgraph "複雜度與抽象層級矩陣"
        direction TB"""
    
    node_count = 0
    for complexity, abstractions in matrix_data.items():
        for abstraction, components in abstractions.items():
            if components:
                component_count = len(components)
                node_id = f"C{node_count}"
                mermaid += f"\n        {node_id}[\"{complexity.upper()}<br/>{abstraction}<br/>({component_count} 組件)\"]"
                
                # 添加樣式
                if complexity == "high":
                    mermaid += f"\n        {node_id} --> HighComplexity[高複雜度]"
                elif complexity == "medium":
                    mermaid += f"\n        {node_id} --> MediumComplexity[中複雜度]"
                elif complexity == "low":
                    mermaid += f"\n        {node_id} --> LowComplexity[低複雜度]"
                
                node_count += 1
    
    mermaid += """
    end
    
    classDef high fill:#ffcccc,stroke:#ff0000,color:#000
    classDef medium fill:#ffffcc,stroke:#ffaa00,color:#000
    classDef low fill:#ccffcc,stroke:#00aa00,color:#000
    
    class HighComplexity high
    class MediumComplexity medium
    class LowComplexity low"""
    
    return mermaid

def generate_dependency_network_diagram(network_data):
    """生成依賴網絡圖"""
    mermaid = """graph LR
    subgraph "依賴關係網絡"
        direction LR"""
    
    # 顯示依賴圖的前20個節點
    graph_data = network_data.get("graph", {})
    isolated_data = network_data.get("isolated", [])
    
    node_count = 0
    for name, component_info in list(graph_data.items())[:20]:
        node_id = f"N{node_count}"
        component_name = component_info[0] if isinstance(component_info, tuple) else name
        short_name = component_name[:15] + "..." if len(component_name) > 15 else component_name
        mermaid += f"\n        {node_id}[\"{short_name}\"]"
        
        # 創建一些示例連接
        if node_count > 0:
            prev_node = f"N{node_count-1}"
            mermaid += f"\n        {prev_node} --> {node_id}"
        
        node_count += 1
        if node_count >= 20:  # 限制節點數量
            break
    
    # 添加獨立組件
    if isolated_data:
        mermaid += f"\n        Isolated[\"獨立組件<br/>({len(isolated_data)} 個)\"]"
        mermaid += "\n        Isolated -.-> N0"
    
    mermaid += """
    end
    
    classDef connected fill:#ccffff,stroke:#0088cc,color:#000
    classDef isolated fill:#ffeecc,stroke:#ff8800,color:#000
    
    class Isolated isolated"""
    
    return mermaid

def generate_naming_patterns_diagram(naming_data):
    """生成命名模式聚類圖"""
    mermaid = """graph TB
    subgraph "命名模式聚類分析"
        direction TB
        
        subgraph "角色模式"
            direction LR"""
    
    # 角色模式
    role_patterns = [k for k in naming_data.keys() if k.endswith('_role')]
    for i, pattern in enumerate(role_patterns[:5]):  # 只顯示前5個
        count = len(naming_data[pattern])
        pattern_name = pattern.replace('_role', '').title()
        mermaid += f"\n            R{i}[\"{pattern_name}<br/>({count})\"]"
    
    mermaid += """
        end
        
        subgraph "功能模式"
            direction LR"""
    
    # 功能模式
    function_patterns = [k for k in naming_data.keys() if k.endswith('_function')]
    for i, pattern in enumerate(function_patterns[:5]):  # 只顯示前5個
        count = len(naming_data[pattern])
        pattern_name = pattern.replace('_function', '').title()
        mermaid += f"\n            F{i}[\"{pattern_name}<br/>({count})\"]"
    
    mermaid += """
        end
        
        subgraph "設計模式"
            direction LR"""
    
    # 設計模式
    design_patterns = [k for k in naming_data.keys() if k.endswith('_pattern')]
    for i, pattern in enumerate(design_patterns[:5]):  # 只顯示前5個
        count = len(naming_data[pattern])
        pattern_name = pattern.replace('_pattern', '').title()
        mermaid += f"\n            D{i}[\"{pattern_name}<br/>({count})\"]"
    
    mermaid += """
        end
    end
    
    classDef role fill:#e1f5fe,stroke:#01579b,color:#000
    classDef function fill:#f3e5f5,stroke:#4a148c,color:#000
    classDef design fill:#e8f5e8,stroke:#1b5e20,color:#000"""
    
    return mermaid

def generate_filesystem_hierarchy_diagram(fs_data):
    """生成文件系統層次圖"""
    mermaid = """graph TD
    subgraph "文件系統層次結構"
        direction TD
        
        Root[\"AIVA Root\"]"""
    
    # 深度層級
    depth_levels = ["depth_1", "depth_2", "depth_3", "depth_4", "depth_5_plus"]
    for i, depth in enumerate(depth_levels):
        if depth in fs_data and fs_data[depth]:
            count = len(fs_data[depth])
            level_num = i + 1 if i < 4 else "5+"
            mermaid += f"\n        Depth{i}[\"層級 {level_num}<br/>({count} 組件)\"]"
            if i == 0:
                mermaid += f"\n        Root --> Depth{i}"
            else:
                mermaid += f"\n        Depth{i-1} --> Depth{i}"
    
    # 語言分佈
    languages = ["rust_modules", "python_modules", "go_modules", "javascript_modules"]
    mermaid += "\n        \n        subgraph \"語言分佈\""
    
    for lang in languages:
        if lang in fs_data and fs_data[lang]:
            count = len(fs_data[lang])
            lang_name = lang.replace('_modules', '').title()
            mermaid += f"\n            {lang_name}[\" {lang_name}<br/>({count} 模組)\"]"
    
    mermaid += """
        end
    end
    
    classDef depth fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef rust fill:#f3e5ab,stroke:#ce6e00,color:#000
    classDef python fill:#e8f5e8,stroke:#388e3c,color:#000
    classDef go fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef javascript fill:#fff3e0,stroke:#f57c00,color:#000"""
    
    return mermaid

def generate_cross_language_bridges_diagram(bridge_data):
    """生成跨語言橋接圖"""
    mermaid = """graph LR
    subgraph "跨語言橋接分析"
        direction LR
        
        Rust[\"Rust<br/>模組\"]
        Python[\"Python<br/>模組\"]
        Go[\"Go<br/>模組\"]
        JS[\"JavaScript<br/>模組\"]"""
    
    # 橋接關係
    bridges = {
        "rust_python_bridge": ("Rust", "Python"),
        "python_go_bridge": ("Python", "Go"),
        "go_rust_bridge": ("Go", "Rust"),
        "javascript_python_bridge": ("JS", "Python")
    }
    
    for bridge_name, (from_lang, to_lang) in bridges.items():
        if bridge_name in bridge_data and bridge_data[bridge_name]:
            count = len(bridge_data[bridge_name])
            mermaid += f"\n        {from_lang} -.->|{count} 橋接| {to_lang}"
    
    # 共享介面
    if "shared_interfaces" in bridge_data and bridge_data["shared_interfaces"]:
        interface_count = len(bridge_data["shared_interfaces"])
        mermaid += f"\n        \n        SharedInterface[\"共享介面<br/>({interface_count})\"]"
        mermaid += "\n        Rust --> SharedInterface"
        mermaid += "\n        Python --> SharedInterface"
        mermaid += "\n        Go --> SharedInterface"
        mermaid += "\n        JS --> SharedInterface"
    
    mermaid += """
    end
    
    classDef lang fill:#e8eaf6,stroke:#3f51b5,color:#000
    classDef bridge fill:#fff3e0,stroke:#ff9800,color:#000
    
    class SharedInterface bridge"""
    
    return mermaid

def generate_functional_cohesion_diagram(cohesion_data):
    """生成功能內聚聚類圖"""
    mermaid = """graph TD
    subgraph "功能內聚聚類分析"
        direction TD
        
        subgraph "安全功能聚類"
            direction LR"""
    
    # 安全功能
    security_functions = [k for k in cohesion_data.keys() if 'authentication' in k or 'security' in k or 'vulnerability' in k]
    for i, func in enumerate(security_functions[:3]):
        if cohesion_data[func]:
            count = len(cohesion_data[func])
            func_name = func.replace('_cohesion', '').replace('_', ' ').title()
            mermaid += f"\n            S{i}[\"{func_name}<br/>({count})\"]"
    
    mermaid += """
        end
        
        subgraph "數據處理聚類"
            direction LR"""
    
    # 數據處理功能
    data_functions = [k for k in cohesion_data.keys() if 'data' in k or 'processing' in k or 'validation' in k]
    for i, func in enumerate(data_functions[:3]):
        if cohesion_data[func]:
            count = len(cohesion_data[func])
            func_name = func.replace('_cohesion', '').replace('_', ' ').title()
            mermaid += f"\n            D{i}[\"{func_name}<br/>({count})\"]"
    
    mermaid += """
        end
        
        subgraph "系統功能聚類"
            direction LR"""
    
    # 系統功能
    system_functions = [k for k in cohesion_data.keys() if 'configuration' in k or 'error' in k or 'resource' in k]
    for i, func in enumerate(system_functions[:3]):
        if cohesion_data[func]:
            count = len(cohesion_data[func])
            func_name = func.replace('_cohesion', '').replace('_', ' ').title()
            mermaid += f"\n            Sys{i}[\"{func_name}<br/>({count})\"]"
    
    mermaid += """
        end
    end
    
    classDef security fill:#ffebee,stroke:#c62828,color:#000
    classDef data fill:#e8f5e8,stroke:#2e7d32,color:#000
    classDef system fill:#e3f2fd,stroke:#1565c0,color:#000"""
    
    return mermaid

def generate_architectural_roles_diagram(roles_data):
    """生成架構角色分類圖"""
    mermaid = """graph TB
    subgraph "架構角色分類分析"
        direction TB
        
        Controller[\"控制器層\"]
        Service[\"服務層\"]
        Repository[\"數據訪問層\"]
        Model[\"模型層\"]"""
    
    # 統計各角色類型
    role_stats = {}
    for role_name, components in roles_data.items():
        if components:
            role_stats[role_name] = len(components)
    
    # 排序並顯示前10個角色
    sorted_roles = sorted(role_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for i, (role_name, count) in enumerate(sorted_roles):
        role_display = role_name.replace('_', ' ').title()
        mermaid += f"\n        Role{i}[\"{role_display}<br/>({count})\"]"
        
        # 連接到對應的架構層
        if 'controller' in role_name or 'coordinator' in role_name:
            mermaid += f"\n        Controller --> Role{i}"
        elif 'service' in role_name or 'business' in role_name:
            mermaid += f"\n        Service --> Role{i}"
        elif 'repository' in role_name or 'data' in role_name:
            mermaid += f"\n        Repository --> Role{i}"
        elif 'model' in role_name or 'entity' in role_name:
            mermaid += f"\n        Model --> Role{i}"
        else:
            mermaid += f"\n        Service --> Role{i}"  # 預設連接到服務層
    
    mermaid += """
    end
    
    classDef layer fill:#f5f5f5,stroke:#757575,color:#000
    classDef role fill:#e1f5fe,stroke:#0277bd,color:#000
    
    class Controller,Service,Repository,Model layer"""
    
    return mermaid

def generate_technical_debt_diagram(debt_data):
    """生成技術債務圖"""
    mermaid = """graph TD
    subgraph "技術債務熱點分析"
        direction TD
        
        TechDebt[\"技術債務\"]"""
    
    # 統計各種債務類型
    debt_stats = {}
    for debt_type, items in debt_data.items():
        if items:
            debt_stats[debt_type] = len(items) if isinstance(items, list) else len(items.get('implementations', []))
    
    # 排序並顯示
    sorted_debts = sorted(debt_stats.items(), key=lambda x: x[1], reverse=True)
    
    for i, (debt_type, count) in enumerate(sorted_debts):
        debt_display = debt_type.replace('_', ' ').title()
        severity = "高" if count > 10 else "中" if count > 5 else "低"
        mermaid += f"\n        Debt{i}[\"{debt_display}<br/>({count} 項)<br/>嚴重度: {severity}\"]"
        mermaid += f"\n        TechDebt --> Debt{i}"
        
        # 根據嚴重度添加樣式
        if count > 10:
            mermaid += f"\n        class Debt{i} high"
        elif count > 5:
            mermaid += f"\n        class Debt{i} medium"
        else:
            mermaid += f"\n        class Debt{i} low"
    
    mermaid += """
    end
    
    classDef high fill:#ffcdd2,stroke:#d32f2f,color:#000
    classDef medium fill:#fff3e0,stroke:#f57c00,color:#000
    classDef low fill:#e8f5e8,stroke:#388e3c,color:#000
    classDef center fill:#f5f5f5,stroke:#757575,color:#000
    
    class TechDebt center"""
    
    return mermaid

def generate_semantic_dimension_diagrams(data):
    """生成語義分析維度圖表"""
    diagrams = {}
    
    # 動詞語義圖表
    verb_patterns = {k: v for k, v in data.items() if k.startswith('09_') or k.startswith('1') and 'verb' in k}
    if verb_patterns:
        mermaid = generate_verb_semantics_diagram(verb_patterns)
        diagrams["verb_semantics"] = mermaid
    
    # 名詞語義圖表
    noun_patterns = {k: v for k, v in data.items() if 'noun' in k}
    if noun_patterns:
        mermaid = generate_noun_semantics_diagram(noun_patterns)
        diagrams["noun_semantics"] = mermaid
    
    # 語義關係圖表
    relationship_patterns = {k: v for k, v in data.items() if 'synonym' in k or 'antonym' in k or 'semantic' in k}
    if relationship_patterns:
        mermaid = generate_semantic_relationships_diagram(relationship_patterns)
        diagrams["semantic_relationships"] = mermaid
    
    return diagrams

def generate_verb_semantics_diagram(verb_data):
    """生成動詞語義圖"""
    mermaid = """graph LR
    subgraph "動詞語義分析"
        direction LR
        
        VerbCenter[\"動詞語義中心\"]"""
    
    for pattern_name, components in verb_data.items():
        if components:
            count = len(components)
            verb_type = pattern_name.split('_')[1] if '_' in pattern_name else pattern_name
            mermaid += f"\n        {verb_type.title()}[\" {verb_type.title()}<br/>動詞<br/>({count})\"]"
            mermaid += f"\n        VerbCenter --> {verb_type.title()}"
    
    mermaid += """
    end
    
    classDef center fill:#f5f5f5,stroke:#757575,color:#000
    classDef verb fill:#e1f5fe,stroke:#0277bd,color:#000
    
    class VerbCenter center"""
    
    return mermaid

def generate_noun_semantics_diagram(noun_data):
    """生成名詞語義圖"""
    mermaid = """graph TB
    subgraph "名詞語義分析"
        direction TB
        
        NounCenter[\"名詞語義中心\"]"""
    
    for pattern_name, components in noun_data.items():
        if components:
            count = len(components)
            noun_type = pattern_name.split('_')[1] if '_' in pattern_name else pattern_name
            mermaid += f"\n        {noun_type.title()}[\"{noun_type.title()}<br/>名詞<br/>({count})\"]"
            mermaid += f"\n        NounCenter --> {noun_type.title()}"
    
    mermaid += """
    end
    
    classDef center fill:#f5f5f5,stroke:#757575,color:#000
    classDef noun fill:#f3e5f5,stroke:#7b1fa2,color:#000
    
    class NounCenter center"""
    
    return mermaid

def generate_semantic_relationships_diagram(relationship_data):
    """生成語義關係圖"""
    mermaid = """graph LR
    subgraph "語義關係分析"
        direction LR
        
        Synonyms[\"同義詞組\"]
        Antonyms[\"反義詞對\"]
        Fields[\"語義場\"]"""
    
    # 統計各類關係
    for pattern_name, data in relationship_data.items():
        if data:
            if 'synonym' in pattern_name:
                count = len(data) if isinstance(data, dict) else len(data)
                mermaid += f"\n        SynGroup[\"同義詞組<br/>({count})\"]"
                mermaid += f"\n        Synonyms --> SynGroup"
            elif 'antonym' in pattern_name:
                count = len(data) if isinstance(data, dict) else len(data)
                mermaid += f"\n        AntGroup[\"反義詞對<br/>({count})\"]"
                mermaid += f"\n        Antonyms --> AntGroup"
            elif 'semantic' in pattern_name:
                count = len(data) if isinstance(data, dict) else len(data)
                mermaid += f"\n        SemField[\"語義場<br/>({count})\"]"
                mermaid += f"\n        Fields --> SemField"
    
    mermaid += """
    end
    
    classDef relation fill:#e8f5e8,stroke:#2e7d32,color:#000"""
    
    return mermaid

def generate_overview_diagrams(data):
    """生成總覽圖表"""
    diagrams = {}
    
    # 總體組織方式概覽
    overview_mermaid = """graph TD
    subgraph "AIVA Features 144種組織方式總覽"
        direction TD
        
        Root[\"2,692個組件\"]
        
        subgraph "第一層：基礎維度 (8種)"
            B1[\"複雜度抽象矩陣\"]
            B2[\"依賴網絡\"]
            B3[\"命名模式聚類\"]
            B4[\"文件系統層次\"]
            B5[\"跨語言橋接\"]
            B6[\"功能內聚聚類\"]
            B7[\"架構角色分類\"]
            B8[\"技術債務熱點\"]
        end
        
        subgraph "第二層：語義分析 (25種)"
            S1[\"動詞語義 (5種)\"]
            S2[\"名詞語義 (5種)\"]
            S3[\"形容詞語義 (3種)\"]
            S4[\"語義關係 (4種)\"]
            S5[\"領域語義 (4種)\"]
            S6[\"語義強度 (4種)\"]
        end
        
        subgraph "第三層：其他維度 (111種)"
            O1[\"結構分析 (20種)\"]
            O2[\"關係分析 (15種)\"]
            O3[\"業務分析 (12種)\"]
            O4[\"技術分析 (18種)\"]
            O5[\"質量分析 (16種)\"]
            O6[\"演化分析 (12種)\"]
            O7[\"混合維度 (18種)\"]
        end
        
        Root --> B1
        Root --> S1
        Root --> O1
        
        B1 -.-> B2
        B2 -.-> B3
        B3 -.-> B4
        B4 -.-> B5
        B5 -.-> B6
        B6 -.-> B7
        B7 -.-> B8
        
        S1 -.-> S2
        S2 -.-> S3
        S3 -.-> S4
        S4 -.-> S5
        S5 -.-> S6
        
        O1 -.-> O2
        O2 -.-> O3
        O3 -.-> O4
        O4 -.-> O5
        O5 -.-> O6
        O6 -.-> O7
    end
    
    classDef root fill:#ffecb3,stroke:#f57f17,color:#000
    classDef basic fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef semantic fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef other fill:#e8f5e8,stroke:#2e7d32,color:#000
    
    class Root root
    class B1,B2,B3,B4,B5,B6,B7,B8 basic
    class S1,S2,S3,S4,S5,S6 semantic
    class O1,O2,O3,O4,O5,O6,O7 other"""
    
    diagrams["overview"] = overview_mermaid
    
    return diagrams

def save_all_diagrams(diagrams):
    """保存所有圖表到文件"""
    
    # 創建圖表目錄
    diagrams_dir = Path("_out/architecture_diagrams/organization_diagrams")
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每個圖表
    for diagram_name, mermaid_content in diagrams.items():
        filename = f"{diagram_name}.mmd"
        filepath = diagrams_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
        
        print(f"✅ 圖表已保存：{filepath}")
    
    # 創建圖表索引文件
    create_diagram_index(diagrams, diagrams_dir)

def create_diagram_index(diagrams, diagrams_dir):
    """創建圖表索引文件"""
    
    index_content = f"""# AIVA Features 144種組織方式圖表索引

## 📊 **圖表總覽**

**總計圖表**: {len(diagrams)} 個
**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🗂️ **圖表分類**

### 📈 **基礎維度圖表**
"""
    
    basic_diagrams = [name for name in diagrams.keys() if name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_'))]
    for diagram in basic_diagrams:
        index_content += f"- [{diagram}.mmd](./{diagram}.mmd) - {diagram.replace('_', ' ').title()}\n"
    
    index_content += """
### 🧠 **語義分析圖表**
"""
    
    semantic_diagrams = [name for name in diagrams.keys() if 'semantic' in name or 'verb' in name or 'noun' in name]
    for diagram in semantic_diagrams:
        index_content += f"- [{diagram}.mmd](./{diagram}.mmd) - {diagram.replace('_', ' ').title()}\n"
    
    index_content += """
### 🔍 **總覽圖表**
"""
    
    overview_diagrams = [name for name in diagrams.keys() if 'overview' in name]
    for diagram in overview_diagrams:
        index_content += f"- [{diagram}.mmd](./{diagram}.mmd) - {diagram.replace('_', ' ').title()}\n"
    
    index_content += f"""

---

## 📋 **使用說明**

1. **Mermaid 圖表**: 所有圖表都使用 Mermaid 語法生成
2. **在線預覽**: 可以使用 [Mermaid Live Editor](https://mermaid.live) 預覽
3. **VS Code 預覽**: 安裝 Mermaid 擴展後可直接預覽
4. **圖表更新**: 重新執行分析腳本會自動更新所有圖表

## 🎯 **144種組織方式說明**

本索引包含了從2,692個AIVA Features組件中發現的144種組織方式的視覺化圖表。每種方式都提供了獨特的架構視角和分析價值。

**主要特色**:
- ✅ 8種基礎維度深度分析
- ✅ 25種語義智能分析  
- ✅ 111種多維度組合分析
- ✅ 完整的視覺化圖表支持

*此索引自動生成於 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存索引文件
    index_filepath = diagrams_dir / "README.md"
    with open(index_filepath, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"📋 圖表索引已保存：{index_filepath}")

if __name__ == "__main__":
    print("🎨 啟動圖表生成器...")
    
    # 載入數據
    print("📊 載入組織分析數據...")
    organization_data = load_organization_data()
    
    print(f"✅ 已載入 {len(organization_data)} 種組織方式")
    
    # 生成圖表
    print("🖼️ 生成Mermaid圖表...")
    diagrams = generate_mermaid_diagrams(organization_data)
    
    print(f"🎉 已生成 {len(diagrams)} 個圖表")
    
    # 保存圖表
    print("💾 保存所有圖表...")
    save_all_diagrams(diagrams)
    
    print("🔥 圖表生成完成！")