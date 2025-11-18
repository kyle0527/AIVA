#!/usr/bin/env python3
"""
AIVA Features 新組圖方案生成器
基於深度分析結果創建創新的組圖方式
"""

import json
from pathlib import Path
from collections import defaultdict

def create_technical_stack_diagrams():
    """創建技術棧組圖"""
    
    # 讀取分析數據
    with open("_out/architecture_diagrams/advanced_analysis_data.json", 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    clusters = analysis['functional_clusters']
    
    # 前端安全棧
    frontend_stack = {
        'xss_cluster': clusters.get('xss_cluster', []),
        'authentication_cluster': [comp for comp in clusters.get('authentication_cluster', []) 
                                 if 'client' in comp[0].lower() or 'js' in comp[1].get('file_path', '').lower()],
        'bypass_cluster': clusters.get('bypass_cluster', [])
    }
    
    # 後端安全棧
    backend_stack = {
        'injection_cluster': clusters.get('injection_cluster', []),
        'ssrf_cluster': clusters.get('ssrf_cluster', []),
        'idor_cluster': clusters.get('idor_cluster', []),
        'sast_cluster': clusters.get('sast_cluster', [])
    }
    
    # 身份驗證棧
    auth_stack = {
        'oauth_cluster': clusters.get('oauth_cluster', []),
        'jwt_cluster': clusters.get('jwt_cluster', []),
        'authentication_cluster': [comp for comp in clusters.get('authentication_cluster', [])
                                 if 'oauth' not in comp[0].lower() and 'jwt' not in comp[0].lower()]
    }
    
    # 基礎設施棧
    infrastructure_stack = {
        'worker_cluster': clusters.get('worker_cluster', []),
        'config_cluster': clusters.get('config_cluster', []),
        'schema_cluster': clusters.get('schema_cluster', []),
        'statistics_cluster': clusters.get('statistics_cluster', []),
        'telemetry_cluster': clusters.get('telemetry_cluster', [])
    }
    
    return {
        'frontend_security_stack': frontend_stack,
        'backend_security_stack': backend_stack,
        'authentication_stack': auth_stack,
        'infrastructure_stack': infrastructure_stack
    }

def create_lifecycle_diagrams():
    """創建生命週期組圖"""
    
    with open("_out/architecture_diagrams/advanced_analysis_data.json", 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    naming_patterns = analysis['naming_patterns']
    
    # 檢測階段
    detection_phase = {
        'detectors': naming_patterns.get('detector_pattern', []),
        'engines': naming_patterns.get('engine_pattern', []),
        'validators': naming_patterns.get('validator_pattern', [])
    }
    
    # 分析階段
    analysis_phase = {
        'analyzers': [comp for comp in naming_patterns.get('test_pattern', []) 
                     if 'analy' in comp[0].lower()],
        'parsers': naming_patterns.get('parser_pattern', []),
        'processors': [comp for comp in naming_patterns.get('executor_pattern', [])
                      if 'process' in comp[0].lower()]
    }
    
    # 報告階段
    reporting_phase = {
        'results': naming_patterns.get('result_pattern', []),
        'formatters': naming_patterns.get('formatter_pattern', []),
        'payloads': naming_patterns.get('payload_pattern', [])
    }
    
    # 管理階段
    management_phase = {
        'managers': naming_patterns.get('manager_pattern', []),
        'controllers': naming_patterns.get('controller_pattern', []),
        'executors': naming_patterns.get('executor_pattern', [])
    }
    
    return {
        'detection_phase': detection_phase,
        'analysis_phase': analysis_phase,
        'reporting_phase': reporting_phase,
        'management_phase': management_phase
    }

def create_design_pattern_diagrams():
    """創建設計模式組圖"""
    
    with open("_out/architecture_diagrams/advanced_analysis_data.json", 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    roles = analysis['architectural_roles']
    naming = analysis['naming_patterns']
    
    # 創建模式
    creational_patterns = {
        'factories': roles.get('factories', []),
        'builders': naming.get('builder_pattern', []),
        'singletons': roles.get('singletons', [])
    }
    
    # 結構模式
    structural_patterns = {
        'adapters': roles.get('adapters', []),
        'decorators': roles.get('decorators', []),
        'facades': roles.get('facades', [])
    }
    
    # 行為模式
    behavioral_patterns = {
        'strategies': roles.get('strategies', []),
        'observers': roles.get('observers', []),
        'handlers': naming.get('handler_pattern', [])
    }
    
    # 併發模式
    concurrency_patterns = {
        'workers': naming.get('worker_pattern', []),
        'executors': naming.get('executor_pattern', []),
        'managers': naming.get('manager_pattern', [])
    }
    
    return {
        'creational_patterns': creational_patterns,
        'structural_patterns': structural_patterns,
        'behavioral_patterns': behavioral_patterns,
        'concurrency_patterns': concurrency_patterns
    }

def create_complexity_matrix_diagram():
    """創建複雜度矩陣組圖"""
    
    with open("_out/architecture_diagrams/advanced_analysis_data.json", 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    complexity_data = analysis['complexity_abstraction']
    
    # 重新組織為矩陣形式
    matrix = {}
    for complexity, abstractions in complexity_data.items():
        matrix[complexity] = {}
        for abstraction, components in abstractions.items():
            # 按語言分組
            by_language = defaultdict(list)
            for comp in components:
                by_language[comp['language']].append(comp)
            matrix[complexity][abstraction] = dict(by_language)
    
    return matrix

def generate_mermaid_diagram(diagram_type, data, title):
    """生成 Mermaid 圖表"""
    
    mermaid_content = f"""---
title: {title}
---
flowchart TD
    subgraph "AIVA Features - {title}"
        direction TB
"""
    
    node_id = 1
    connections = []
    
    if diagram_type == "technical_stack":
        for stack_name, stack_data in data.items():
            stack_id = f"S{node_id}"
            node_id += 1
            
            # 創建棧節點
            clean_name = stack_name.replace('_', ' ').title()
            mermaid_content += f'        {stack_id}["{clean_name}"]\n'
            mermaid_content += f'        {stack_id} --> {stack_id}_DETAILS\n'
            mermaid_content += f'        subgraph {stack_id}_DETAILS["{clean_name} Details"]\n'
            
            for cluster_name, components in stack_data.items():
                if components:
                    cluster_id = f"C{node_id}"
                    node_id += 1
                    
                    cluster_display = cluster_name.replace('_', ' ').replace(' cluster', '').title()
                    component_count = len(components)
                    
                    mermaid_content += f'            {cluster_id}["{cluster_display}<br/>{component_count} 組件"]\n'
                    
                    # 添加主要語言信息
                    languages = {}
                    for comp in components:
                        lang = comp[1].get('language', 'unknown')
                        languages[lang] = languages.get(lang, 0) + 1
                    
                    if languages:
                        main_lang = max(languages, key=languages.get)
                        mermaid_content += f'            {cluster_id} -.-> L{node_id}["{main_lang}: {languages[main_lang]}"]\n'
                        node_id += 1
            
            mermaid_content += "        end\n"
    
    elif diagram_type == "lifecycle":
        for phase_name, phase_data in data.items():
            phase_id = f"P{node_id}"
            node_id += 1
            
            clean_name = phase_name.replace('_', ' ').title()
            mermaid_content += f'        {phase_id}["{clean_name}"]\n'
            
            for category_name, components in phase_data.items():
                if components:
                    category_id = f"CAT{node_id}"
                    node_id += 1
                    
                    category_display = category_name.replace('_', ' ').title()
                    component_count = len(components)
                    
                    mermaid_content += f'        {category_id}["{category_display}<br/>{component_count} 組件"]\n'
                    connections.append(f'        {phase_id} --> {category_id}')
    
    elif diagram_type == "design_patterns":
        for pattern_type, pattern_data in data.items():
            pattern_id = f"PT{node_id}"
            node_id += 1
            
            clean_name = pattern_type.replace('_', ' ').title()
            mermaid_content += f'        {pattern_id}["{clean_name}"]\n'
            
            for role_name, components in pattern_data.items():
                if components:
                    role_id = f"R{node_id}"
                    node_id += 1
                    
                    role_display = role_name.replace('_', ' ').title()
                    component_count = len(components)
                    
                    mermaid_content += f'        {role_id}["{role_display}<br/>{component_count} 組件"]\n'
                    connections.append(f'        {pattern_id} --> {role_id}')
    
    elif diagram_type == "complexity_matrix":
        for complexity, abstractions in data.items():
            complexity_id = f"COMP{node_id}"
            node_id += 1
            
            mermaid_content += f'        {complexity_id}["{complexity.upper()} Complexity"]\n'
            
            for abstraction, languages in abstractions.items():
                abstraction_id = f"ABS{node_id}"
                node_id += 1
                
                total_components = sum(len(comps) for comps in languages.values())
                mermaid_content += f'        {abstraction_id}["{abstraction.title()}<br/>{total_components} 組件"]\n'
                connections.append(f'        {complexity_id} --> {abstraction_id}')
                
                for language, components in languages.items():
                    lang_id = f"L{node_id}"
                    node_id += 1
                    
                    mermaid_content += f'        {lang_id}["{language}: {len(components)}"]\n'
                    connections.append(f'        {abstraction_id} --> {lang_id}')
    
    # 添加連接
    for connection in connections:
        mermaid_content += f"{connection}\n"
    
    # 添加樣式
    mermaid_content += """
    end

    classDef stackStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef clusterStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px  
    classDef componentStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef languageStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
"""
    
    return mermaid_content

def generate_all_new_diagrams():
    """生成所有新的組圖方案"""
    
    print("🎯 生成技術棧組圖...")
    tech_stacks = create_technical_stack_diagrams()
    
    print("🔄 生成生命週期組圖...")
    lifecycle = create_lifecycle_diagrams()
    
    print("🎨 生成設計模式組圖...")
    design_patterns = create_design_pattern_diagrams()
    
    print("📊 生成複雜度矩陣組圖...")
    complexity_matrix = create_complexity_matrix_diagram()
    
    # 生成 Mermaid 圖表
    diagrams = [
        ("technical_stack", tech_stacks, "技術棧架構", "FEATURES_TECHNICAL_STACKS"),
        ("lifecycle", lifecycle, "生命週期架構", "FEATURES_LIFECYCLE_PHASES"),
        ("design_patterns", design_patterns, "設計模式架構", "FEATURES_DESIGN_PATTERNS"),
        ("complexity_matrix", complexity_matrix, "複雜度矩陣", "FEATURES_COMPLEXITY_MATRIX")
    ]
    
    output_dir = Path("_out/architecture_diagrams/advanced")
    output_dir.mkdir(exist_ok=True)
    
    generated_files = []
    
    for diagram_type, data, title, filename in diagrams:
        print(f"📈 生成 {title} 圖表...")
        
        mermaid_content = generate_mermaid_diagram(diagram_type, data, title)
        
        output_file = output_dir / f"{filename}.mmd"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
        
        generated_files.append(str(output_file))
        print(f"✅ 已生成: {output_file}")
    
    return generated_files

if __name__ == "__main__":
    print("🚀 開始生成新的組圖方案...")
    
    generated_files = generate_all_new_diagrams()
    
    print(f"\n✨ 完成！共生成 {len(generated_files)} 個新組圖:")
    for file in generated_files:
        print(f"  📊 {file}")
    
    print("\n💡 新組圖方案包括:")
    print("  🎯 技術棧架構 - 按前端/後端/認證/基礎設施分組")
    print("  🔄 生命週期架構 - 按檢測/分析/報告/管理階段分組") 
    print("  🎨 設計模式架構 - 按創建/結構/行為/併發模式分組")
    print("  📊 複雜度矩陣 - 按複雜度和抽象層級交叉分組")