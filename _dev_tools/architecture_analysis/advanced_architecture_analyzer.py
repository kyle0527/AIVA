#!/usr/bin/env python3
"""
AIVA Features 深度組圖分析器
發現隱藏的組織能力和架構模式
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import re

def analyze_advanced_patterns():
    """進行深度架構模式分析"""
    
    # 讀取分類數據
    classification_file = Path("_out/architecture_diagrams/features_diagram_classification.json")
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classifications = data.get('classifications', {})
    
    # 1. 按複雜度和抽象層級組圖
    complexity_abstraction_map = analyze_complexity_abstraction_patterns(classifications)
    
    # 2. 按依賴關係組圖 
    dependency_patterns = analyze_dependency_patterns(classifications)
    
    # 3. 按命名模式組圖
    naming_patterns = analyze_naming_patterns(classifications)
    
    # 4. 按文件路徑模式組圖
    path_patterns = analyze_path_patterns(classifications)
    
    # 5. 按跨語言協作模式組圖
    cross_language_patterns = analyze_cross_language_patterns(classifications)
    
    # 6. 按功能聚類組圖
    functional_clusters = analyze_functional_clusters(classifications)
    
    # 7. 按架構角色組圖
    architectural_roles = analyze_architectural_roles(classifications)
    
    # 8. 按技術債務模式組圖
    technical_debt_patterns = analyze_technical_debt_patterns(classifications)
    
    return {
        "complexity_abstraction": complexity_abstraction_map,
        "dependency_patterns": dependency_patterns,
        "naming_patterns": naming_patterns,
        "path_patterns": path_patterns,
        "cross_language_patterns": cross_language_patterns,
        "functional_clusters": functional_clusters,
        "architectural_roles": architectural_roles,
        "technical_debt_patterns": technical_debt_patterns
    }

def analyze_complexity_abstraction_patterns(classifications):
    """分析複雜度與抽象層級的組合模式"""
    patterns = defaultdict(lambda: defaultdict(list))
    
    for name, info in classifications.items():
        complexity = info.get('complexity', 'unknown')
        abstraction = info.get('abstraction_level', 'unknown')
        language = info.get('language', 'unknown')
        priority = info.get('priority', 5)
        
        patterns[complexity][abstraction].append({
            'name': name,
            'language': language,
            'priority': priority,
            'file_path': info.get('file_path', '')
        })
    
    return dict(patterns)

def analyze_dependency_patterns(classifications):
    """分析依賴關係模式"""
    dependency_graph = defaultdict(list)
    isolated_components = []
    
    for name, info in classifications.items():
        deps = info.get('dependencies', [])
        cross_lang_deps = info.get('cross_language_dependencies')
        
        if not deps and not cross_lang_deps:
            isolated_components.append({
                'name': name,
                'language': info.get('language'),
                'category': info.get('category'),
                'complexity': info.get('complexity')
            })
        else:
            dependency_graph[name] = {
                'same_language_deps': deps,
                'cross_language_deps': cross_lang_deps,
                'info': info
            }
    
    return {
        'dependency_graph': dict(dependency_graph),
        'isolated_components': isolated_components
    }

def analyze_naming_patterns(classifications):
    """分析命名模式"""
    patterns = {
        'manager_pattern': [],
        'worker_pattern': [],  
        'config_pattern': [],
        'detector_pattern': [],
        'engine_pattern': [],
        'handler_pattern': [],
        'helper_pattern': [],
        'test_pattern': [],
        'schema_pattern': [],
        'model_pattern': [],
        'payload_pattern': [],
        'result_pattern': [],
        'factory_pattern': [],
        'builder_pattern': [],
        'adapter_pattern': [],
        'validator_pattern': [],
        'parser_pattern': [],
        'formatter_pattern': [],
        'executor_pattern': [],
        'controller_pattern': []
    }
    
    for name, info in classifications.items():
        lower_name = name.lower()
        
        # 檢查各種命名模式
        if 'manager' in lower_name:
            patterns['manager_pattern'].append((name, info))
        if 'worker' in lower_name:
            patterns['worker_pattern'].append((name, info))
        if 'config' in lower_name:
            patterns['config_pattern'].append((name, info))
        if 'detect' in lower_name:
            patterns['detector_pattern'].append((name, info))
        if 'engine' in lower_name:
            patterns['engine_pattern'].append((name, info))
        if 'handler' in lower_name or 'handle' in lower_name:
            patterns['handler_pattern'].append((name, info))
        if 'helper' in lower_name or 'util' in lower_name:
            patterns['helper_pattern'].append((name, info))
        if 'test' in lower_name:
            patterns['test_pattern'].append((name, info))
        if 'schema' in lower_name:
            patterns['schema_pattern'].append((name, info))
        if 'model' in lower_name:
            patterns['model_pattern'].append((name, info))
        if 'payload' in lower_name:
            patterns['payload_pattern'].append((name, info))
        if 'result' in lower_name:
            patterns['result_pattern'].append((name, info))
        if 'factory' in lower_name:
            patterns['factory_pattern'].append((name, info))
        if 'builder' in lower_name:
            patterns['builder_pattern'].append((name, info))
        if 'adapter' in lower_name:
            patterns['adapter_pattern'].append((name, info))
        if 'valid' in lower_name:
            patterns['validator_pattern'].append((name, info))
        if 'pars' in lower_name:
            patterns['parser_pattern'].append((name, info))
        if 'format' in lower_name:
            patterns['formatter_pattern'].append((name, info))
        if 'execut' in lower_name:
            patterns['executor_pattern'].append((name, info))
        if 'control' in lower_name:
            patterns['controller_pattern'].append((name, info))
    
    # 過濾空的模式
    return {k: v for k, v in patterns.items() if v}

def analyze_path_patterns(classifications):
    """分析文件路徑模式"""
    path_clusters = defaultdict(list)
    
    for name, info in classifications.items():
        file_path = info.get('file_path', '')
        if file_path:
            # 提取目錄結構
            path_parts = file_path.replace('\\', '/').split('/')
            
            # 分析不同層級的目錄模式
            if len(path_parts) >= 3:
                # services/features/xxx
                if len(path_parts) >= 4:
                    module_dir = path_parts[2]  # function_xxx
                    path_clusters[f"module_{module_dir}"].append((name, info))
                
                # 按功能目錄分類
                for i, part in enumerate(path_parts[2:], 2):
                    if part.startswith('function_'):
                        path_clusters[f"function_module_{part}"].append((name, info))
                    elif part in ['common', 'base']:
                        path_clusters[f"shared_module_{part}"].append((name, info))
    
    return dict(path_clusters)

def analyze_cross_language_patterns(classifications):
    """分析跨語言協作模式"""
    language_interfaces = defaultdict(list)
    shared_concepts = defaultdict(list)
    
    # 收集相同名稱但不同語言的組件
    name_language_map = defaultdict(list)
    
    for name, info in classifications.items():
        language = info.get('language', 'unknown')
        name_language_map[name].append((language, info))
    
    # 找出跨語言的相同概念
    for name, lang_infos in name_language_map.items():
        if len(lang_infos) > 1:
            shared_concepts[name] = lang_infos
    
    # 分析接口模式
    for name, info in classifications.items():
        if info.get('cross_language_dependencies'):
            language_interfaces[info.get('language', 'unknown')].append((name, info))
    
    return {
        'shared_concepts': dict(shared_concepts),
        'language_interfaces': dict(language_interfaces)
    }

def analyze_functional_clusters(classifications):
    """分析功能聚類模式"""
    clusters = {
        'authentication_cluster': [],
        'detection_cluster': [],
        'injection_cluster': [],
        'ssrf_cluster': [],
        'xss_cluster': [],
        'idor_cluster': [],
        'oauth_cluster': [],
        'jwt_cluster': [],
        'sast_cluster': [],
        'config_cluster': [],
        'schema_cluster': [],
        'worker_cluster': [],
        'telemetry_cluster': [],
        'statistics_cluster': [],
        'validation_cluster': [],
        'analysis_cluster': [],
        'bypass_cluster': [],
        'exploit_cluster': [],
        'payload_cluster': []
    }
    
    for name, info in classifications.items():
        lower_name = name.lower()
        file_path = info.get('file_path', '').lower()
        
        # 根據名稱和路徑分類到功能聚類
        if 'auth' in lower_name or 'auth' in file_path:
            clusters['authentication_cluster'].append((name, info))
        if 'detect' in lower_name or 'detect' in file_path:
            clusters['detection_cluster'].append((name, info))
        if 'injection' in lower_name or 'sqli' in lower_name or 'sql' in file_path:
            clusters['injection_cluster'].append((name, info))
        if 'ssrf' in lower_name or 'ssrf' in file_path:
            clusters['ssrf_cluster'].append((name, info))
        if 'xss' in lower_name or 'xss' in file_path:
            clusters['xss_cluster'].append((name, info))
        if 'idor' in lower_name or 'idor' in file_path:
            clusters['idor_cluster'].append((name, info))
        if 'oauth' in lower_name or 'oauth' in file_path:
            clusters['oauth_cluster'].append((name, info))
        if 'jwt' in lower_name or 'jwt' in file_path:
            clusters['jwt_cluster'].append((name, info))
        if 'sast' in lower_name or 'sast' in file_path:
            clusters['sast_cluster'].append((name, info))
        if 'config' in lower_name:
            clusters['config_cluster'].append((name, info))
        if 'schema' in lower_name:
            clusters['schema_cluster'].append((name, info))
        if 'worker' in lower_name:
            clusters['worker_cluster'].append((name, info))
        if 'telemetry' in lower_name or 'metric' in lower_name:
            clusters['telemetry_cluster'].append((name, info))
        if 'statistic' in lower_name:
            clusters['statistics_cluster'].append((name, info))
        if 'valid' in lower_name:
            clusters['validation_cluster'].append((name, info))
        if 'analys' in lower_name:
            clusters['analysis_cluster'].append((name, info))
        if 'bypass' in lower_name:
            clusters['bypass_cluster'].append((name, info))
        if 'exploit' in lower_name:
            clusters['exploit_cluster'].append((name, info))
        if 'payload' in lower_name:
            clusters['payload_cluster'].append((name, info))
    
    # 過濾空的聚類
    return {k: v for k, v in clusters.items() if v}

def analyze_architectural_roles(classifications):
    """分析架構角色模式"""
    roles = {
        'coordinators': [],      # 協調者 - Manager, Controller
        'processors': [],        # 處理者 - Worker, Engine, Processor  
        'validators': [],        # 驗證者 - Validator, Checker
        'adapters': [],         # 適配者 - Adapter, Converter
        'repositories': [],     # 存儲者 - Repository, Store
        'factories': [],        # 工廠 - Factory, Builder, Creator
        'observers': [],        # 觀察者 - Monitor, Tracker, Listener
        'strategies': [],       # 策略 - Strategy, Policy
        'utilities': [],        # 工具 - Utils, Helper, Tool
        'models': [],          # 模型 - Model, Schema, Entity
        'interfaces': [],      # 介面 - Interface, API, Contract
        'decorators': [],      # 裝飾者 - Decorator, Wrapper
        'singletons': [],      # 單例 - Singleton, Global
        'facades': []          # 門面 - Facade, Gateway
    }
    
    for name, info in classifications.items():
        lower_name = name.lower()
        abstraction = info.get('abstraction_level', '')
        
        # 根據命名和抽象層級判斷架構角色
        if any(x in lower_name for x in ['manager', 'coordinator', 'controller']):
            roles['coordinators'].append((name, info))
        elif any(x in lower_name for x in ['worker', 'engine', 'processor', 'executor']):
            roles['processors'].append((name, info))
        elif any(x in lower_name for x in ['valid', 'check', 'verify']):
            roles['validators'].append((name, info))
        elif any(x in lower_name for x in ['adapter', 'convert', 'transform']):
            roles['adapters'].append((name, info))
        elif any(x in lower_name for x in ['repository', 'store', 'cache']):
            roles['repositories'].append((name, info))
        elif any(x in lower_name for x in ['factory', 'builder', 'creator']):
            roles['factories'].append((name, info))
        elif any(x in lower_name for x in ['monitor', 'track', 'listen', 'observer']):
            roles['observers'].append((name, info))
        elif any(x in lower_name for x in ['strategy', 'policy']):
            roles['strategies'].append((name, info))
        elif any(x in lower_name for x in ['util', 'helper', 'tool']):
            roles['utilities'].append((name, info))
        elif any(x in lower_name for x in ['model', 'schema', 'entity']) or abstraction == 'component':
            roles['models'].append((name, info))
        elif any(x in lower_name for x in ['interface', 'api', 'contract']):
            roles['interfaces'].append((name, info))
        elif any(x in lower_name for x in ['decorator', 'wrapper']):
            roles['decorators'].append((name, info))
        elif any(x in lower_name for x in ['singleton', 'global', 'instance']):
            roles['singletons'].append((name, info))
        elif any(x in lower_name for x in ['facade', 'gateway']):
            roles['facades'].append((name, info))
    
    return {k: v for k, v in roles.items() if v}

def analyze_technical_debt_patterns(classifications):
    """分析技術債務模式"""
    debt_patterns = {
        'duplicate_implementations': [],
        'inconsistent_naming': [],
        'missing_abstractions': [],
        'god_objects': [],
        'scattered_responsibilities': [],
        'language_inconsistencies': []
    }
    
    # 檢測重複實現
    name_groups = defaultdict(list)
    for name, info in classifications.items():
        # 提取核心名稱（去除前綴後綴）
        core_name = re.sub(r'^(get_|set_|create_|build_)', '', name.lower())
        core_name = re.sub(r'(worker|manager|config|engine)$', '', core_name)
        name_groups[core_name].append((name, info))
    
    for core_name, items in name_groups.items():
        if len(items) > 1 and core_name.strip():
            # 檢查是否為真正的重複實現
            languages = set(item[1].get('language') for item in items)
            categories = set(item[1].get('category') for item in items)
            
            if len(languages) > 1 or len(categories) > 1:
                debt_patterns['duplicate_implementations'].append({
                    'core_concept': core_name,
                    'implementations': items,
                    'languages': list(languages),
                    'categories': list(categories)
                })
    
    # 檢測命名不一致
    naming_styles = defaultdict(list)
    for name, info in classifications.items():
        if '_' in name:
            naming_styles['snake_case'].append((name, info))
        elif any(c.isupper() for c in name[1:]):
            naming_styles['camelCase'].append((name, info))
        else:
            naming_styles['lowercase'].append((name, info))
    
    if len(naming_styles) > 1:
        debt_patterns['inconsistent_naming'] = dict(naming_styles)
    
    # 檢測缺失的抽象
    function_groups = defaultdict(list)
    for name, info in classifications.items():
        if info.get('abstraction_level') == 'function':
            category = info.get('category', 'unknown')
            function_groups[category].append((name, info))
    
    for category, functions in function_groups.items():
        if len(functions) > 10:  # 如果某個類別有太多函數級組件
            debt_patterns['missing_abstractions'].append({
                'category': category,
                'function_count': len(functions),
                'functions': functions[:5]  # 只顯示前5個作為範例
            })
    
    # 檢測上帝物件
    for name, info in classifications.items():
        if (info.get('complexity') == 'high' and 
            info.get('abstraction_level') == 'service' and
            'manager' in name.lower()):
            debt_patterns['god_objects'].append((name, info))
    
    return debt_patterns

def generate_advanced_analysis_report(analysis_results):
    """生成深度分析報告"""
    
    report = """# AIVA Features 深度架構分析報告

## 🔍 **發現的隱藏組織能力**

### 1. 複雜度與抽象層級矩陣分析

"""
    
    # 複雜度抽象層級分析
    complexity_data = analysis_results['complexity_abstraction']
    for complexity, abstractions in complexity_data.items():
        report += f"#### **{complexity.upper()} 複雜度組件**\n"
        for abstraction, components in abstractions.items():
            if components:
                report += f"- **{abstraction}** 層級: {len(components)} 個組件\n"
                
                # 按語言統計
                lang_count = Counter(comp['language'] for comp in components)
                lang_stats = ', '.join(f"{lang}: {count}" for lang, count in lang_count.items())
                report += f"  - 語言分佈: {lang_stats}\n"
                
                # 高優先級組件
                high_priority = [comp for comp in components if comp['priority'] <= 2]
                if high_priority:
                    report += f"  - 高優先級組件: {', '.join(comp['name'] for comp in high_priority[:3])}\n"
        report += "\n"
    
    report += """### 2. 功能聚類分析

"""
    
    # 功能聚類分析
    clusters = analysis_results['functional_clusters']
    for cluster_name, components in clusters.items():
        if len(components) >= 3:  # 只報告有意義的聚類
            report += f"#### **{cluster_name.replace('_', ' ').title()}**\n"
            report += f"- 組件數量: {len(components)}\n"
            
            # 語言分佈
            languages = Counter(comp[1].get('language') for comp in components)
            report += f"- 主要語言: {', '.join(f'{lang}({count})' for lang, count in languages.most_common(3))}\n"
            
            # 複雜度分佈
            complexities = Counter(comp[1].get('complexity') for comp in components)
            report += f"- 複雜度分佈: {', '.join(f'{comp}({count})' for comp, count in complexities.items())}\n"
            
            # 核心組件
            high_priority_components = [comp for comp in components if comp[1].get('priority', 5) <= 2]
            if high_priority_components:
                report += f"- 核心組件: {', '.join(comp[0] for comp in high_priority_components[:3])}\n"
            
            report += "\n"
    
    report += """### 3. 架構角色模式分析

"""
    
    # 架構角色分析
    roles = analysis_results['architectural_roles']
    for role_name, components in roles.items():
        if components:
            report += f"#### **{role_name.replace('_', ' ').title()}** ({len(components)} 組件)\n"
            
            # 語言偏好
            languages = Counter(comp[1].get('language') for comp in components)
            dominant_lang = languages.most_common(1)[0] if languages else ('unknown', 0)
            report += f"- 主導語言: {dominant_lang[0]} ({dominant_lang[1]}/{len(components)})\n"
            
            # 示例組件
            examples = [comp[0] for comp in components[:3]]
            report += f"- 典型組件: {', '.join(examples)}\n\n"
    
    report += """### 4. 技術債務分析

"""
    
    # 技術債務分析
    debt = analysis_results['technical_debt_patterns']
    
    if debt.get('duplicate_implementations'):
        report += "#### **🚨 重複實現問題**\n"
        for dup in debt['duplicate_implementations'][:5]:  # 只顯示前5個
            report += f"- **{dup['core_concept']}**: {len(dup['implementations'])} 個實現\n"
            report += f"  - 涉及語言: {', '.join(dup['languages'])}\n"
            report += f"  - 跨層級: {', '.join(dup['categories'])}\n"
        report += "\n"
    
    if debt.get('inconsistent_naming'):
        report += "#### **📝 命名風格不一致**\n"
        for style, components in debt['inconsistent_naming'].items():
            report += f"- **{style}**: {len(components)} 個組件\n"
        report += "\n"
    
    if debt.get('missing_abstractions'):
        report += "#### **🏗️ 缺失抽象層**\n"
        for missing in debt['missing_abstractions']:
            report += f"- **{missing['category']}** 類別: {missing['function_count']} 個函數級組件，需要抽象化\n"
        report += "\n"
    
    if debt.get('god_objects'):
        report += "#### **👹 上帝物件**\n"
        for god_obj, info in debt['god_objects']:
            report += f"- **{god_obj}**: 高複雜度服務級組件，建議拆分\n"
        report += "\n"
    
    report += """### 5. 跨語言協作模式

"""
    
    # 跨語言分析
    cross_lang = analysis_results['cross_language_patterns']
    
    if cross_lang.get('shared_concepts'):
        report += "#### **🔗 共享概念**\n"
        for concept, implementations in cross_lang['shared_concepts'].items():
            if len(implementations) > 1:
                languages = [impl[0] for impl in implementations]
                report += f"- **{concept}**: 在 {', '.join(languages)} 中都有實現\n"
        report += "\n"
    
    report += """### 6. 命名模式統計

"""
    
    # 命名模式分析
    naming = analysis_results['naming_patterns']
    pattern_stats = {k: len(v) for k, v in naming.items()}
    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True)
    
    for pattern, count in sorted_patterns[:10]:  # 顯示前10個最常見的模式
        if count > 0:
            report += f"- **{pattern.replace('_', ' ').title()}**: {count} 個組件\n"
    
    report += """

## 💡 **新發現的組織建議**

### 🎯 **按技術棧重新組織**
1. **前端安全棧**: JavaScript 分析、XSS 檢測、客戶端繞過
2. **後端安全棧**: SQL 注入、SSRF、IDOR 檢測  
3. **身份驗證棧**: JWT、OAuth、認證繞過
4. **基礎設施棧**: Worker、配置、統計、Schema

### 🔄 **按生命週期組織**
1. **檢測階段**: 各種 Detector 和 Engine
2. **分析階段**: 各種 Analyzer 和 Parser
3. **報告階段**: 各種 Reporter 和 Formatter
4. **管理階段**: 各種 Manager 和 Controller

### 📊 **按數據流組織**
1. **輸入處理**: Parser、Validator、Converter
2. **核心處理**: Engine、Processor、Detector
3. **結果處理**: Formatter、Reporter、Exporter
4. **狀態管理**: Statistics、Telemetry、Monitor

### 🎨 **按設計模式組織**
1. **創建模式**: Factory、Builder、Singleton
2. **結構模式**: Adapter、Decorator、Facade  
3. **行為模式**: Strategy、Observer、Command
4. **併發模式**: Worker、Queue、Pool

---

**📊 分析統計**:
- 發現 **{total_components}** 個組件
- 識別 **{total_patterns}** 種架構模式
- 檢測 **{debt_issues}** 個技術債務問題
- 建議 **4** 種新的組織方式

*這份深度分析揭示了 AIVA Features 模組的隱藏組織潛力和架構優化機會。*
""".format(
        total_components=sum(len(v) if isinstance(v, list) else sum(len(vv) for vv in v.values()) 
                           for v in analysis_results.values() if v),
        total_patterns=len([k for k, v in analysis_results.items() if v]),
        debt_issues=len(debt.get('duplicate_implementations', [])) + 
                   len(debt.get('missing_abstractions', [])) + 
                   len(debt.get('god_objects', []))
    )
    
    return report

if __name__ == "__main__":
    print("🔍 開始深度架構分析...")
    
    analysis_results = analyze_advanced_patterns()
    
    print("📊 生成分析報告...")
    report = generate_advanced_analysis_report(analysis_results)
    
    # 保存報告
    output_file = Path("services/features/ADVANCED_ARCHITECTURE_ANALYSIS_REPORT.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 深度分析完成！報告已保存到: {output_file}")
    
    # 保存詳細分析數據
    analysis_data_file = Path("_out/architecture_diagrams/advanced_analysis_data.json")
    with open(analysis_data_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"📋 詳細數據已保存到: {analysis_data_file}")