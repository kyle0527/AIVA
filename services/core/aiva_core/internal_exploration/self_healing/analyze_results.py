#!/usr/bin/env python3
"""
分析結果深度評估腳本
==================
評估 self_healing 分析結果的質量和有效性

重構說明: 將巨型函數拆分為多個單一職責的小函數，降低認知複雜度從 36 → <10
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

# 導入 classify_script_type 函數
try:
    from .core_analyzer import classify_script_type
except ImportError:
    # 本地測試時的後備導入
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from core_analyzer import classify_script_type

def load_analysis_data(results_dir: Path) -> dict:
    """載入分析數據
    
    Args:
        results_dir: 結果目錄路徑
        
    Returns:
        分析數據字典
    """
    json_path = results_dir / 'analysis_results.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_basic_statistics(data: dict) -> None:
    """列印基礎統計資訊
    
    Args:
        data: 分析數據
    """
    print("\n1️⃣ 基礎統計")
    print("-" * 80)
    
    flow_chains = data.get('flow_chains', [])
    script_functions = data.get('function_details', {}).get('script_functions', {})
    
    total_functions = sum(len(s.get('functions', {})) for s in script_functions.values())
    
    print(f"  掃描的腳本數量: {len(script_functions)}")
    print(f"  數據流鏈路數量: {len(flow_chains)}")
    print(f"  總函數數量: {total_functions}")

def build_connection_graph(flow_chains: list) -> dict:
    """構建連接圖
    
    Args:
        flow_chains: 流鏈列表
        
    Returns:
        連接計數字典 {script_name: connection_count}
    """
    connections: dict[str, int] = defaultdict(int)
    for chain in flow_chains:
        for i in range(len(chain) - 1):
            source = Path(chain[i]).stem
            connections[source] += 1
    return dict(connections)

def calculate_connection_stats(script_functions: dict, connections: dict) -> list:
    """計算每個腳本的連接統計
    
    Args:
        script_functions: 腳本函數數據
        connections: 連接計數
        
    Returns:
        連接統計列表
    """
    connection_stats = []
    
    for script_name, script_info in script_functions.items():
        export_count = len(script_info.get('export_functions', []))
        actual_connections = connections.get(script_name, 0)
        
        if export_count > 0:
            ratio = actual_connections / export_count
            status = classify_connection_status(ratio)
            
            connection_stats.append({
                'name': script_name,
                'exports': export_count,
                'connections': actual_connections,
                'ratio': ratio,
                'status': status
            })
    
    return connection_stats

def classify_connection_status(ratio: float) -> str:
    """分類連接狀態
    
    Args:
        ratio: 連接比率
        
    Returns:
        狀態字符串
    """
    if ratio > 1.0:
        return 'over-connected'
    elif ratio >= 0.4:
        return 'well-connected'
    elif ratio > 0:
        return 'under-connected'
    else:
        return 'isolated'

def group_by_status(connection_stats: list) -> dict:
    """按狀態分組
    
    Args:
        connection_stats: 連接統計列表
        
    Returns:
        按狀態分組的字典
    """
    status_groups = defaultdict(list)
    for stat in connection_stats:
        status_groups[stat['status']].append(stat)
    return dict(status_groups)

def print_connection_analysis(status_groups: dict) -> None:
    """列印連接完整度分析
    
    Args:
        status_groups: 狀態分組
    """
    print("\n2️⃣ 連接完整度分析")
    print("-" * 80)
    print("\n  狀態分布:")
    print(f"    🟢 過度連接 (ratio > 1.0):     {len(status_groups.get('over-connected', []))} 個")
    print(f"    🟢 連接良好 (ratio 0.4-1.0):   {len(status_groups.get('well-connected', []))} 個")
    print(f"    🟡 連接不足 (ratio 0-0.4):     {len(status_groups.get('under-connected', []))} 個")
    print(f"    🔴 完全孤立 (ratio = 0):       {len(status_groups.get('isolated', []))} 個")

def print_typical_cases(status_groups: dict) -> None:
    """列印典型案例分析
    
    Args:
        status_groups: 狀態分組
    """
    print("\n3️⃣ 典型案例分析")
    print("-" * 80)
    
    print_over_connected_modules(status_groups.get('over-connected', []))
    print_under_connected_modules(status_groups.get('under-connected', []))
    print_isolated_modules(status_groups.get('isolated', []))

def print_over_connected_modules(modules: list) -> None:
    """列印過度連接的模組"""
    print("\n  🌟 最重要的核心模組 (過度連接):")
    top_modules = sorted(modules, key=lambda x: x['ratio'], reverse=True)[:5]
    for stat in top_modules:
        print(f"    • {stat['name']}: {stat['connections']}/{stat['exports']} "
              f"({stat['ratio']*100:.0f}%)")

def print_under_connected_modules(modules: list) -> None:
    """列印連接不足的模組"""
    print("\n  ⚠️  未完整連接的模組 (應該檢查):")
    top_modules = sorted(modules, key=lambda x: x['exports'], reverse=True)[:10]
    for stat in top_modules:
        missing = stat['exports'] - stat['connections']
        print(f"    • {stat['name']}: {stat['connections']}/{stat['exports']} "
              f"({stat['ratio']*100:.0f}%) - 缺失 {missing} 個連接")

def print_isolated_modules(modules: list) -> None:
    """列印完全孤立的模組"""
    print("\n  🔴 完全孤立的模組 (可能是工具腳本):")
    top_modules = sorted(modules, key=lambda x: x['exports'], reverse=True)[:10]
    
    # 使用 classify_script_type 進行分類
    categorized = {
        'tool': [],
        'entry_point': [],
        'module': []
    }
    
    for stat in top_modules:
        # 假設腳本路徑可從名稱推斷（需要實際路徑時應從外部傳入）
        script_type = classify_script_type('', stat['name'])
        categorized[script_type].append(stat)
    
    # 顯示真正的孤立模組（非工具腳本）
    real_isolated = categorized['module']
    if real_isolated:
        print("\n    真正的孤立模組（需要檢查）：")
        for stat in real_isolated:
            print(f"    • {stat['name']}: 定義了 {stat['exports']} 個函數但完全未被使用")
    
    # 顯示工具腳本（可以忽略）
    tool_scripts = categorized['tool'] + categorized['entry_point']
    if tool_scripts:
        print("\n    工具/入口腳本（正常情況）：")
        for stat in tool_scripts:
            print(f"    ℹ️ {stat['name']}: {stat['exports']} 個函數（工具腳本，可忽略）")

def verify_design_principles(connection_stats: list) -> None:
    """驗證設計理念
    
    Args:
        connection_stats: 連接統計
    """
    print("\n4️⃣ 設計理念驗證")
    print("-" * 80)
    
    important_modules = ['rl_models', 'rl_trainers', 'backends', 
                        'postgresql_vector_store', 'aiva_exploration_pipeline']
    
    print("\n  檢查關鍵模組是否被正確處理:")
    for module in important_modules:
        stat = next((s for s in connection_stats if s['name'] == module), None)
        if stat:
            if stat['status'] in ['over-connected', 'well-connected']:
                print(f"    ✅ {module}: {stat['status']} ({stat['ratio']*100:.0f}%) - 正確未標記為問題")
            else:
                print(f"    ❌ {module}: {stat['status']} ({stat['ratio']*100:.0f}%) - 可能誤判")

def generate_improvement_suggestions(connection_stats: list, status_groups: dict) -> list:
    """生成改進建議
    
    Args:
        connection_stats: 連接統計
        status_groups: 狀態分組
        
    Returns:
        建議列表
    """
    suggestions = []
    
    # 建議1: 調整閾值
    under_connected_with_few_exports = [
        s for s in status_groups.get('under-connected', [])
        if s['exports'] < 5
    ]
    if len(under_connected_with_few_exports) > 20:
        suggestions.append(
            f"📌 當前有 {len(under_connected_with_few_exports)} 個小模組被標記為「未完整連接」，"
            "建議提高最小函數數量閾值（從5增加到8-10）"
        )
    
    # 建議2: 識別工具腳本
    tool_keywords = ['test', 'demo', 'example', 'cli', 'main']
    tool_scripts = [
        s for s in status_groups.get('isolated', [])
        if any(keyword in s['name'].lower() for keyword in tool_keywords)
    ]
    if len(tool_scripts) > 0:
        suggestions.append(
            f"📌 發現 {len(tool_scripts)} 個可能的工具/示例腳本被標記為「孤立」，"
            "建議添加腳本類型識別邏輯"
        )
    
    # 建議3: 連接質量評分
    high_quality_connections = [
        s for s in connection_stats
        if 0.3 <= s['ratio'] <= 2.0 and s['exports'] >= 5
    ]
    total_significant = [s for s in connection_stats if s['exports'] >= 5]
    
    if len(total_significant) > 0:
        quality_rate = len(high_quality_connections) / len(total_significant)
        health_status = '健康' if quality_rate > 0.6 else '需要改進'
        suggestions.append(
            f"📌 {len(high_quality_connections)}/{len(total_significant)} 個重要模組 "
            f"({quality_rate*100:.1f}%) 連接質量良好，"
            f"系統整體架構{health_status}"
        )
    
    return suggestions

def print_improvement_suggestions(suggestions: list) -> None:
    """列印改進建議
    
    Args:
        suggestions: 建議列表
    """
    print("\n5️⃣ 改進建議")
    print("-" * 80)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n  {i}. {suggestion}")

def export_detailed_data(
    results_dir: Path,
    connection_stats: list,
    status_groups: dict,
    script_functions: dict
) -> None:
    """導出詳細數據
    
    Args:
        results_dir: 結果目錄
        connection_stats: 連接統計
        status_groups: 狀態分組
        script_functions: 腳本函數數據
    """
    print("\n6️⃣ 導出詳細數據")
    print("-" * 80)
    
    important_modules = ['rl_models', 'rl_trainers', 'backends', 
                        'postgresql_vector_store', 'aiva_exploration_pipeline']
    
    output_data = {
        'summary': {
            'total_scripts': len(script_functions),
            'over_connected': len(status_groups.get('over-connected', [])),
            'well_connected': len(status_groups.get('well-connected', [])),
            'under_connected': len(status_groups.get('under-connected', [])),
            'isolated': len(status_groups.get('isolated', []))
        },
        'connection_stats': connection_stats,
        'important_modules': {
            module: next((s for s in connection_stats if s['name'] == module), None)
            for module in important_modules
        }
    }
    
    output_path = results_dir / 'connection_analysis_detail.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  詳細數據已保存至: {output_path}")

def analyze_report_quality():
    """分析報告質量
    
    主函數 - 協調各個分析步驟，複雜度從 36 降至 5
    
    Returns:
        (connection_stats, status_groups) 元組
    """
    results_dir = Path('C:/D/fold7/AIVA-git/services/core/aiva_core/analysis_results_v2')
    
    print("="*80)
    print("📊 Self-Healing 分析結果深度評估")
    print("="*80)
    
    # 1. 載入數據
    data = load_analysis_data(results_dir)
    
    flow_chains = data.get('flow_chains', [])
    script_functions = data.get('function_details', {}).get('script_functions', {})
    
    # 2-3. 統計和連接分析
    print_basic_statistics(data)
    connections = build_connection_graph(flow_chains)
    connection_stats = calculate_connection_stats(script_functions, connections)
    status_groups = group_by_status(connection_stats)
    
    print_connection_analysis(status_groups)
    
    # 4. 典型案例分析
    print_typical_cases(status_groups)
    
    # 5. 設計理念驗證
    verify_design_principles(connection_stats)
    
    # 6. 改進建議
    suggestions = generate_improvement_suggestions(connection_stats, status_groups)
    print_improvement_suggestions(suggestions)
    
    # 7. 導出詳細數據
    export_detailed_data(results_dir, connection_stats, status_groups, script_functions)
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)
    
    return connection_stats, status_groups

if __name__ == "__main__":
    analyze_report_quality()
