#!/usr/bin/env python3
"""
提取並分析 aiva_core 的 29 個內部可執行指令
"""
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 29 個內部可執行腳本
INTERNAL_COMMANDS = [
    "services/core/aiva_core/cognitive_core/ai_capability_query.py",
    "services/core/aiva_core/cognitive_core/anti_hallucination/anti_hallucination_module.py",
    "services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py",
    "services/core/aiva_core/cognitive_core/internal_loop_connector.py",
    "services/core/aiva_core/cognitive_core/rag/postgresql_vector_store.py",
    "services/core/aiva_core/cognitive_core/test_scope_management.py",
    "services/core/aiva_core/core_capabilities/attack/bizlogic_attack_executor.py",
    "services/core/aiva_core/core_capabilities/capability_registry.py",
    "services/core/aiva_core/external_learning/ai_model/train_classifier.py",
    "services/core/aiva_core/external_learning/event_listener.py",
    "services/core/aiva_core/external_learning/experience_manager.py",
    "services/core/aiva_core/internal_exploration/aiva_exploration_pipeline.py",
    "services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py",
    "services/core/aiva_core/internal_exploration/python_tools/aiva_exploration_pipeline.py",
    "services/core/aiva_core/internal_exploration/python_tools/aiva_flow_analyzer.py",
    "services/core/aiva_core/internal_exploration/python_tools/aiva_flow_classifier.py",
    "services/core/aiva_core/internal_exploration/self_healing/analyze_connection_recommendations.py",
    "services/core/aiva_core/internal_exploration/self_healing/analyze_dataflow_breakpoints.py",
    "services/core/aiva_core/internal_exploration/self_healing/analyze_missing_function_connections.py",
    "services/core/aiva_core/internal_exploration/self_healing/analyze_results.py",
    "services/core/aiva_core/internal_exploration/self_healing/core_analyzer.py",
    "services/core/aiva_core/internal_exploration/self_healing/practical_analyzer.py",
    "services/core/aiva_core/internal_exploration/self_healing/run_analysis.py",
    "services/core/aiva_core/service_backbone/api/unified_function_caller.py",
    "services/core/aiva_core/service_backbone/authz/authz_mapper.py",
    "services/core/aiva_core/service_backbone/authz/matrix_visualizer.py",
    "services/core/aiva_core/service_backbone/authz/permission_matrix.py",
    "services/core/aiva_core/service_backbone/coordination/ai_controller.py",
    "services/core/aiva_core/service_backbone/coordination/optimized_core.py",
    "services/core/aiva_core/service_backbone/storage/examples/cli_integration_example.py",
]

def load_capabilities():
    """加載 aiva_core 能力數據"""
    data_file = Path("data/analysis_results/capabilities_aiva_core_classified_20251215_074111.json")
    if not data_file.exists():
        print(f"❌ 找不到數據文件: {data_file}")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_internal_commands(capabilities: List[Dict]) -> List[Dict]:
    """提取內部指令的能力數據"""
    internal_cmds = []
    found_files = set()  # 追蹤已找到的文件，避免重複
    
    for cap in capabilities:
        file_path = cap.get('file_path', '')
        # 標準化路徑（移除盤符和絕對路徑前綴）
        normalized = file_path.replace('\\', '/').replace('C:/D/fold7/AIVA-git/', '')
        
        # 檢查是否在內部指令列表中（精確匹配文件路徑）
        for cmd_path in INTERNAL_COMMANDS:
            if normalized == cmd_path:
                # 只在文件第一次出現時標記（避免一個文件的多個能力都被標記）
                if normalized not in found_files:
                    found_files.add(normalized)
                    # 標記為內部指令
                    cap['is_internal_command'] = True
                    cap['command_type'] = categorize_command(cmd_path)
                    cap['file_normalized'] = normalized
                    internal_cmds.append(cap)
                    break
    
    return internal_cmds

def categorize_command(file_path: str) -> str:
    """分類指令類型"""
    if 'test_' in file_path or 'example' in file_path:
        return 'testing'
    elif 'analyze' in file_path or 'analyzer' in file_path or 'classifier' in file_path:
        return 'tool'
    elif 'manager' in file_path or 'controller' in file_path or 'registry' in file_path:
        return 'management'
    else:
        return 'other'

def analyze_internal_commands(commands: List[Dict]) -> Dict:
    """分析內部指令"""
    stats = {
        'total': len(commands),
        'by_type': {},
        'by_module': {},
        'by_scope': {},
        'by_visibility': {},
        'missing': []
    }
    
    for cmd in commands:
        # 按類型統計
        cmd_type = cmd.get('command_type', 'unknown')
        stats['by_type'][cmd_type] = stats['by_type'].get(cmd_type, 0) + 1
        
        # 按模組統計
        # 從路徑中提取模組名
        file_path = cmd.get('file_normalized', cmd.get('file_path', ''))
        if 'cognitive_core' in file_path:
            sub_module = 'cognitive_core'
        elif 'core_capabilities' in file_path:
            sub_module = 'core_capabilities'
        elif 'external_learning' in file_path:
            sub_module = 'external_learning'
        elif 'internal_exploration' in file_path:
            sub_module = 'internal_exploration'
        elif 'service_backbone' in file_path:
            sub_module = 'service_backbone'
        else:
            sub_module = 'unknown'
        
        stats['by_module'][sub_module] = stats['by_module'].get(sub_module, 0) + 1
        
        # 按範圍統計
        scope = cmd.get('scope', 'unknown')
        stats['by_scope'][scope] = stats['by_scope'].get(scope, 0) + 1
        
        # 按可見性統計
        visibility = cmd.get('visibility', 'unknown')
        stats['by_visibility'][visibility] = stats['by_visibility'].get(visibility, 0) + 1
    
    # 檢查缺失的指令
    found_paths = [cmd.get('file_normalized', '') for cmd in commands]
    for cmd_path in INTERNAL_COMMANDS:
        if cmd_path not in found_paths:
            stats['missing'].append(cmd_path)
    
    return stats

def main():
    print("=" * 60)
    print("AIVA Core 內部指令提取與分析")
    print("=" * 60)
    
    # 1. 加載能力數據
    print("\n📂 加載能力數據...")
    capabilities = load_capabilities()
    if not capabilities:
        return
    
    print(f"✅ 已加載 {len(capabilities)} 個能力")
    
    # 2. 提取內部指令
    print("\n🔍 提取內部指令...")
    internal_commands = extract_internal_commands(capabilities)
    print(f"✅ 找到 {len(internal_commands)} 個內部指令")
    
    # 3. 分析統計
    print("\n📊 分析內部指令...")
    stats = analyze_internal_commands(internal_commands)
    
    print(f"\n總數: {stats['total']}")
    print(f"預期: 29")
    print(f"缺失: {len(stats['missing'])}")
    
    print("\n按類型分佈:")
    for cmd_type, count in sorted(stats['by_type'].items()):
        print(f"  {cmd_type}: {count}")
    
    print("\n按模組分佈:")
    for module, count in sorted(stats['by_module'].items()):
        print(f"  {module}: {count}")
    
    print("\n按範圍分佈:")
    for scope, count in sorted(stats['by_scope'].items()):
        print(f"  {scope}: {count}")
    
    print("\n按可見性分佈:")
    for visibility, count in sorted(stats['by_visibility'].items()):
        print(f"  {visibility}: {count}")
    
    if stats['missing']:
        print(f"\n⚠️ 缺失的指令 ({len(stats['missing'])}):")
        for missing in stats['missing']:
            print(f"  - {missing}")
    
    # 4. 保存結果
    output_file = Path(f"data/analysis_results/internal_commands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'total_capabilities': len(capabilities),
        'internal_commands': internal_commands,
        'statistics': stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 結果已保存: {output_file}")
    
    # 5. 生成報告
    report_file = output_file.with_suffix('.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# AIVA Core 內部指令分析報告\n\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 統計摘要\n\n")
        f.write(f"- 總能力數: {len(capabilities)}\n")
        f.write(f"- 內部指令: {stats['total']}\n")
        f.write(f"- 預期數量: 29\n")
        f.write(f"- 缺失數量: {len(stats['missing'])}\n\n")
        
        f.write("## 分類統計\n\n")
        f.write("### 按類型\n\n")
        for cmd_type, count in sorted(stats['by_type'].items()):
            f.write(f"- {cmd_type}: {count}\n")
        
        f.write("\n### 按模組\n\n")
        for module, count in sorted(stats['by_module'].items()):
            f.write(f"- {module}: {count}\n")
        
        f.write("\n### 按範圍\n\n")
        for scope, count in sorted(stats['by_scope'].items()):
            f.write(f"- {scope}: {count}\n")
        
        f.write("\n### 按可見性\n\n")
        for visibility, count in sorted(stats['by_visibility'].items()):
            f.write(f"- {visibility}: {count}\n")
        
        if stats['missing']:
            f.write(f"\n## 缺失的指令\n\n")
            for missing in stats['missing']:
                f.write(f"- {missing}\n")
        
        f.write("\n## 內部指令清單\n\n")
        for cmd in sorted(internal_commands, key=lambda x: x.get('sub_module', '')):
            f.write(f"### {cmd.get('name', 'Unknown')}\n\n")
            f.write(f"- 文件: `{cmd.get('file_path', 'Unknown')}`\n")
            f.write(f"- 模組: {cmd.get('sub_module', 'Unknown')}\n")
            f.write(f"- 類型: {cmd.get('command_type', 'Unknown')}\n")
            f.write(f"- 範圍: {cmd.get('scope', 'Unknown')}\n")
            f.write(f"- 可見性: {cmd.get('visibility', 'Unknown')}\n")
            if cmd.get('capabilities'):
                f.write(f"- 能力數: {len(cmd['capabilities'])}\n")
            f.write("\n")
    
    print(f"📄 報告已生成: {report_file}")
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()
