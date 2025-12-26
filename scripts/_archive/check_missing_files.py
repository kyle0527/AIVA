#!/usr/bin/env python3
"""檢查缺失的內部指令文件"""
import json
from pathlib import Path

# 加載數據
data = json.load(open('data/analysis_results/capabilities_aiva_core_classified_20251215_074111.json', encoding='utf-8'))

# 提取所有唯一文件
files_in_data = set([
    cap['file_path'].replace('\\', '/').replace('C:/D/fold7/AIVA-git/', '') 
    for cap in data
])

print(f"📊 能力數據中總共有 {len(files_in_data)} 個唯一文件")
print(f"📊 總能力數: {len(data)}")

# 29 個包含 if __name__ == "__main__" 的文件
target_files = [
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

print(f"\n🔍 檢查 29 個可執行文件:\n")

found = []
missing = []

for f in target_files:
    if f in files_in_data:
        found.append(f)
        # 計算該文件有多少能力
        cap_count = sum(1 for cap in data if f in cap['file_path'].replace('\\', '/'))
        print(f"  ✅ {f} ({cap_count} 個能力)")
    else:
        missing.append(f)
        print(f"  ❌ {f}")

print(f"\n📊 統計:")
print(f"  找到: {len(found)}")
print(f"  缺失: {len(missing)}")

if missing:
    print(f"\n⚠️ 這 {len(missing)} 個文件可能:")
    print("  1. 只有 if __name__ == '__main__' 入口代碼，沒有函數/類")
    print("  2. 掃描時被跳過")
    print("  3. 路徑格式不匹配")
    
    print(f"\n❓ 建議檢查這些文件的實際內容")
