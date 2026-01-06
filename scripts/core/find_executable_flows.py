"""
尋找可以無參數實例化的類別進行測試
"""
import json
import sys
import os

# 設定 PYTHONPATH
sys.path.insert(0, r'C:\D\fold7\AIVA-git\services')
sys.path.insert(0, r'C:\D\fold7\AIVA-git\services\core')

with open('C:/Users/User/Downloads/data/internal_exploration/latest_classification.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 測試可實例化的模組
testable_modules = [
    ('aiva_core.cognitive_core.ai_capability_query', 'AICapabilityQuery'),
    ('aiva_core.cognitive_core.internal_loop_connector', 'InternalLoopConnector'),
    ('aiva_core.cognitive_core.capability_orchestrator', 'CapabilityOrchestrator'),
    ('aiva_core.core_capabilities.capability_registry', 'CapabilityRegistry'),
    ('aiva_core.external_learning.experience_manager', 'ExperienceManager'),
    ('aiva_core.task_planning.planner.task_converter', 'TaskConverter'),
    ('aiva_core.cognitive_core.rag.vector_store', 'VectorStore'),
    ('aiva_core.cognitive_core.decision.skill_graph', 'AIVASkillGraph'),
    ('aiva_core.service_backbone.api.enhanced_unified_caller', 'EnhancedUnifiedFunctionCaller'),
    ('aiva_core.task_planning.command_builder', 'CommandBuilder'),
]

print("=== 測試可無參數實例化的類別 ===\n")
successful = []

for module_path, class_name in testable_modules:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls()
        print(f"✅ {class_name}: 成功實例化")
        successful.append((module_path, class_name))
    except TypeError as e:
        if 'missing' in str(e) and 'argument' in str(e):
            print(f"❌ {class_name}: 需要參數 - {e}")
        else:
            print(f"❌ {class_name}: TypeError - {e}")
    except Exception as e:
        print(f"❌ {class_name}: {type(e).__name__} - {e}")

print(f"\n成功: {len(successful)}/{len(testable_modules)}")

# 找出包含這些類別的 Flow
print("\n=== 可執行的 Flow ===")
for module_path, class_name in successful:
    short_name = module_path.split('.')[-1]
    matching_flows = [f for f in data['flows'] if short_name in f['path']]
    if matching_flows:
        print(f"\n{class_name} ({short_name}):")
        for flow in matching_flows[:3]:
            print(f"  Flow {flow['id']}: {' -> '.join(flow['path'])}")
