import json

def analyze_classification_comparison():
    with open('classification_data_v8_endpoint.json/classification_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('=== Flow 模式驗證 ===')
    print()

    # 檢查一些有趣的例子
    examples = {
        'service_backbone終點': [],
        'cognitive_core終點': [],
        'task_planning終點': []
    }

    for flow in data['flows']:
        primary = flow['primary_module']
        majority = flow.get('majority_module', 'unknown')
        
        if primary == 'service_backbone' and majority == 'external_learning':
            examples['service_backbone終點'].append(flow)
        elif primary == 'cognitive_core' and majority == 'external_learning':
            examples['cognitive_core終點'].append(flow)
        elif primary == 'task_planning' and majority == 'external_learning':
            examples['task_planning終點'].append(flow)

    # 顯示例子
    for category, flows in examples.items():
        if flows:
            print(f'📋 {category} 例子 (前3個):')
            for i, flow in enumerate(flows[:3]):
                module_path = ' → '.join(flow['modules'])
                start_func = flow['start']
                end_func = flow['end']
                maj_module = flow.get('majority_module', 'N/A')
                prim_module = flow['primary_module']
                print(f'  {i+1}. {start_func} → {end_func}')
                print(f'     模組路徑: {module_path}')
                print(f'     多數決: {maj_module}, 終點: {prim_module}')
            print()

    print('統計：')
    print(f'service_backbone終點且external_learning為多數: {len(examples["service_backbone終點"])}')
    print(f'cognitive_core終點且external_learning為多數: {len(examples["cognitive_core終點"])}')
    print(f'task_planning終點且external_learning為多數: {len(examples["task_planning終點"])}')
    
    # 額外分析：純external_learning flows
    pure_external = []
    mixed_flows = []
    
    for flow in data['flows']:
        if flow['primary_module'] == 'external_learning' and flow.get('majority_module') == 'external_learning':
            pure_external.append(flow)
        elif flow['primary_module'] != flow.get('majority_module'):
            mixed_flows.append(flow)
    
    print()
    print('🎯 深度分析：')
    print(f'純external_learning flows (始終都是): {len(pure_external)}')
    print(f'分類不一致的flows: {len(mixed_flows)}')
    print(f'分類一致性: {(len(data["flows"]) - len(mixed_flows)) / len(data["flows"]) * 100:.1f}%')

if __name__ == '__main__':
    analyze_classification_comparison()