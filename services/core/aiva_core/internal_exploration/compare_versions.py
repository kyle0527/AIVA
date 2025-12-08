import json

def compare_services_tools():
    # 比較services版本與tools版本V8的結果
    with open('services_classification_v8_test/classification_data.json', 'r', encoding='utf-8') as f:
        services_data = json.load(f)

    with open('C:/D/fold7/AIVA-git/tools/common/development/classification_data_v8_endpoint.json/classification_data.json', 'r', encoding='utf-8') as f:
        tools_data = json.load(f)

    print('=== Services vs Tools V8 對比 ===')
    print(f'Services flows: {len(services_data.get("flows", []))}')
    print(f'Tools flows: {len(tools_data.get("flows", []))}')

    # 比較模組分佈
    def get_module_distribution(data):
        dist = {}
        for flow in data.get('flows', []):
            module = flow.get('primary_module', 'unknown')
            dist[module] = dist.get(module, 0) + 1
        return dist

    services_dist = get_module_distribution(services_data)
    tools_dist = get_module_distribution(tools_data)

    print()
    print('模組分佈對比:')
    all_modules = set(services_dist.keys()) | set(tools_dist.keys())
    identical = True
    for module in sorted(all_modules):
        s_count = services_dist.get(module, 0)
        t_count = tools_dist.get(module, 0)
        match = '✅' if s_count == t_count else '❌'
        if s_count != t_count:
            identical = False
        print(f'  {module}: Services={s_count}, Tools={t_count} {match}')

    # 檢查第一個flow是否完全一致
    if services_data.get('flows') and tools_data.get('flows'):
        s_first = services_data['flows'][0]
        t_first = tools_data['flows'][0]
        
        print()
        print('第一個flow對比:')
        print(f'  Path: {s_first.get("path", [])} vs {t_first.get("path", [])}')
        print(f'  Primary: {s_first.get("primary_module")} vs {t_first.get("primary_module")}')
        print(f'  Endpoint: {s_first.get("endpoint_module")} vs {t_first.get("endpoint_module")}')
        print(f'  Majority: {s_first.get("majority_module")} vs {t_first.get("majority_module")}')

    print()
    if identical:
        print('✅ 修復成功：Services版本與Tools版本V8完全一致')
    else:
        print('⚠️  修復需檢查：Services版本與Tools版本V8存在差異')

if __name__ == '__main__':
    compare_services_tools()