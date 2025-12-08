import json

# 讀取分類數據
with open('classification_data.json', encoding='utf-8') as f:
    data = json.load(f)

print(f'總數據流: {len(data["flows"])}')
print(f'元數據中程式組件數量: {data["metadata"]["component_type_distribution"]["程式組件"]}')

# 找出我聲稱的11條流程
prog_flows = []
for flow in data['flows']:
    all_prog = all(c['component_type'] == '程式組件' for c in flow['classifications'])
    suitable_length = 2 <= flow['length'] <= 5
    excludes = ['capability_cli', 'path_difference_analyzer']
    no_excludes = not any(exc in flow['start'] or exc in flow['end'] for exc in excludes)
    
    if all_prog and suitable_length and no_excludes:
        prog_flows.append(flow)

print(f'\n實際找到的純程式組件流程: {len(prog_flows)}條')
print('\n確認前5條:')
for i, flow in enumerate(prog_flows[:5], 1):
    print(f'{i}. Flow {flow["id"]}: {">>".join(flow["path"])}')
    print(f'   長度: {flow["length"]}, 全部組件類型:')
    for cls in flow['classifications']:
        print(f'     - {cls["script"]}: {cls["component_type"]}')
    print()

# 檢查我提到的特定流程是否真的在分類檔案中
target_flows = [429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439]
print('\n驗證我提到的流程:')
for target_id in target_flows:
    target_flow = next((f for f in data['flows'] if f['id'] == target_id), None)
    if target_flow:
        all_prog = all(c['component_type'] == '程式組件' for c in target_flow['classifications'])
        print(f'Flow {target_id}: {">>".join(target_flow["path"])} - 純程式組件: {all_prog}')
    else:
        print(f'Flow {target_id}: 未找到!')