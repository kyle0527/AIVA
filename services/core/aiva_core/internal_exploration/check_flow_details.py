import json

with open('classification_data.json', encoding='utf-8') as f:
    data = json.load(f)

# 檢查Flow 439的詳細分類
flow_439 = next(f for f in data['flows'] if f['id'] == 439)
print('Flow 439詳細內容:')
print(f'路徑: {" → ".join(flow_439["path"])}')
print('分類詳情:')
for cls in flow_439['classifications']:
    print(f'  {cls["script"]}:')
    print(f'    模組: {cls["module"]}')
    print(f'    類型: {cls["component_type"]}')  
    print(f'    描述: {cls["description"]}')
    print()

print('='*50)
print('Flow 434詳細內容:')
flow_434 = next(f for f in data['flows'] if f['id'] == 434)
print(f'路徑: {" → ".join(flow_434["path"])}')
print('分類詳情:')
for cls in flow_434['classifications']:
    print(f'  {cls["script"]}:')
    print(f'    模組: {cls["module"]}')
    print(f'    類型: {cls["component_type"]}')  
    print(f'    描述: {cls["description"]}')
    print()