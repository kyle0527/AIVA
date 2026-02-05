import json

with open('analysis_results.json', encoding='utf-8') as f:
    data = json.load(f)

graphs = {g['name']: g['file_path'] for g in data['graphs']}
same_file = 0
diff_file = 0

for conn in data['connections']:
    source_file = graphs.get(conn['source_graph'])
    target_file = graphs.get(conn['target_graph'])
    
    if source_file == target_file:
        same_file += 1
    else:
        diff_file += 1

print(f"总连接数: {len(data['connections'])}")
print(f"同脚本连接: {same_file}")
print(f"跨脚本连接: {diff_file}")
print(f"\n同脚本连接占比: {same_file/len(data['connections'])*100:.1f}%")
print(f"跨脚本连接占比: {diff_file/len(data['connections'])*100:.1f}%")
