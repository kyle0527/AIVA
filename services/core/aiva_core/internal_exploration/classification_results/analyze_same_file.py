import json

with open('analysis_results.json', encoding='utf-8') as f:
    data = json.load(f)

graphs = {g['name']: g['file_path'] for g in data['graphs']}

print("=== 分析同脚本连接的意义 ===\n")

# 找一个有同脚本连接的例子
file = "analyze_dataflow_breakpoints.py"
file_graphs = [g for g in data['graphs'] if file in g['file_path']]
file_graph_names = [g['name'] for g in file_graphs]

same_file_conns = [c for c in data['connections'] 
                   if c['source_graph'] in file_graph_names 
                   and c['target_graph'] in file_graph_names]

print(f"文件: {file}")
print(f"函数数量: {len(file_graphs)}")
print(f"同脚本内连接: {len(same_file_conns)}\n")

print("这些连接的源和目标：")
for conn in same_file_conns[:5]:
    print(f"  {conn['source_graph']}")
    print(f"    → {conn['target_graph']}")
    print()

# 分析 flow_chains 中是否包含同脚本连接
print("\n=== flow_chains 中的同脚本段 ===")
for i, chain in enumerate(data['flow_chains'][:5], 1):
    print(f"\nChain {i}: {len(chain)} 步")
    for j, func in enumerate(chain):
        file_path = graphs.get(func, "unknown")
        file_name = file_path.split('\\')[-1] if '\\' in file_path else file_path
        print(f"  {j+1}. {func} [{file_name}]")
