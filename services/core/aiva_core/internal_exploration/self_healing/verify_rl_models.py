import json
from pathlib import Path

json_path = Path('C:/D/fold7/AIVA-git/services/core/aiva_core/analysis_results_v2/analysis_results.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查找 rl_models
script_functions = data.get('function_details', {}).get('script_functions', {})
rl_models = script_functions.get('rl_models', {})

print('='*60)
print('rl_models.py 分析')
print('='*60)
print(f"export_functions 數量: {len(rl_models.get('export_functions', []))}")
print(f"entry_points 數量: {len(rl_models.get('entry_points', []))}")
print(f"functions 數量: {len(rl_models.get('functions', {}))}")
print()

# 查找在 flow_chains 中的連接
flow_chains = data.get('flow_chains', [])
rl_models_as_source = 0
rl_models_as_target = 0

for chain in flow_chains:
    for i, script in enumerate(chain):
        if 'rl_models.py' in script:
            # 作為數據源（有輸出）
            if i < len(chain) - 1:
                rl_models_as_source += 1
            # 作為數據消費者（有輸入）
            if i > 0:
                rl_models_as_target += 1

print("在 flow_chains 中:")
print(f"  作為數據源的連接數（輸出）: {rl_models_as_source}")
print(f"  作為消費者的連接數（輸入）: {rl_models_as_target}")
print()

# 計算比率
potential_outputs = len(rl_models.get('export_functions', []))
actual_outputs = rl_models_as_source

if potential_outputs > 0:
    ratio = actual_outputs / potential_outputs
    print(f"連接完整度: {actual_outputs}/{potential_outputs} = {ratio*100:.1f}%")
    print()
    
    # 檢查是否應該被標記為「未完整連接」
    if potential_outputs >= 5 and actual_outputs < potential_outputs * 0.4:
        print("✅ 應該被標記為「未完整連接的輸出接口」")
        print(f"   因為: 定義了 {potential_outputs} 個函數，但只有 {actual_outputs} 個被實際使用 ({ratio*100:.1f}% < 40%)")
    else:
        print("❌ 不應該被標記為問題")
        if potential_outputs < 5:
            print(f"   原因: 函數數量太少 ({potential_outputs} < 5)")
        else:
            print(f"   原因: 連接完整度足夠 ({ratio*100:.1f}% >= 40%)")
