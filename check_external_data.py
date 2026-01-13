"""檢查外部模組數據結構"""
import json
from pathlib import Path

files = {
    'function_sqli': 'module_analysis/function_sqli/analysis_results.json',
    'typescript': 'services/scan/typescript_engine/analysis_output/analysis_results.json',
    'rust': 'services/integration/data/internal_exploration/analysis_results/rust/analysis_results.json',
    'go': 'services/features/function_authn_go/analysis_output/analysis_results.json'
}

print("="*60)
print("外部模組數據結構分析")
print("="*60)

for name, file_path in files.items():
    print(f"\n[{name}]")
    try:
        data = json.load(open(file_path, encoding='utf-8'))
        print(f"  頂層keys: {list(data.keys())[:8]}")
        
        # 檢查目標路徑
        target = data.get('target') or data.get('metadata', {}).get('project_path', 'N/A')
        print(f"  目標路徑: {target}")
        
        # 檢查流程數據
        flows = data.get('flows') or data.get('flow_chains', [])
        if flows:
            first_flow = flows[0]
            print(f"  Flow結構: {type(first_flow).__name__}")
            if isinstance(first_flow, dict):
                print(f"    Flow keys: {list(first_flow.keys())[:10]}")
                # 檢查路徑信息
                if 'full_path' in first_flow:
                    print(f"    路徑範例: {first_flow['full_path'][0] if first_flow['full_path'] else 'N/A'}")
                if 'path' in first_flow and isinstance(first_flow['path'], list):
                    print(f"    函數路徑: {first_flow['path'][0] if first_flow['path'] else 'N/A'}")
            elif isinstance(first_flow, list):
                print(f"    路徑範例: {first_flow[0] if first_flow else 'N/A'}")
                
    except Exception as e:
        print(f"  [錯誤] {e}")

print("\n" + "="*60)
print("結論：如何從數據提取模組名稱")
print("="*60)
print("""
1. Python數據: 從 'target' 欄位提取模組名稱
   範例: "services/features/function_sqli" -> function_sqli

2. TypeScript數據: 從 'metadata.project_path' 或路徑中提取
   範例: "src\\services\\scan-service.ts" -> scan_*

3. Rust/Go數據: 從 'target' 或文件路徑提取
   範例: "services/integration/..." -> function_*

策略：從路徑中動態提取 'function_*' 或 'scan_*' 模式
""")
