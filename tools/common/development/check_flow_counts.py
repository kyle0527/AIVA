import json

def check_flow_counts():
    print('=== 檢查各版本的flow數量 ===')

    # 檢查V3原始數據
    with open('aiva_core_analysis_v3/analysis_results.json', 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    print(f'V3原始分析: {len(v3_data.get("flow_chains", []))} flows')

    # 檢查V5原始數據 
    try:
        with open('C:/D/fold7/AIVA-git/services/core/aiva_core/internal_exploration/aiva_core_analysis_v5/analysis_results.json', 'r', encoding='utf-8') as f:
            v5_data = json.load(f)
        print(f'V5原始分析: {len(v5_data.get("flow_chains", []))} flows')
    except Exception as e:
        print(f'V5原始分析: 讀取失敗 - {e}')

    # 檢查V6分類結果
    try:
        with open('aiva_core_classification_v6_endpoint/classification_data.json', 'r', encoding='utf-8') as f:
            v6_data = json.load(f)
        print(f'V6分類結果 (endpoint): {len(v6_data.get("flows", []))} flows')
    except Exception as e:
        print(f'V6分類結果: 不存在 - {e}')

    # 檢查V8分類結果 
    try:
        with open('classification_data_v8_endpoint.json/classification_data.json', 'r', encoding='utf-8') as f:
            v8_data = json.load(f)
        print(f'V8分類結果 (endpoint): {len(v8_data.get("flows", []))} flows')
    except Exception as e:
        print(f'V8分類結果: 不存在 - {e}')

    # 檢查services的測試結果
    try:
        with open('C:/D/fold7/AIVA-git/services/core/aiva_core/internal_exploration/services_classification_v8_test/classification_data.json', 'r', encoding='utf-8') as f:
            services_data = json.load(f)
        print(f'Services測試結果: {len(services_data.get("flows", []))} flows')
    except Exception as e:
        print(f'Services測試結果: 不存在 - {e}')

    print()
    print('=== 檢查數據來源差異 ===')
    
    # 比較V3和V5的差異
    print(f'V3 (278 flows) vs V5 (可能321 flows) 差異: 可能是不同時期的分析結果')
    print('V6原本使用V5數據源 (321 flows)')
    print('V8和Services修復版使用V3數據源 (278 flows)')
    
    # 檢查原本V8的位置
    print()
    print('=== 檢查原本V8數據位置 ===')
    import os
    
    v8_possible_paths = [
        'classification_data_v8_endpoint.json/classification_data.json',
        '../classification_data_v8_endpoint.json/classification_data.json',
        'classification_data_v8_endpoint/classification_data.json'
    ]
    
    for path in v8_possible_paths:
        if os.path.exists(path):
            print(f'✅ 找到V8數據: {path}')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f'   Flows: {len(data.get("flows", []))}')
            except Exception as e:
                print(f'   ❌ 讀取失敗: {e}')
        else:
            print(f'❌ 不存在: {path}')

if __name__ == '__main__':
    check_flow_counts()