"""
AIVA 分類數據轉換器
將增強分類處理器的輸出轉換為執行器可讀的格式
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def convert_enhanced_to_executor_format(enhanced_data_path: str, output_path: Optional[str] = None):
    """將增強分類數據轉換為執行器格式"""
    
    # 讀取增強數據
    with open(enhanced_data_path, 'r', encoding='utf-8') as f:
        enhanced_data = json.load(f)
    
    # 轉換為執行器期望的格式
    executor_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source': 'enhanced_classifier_processor',
            'total_flows': enhanced_data['metadata']['total_capabilities'],
            'schema_version': '2.0'
        },
        'flows': []
    }
    
    # 從模組能力中提取flows
    for module_name, module_data in enhanced_data.get('module_capabilities', {}).items():
        for group_key, group_data in module_data['capability_groups'].items():
            for capability in group_data['capabilities']:
                flow_entry = {
                    'id': capability['flow_id'],
                    'name': f"flow_{capability['flow_id']}",
                    'path': capability['technical_details']['execution_path'],
                    'start': capability['technical_details']['start_point'],
                    'end': capability['technical_details']['end_point'],
                    'length': len(capability['technical_details']['execution_path']),
                    'primary_module': module_data['module_info']['module_id'],
                    'display_name': capability['display_name'],
                    'description': capability['ai_description']['purpose'],
                    'complexity': capability['ai_description']['complexity'],
                    'risk_level': capability['usage_guide']['risk_level'],
                    'example_commands': capability['usage_guide']['example_commands']
                }
                executor_data['flows'].append(flow_entry)
    
    # 排序flows by id
    executor_data['flows'].sort(key=lambda x: x['id'])
    
    # 保存轉換結果
    if not output_path:
        output_path = "classification_data_for_executor.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(executor_data, f, indent=2, ensure_ascii=False)
    
    print(f"[Convert] 轉換完成：{len(executor_data['flows'])} 個flows")
    print(f"📁 輸出文件：{output_path}")
    
    return output_path

if __name__ == "__main__":
    enhanced_data_path = "test_enhanced_output/4_standardization/module_grouped_capabilities.json"
    output_path = "classification_data_for_executor.json"
    
    convert_enhanced_to_executor_format(enhanced_data_path, output_path)