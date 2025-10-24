"""
提取 Enhanced 類別
"""

import re
from pathlib import Path

# 使用相對路徑，動態定位 schemas.py
project_root = Path(__file__).parent.parent.parent
schemas_file = project_root / 'schemas.py'

if not schemas_file.exists():
    schemas_file = project_root / 'services' / 'aiva_common' / 'schemas.py'
    
if not schemas_file.exists():
    print("❌ 找不到 schemas.py 檔案")
    exit(1)

content = schemas_file.read_text(encoding='utf-8')

enhanced_classes = [
    'EnhancedFindingPayload',
    'EnhancedScanScope',
    'EnhancedScanRequest',
    'EnhancedFunctionTaskTarget',
    'EnhancedIOCRecord',
    'EnhancedRiskAssessment',
    'EnhancedAttackPathNode',
    'EnhancedAttackPath',
    'EnhancedTaskExecution',
    'EnhancedVulnerabilityCorrelation',
]

output = []
output.append('"""')
output.append('Enhanced 版本 Schemas')
output.append('')
output.append('此模組定義了各種增強版本的資料模型,提供更詳細的字段和擴展功能。')
output.append('"""')
output.append('')
output.append('from __future__ import annotations')
output.append('')
output.append('from datetime import UTC, datetime')
output.append('from typing import Any')
output.append('')
output.append('from pydantic import BaseModel, Field')
output.append('')
output.append('from ..enums import Confidence, ModuleName, Severity, TestStatus, VulnerabilityType')
output.append('')
output.append('# ============================================================================')
output.append('# Enhanced 版本')
output.append('# ============================================================================')
output.append('')

for cls_name in enhanced_classes:
    # 找到類別定義開始
    pattern = rf'^class {cls_name}\(BaseModel\):'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        print(f'找不到 {cls_name}')
        continue
    
    start = match.start()
    
    # 找到下一個類別或檔案結尾
    next_class = re.search(r'\n\nclass \w+\(', content[start + 10:])
    if next_class:
        end = start + next_class.start() + 10
    else:
        end = len(content)
    
    class_code = content[start:end].rstrip()
    output.append(class_code)
    output.append('')
    output.append('')

# 寫入檔案
output_file = project_root / 'schemas' / 'enhanced.py'
output_file.parent.mkdir(exist_ok=True)
output_file.write_text('\n'.join(output), encoding='utf-8')

print(f'✅ 已創建 {output_file}')
print(f'✅ 提取了 {len(enhanced_classes)} 個 Enhanced 類別')
