"""
自動提取 enums.py 並創建模組化結構
"""

import re
from pathlib import Path

# 讀取原始 enums.py
enums_file = Path('enums.py')
content = enums_file.read_text(encoding='utf-8')

# 枚舉分類配置
classifications = {
    'common.py': {
        'description': '通用枚舉 - 狀態、級別、類型等基礎枚舉',
        'enums': [
            'Severity', 'Confidence', 'TaskStatus', 'TestStatus',
            'ScanStatus', 'ThreatLevel', 'RiskLevel', 'RemediationStatus'
        ]
    },
    'modules.py': {
        'description': '模組相關枚舉 - 模組名稱、主題等',
        'enums': ['ModuleName', 'Topic']
    },
    'security.py': {
        'description': '安全測試相關枚舉 - 漏洞、攻擊、權限等',
        'enums': [
            'VulnerabilityType', 'VulnerabilityStatus', 'Location',
            'SensitiveInfoType', 'IntelSource', 'IOCType', 'RemediationType',
            'Permission', 'AccessDecision', 'PostExTestType', 'PersistenceType',
            'Exploitability', 'AttackPathNodeType', 'AttackPathEdgeType'
        ]
    },
    'assets.py': {
        'description': '資產管理相關枚舉 - 資產類型、環境、合規等',
        'enums': [
            'BusinessCriticality', 'Environment', 'AssetType', 'AssetStatus',
            'DataSensitivity', 'AssetExposure', 'ComplianceFramework'
        ]
    }
}

def extract_enum_code(enum_name, content):
    """提取單個枚舉的完整定義"""
    # 找到枚舉定義開始
    pattern = rf'^class {enum_name}\([^)]+\):'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        return None
    
    start = match.start()
    
    # 找到下一個類別或檔案結尾
    next_class = re.search(r'\n\nclass \w+\(', content[start + 10:])
    if next_class:
        end = start + next_class.start() + 10
    else:
        end = len(content)
    
    return content[start:end].rstrip()

# 創建 enums 資料夾
enums_dir = Path('enums')
enums_dir.mkdir(exist_ok=True)

print("=" * 70)
print("開始創建 enums 模組化結構")
print("=" * 70)

# 生成各個模組檔案
for filename, info in classifications.items():
    output = []
    output.append('"""')
    output.append(info['description'])
    output.append('"""')
    output.append('')
    output.append('from __future__ import annotations')
    output.append('')
    output.append('from enum import Enum')
    output.append('')
    output.append('')
    
    enum_count = 0
    for enum_name in info['enums']:
        enum_code = extract_enum_code(enum_name, content)
        if enum_code:
            output.append(enum_code)
            output.append('')
            output.append('')
            enum_count += 1
        else:
            print(f"⚠️  找不到 {enum_name}")
    
    # 寫入檔案
    output_file = enums_dir / filename
    output_file.write_text('\n'.join(output), encoding='utf-8')
    
    print(f"✅ 創建 enums/{filename} ({enum_count} 個枚舉)")

# 創建 __init__.py
print("\n" + "=" * 70)
print("創建 enums/__init__.py")
print("=" * 70)

init_content = []
init_content.append('"""')
init_content.append('AIVA Common Enums Package')
init_content.append('')
init_content.append('此套件提供了 AIVA 微服務生態系統中所有枚舉類型的統一介面。')
init_content.append('')
init_content.append('使用方式:')
init_content.append('    from aiva_common.enums import ModuleName, Severity, VulnerabilityType')
init_content.append('')
init_content.append('架構說明:')
init_content.append('    - common.py: 通用枚舉 (狀態、級別等)')
init_content.append('    - modules.py: 模組相關枚舉 (模組名稱、主題)')
init_content.append('    - security.py: 安全測試枚舉 (漏洞、攻擊類型等)')
init_content.append('    - assets.py: 資產管理枚舉 (資產類型、環境等)')
init_content.append('"""')
init_content.append('')

# 生成導入語句
init_content.append('# ==================== 通用枚舉 ====================')
init_content.append('from .common import (')
for enum in classifications['common.py']['enums']:
    init_content.append(f'    {enum},')
init_content.append(')')
init_content.append('')

init_content.append('# ==================== 模組相關 ====================')
init_content.append('from .modules import (')
for enum in classifications['modules.py']['enums']:
    init_content.append(f'    {enum},')
init_content.append(')')
init_content.append('')

init_content.append('# ==================== 安全測試 ====================')
init_content.append('from .security import (')
for enum in classifications['security.py']['enums']:
    init_content.append(f'    {enum},')
init_content.append(')')
init_content.append('')

init_content.append('# ==================== 資產管理 ====================')
init_content.append('from .assets import (')
for enum in classifications['assets.py']['enums']:
    init_content.append(f'    {enum},')
init_content.append(')')
init_content.append('')

# 生成 __all__ 列表
init_content.append('# 為了保持向後相容，明確匯出所有公開介面')
init_content.append('__all__ = [')

all_enums = []
for info in classifications.values():
    all_enums.extend(info['enums'])

for enum in sorted(all_enums):
    init_content.append(f'    "{enum}",')

init_content.append(']')
init_content.append('')
init_content.append('# 版本資訊')
init_content.append('__version__ = "2.0.0"')

# 寫入 __init__.py
init_file = enums_dir / '__init__.py'
init_file.write_text('\n'.join(init_content), encoding='utf-8')

print(f"✅ 創建 enums/__init__.py (導出 {len(all_enums)} 個枚舉)")

print("\n" + "=" * 70)
print("Enums 模組化結構創建完成!")
print("=" * 70)
print(f"""
創建的檔案:
  - enums/common.py       ({len(classifications['common.py']['enums'])} 個枚舉)
  - enums/modules.py      ({len(classifications['modules.py']['enums'])} 個枚舉)
  - enums/security.py     ({len(classifications['security.py']['enums'])} 個枚舉)
  - enums/assets.py       ({len(classifications['assets.py']['enums'])} 個枚舉)
  - enums/__init__.py     (統一導出介面)

總計: {len(all_enums)} 個枚舉
""")
