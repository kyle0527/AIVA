"""
比對舊檔案與新 schemas/ 資料夾的內容
"""

import re
from pathlib import Path

def extract_classes(content):
    """提取所有類別名稱"""
    pattern = r'^class\s+(\w+)\s*\('
    return set(re.findall(pattern, content, re.MULTILINE))

def compare_files(old_path, new_path, name):
    """比較兩個檔案的類別"""
    old_file = Path(old_path)
    new_file = Path(new_path)
    
    if not old_file.exists():
        print(f"[FAIL] {old_path} 不存在")
        return False
    
    if not new_file.exists():
        print(f"[FAIL] {new_path} 不存在")
        return False
    
    old_content = old_file.read_text(encoding='utf-8')
    new_content = new_file.read_text(encoding='utf-8')
    
    old_classes = extract_classes(old_content)
    new_classes = extract_classes(new_content)
    
    print(f'\n[STATS] {name} 分析:')
    print(f'  舊檔案 ({old_path}):')
    print(f'    - 類別數量: {len(old_classes)}')
    
    print(f'  新檔案 ({new_path}):')
    print(f'    - 類別數量: {len(new_classes)}')
    
    only_in_old = old_classes - new_classes
    only_in_new = new_classes - old_classes
    common = old_classes & new_classes
    
    print(f'\n  [SEARCH] 比對結果:')
    print(f'    - 共同類別: {len(common)}')
    
    if only_in_old:
        print(f'    - [WARN]  僅在舊檔案: {len(only_in_old)} 個')
        print(f'      {sorted(only_in_old)}')
        return False
    else:
        print(f'    - [OK] 舊檔案的所有類別已遷移')
    
    if only_in_new:
        print(f'    - [NOTE] 新增類別: {len(only_in_new)} 個')
        
    return True

# 主要比對
print("=" * 60)
print("檔案遷移比對報告")
print("=" * 60)

# 比對 ai_schemas.py
ai_migrated = compare_files(
    'ai_schemas.py',
    'schemas/ai.py',
    'AI Schemas'
)

# 比對 schemas.py (檢查是否所有內容都已拆分)
print("\n" + "=" * 60)
print("schemas.py 拆分檢查")
print("=" * 60)

schemas_content = Path('schemas.py').read_text(encoding='utf-8')
schemas_classes = extract_classes(schemas_content)

# 檢查新 schemas/ 資料夾中的所有類別
new_schemas_dir = Path('schemas')
all_new_classes = set()

for py_file in new_schemas_dir.glob('*.py'):
    if py_file.name == '__init__.py':
        continue
    content = py_file.read_text(encoding='utf-8')
    classes = extract_classes(content)
    all_new_classes.update(classes)
    print(f'\n  [U+1F4C4] {py_file.name}: {len(classes)} 個類別')

print(f'\n[STATS] 總計:')
print(f'  - 原 schemas.py: {len(schemas_classes)} 個類別')
print(f'  - 新 schemas/: {len(all_new_classes)} 個類別')

missing = schemas_classes - all_new_classes
if missing:
    print(f'\n[WARN]  尚未遷移的類別 ({len(missing)} 個):')
    for cls in sorted(missing):
        print(f'    - {cls}')
else:
    print(f'\n[OK] schemas.py 的所有類別已遷移')

# 可以刪除的檔案列表
print("\n" + "=" * 60)
print("建議刪除的檔案")
print("=" * 60)

can_delete = []

# 備份檔案
backup_patterns = [
    '*backup*.py',
    '*old*.py',
    '*broken*.py',
    '*fixed*.py',
    '*compat*.py',
    '*.disabled',
]

for pattern in backup_patterns:
    for f in Path('.').glob(pattern):
        if f.is_file() and f.name != '__init__.py':
            can_delete.append(f)

if ai_migrated:
    can_delete.append(Path('ai_schemas.py'))

if not missing:  # 如果 schemas.py 已完全遷移
    can_delete.append(Path('schemas.py'))

print('\n可以安全刪除的檔案:')
for f in sorted(can_delete):
    size = f.stat().st_size if f.exists() else 0
    print(f'  - {f.name} ({size:,} bytes)')

print(f'\n總計: {len(can_delete)} 個檔案可以刪除')
