"""
全面驗證 schemas 和 enums 遷移完整性
"""

import sys
from pathlib import Path

# 設定路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "services"))

print("=" * 80)
print("SCHEMAS 和 ENUMS 遷移完整性驗證")
print("=" * 80)

# ============================================================================
# 1. 檢查 enums 遷移完整性
# ============================================================================
print("\n" + "=" * 80)
print("1[U+FE0F][U+20E3]  ENUMS 遷移完整性檢查")
print("=" * 80)

aiva_common = project_root / "services" / "aiva_common"

# 讀取舊的 enums.py
old_enums_file = aiva_common / "enums.py"
if old_enums_file.exists():
    import re
    old_content = old_enums_file.read_text(encoding='utf-8')
    old_enum_classes = re.findall(r'^class (\w+)\(', old_content, re.MULTILINE)
    old_enum_set = set(old_enum_classes)
    print(f"\n[U+1F4C4] 舊 enums.py: {len(old_enum_set)} 個枚舉類別")
    print(f"   {', '.join(sorted(old_enum_set)[:5])}...")
else:
    print("\n[WARN]  舊 enums.py 不存在")
    old_enum_set = set()

# 檢查新的 enums 模組
try:
    from aiva_common import enums
    
    new_enum_classes = [
        name for name in dir(enums)
        if not name.startswith('_') and name[0].isupper()
    ]
    new_enum_set = set(new_enum_classes)
    
    print(f"\n[U+1F4C1] 新 enums/ 模組: {len(new_enum_set)} 個枚舉類別")
    print(f"   {', '.join(sorted(new_enum_set)[:5])}...")
    
    # 比較差異
    if old_enum_set:
        missing = old_enum_set - new_enum_set
        extra = new_enum_set - old_enum_set
        
        if missing:
            print(f"\n[WARN]  缺失的枚舉 ({len(missing)} 個): {', '.join(sorted(missing))}")
        if extra:
            print(f"\n[U+2795] 新增的枚舉 ({len(extra)} 個): {', '.join(sorted(extra))}")
        if not missing and not extra:
            print(f"\n[OK] 完全匹配！所有 {len(old_enum_set)} 個枚舉都已遷移")
    else:
        print(f"\n[OK] 新 enums 模組有 {len(new_enum_set)} 個枚舉")
        
except Exception as e:
    print(f"\n[FAIL] 導入 enums 失敗: {e}")
    new_enum_set = set()

# ============================================================================
# 2. 檢查 schemas 遷移完整性
# ============================================================================
print("\n" + "=" * 80)
print("2[U+FE0F][U+20E3]  SCHEMAS 遷移完整性檢查")
print("=" * 80)

# 讀取舊的 schemas.py
old_schemas_file = aiva_common / "schemas.py"
if old_schemas_file.exists():
    import re
    old_content = old_schemas_file.read_text(encoding='utf-8')
    old_schema_classes = re.findall(r'^class (\w+)\(', old_content, re.MULTILINE)
    old_schema_set = set(old_schema_classes)
    print(f"\n[U+1F4C4] 舊 schemas.py: {len(old_schema_set)} 個 Schema 類別")
    print(f"   {', '.join(sorted(old_schema_set)[:5])}...")
else:
    print("\n[WARN]  舊 schemas.py 不存在")
    old_schema_set = set()

# 檢查舊的 ai_schemas.py
old_ai_schemas_file = aiva_common / "ai_schemas.py"
if old_ai_schemas_file.exists():
    import re
    old_ai_content = old_ai_schemas_file.read_text(encoding='utf-8')
    old_ai_schema_classes = re.findall(r'^class (\w+)\(', old_ai_content, re.MULTILINE)
    old_ai_schema_set = set(old_ai_schema_classes)
    print(f"\n[U+1F4C4] 舊 ai_schemas.py: {len(old_ai_schema_set)} 個 Schema 類別")
    print(f"   {', '.join(sorted(old_ai_schema_set)[:5])}...")
    
    # 合併到總數
    old_schema_set = old_schema_set | old_ai_schema_set
    print(f"\n[STATS] 舊 schemas 總計: {len(old_schema_set)} 個類別")
else:
    print("\n[WARN]  舊 ai_schemas.py 不存在")

# 檢查新的 schemas 模組
try:
    from aiva_common import schemas
    
    new_schema_classes = [
        name for name in dir(schemas)
        if not name.startswith('_') and name[0].isupper()
    ]
    new_schema_set = set(new_schema_classes)
    
    print(f"\n[U+1F4C1] 新 schemas/ 模組: {len(new_schema_set)} 個 Schema 類別")
    print(f"   {', '.join(sorted(new_schema_set)[:5])}...")
    
    # 比較差異
    if old_schema_set:
        missing = old_schema_set - new_schema_set
        extra = new_schema_set - old_schema_set
        
        if missing:
            print(f"\n[WARN]  缺失的 Schema ({len(missing)} 個):")
            for m in sorted(missing):
                print(f"   - {m}")
        if extra:
            print(f"\n[U+2795] 新增的 Schema ({len(extra)} 個):")
            for e in sorted(extra):
                print(f"   - {e}")
        if not missing and not extra:
            print(f"\n[OK] 完全匹配！所有 {len(old_schema_set)} 個 Schema 都已遷移")
    else:
        print(f"\n[OK] 新 schemas 模組有 {len(new_schema_set)} 個 Schema")
        
except Exception as e:
    print(f"\n[FAIL] 導入 schemas 失敗: {e}")
    import traceback
    traceback.print_exc()
    new_schema_set = set()

# ============================================================================
# 3. 檢查檔案結構
# ============================================================================
print("\n" + "=" * 80)
print("3[U+FE0F][U+20E3]  檔案結構檢查")
print("=" * 80)

# 檢查 schemas 資料夾
schemas_dir = aiva_common / "schemas"
if schemas_dir.exists():
    schema_files = list(schemas_dir.glob("*.py"))
    print(f"\n[U+1F4C1] schemas/ 資料夾: {len(schema_files)} 個檔案")
    for f in sorted(schema_files):
        size = f.stat().st_size
        print(f"   - {f.name:30s} ({size:>8,} bytes)")
else:
    print("\n[FAIL] schemas/ 資料夾不存在")

# 檢查 enums 資料夾
enums_dir = aiva_common / "enums"
if enums_dir.exists():
    enum_files = list(enums_dir.glob("*.py"))
    print(f"\n[U+1F4C1] enums/ 資料夾: {len(enum_files)} 個檔案")
    for f in sorted(enum_files):
        size = f.stat().st_size
        print(f"   - {f.name:30s} ({size:>8,} bytes)")
else:
    print("\n[FAIL] enums/ 資料夾不存在")

# ============================================================================
# 4. 舊檔案檢查
# ============================================================================
print("\n" + "=" * 80)
print("4[U+FE0F][U+20E3]  舊檔案狀態")
print("=" * 80)

old_files = {
    "schemas.py": old_schemas_file,
    "enums.py": old_enums_file,
    "ai_schemas.py": old_ai_schemas_file
}

total_old_size = 0
for name, path in old_files.items():
    if path.exists():
        size = path.stat().st_size
        total_old_size += size
        print(f"[WARN]  {name:20s} 存在 ({size:>10,} bytes)")
    else:
        print(f"[OK] {name:20s} 已刪除")

if total_old_size > 0:
    print(f"\n[STATS] 舊檔案總大小: {total_old_size:,} bytes ({total_old_size / 1024:.1f} KB)")

# ============================================================================
# 5. 整合測試
# ============================================================================
print("\n" + "=" * 80)
print("5[U+FE0F][U+20E3]  導入測試")
print("=" * 80)

test_cases = [
    # Enums 測試
    ("from aiva_common.enums import ModuleName", "ModuleName"),
    ("from aiva_common.enums import Severity", "Severity"),
    ("from aiva_common.enums import Topic", "Topic"),
    ("from aiva_common.enums import VulnerabilityType", "VulnerabilityType"),
    
    # Schemas 測試 - 先測試不依賴其他的
    ("from aiva_common.schemas.base import MessageHeader", "MessageHeader"),
    ("from aiva_common.schemas.base import Authentication", "Authentication"),
]

passed = 0
failed = 0

for test_code, expected_name in test_cases:
    try:
        exec(test_code)
        print(f"[OK] {test_code}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] {test_code}")
        print(f"   錯誤: {str(e)[:60]}")
        failed += 1

print(f"\n[STATS] 測試結果: {passed}/{passed + failed} 通過")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("[LIST] 遷移完整性總結")
print("=" * 80)

enums_complete = len(new_enum_set) >= 31 if new_enum_set else False
schemas_complete = len(new_schema_set) >= 126 if new_schema_set else False

print(f"""
Enums 遷移:
  - 舊檔案: {len(old_enum_set)} 個枚舉
  - 新模組: {len(new_enum_set)} 個枚舉
  - 狀態: {'[OK] 完整' if enums_complete else '[WARN] 不完整'}

Schemas 遷移:
  - 舊檔案: {len(old_schema_set)} 個 Schema
  - 新模組: {len(new_schema_set)} 個 Schema
  - 狀態: {'[OK] 完整' if schemas_complete else '[WARN] 不完整'}

檔案結構:
  - schemas/ 資料夾: {'[OK] 存在' if schemas_dir.exists() else '[FAIL] 不存在'}
  - enums/ 資料夾: {'[OK] 存在' if enums_dir.exists() else '[FAIL] 不存在'}

舊檔案清理:
  - schemas.py: {'[WARN] 需要刪除' if old_schemas_file.exists() else '[OK] 已刪除'}
  - enums.py: {'[WARN] 需要刪除' if old_enums_file.exists() else '[OK] 已刪除'}
  - ai_schemas.py: {'[WARN] 需要刪除' if old_ai_schemas_file.exists() else '[OK] 已刪除'}

下一步行動:
""")

if not enums_complete or not schemas_complete:
    print("  [WARN] 需要完成遷移缺失的類別")
if old_schemas_file.exists() or old_enums_file.exists() or old_ai_schemas_file.exists():
    print("  [U+1F5D1][U+FE0F]  需要刪除舊檔案")
if passed < len(test_cases):
    print("  [CONFIG] 需要修復導入問題")
if enums_complete and schemas_complete and not (old_schemas_file.exists() or old_enums_file.exists()):
    print("  [OK] 遷移完全完成！")

print("=" * 80)
