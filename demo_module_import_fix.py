#!/usr/bin/env python3
"""
AIVA 模組導入問題演示

這個腳本展示了修復前後的差異。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("AIVA 模組導入修復演示")
print("=" * 70)

print("\n問題說明:")
print("-" * 70)
print("修復前的問題:")
print("1. models.py 和 schemas.py 都定義了相同的類（重複定義）")
print("2. __init__.py 從兩個文件導入，造成混亂")
print("3. 服務模組不確定應該從哪個文件導入")
print("4. 維護困難：修改一個類需要改兩個地方")

print("\n修復後的架構:")
print("-" * 70)
print("1. schemas.py 是唯一的數據源（單一來源原則）")
print("2. models.py 重新導出 schemas.py 的類（向後兼容）")
print("3. __init__.py 統一從 schemas.py 導入")
print("4. 所有服務從 schemas.py 或 aiva_common 包導入")

print("\n✅ 導入方式（推薦順序）:")
print("-" * 70)
print("1. 從 aiva_common 包導入（最佳）:")
print("   from services.aiva_common import MessageHeader, CVSSv3Metrics")
print()
print("2. 從 schemas.py 直接導入（明確）:")
print("   from services.aiva_common.schemas import MessageHeader, CVSSv3Metrics")
print()
print("3. 從 models.py 導入（向後兼容，但不推薦）:")
print("   from services.aiva_common.models import MessageHeader, CVSSv3Metrics")

print("\n🔄 向後兼容性:")
print("-" * 70)
print("舊代碼仍然可以工作，因為 models.py 現在重新導出 schemas.py 的類。")
print("但建議逐步遷移到從 aiva_common 或 schemas.py 導入。")

print("\n📊 統計信息:")
print("-" * 70)

# Count classes in schemas.py
try:
    with open("services/aiva_common/schemas.py", "r", encoding="utf-8") as f:
        schema_content = f.read()
        schema_classes = schema_content.count("class ")
        schema_lines = len(schema_content.split("\n"))
    print(f"schemas.py: {schema_classes} 個類, {schema_lines} 行")
except:
    print("schemas.py: 無法讀取")

# Count classes in models.py
try:
    with open("services/aiva_common/models.py", "r", encoding="utf-8") as f:
        models_content = f.read()
        # Count import lines instead of class definitions
        import_lines = [line for line in models_content.split("\n") if "from .schemas import" in line]
        print(f"models.py: 重新導出層（向後兼容），{len(import_lines)} 個導入語句")
except:
    print("models.py: 無法讀取")

print("\n📝 修改的文件:")
print("-" * 70)
modified_files = [
    "services/aiva_common/__init__.py",
    "services/aiva_common/models.py",
    "services/aiva_common/schemas.py",
    "services/scan/__init__.py",
    "services/scan/models.py",
    "services/core/aiva_core/__init__.py",
    "services/core/models.py",
    "services/function/__init__.py",
]

for f in modified_files:
    print(f"  - {f}")

print("\n📚 新增文件:")
print("-" * 70)
print("  - test_module_imports.py (綜合測試)")
print("  - MODULE_IMPORT_FIX_REPORT.md (詳細報告)")

print("\n" + "=" * 70)
print("✨ 修復完成！")
print("=" * 70)
print()
print("下一步:")
print("1. 安裝依賴: pip install -r requirements.txt")
print("2. 運行測試: python test_module_imports.py")
print("3. 閱讀報告: MODULE_IMPORT_FIX_REPORT.md")
print()
