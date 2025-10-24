"""
診斷並修復 aiva_common 導入問題

問題：
1. schemas/tasks.py 導入了不存在的 AuthType 和 CrawlingStrategy
2. 這導致所有 schemas 導入失敗

解決方案：
1. 檢查這些枚舉是否真的需要
2. 如果需要，創建它們
3. 如果不需要，移除無用的導入
"""

import re
from pathlib import Path

aiva_common = Path(__file__).parent.parent.parent / "services" / "aiva_common"

print("=" * 80)
print("診斷 AuthType 和 CrawlingStrategy 使用情況")
print("=" * 80)

# 讀取 tasks.py
tasks_file = aiva_common / "schemas" / "tasks.py"
tasks_content = tasks_file.read_text(encoding="utf-8")

# 檢查是否實際使用這些枚舉
print("\n🔍 檢查 AuthType 使用情況:")
auth_type_usage = re.findall(r'\bAuthType\b', tasks_content)
print(f"   找到 {len(auth_type_usage)} 處引用")
for i, match in enumerate(auth_type_usage[:5], 1):
    lines = tasks_content.split('\n')
    for line_no, line in enumerate(lines, 1):
        if 'AuthType' in line and i == 1:
            print(f"   第 {line_no} 行: {line.strip()}")
            i += 1

print("\n🔍 檢查 CrawlingStrategy 使用情況:")
crawl_usage = re.findall(r'\bCrawlingStrategy\b', tasks_content)
print(f"   找到 {len(crawl_usage)} 處引用")
for i, match in enumerate(crawl_usage[:5], 1):
    lines = tasks_content.split('\n')
    for line_no, line in enumerate(lines, 1):
        if 'CrawlingStrategy' in line and i == 1:
            print(f"   第 {line_no} 行: {line.strip()}")
            i += 1

# 檢查導入區塊
print("\n📋 當前導入區塊:")
import_section = re.search(r'from \.\.enums import \((.*?)\)', tasks_content, re.DOTALL)
if import_section:
    imports = [i.strip().rstrip(',') for i in import_section.group(1).split('\n') if i.strip()]
    for imp in imports:
        print(f"   - {imp}")

# 檢查實際在程式碼中使用的枚舉
print("\n🔍 實際使用的枚舉 (在類型提示或賦值中):")
used_enums = set()

# 找出所有在 Field 或類型提示中使用的枚舉
type_hints = re.findall(r':\s*([A-Z]\w+)(?:\s*\||\s*=|\s*$)', tasks_content)
field_types = re.findall(r':\s*([A-Z]\w+)\s*=\s*Field', tasks_content)

all_potential = set(type_hints + field_types)
for enum in all_potential:
    if enum in import_section.group(1):
        used_enums.add(enum)

print(f"   實際使用: {', '.join(sorted(used_enums)) if used_enums else '無'}")

# 找出未使用的導入
imported_enums = set(imp.strip().rstrip(',') for imp in imports if imp)
unused_imports = imported_enums - used_enums

if 'AuthType' in unused_imports or 'CrawlingStrategy' in unused_imports:
    print(f"\n⚠️  未使用的導入: {', '.join(unused_imports)}")
    print("\n建議:")
    print("   1. 從 schemas/tasks.py 的導入中移除這些未使用的枚舉")
    print("   2. 如果將來需要，可以重新添加")
    
    # 生成修復後的導入
    print("\n✅ 修復後的導入區塊:")
    clean_imports = sorted(used_enums - {'AuthType', 'CrawlingStrategy'})
    print("from ..enums import (")
    for imp in clean_imports:
        print(f"    {imp},")
    print(")")
else:
    print("\n✅ 所有導入的枚舉都有使用")

# 檢查是否需要創建這些枚舉
print("\n" + "=" * 80)
print("是否需要創建缺失的枚舉？")
print("=" * 80)

print("\n建議的 AuthType 定義 (如果需要):")
print("""
class AuthType(str, Enum):
    \"\"\"認證類型\"\"\"
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    DIGEST = "digest"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    SESSION = "session"
""")

print("\n建議的 CrawlingStrategy 定義 (如果需要):")
print("""
class CrawlingStrategy(str, Enum):
    \"\"\"爬蟲策略\"\"\"
    QUICK = "quick"      # 快速掃描
    NORMAL = "normal"    # 正常掃描
    DEEP = "deep"        # 深度掃描
    FULL = "full"        # 完整掃描
    CUSTOM = "custom"    # 自訂策略
""")

print("\n" + "=" * 80)
print("診斷完成")
print("=" * 80)
