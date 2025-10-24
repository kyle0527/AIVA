"""
移除 __init__.py 的功能標記
因為初始化腳本本來就可能是空的或只有簡單匯入
"""
import re
from pathlib import Path

# 使用相對路徑，從項目根目錄計算
project_root = Path(__file__).parent.parent.parent
tree_path = project_root / '_out' / 'tree_ultimate_chinese_20251019_082355.txt'

with open(tree_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 統計移除前的數量
before_no = content.count('❌')
before_min = content.count('🔶')
before_part = content.count('⚠️')
before_total = before_no + before_min + before_part

print(f'移除前總標記數: {before_total}')
print(f'  ❌ 無功能: {before_no}')
print(f'  🔶 基本架構: {before_min}')
print(f'  ⚠️ 部分功能: {before_part}')

# 移除 __init__.py 的所有標記
# 匹配模式: __init__.py # xxx ❌/🔶/⚠️
content = re.sub(r'(__init__\.py\s+#[^❌🔶⚠️\n]+)\s*[❌🔶⚠️]', r'\1', content)

# 統計移除後的數量
after_no = content.count('❌')
after_min = content.count('🔶')
after_part = content.count('⚠️')
after_total = after_no + after_min + after_part

print(f'\n移除後總標記數: {after_total}')
print(f'  ❌ 無功能: {after_no}')
print(f'  🔶 基本架構: {after_min}')
print(f'  ⚠️ 部分功能: {after_part}')

removed = before_total - after_total
print(f'\n已移除 {removed} 個 __init__.py 的標記')
print(f'  移除的 ❌: {before_no - after_no}')
print(f'  移除的 🔶: {before_min - after_min}')
print(f'  移除的 ⚠️: {before_part - after_part}')

# 寫回檔案
with open(tree_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\n✅ 已更新: {tree_path}')
