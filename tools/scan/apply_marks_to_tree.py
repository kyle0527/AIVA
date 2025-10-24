"""
直接在樹狀圖上標註功能狀態
"""
import json
import re
from pathlib import Path

# 使用相對路徑，從項目根目錄計算
project_root = Path(__file__).parent.parent.parent

# 讀取功能分析結果
with open(project_root / '_out' / 'script_functionality_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 讀取樹狀圖
tree_path = project_root / '_out' / 'tree_ultimate_chinese_20251019_082355.txt'
with open(tree_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 建立檔案標記映射
marks = {}
for status, symbol in [
    ('no_functionality', '❌'),
    ('minimal_functionality', '🔶'),
    ('partial_functionality', '⚠️')
]:
    for item in data.get(status, []):
        path = item['file'].replace(str(project_root), '').replace('\\', '/').lstrip('/')
        filename = path.split('/')[-1]
        if filename not in marks:  # 只保留第一個狀態
            marks[filename] = symbol

# 標註每一行
marked_count = 0
new_lines = []
for line in lines:
    # 找出檔案名稱
    match = re.search(r'([─├└│\s]+)([^\s─├└│]+\.(py|ps1))(\s+#\s+[^❌🔶⚠️\n]+)', line)
    if match:
        prefix, filename, ext, comment = match.groups()
        if filename in marks and marks[filename] not in line:
            # 在註解後加上標記
            new_line = f"{prefix}{filename}{comment} {marks[filename]}\n"
            new_lines.append(new_line)
            marked_count += 1
            continue
    
    new_lines.append(line)

print(f'已標記 {marked_count} 個檔案')

# 寫回檔案
with open(tree_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'已更新: {tree_path}')
print(f'\n標記說明:')
print(f'  ❌ = 無功能 ({len(data["no_functionality"])} 個)')
print(f'  🔶 = 基本架構 ({len(data["minimal_functionality"])} 個)')
print(f'  ⚠️ = 部分功能 ({len(data["partial_functionality"])} 個)')
