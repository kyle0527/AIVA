#!/usr/bin/env python3
"""生成專案統計報告"""
import os
from pathlib import Path
from collections import Counter

# 設定路徑
project_root = Path('C:/F/AIVA')
output_dir = Path('C:/F/AIVA/_out1101016')

# 排除目錄
exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '_archive', '_out', '_out1101016', 'aiva_platform_integrated.egg-info'}

# 收集所有程式碼檔案
all_files = []
for root, dirs, files in os.walk(project_root):
    # 排除特定目錄
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        file_path = Path(root) / file
        ext = file_path.suffix.lower()
        if ext in ['.py', '.go', '.rs', '.ts', '.js', '.sql', '.md']:
            try:
                size = file_path.stat().st_size
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                all_files.append({
                    'path': str(file_path.relative_to(project_root)),
                    'ext': ext,
                    'size': size,
                    'lines': lines
                })
            except:
                pass

# 生成副檔名統計
ext_counter = Counter([f['ext'] for f in all_files])
with open(output_dir / 'ext_counts.csv', 'w', encoding='utf-8') as f:
    f.write('Extension,Count\n')
    for ext, count in sorted(ext_counter.items()):
        f.write(f'{ext},{count}\n')

# 生成行數統計
ext_lines = {}
for f in all_files:
    ext = f['ext']
    if ext not in ext_lines:
        ext_lines[ext] = {'files': 0, 'lines': 0}
    ext_lines[ext]['files'] += 1
    ext_lines[ext]['lines'] += f['lines']

with open(output_dir / 'loc_by_ext.csv', 'w', encoding='utf-8') as f:
    f.write('Extension,Files,Lines,AvgLines\n')
    for ext in sorted(ext_lines.keys()):
        data = ext_lines[ext]
        avg = data['lines'] / data['files'] if data['files'] > 0 else 0
        f.write(f'{ext},{data["files"]},{data["lines"]},{avg:.1f}\n')

print('[OK] 統計檔案生成完成')
print(f'  • ext_counts.csv: {len(ext_counter)} 種副檔名')
print(f'  • loc_by_ext.csv: {sum(d["files"] for d in ext_lines.values())} 個檔案')
print(f'  • 總行數: {sum(d["lines"] for d in ext_lines.values()):,}')
