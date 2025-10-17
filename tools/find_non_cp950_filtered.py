#!/usr/bin/env python3
"""
CP950 編碼兼容性檢查工具

檢查 Python 文件中無法用 CP950 編碼的字符，
主要用於確保代碼在 Windows 中文環境下的兼容性。
"""

from pathlib import Path

# 使用當前腳本所在目錄作為基準
script_dir = Path(__file__).parent
root = script_dir.parent  # AIVA 根目錄
out = script_dir / "non_cp950_filtered_report.txt"

print(f"掃描目錄: {root}")
print(f"輸出報告: {out}")
print("-" * 50)

files_checked = 0
issues = []

# 收集所有 Python 文件
py_files = list(root.rglob('*.py'))
total_files = len(py_files)
print(f"找到 {total_files} 個 Python 文件")

for p in py_files:
    # 跳過備份目錄
    if any(skip_dir in str(p) for skip_dir in ['emoji_backups', 'emoji_backups2', '__pycache__', '.git']):
        continue

    files_checked += 1

    # 進度顯示
    if files_checked % 50 == 0:
        print(f"已檢查: {files_checked}/{total_files}")

    try:
        text = p.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError, PermissionError) as e:
        print(f"警告: 無法讀取 {p}: {e}")
        continue

    # 檢查每一行
    for i, line in enumerate(text.splitlines(), start=1):
        try:
            line.encode('cp950')
        except UnicodeEncodeError as e:
            issues.append((str(p.relative_to(root)), i, line.strip(), str(e)))

# 生成報告
print("\n檢查完成！")
print(f"檢查文件: {files_checked}")
print(f"發現問題: {len(issues)}")

with out.open('w', encoding='utf-8') as f:
    f.write('CP950 編碼兼容性檢查報告\n')
    f.write(f'掃描目錄: {root}\n')
    f.write(f'檢查時間: {Path(__file__).stat().st_mtime}\n')
    f.write('=' * 60 + '\n\n')
    f.write('檢查統計:\n')
    f.write(f'  總文件數: {files_checked}\n')
    f.write(f'  問題行數: {len(issues)}\n\n')

    if issues:
        f.write('問題詳情:\n')
        f.write(f'{"文件路徑":<50} {"行號":>6} {"問題內容"}\n')
        f.write('-' * 100 + '\n')

        for path_str, line_num, line_content, _error_msg in issues:
            # 截斷過長的行內容
            display_line = line_content[:60] + '...' if len(line_content) > 60 else line_content
            f.write(f'{path_str:<50} {line_num:>6} {display_line}\n')
    else:
        f.write('[SUCCESS] 所有文件都與 CP950 編碼兼容！\n')

print(f'報告已生成: {out}')
