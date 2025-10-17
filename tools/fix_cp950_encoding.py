#!/usr/bin/env python3
"""
修正 CP950 編碼問題
將所有無法用 CP950 編碼的字符替換為 ASCII 兼容的文字描述
"""

from pathlib import Path
import shutil
from datetime import datetime

# 使用當前腳本所在目錄作為基準
script_dir = Path(__file__).parent
root = script_dir.parent  # AIVA 根目錄
backup_root = root / 'emoji_backups_cp950'
backup_root.mkdir(exist_ok=True)

print(f"掃描目錄: {root}")
print(f"備份目錄: {backup_root}")
print("-" * 60)

# Emoji 和特殊字符的映射表
EMOJI_MAP = {
    '[OK]': '[OK]',
    '[FAIL]': '[FAIL]',
    '[WARN]': '[WARN]',
    '[TEST]': '[TEST]',
    '[START]': '[START]',
    '[STATS]': '[STATS]',
    '[LIST]': '[LIST]',
    '[NOTE]': '[NOTE]',
    '[SAVE]': '[SAVE]',
    '[SEARCH]': '[SEARCH]',
    '[CONFIG]': '[CONFIG]',
    '[TARGET]': '[TARGET]',
    '[UI]': '[UI]',
    '[AI]': '[AI]',
    '[CHAT]': '[CHAT]',
    '[MIX]': '[MIX]',
    '[BRAIN]': '[BRAIN]',
    '[LOG]': '[LOG]',
    '[RELOAD]': '[RELOAD]',
    '[SPARKLE]': '[SPARKLE]',
    '[DOCS]': '[DOCS]',
    '[TIP]': '[TIP]',
    '[STOP]': '[STOP]',
    '[TIME]': '[TIME]',
    '[SUCCESS]': '[SUCCESS]',
    '[INFO]': '[INFO]',
    '[IMAGE]': '[IMAGE]',
    '[SHIELD]': '[SHIELD]',
    '[FAST]': '[FAST]',
    '[SECURE]': '[SECURE]',
    '[CLOUD]': '[CLOUD]',
    '[ALERT]': '[ALERT]',
    '[LOCK]': '[LOCK]',
    '[SPY]': '[SPY]',
    '[PIN]': '[PIN]',
    '[YELLOW]': '[YELLOW]',
    '[RED]': '[RED]',
    '[CHECK]': '[CHECK]',
    '<->': '<->',
    '→': '->',
    '←': '<-',
}

py_files = [
    p for p in root.rglob('*.py') 
    if not any(skip in str(p) for skip in ['emoji_backups', '__pycache__', '.git', 'node_modules'])
]

modified = []
total_files = len(py_files)
files_checked = 0

print(f"找到 {total_files} 個 Python 文件")
print("開始處理...\n")

for p in py_files:
    files_checked += 1
    
    # 進度顯示
    if files_checked % 50 == 0:
        print(f"處理進度: {files_checked}/{total_files}")
    
    try:
        text = p.read_text(encoding='utf-8')
    except Exception as e:
        print(f"[U+26A0] 無法讀取 {p}: {e}")
        continue
    
    lines = text.splitlines()
    changed = False
    new_lines = []
    
    for line in lines:
        try:
            line.encode('cp950')
            new_lines.append(line)
        except UnicodeEncodeError:
            # 先嘗試使用映射表替換
            new_line = line
            for emoji, replacement in EMOJI_MAP.items():
                if emoji in new_line:
                    new_line = new_line.replace(emoji, replacement)
                    changed = True
            
            # 檢查是否還有無法編碼的字符
            try:
                new_line.encode('cp950')
                new_lines.append(new_line)
            except UnicodeEncodeError:
                # 替換剩餘無法編碼的字符
                new_line_chars = []
                for ch in new_line:
                    try:
                        ch.encode('cp950')
                        new_line_chars.append(ch)
                    except UnicodeEncodeError:
                        # 替換為 Unicode 碼點
                        new_line_chars.append(f'[U+{ord(ch):04X}]')
                        changed = True
                new_lines.append(''.join(new_line_chars))
    
    if changed:
        # 備份原文件
        rel = p.relative_to(root)
        backup_path = backup_root / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, backup_path)
        
        # 寫入修正後的內容
        new_text = '\n'.join(new_lines)
        if text.endswith('\n'):
            new_text += '\n'
        p.write_text(new_text, encoding='utf-8')
        modified.append(str(p.relative_to(root)))
        print(f"[CHECK] 修正: {p.relative_to(root)}")

# 生成報告
print(f"\n處理完成！")
print(f"檢查文件: {files_checked}")
print(f"修正文件: {len(modified)}")

report_path = root / 'tools' / 'cp950_fix_report.txt'
with report_path.open('w', encoding='utf-8') as f:
    f.write('CP950 編碼修正報告\n')
    f.write('=' * 60 + '\n')
    f.write(f'處理時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'掃描目錄: {root}\n')
    f.write(f'備份目錄: {backup_root}\n')
    f.write('\n')
    f.write(f'檢查文件數: {files_checked}\n')
    f.write(f'修正文件數: {len(modified)}\n')
    f.write('\n')
    
    if modified:
        f.write('修正的文件:\n')
        f.write('-' * 60 + '\n')
        for file_path in sorted(modified):
            f.write(f'{file_path}\n')
    else:
        f.write('沒有文件需要修正\n')

print(f'\n報告已生成: {report_path}')
print(f'備份位置: {backup_root}')
