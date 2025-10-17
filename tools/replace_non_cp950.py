from pathlib import Path
import shutil

# 使用當前腳本所在目錄作為基準
script_dir = Path(__file__).parent
root = script_dir.parent  # AIVA 根目錄
backup_root = root / 'emoji_backups2'
backup_root.mkdir(exist_ok=True)

print(f"掃描目錄: {root}")
print(f"備份目錄: {backup_root}")

py_files = [p for p in root.rglob('*.py') if 'emoji_backups' not in str(p) and 'emoji_backups2' not in str(p)]
modified = []

for p in py_files:
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        continue
    lines = text.splitlines()
    changed = False
    new_lines = []
    for line in lines:
        try:
            line.encode('cp950')
            new_lines.append(line)
        except UnicodeEncodeError:
            # replace each offending character
            new_line_chars = []
            for ch in line:
                try:
                    ch.encode('cp950')
                    new_line_chars.append(ch)
                except UnicodeEncodeError:
                    # replace with bracketed unicode codepoint
                    new_line_chars.append(f'[U+{ord(ch):04X}]')
                    changed = True
            new_lines.append(''.join(new_line_chars))
    if changed:
        # backup
        rel = p.relative_to(root)
        backup_path = backup_root / (str(rel).replace('\\', '__'))
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, backup_path)
        p.write_text('\n'.join(new_lines) + ('\n' if text.endswith('\n') else ''), encoding='utf-8')
        modified.append(str(p))

out = root / 'tools' / 'replace_non_cp950_out.txt'
with out.open('w', encoding='utf-8') as f:
    f.write(f'files_modified: {len(modified)}\n')
    for m in modified:
        f.write(m + '\n')

print(f'已修正 {len(modified)} 個文件')
print('報告已生成:', out)
