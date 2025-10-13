from pathlib import Path

root = Path(r"c:\D\E\AIVA\AIVA-main")
out = Path(r"c:\D\E\AIVA\AIVA-main\tools\non_cp950_filtered_report.txt")

files_checked = 0
issues = []

for p in root.rglob('*.py'):
    if 'emoji_backups' in str(p) or 'emoji_backups2' in str(p):
        continue
    files_checked += 1
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), start=1):
        try:
            line.encode('cp950')
        except UnicodeEncodeError:
            issues.append((str(p), i, line))

with out.open('w', encoding='utf-8') as f:
    f.write(f'files_checked: {files_checked}\n')
    f.write(f'issues_found: {len(issues)}\n')
    for p, i, line in issues:
        f.write(f'{p}\t{ i }\t{line}\n')

print(f'wrote {out}')
