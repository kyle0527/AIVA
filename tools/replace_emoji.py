from pathlib import Path
import shutil

root = Path(r"c:\D\E\AIVA\AIVA-main")
backup_root = root / 'emoji_backups'
backup_root.mkdir(exist_ok=True)

# mapping: emoji -> replacement (prefer Chinese or ASCII)
repl = {
    '[已]': '[已]',
    '[失敗]': '[失敗]',
    '[警告]': '[警告]',
    '[警告]': '[警告]',
    '[目錄]': '[目錄]',
    '[完成]': '[完成]',
    '[啟動]': '[啟動]',
    '[統計]': '[統計]',
    '[循環]': '[循環]',
    '[API]': '[連線]',
    '[接收]': '[接收]',
    '[列表]': '[列表]',
    '[目標]': '[目標]',
    '[記錄]': '[記錄]',
    '[設定]': '[設定]',
    '[設定]': '[設定]',
    '[調整]': '[調整]',
    '[快速]': '[快速]',
    '[監控]': '[監控]',
    '[資訊]': '[資訊]',
    '[X]': '[X]',
    '[V]': '[V]',
    '[資料]': '[資料]',
    '[數值]': '[數值]',
    '[神經網路]': '[神經網路]',
    '[警報]': '[警報]',
    '[說明]': '[說明]',
    '[API]': '[API]',
}

py_files = list(root.rglob('*.py'))
modified = []
for p in py_files:
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        continue
    new_text = text
    for k, v in repl.items():
        if k in new_text:
            new_text = new_text.replace(k, v)
    if new_text != text:
        # backup
        rel = p.relative_to(root)
        backup_path = backup_root / (str(rel).replace('\\', '__'))
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, backup_path)
        p.write_text(new_text, encoding='utf-8')
        modified.append(str(p))

out = root / 'tools' / 'replace_emoji_out.txt'
with out.open('w', encoding='utf-8') as f:
    f.write(f'files_modified: {len(modified)}\n')
    for m in modified:
        f.write(m + '\n')

print('wrote', out)
