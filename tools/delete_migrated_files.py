"""
安全刪除已遷移和備份的檔案
"""

from pathlib import Path
import shutil
from datetime import datetime

# 可以安全刪除的檔案列表
files_to_delete = [
    # 備份檔案
    '__init___backup.py',
    '__init___fixed.py',
    '__init___old.py',
    'schemas_backup.py',
    'schemas_backup_20251016_072549.py',
    'schemas_broken.py',
    'schemas_compat.py',
    'schemas_current_backup.py',
    'schemas_fixed.py',
    'schemas_master_backup_1.py',
    'schemas_master_backup_2.py.disabled',
    # 已遷移檔案
    'ai_schemas.py',  # 已完全遷移到 schemas/ai.py
]

aiva_common = Path('.')

print("=" * 70)
print("檔案刪除確認")
print("=" * 70)

# 顯示將要刪除的檔案
total_size = 0
existing_files = []

for filename in files_to_delete:
    file_path = aiva_common / filename
    if file_path.exists():
        size = file_path.stat().st_size
        total_size += size
        existing_files.append((filename, size))
        print(f"[CHECK] {filename:50s} {size:>10,} bytes")
    else:
        print(f"[U+2717] {filename:50s} (不存在)")

print("=" * 70)
print(f"總計: {len(existing_files)} 個檔案, {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
print("=" * 70)

# 確認刪除
response = input("\n確定要刪除這些檔案嗎? (yes/no): ").strip().lower()

if response == 'yes':
    # 創建備份目錄
    backup_dir = Path('_deleted_backups') / datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    deleted_count = 0
    for filename, size in existing_files:
        file_path = aiva_common / filename
        backup_path = backup_dir / filename
        
        try:
            # 先備份到 _deleted_backups 以防萬一
            shutil.copy2(file_path, backup_path)
            # 刪除原檔案
            file_path.unlink()
            print(f"[CHECK] 已刪除: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"[U+2717] 刪除失敗 {filename}: {e}")
    
    print("\n" + "=" * 70)
    print(f"完成! 已刪除 {deleted_count} 個檔案")
    print(f"備份位置: {backup_dir.absolute()}")
    print("=" * 70)
else:
    print("\n取消刪除操作")
