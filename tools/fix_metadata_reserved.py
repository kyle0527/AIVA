"""修復 storage/models.py 中的 metadata 保留字問題"""
import re

file_path = r'C:\D\fold7\AIVA-git\services\core\aiva_core\storage\models.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替換所有的 'metadata = Column' 為 'extra_metadata = Column("metadata",'
# 使用正則表達式
content = re.sub(
    r'^(\s+)metadata = Column\(JSON, nullable=True\)',
    r'\1extra_metadata = Column("metadata", JSON, nullable=True)',
    content,
    flags=re.MULTILINE
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 已修復所有 metadata 欄位為 extra_metadata')
print(f'   檔案: {file_path}')

# 統計修復數量
count = content.count('extra_metadata = Column("metadata"')
print(f'   共修復: {count} 處')
