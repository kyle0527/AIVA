"""列出需要實作的無功能檔案"""
import json

# 讀取功能分析報告
with open(r'C:\D\fold7\AIVA-git\_out\script_functionality_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 取得無功能的檔案 (排除 __init__.py)
no_func_files = []
for item in data.get('no_functionality', []):
    file_path = item['file']
    if '__init__.py' not in file_path:
        no_func_files.append({
            'path': file_path,
            'reason': item.get('reason', ''),
            'lines': item.get('code_lines', 0)
        })

print(f'需要實作的檔案數量: {len(no_func_files)}')
print('\n檔案清單:')
for i, file_info in enumerate(no_func_files, 1):
    rel_path = file_info['path'].replace('C:\\D\\fold7\\AIVA-git\\', '')
    print(f'{i:2d}. {rel_path}')
    print(f'    原因: {file_info["reason"]}')
    print()
