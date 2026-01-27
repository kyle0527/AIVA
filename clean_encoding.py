import re

file_path = r"C:\D\fold7\AIVA-git\services\features\function_info_leak\sensitive_info_detector.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替換所有亂碼註解
replacements = {
    '"""瑼Ｘ葫批捆銝剔靽⊥瘣拚': '"""Sensitive Information Detector',
    '瑼Ｘ葫批捆銝剔靽⊥瘣拚': 'Sensitive Information Detector',
    '銝餉:': 'Features:',
    '- 瑼Ｘ葫 API keys, tokens, passwords 蝑霅霅': '- Detect API keys, tokens, passwords',
    '- 瑼Ｘ葫犖霅靽⊥ (PII): email, phone, SSN, credit card': '- Detect PII: email, phone, SSN, credit card',
    '- 瑼Ｘ葫脫霅(AWS, GCP, Azure)': '- Detect cloud credentials (AWS, GCP, Azure)',
    '- 瑼Ｘ葫豢摨恍摮泵銝': '- Detect database connection strings',
    '- 瑼Ｘ葫折頝臬頂蝯曹縑': '- Detect internal IP addresses',
    '- 瑼Ｘ葫 debug 靽⊥隤文璉': '- Detect debug information and stack traces',
    '- 瑼Ｘ葫 HTML 閮駁銝剔靽⊥': '- Detect sensitive info in HTML comments',
    '雿輻蝭:': 'Example:',
    '# 瑼Ｘ葫 HTML 批捆': '# Detect in HTML',
    '# 瑼Ｘ葫踵': '# Detect in headers',
    '# 瑼Ｘ葫 JavaScript': '# Detect in JavaScript',
    '炎皜砍': 'Initialize detector',
    'Args:': 'Args:',
    'min_severity: 雿摨雿甇斤亦寥撠◤蕪': 'min_severity: Minimum severity level to report',
    '"""': '"""'
}

# 替換所有包含亂碼的註解為英文
lines = content.split('\n')
cleaned_lines = []

for line in lines:
    # 檢查是否包含亂碼字元（非 ASCII 且非常見中文）
    if re.search(r'[\u4e00-\u9fff]', line):
        # 如果是註解行或 docstring，替換為英文
        if '"""' in line or "'''" in line or line.strip().startswith('#'):
            # 移除亂碼，保留結構
            if '"""' in line:
                indent = len(line) - len(line.lstrip())
                if line.strip() == '"""':
                    cleaned_lines.append(line)
                else:
                    # 簡單替換為 placeholder
                    cleaned_lines.append(' ' * indent + '"""TODO: Add docstring"""')
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    else:
        cleaned_lines.append(line)

cleaned_content = '\n'.join(cleaned_lines)

# 備份原文件
import shutil
shutil.copy(file_path, file_path + '.backup')

# 寫入清理後的內容
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("✅ 文件已清理")
print("📁 備份: sensitive_info_detector.py.backup")
