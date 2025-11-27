#!/usr/bin/env python3
"""修正錯誤的連結替換"""

import re
from pathlib import Path

# 需要修正的文件和錯誤連結映射
fixes = {
    r'C:\D\fold7\AIVA-git\services\scan\README.md': [
        {
            'old': 'docs/guides/services/rust_engine_USAGE_GUIDE.md',
            'new': '../../reports/architecture/COORDINATOR_USAGE_GUIDE.md',
            'context': 'COORDINATOR_USAGE_GUIDE'
        }
    ],
    r'C:\D\fold7\AIVA-git\services\features\README.md': [
        {
            'old': '../../docs/guides/services/rust_engine_USAGE_GUIDE.md',
            'new': '../../guides/development/METRICS_USAGE_GUIDE.md',
            'context': '性能監控規範'
        }
    ],
    r'C:\D\fold7\AIVA-git\services\scan\coordinators\README.md': [
        {
            'old': '../../../docs/guides/services/rust_engine_USAGE_GUIDE.md',
            'new': './COORDINATOR_USAGE_GUIDE.md',
            'line_contains': '完整指南'
        },
        {
            'old': '../../../docs/guides/services/rust_engine_USAGE_GUIDE.md',
            'new': './COORDINATOR_USAGE_GUIDE.md',
            'line_contains': '完整使用指南'
        },
        {
            'old': '../../../docs/guides/services/rust_engine_USAGE_GUIDE.md',
            'new': './PYTHON_ENGINE_USAGE_GUIDE.md',
            'line_contains': 'PYTHON_ENGINE_USAGE_GUIDE'
        },
    ],
    r'C:\D\fold7\AIVA-git\services\integration\docs\README.md': [
        {
            'old': '../../../docs/guides/services/rust_engine_USAGE_GUIDE.md',
            'new': '../../features/INTEGRATION_USAGE_GUIDE.md',
            'context': 'Features 模組整合指南'
        }
    ],
}

def fix_file(file_path: str, replacements: list):
    """修正單個文件"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"⚠️ 文件不存在: {file_path}")
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 無法讀取 {file_path}: {e}")
        return False
    
    original_content = content
    fixed_count = 0
    
    for replacement in replacements:
        old = replacement['old']
        new = replacement['new']
        
        # 如果有特定行上下文，只替換符合的行
        if 'line_contains' in replacement:
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if replacement['line_contains'] in line and old in line:
                    new_lines.append(line.replace(old, new))
                    fixed_count += 1
                else:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
        elif 'context' in replacement:
            # 查找包含上下文的段落並替換
            context = replacement['context']
            # 使用更精確的匹配
            pattern = rf'(\[{re.escape(context)}[^\]]*\]\()({re.escape(old)})(\))'
            if re.search(pattern, content):
                content = re.sub(pattern, rf'\1{new}\3', content)
                fixed_count += 1
            # 也處理表格中的情況
            pattern2 = rf'(\| .{0,50}{re.escape(context)}.{{0,50}}\| ){re.escape(old)}'
            if re.search(pattern2, content):
                content = re.sub(pattern2, rf'\1{new}', content)
                fixed_count += 1
    
    if content != original_content:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ {path.name} - 修正 {fixed_count} 處")
            return True
        except Exception as e:
            print(f"⚠️ 無法寫入 {file_path}: {e}")
            return False
    else:
        print(f"ℹ️ {path.name} - 無需修正")
        return False

def main():
    print("="*80)
    print("🔧 修正錯誤的連結替換")
    print("="*80)
    print()
    
    fixed_files = 0
    total_files = len(fixes)
    
    for file_path, replacements in fixes.items():
        if fix_file(file_path, replacements):
            fixed_files += 1
    
    print()
    print("="*80)
    print(f"📊 修正完成: {fixed_files}/{total_files} 個文件")
    print("="*80)

if __name__ == "__main__":
    main()
