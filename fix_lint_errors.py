#!/usr/bin/env python3
"""
批量修復 Lint 錯誤腳本
快速修復剩餘的 lint 問題
"""

import re
from pathlib import Path

def fix_enum_init_comments():
    """修復 enums/__init__.py 中的註釋"""
    file_path = Path("services/aiva_common/enums/__init__.py")
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除導入行中的註釋
    content = re.sub(
        r'from \.[a-z_]+ import \({2}# [^)]+',
        lambda m: m.group(0).split(' # ')[0],
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已修復 {file_path}")

def fix_health_check_exceptions():
    """修復 health_check.py 中的異常處理"""
    file_path = Path("scripts/utilities/health_check.py")
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替換空的 except 為具體異常
    content = content.replace('    except:', '    except Exception:')
    content = content.replace('            except:', '            except Exception:')
    
    # 定義錯誤消息常量
    error_constant = '❌ 未安裝或不可用'
    content = content.replace(f'"{error_constant}"', 'NOT_AVAILABLE')
    
    # 在文件開頭添加常量
    if 'NOT_AVAILABLE = ' not in content:
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                insert_pos = i + 1
        
        lines.insert(insert_pos, f'NOT_AVAILABLE = "{error_constant}"')
        content = '\n'.join(lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已修復 {file_path}")

def main():
    """主函數"""
    print("🔧 開始批量修復 lint 錯誤...")
    
    try:
        fix_enum_init_comments()
        fix_health_check_exceptions()
        print("✅ 所有修復完成！")
        
    except Exception as e:
        print(f"❌ 修復過程中出現錯誤: {e}")

if __name__ == "__main__":
    main()