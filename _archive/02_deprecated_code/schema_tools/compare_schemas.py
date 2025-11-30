"""
比對舊檔案與新 schemas/ 資料夾的內容
"""

import re
from pathlib import Path


def extract_classes(content):
    """提取所有類別名稱"""
    pattern = r'^class\s+(\w+)\s*\('
    return set(re.findall(pattern, content, re.MULTILINE))


def compare_files(old_name, new_name, description):
    """比對兩個檔案的類別差異"""
    print(f"\n📄 比對 {description}")
    print("-" * 40)
    
    old_path = Path(old_name)
    new_path = Path(new_name)
    
    if not old_path.exists():
        print(f"⚠️ 舊檔案不存在: {old_path}")
        return False
        
    if not new_path.exists():
        print(f"⚠️ 新檔案不存在: {new_path}")
        return False
    
    old_content = old_path.read_text(encoding='utf-8')
    new_content = new_path.read_text(encoding='utf-8')
    
    old_classes = extract_classes(old_content)
    new_classes = extract_classes(new_content)
    
    print(f'  舊檔案 ({old_path}):')
    print(f'    - 類別數量: {len(old_classes)}')
    
    print(f'  新檔案 ({new_path}):')
    print(f'    - 類別數量: {len(new_classes)}')
    
    only_in_old = old_classes - new_classes
    only_in_new = new_classes - old_classes
    common = old_classes & new_classes
    
    print('\n  🔍 比對結果:')
    print(f'    - 共同類別: {len(common)}')
    
    if only_in_old:
        print(f'    - ⚠️  僅在舊檔案: {len(only_in_old)} 個')
        print(f'      {sorted(only_in_old)}')
        return False
    else:
        print('    - ✅ 舊檔案的所有類別已遷移')
    
    if only_in_new:
        print(f'    - 📝 新增類別: {len(only_in_new)} 個')
        
    return True


def main():
    """主要比對功能"""
    # 使用相對路徑，從項目根目錄計算
    project_root = Path(__file__).parent.parent.parent
    aiva_common = project_root / "services" / "aiva_common"
    
    # 切換到正確的工作目錄
    import os
    os.chdir(aiva_common)
    
    print("=" * 60)
    print("檔案遷移比對報告")
    print("=" * 60)

    # 比對 ai_schemas.py
    compare_files(
        'ai_schemas.py',
        'schemas/ai.py',
        'AI Schemas'
    )

    # 比對 schemas.py (檢查是否所有內容都已拆分)
    print("\n" + "=" * 60)
    print("schemas.py 拆分檢查")
    print("=" * 60)

    schemas_path = Path('schemas.py')
    if not schemas_path.exists():
        print("⚠️ schemas.py 檔案不存在")
        return

    schemas_content = schemas_path.read_text(encoding='utf-8')
    schemas_classes = extract_classes(schemas_content)

    # 檢查新 schemas/ 資料夾中的所有類別
    new_schemas_dir = Path('schemas')
    all_new_classes = set()

    if new_schemas_dir.exists():
        for py_file in new_schemas_dir.glob('*.py'):
            if py_file.name == '__init__.py':
                continue
            content = py_file.read_text(encoding='utf-8')
            classes = extract_classes(content)
            all_new_classes.update(classes)
            print(f'\n  📄 {py_file.name}: {len(classes)} 個類別')
    
    print('\n📊 總計:')
    print(f'  - 原 schemas.py: {len(schemas_classes)} 個類別')
    print(f'  - 新 schemas/: {len(all_new_classes)} 個類別')

    missing = schemas_classes - all_new_classes
    if missing:
        print(f'\n⚠️  尚未遷移的類別 ({len(missing)} 個):')
        for cls in sorted(missing):
            print(f'    - {cls}')
    else:
        print('\n✅ schemas.py 的所有類別已遷移')

    # 可以刪除的檔案列表
    print("\n" + "=" * 60)
    print("建議刪除的檔案")
    print("=" * 60)
    
    # 檢查可以刪除的舊檔案
    deletable_files = [
        'ai_schemas.py',
        'schemas.py'
    ]
    
    for file_name in deletable_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  - ✅ {file_name}")
        else:
            print(f"  - ⚠️ {file_name} (已不存在)")


if __name__ == "__main__":
    main()