#!/usr/bin/env python3
"""
AIVA 導入路徑修復工具
用途: 修復移動後腳本的導入路徑問題
"""

import os
import re
import sys
from pathlib import Path

def fix_import_paths():
    """修復所有腳本的導入路徑"""
    
    project_root = Path(__file__).parent.parent.parent
    scripts_dir = project_root / "scripts"
    
    print("🔧 AIVA 導入路徑修復工具")
    print("=" * 50)
    print(f"📁 項目根目錄: {project_root}")
    
    # 需要修復的文件模式
    files_to_fix = []
    
    # 掃描所有Python文件
    for root, dirs, files in os.walk(scripts_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                files_to_fix.append(file_path)
    
    print(f"📄 找到 {len(files_to_fix)} 個Python文件需要檢查")
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修復常見的路徑問題
            patterns = [
                # 修復相對路徑導入
                (r'sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\)\)', 
                 'project_root = Path(__file__).parent.parent.parent\nsys.path.insert(0, str(project_root))'),
                
                (r'sys\.path\.append\(str\(Path\(__file__\)\.parent\.parent\.parent\)\)',
                 'project_root = Path(__file__).parent.parent.parent\nsys.path.append(str(project_root))'),
                
                # 修復直接導入
                (r'from scripts.testing.real_attack_executor import',
                 'from scripts.testing.real_attack_executor import'),
                
                (r'from scripts.testing.enhanced_real_ai_attack_system import',
                 'from scripts.testing.enhanced_real_ai_attack_system import'),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
            
            # 如果有變化，寫回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 修復: {file_path.relative_to(project_root)}")
                fixed_count += 1
            else:
                print(f"⚪ 無需修復: {file_path.relative_to(project_root)}")
                
        except Exception as e:
            print(f"❌ 修復失敗 {file_path.relative_to(project_root)}: {e}")
    
    print()
    print(f"🎯 修復完成! 共修復 {fixed_count} 個文件")
    
    # 檢查是否有遺漏的導入問題
    print("\n🔍 檢查導入問題...")
    test_imports()

def test_imports():
    """測試關鍵模組的導入"""
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    test_modules = [
        'services.aiva_common.enums.modules',
        'services.scan.aiva_scan',
        'services.features.high_value_manager'
    ]
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

if __name__ == "__main__":
    fix_import_paths()