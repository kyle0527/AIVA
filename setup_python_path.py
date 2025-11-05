#!/usr/bin/env python3
"""
AIVA Python 路徑設定腳本
設定正確的 PYTHONPATH 以確保所有模組可以正確導入
"""

import os
import sys
from pathlib import Path

def setup_aiva_python_path():
    """設定 AIVA 專案的 Python 路徑"""
    # 獲取專案根目錄
    project_root = Path(__file__).parent.absolute()
    
    # 需要添加到 PYTHONPATH 的路徑列表
    paths_to_add = [
        project_root,  # 專案根目錄
        project_root / "services",  # services 目錄
        project_root / "services" / "features",  # features 目錄
        project_root / "services" / "aiva_common",  # aiva_common 目錄
        project_root / "api",  # API 目錄
        project_root / "cli",  # CLI 目錄
        project_root / "config",  # 配置目錄
    ]
    
    # 添加到 sys.path
    for path in paths_to_add:
        path_str = str(path)
        if path_str not in sys.path and path.exists():
            sys.path.insert(0, path_str)
            print(f"✅ 已添加到 PYTHONPATH: {path_str}")
    
    # 設定環境變數
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [str(p) for p in paths_to_add if p.exists()]
    
    if current_pythonpath:
        new_pythonpath = os.pathsep.join(new_paths + [current_pythonpath])
    else:
        new_pythonpath = os.pathsep.join(new_paths)
    
    os.environ['PYTHONPATH'] = new_pythonpath
    
    print("\n📋 新的 PYTHONPATH 已設定:")
    for path in new_paths:
        print(f"   • {path}")
    
    return new_paths

def test_imports():
    """測試關鍵模組的導入"""
    print("\n🧪 測試關鍵模組導入...")
    
    test_modules = [
        "services.aiva_common.utils.logging",
        "services.aiva_common.schemas",
        "services.features.function_sqli",
        "services.features.function_xss", 
        "services.features.function_idor",
        "services.features.function_ssrf",
    ]
    
    success_count = 0
    for module_name in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
    
    print(f"\n📊 導入測試結果: {success_count}/{len(test_modules)} 成功")
    return success_count == len(test_modules)

def generate_powershell_script():
    """生成 PowerShell 環境設定腳本"""
    project_root = Path(__file__).parent.absolute()
    
    paths_to_add = [
        project_root,
        project_root / "services",
        project_root / "services" / "features", 
        project_root / "services" / "aiva_common",
        project_root / "api",
        project_root / "cli",
        project_root / "config",
    ]
    
    existing_paths = [str(p) for p in paths_to_add if p.exists()]
    pythonpath_value = ";".join(existing_paths)
    
    ps_script = f'''# AIVA Python 環境設定腳本
# 執行此腳本以設定正確的 PYTHONPATH

$env:PYTHONPATH = "{pythonpath_value}"
Write-Host "✅ PYTHONPATH 已設定為:" -ForegroundColor Green
Write-Host $env:PYTHONPATH -ForegroundColor Yellow

# 驗證設定
python -c "import sys; print('\\n📋 當前 Python 路徑:'); [print(f'  • {{p}}') for p in sys.path[:10]]"
'''
    
    script_path = project_root / "setup_env.ps1"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(ps_script)
    
    print(f"\n💾 PowerShell 腳本已生成: {script_path}")
    return script_path

if __name__ == "__main__":
    print("🚀 AIVA Python 路徑設定工具")
    print("=" * 50)
    
    # 設定路徑
    paths = setup_aiva_python_path()
    
    # 測試導入
    import_success = test_imports()
    
    # 生成 PowerShell 腳本
    ps_script_path = generate_powershell_script()
    
    print("\n" + "=" * 50)
    if import_success:
        print("✅ Python 路徑設定成功！所有關鍵模組都可以正常導入。")
    else:
        print("⚠️  Python 路徑設定完成，但部分模組導入失敗，可能需要進一步檢查。")
    
    print("\n💡 提示: 執行以下命令設定環境變數:")
    print("   PowerShell: .\\setup_env.ps1")
    print("   或直接執行: python setup_python_path.py")