"""
三個功能模組命令系統驗證腳本

驗證 handler.py 文件的結構和接口是否符合規範
"""

import ast
import os
from pathlib import Path


def validate_handler_file(handler_path: str, expected_class_name: str) -> dict:
    """驗證 handler 文件結構"""
    result = {
        "file_exists": False,
        "class_found": False,
        "has_handle_command": False,
        "imports_aiva_common": False,
        "imports_manager": False,
        "errors": []
    }
    
    # 檢查文件是否存在
    if not os.path.exists(handler_path):
        result["errors"].append(f"文件不存在: {handler_path}")
        return result
    
    result["file_exists"] = True
    
    try:
        # 讀取並解析文件
        with open(handler_path, encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # 檢查導入
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'aiva_common' in node.module:
                    result["imports_aiva_common"] = True
                if node.module and 'manager' in node.module:
                    result["imports_manager"] = True
        
        # 檢查類定義
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == expected_class_name:
                    result["class_found"] = True
                    
                    # 檢查是否有 handle_command 方法
                    for item in node.body:
                        if isinstance(item, ast.AsyncFunctionDef):
                            if item.name == "handle_command":
                                result["has_handle_command"] = True
                                
                                # 檢查參數
                                args = [arg.arg for arg in item.args.args]
                                if 'self' in args and 'command' in args:
                                    result["handle_command_params"] = "✅ 正確"
                                else:
                                    result["handle_command_params"] = "❌ 參數不完整"
        
        if not result["class_found"]:
            result["errors"].append(f"未找到類: {expected_class_name}")
        
        if result["class_found"] and not result["has_handle_command"]:
            result["errors"].append(f"類 {expected_class_name} 缺少 handle_command 方法")
        
    except SyntaxError as e:
        result["errors"].append(f"語法錯誤: {e}")
    except Exception as e:
        result["errors"].append(f"解析錯誤: {e}")
    
    return result


def main():
    """主驗證流程"""
    base_path = Path(__file__).parent
    
    modules = [
        {
            "name": "function_payload_generator",
            "handler_class": "PayloadGeneratorCommandHandler",
            "path": base_path / "function_payload_generator" / "handler.py"
        },
        {
            "name": "function_wordlist_generator",
            "handler_class": "WordlistGeneratorCommandHandler",
            "path": base_path / "function_wordlist_generator" / "handler.py"
        },
        {
            "name": "function_social_engineering",
            "handler_class": "SocialEngineeringCommandHandler",
            "path": base_path / "function_social_engineering" / "handler.py"
        }
    ]
    
    print("=" * 80)
    print("三個功能模組命令系統結構驗證")
    print("=" * 80)
    print()
    
    all_passed = True
    
    for module in modules:
        print(f"📦 驗證模組: {module['name']}")
        print("-" * 80)
        
        result = validate_handler_file(str(module['path']), module['handler_class'])
        
        # 顯示結果
        status_icon = "✅" if result["file_exists"] else "❌"
        print(f"  {status_icon} 文件存在: {result['file_exists']}")
        
        status_icon = "✅" if result["class_found"] else "❌"
        print(f"  {status_icon} 類定義: {result['class_found']} ({module['handler_class']})")
        
        status_icon = "✅" if result["has_handle_command"] else "❌"
        print(f"  {status_icon} handle_command 方法: {result['has_handle_command']}")
        
        if "handle_command_params" in result:
            print(f"    └─ 參數檢查: {result['handle_command_params']}")
        
        status_icon = "✅" if result["imports_aiva_common"] else "❌"
        print(f"  {status_icon} 導入 aiva_common: {result['imports_aiva_common']}")
        
        status_icon = "✅" if result["imports_manager"] else "❌"
        print(f"  {status_icon} 導入 manager: {result['imports_manager']}")
        
        if result["errors"]:
            print("  ⚠️  錯誤:")
            for error in result["errors"]:
                print(f"    - {error}")
            all_passed = False
        
        print()
    
    # 總結
    print("=" * 80)
    if all_passed:
        print("✅ 所有模組結構驗證通過！")
    else:
        print("❌ 部分模組存在問題，請查看上方錯誤信息")
    print("=" * 80)
    
    # 檢查 CommandType 擴展
    print("\n📋 檢查 CommandType 擴展")
    print("-" * 80)
    
    commands_path = base_path.parent / "aiva_common" / "schemas" / "commands.py"
    required_types = [
        "FEATURE_PAYLOAD_GENERATE",
        "FEATURE_WORDLIST_GENERATE",
        "FEATURE_PHISHING_CAMPAIGN",
        "FEATURE_OSINT_COLLECT"
    ]
    
    if commands_path.exists():
        with open(commands_path, encoding='utf-8') as f:
            content = f.read()
        
        for cmd_type in required_types:
            if cmd_type in content:
                print(f"  ✅ {cmd_type}")
            else:
                print(f"  ❌ {cmd_type} (未找到)")
                all_passed = False
    else:
        print(f"  ⚠️  commands.py 文件不存在: {commands_path}")
    
    print()
    print("=" * 80)
    print("驗證完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
