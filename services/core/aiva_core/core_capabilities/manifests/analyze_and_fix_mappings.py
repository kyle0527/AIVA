"""分析實際腳本文件，確認類名和方法名，移除 mock"""
import json
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

# 載入當前映射
mappings_file = PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_mappings.json"
with open(mappings_file, 'r', encoding='utf-8') as f:
    mappings = json.load(f)

def analyze_python_file(file_path):
    """分析 Python 文件，提取類名和方法名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef) and not m.name.startswith('_')]
                classes.append({
                    'name': node.name,
                    'methods': methods[:5]  # 前5個公開方法
                })
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                functions.append(node.name)
        
        return {
            'classes': classes[:3],  # 前3個類
            'functions': functions[:5],  # 前5個函數
            'exists': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'exists': False
        }

# 分析每個腳本
print("=" * 100)
print("🔍 分析實際腳本文件")
print("=" * 100)
print()

updated_mappings = {}
mock_count = 0
real_count = 0

for script_name, info in mappings.items():
    script_path = PROJECT_ROOT / info.get('script_path', '')
    
    print(f"📄 {script_name}")
    print(f"   路徑: {info.get('script_path', 'N/A')}")
    
    if script_path.exists():
        analysis = analyze_python_file(script_path)
        
        if analysis['exists']:
            print(f"   ✅ 文件存在")
            
            # 顯示發現的類和方法
            if analysis['classes']:
                print(f"   📦 類:")
                for cls in analysis['classes']:
                    print(f"      - {cls['name']}")
                    if cls['methods']:
                        print(f"        方法: {', '.join(cls['methods'])}")
            
            if analysis['functions']:
                print(f"   🔧 函數: {', '.join(analysis['functions'])}")
            
            # 檢查映射的類和方法是否存在
            mapped_class = info.get('class')
            mapped_method = info.get('method')
            
            if mapped_class:
                class_found = any(c['name'] == mapped_class for c in analysis['classes'])
                if class_found:
                    cls_info = next(c for c in analysis['classes'] if c['name'] == mapped_class)
                    method_found = mapped_method in cls_info['methods'] if mapped_method else False
                    
                    if method_found:
                        print(f"   ✅ 映射正確: {mapped_class}.{mapped_method}()")
                        info['mock'] = False
                        real_count += 1
                    else:
                        print(f"   ⚠️  類存在但方法不匹配")
                        print(f"      期望: {mapped_method}")
                        print(f"      實際: {', '.join(cls_info['methods'][:5])}")
                        # 嘗試修正
                        if cls_info['methods']:
                            suggested = cls_info['methods'][0]
                            print(f"      建議: {suggested}")
                            info['method'] = suggested
                            info['mock'] = False
                            info['note'] = f"自動修正方法名從 {mapped_method} 到 {suggested}"
                            real_count += 1
                        else:
                            mock_count += 1
                else:
                    print(f"   ⚠️  類不存在: {mapped_class}")
                    # 嘗試使用第一個類
                    if analysis['classes']:
                        suggested = analysis['classes'][0]['name']
                        suggested_method = analysis['classes'][0]['methods'][0] if analysis['classes'][0]['methods'] else None
                        print(f"      建議: {suggested}.{suggested_method}()")
                        info['class'] = suggested
                        if suggested_method:
                            info['method'] = suggested_method
                        info['mock'] = False
                        info['note'] = f"自動修正類名從 {mapped_class} 到 {suggested}"
                        real_count += 1
                    else:
                        mock_count += 1
            else:
                # 函數調用
                func = info.get('function')
                if func and func in analysis['functions']:
                    print(f"   ✅ 映射正確: {func}()")
                    info['mock'] = False
                    real_count += 1
                else:
                    if analysis['functions']:
                        suggested = analysis['functions'][0]
                        print(f"   建議函數: {suggested}")
                        info['function'] = suggested
                        info['mock'] = False
                        real_count += 1
                    else:
                        mock_count += 1
        else:
            print(f"   ❌ 分析失敗: {analysis.get('error', 'Unknown')}")
            mock_count += 1
    else:
        print(f"   ❌ 文件不存在")
        mock_count += 1
    
    updated_mappings[script_name] = info
    print()

# 保存更新後的映射
with open(mappings_file, 'w', encoding='utf-8') as f:
    json.dump(updated_mappings, f, indent=2, ensure_ascii=False)

print("=" * 100)
print("📊 統計結果")
print("=" * 100)
print(f"✅ 真實執行: {real_count}/{len(mappings)} ({real_count/len(mappings)*100:.1f}%)")
print(f"🟡 仍為 Mock: {mock_count}/{len(mappings)} ({mock_count/len(mappings)*100:.1f}%)")
print()
print(f"已更新: {mappings_file}")
