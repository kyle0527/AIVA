#!/usr/bin/env python3
"""測試 plugins 各組件的導入功能"""
import sys
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_sarif_converter():
    """測試 SARIF 轉換器"""
    try:
        from plugins.aiva_converters.converters.sarif_converter import SARIFConverter
        print("✅ SARIFConverter 導入成功")
        print(f"   - 工具名稱: {SARIFConverter.TOOL_NAME}")
        print(f"   - 版本: {SARIFConverter.TOOL_VERSION}")
        return True
    except Exception as e:
        print(f"❌ SARIFConverter 導入失敗: {e}")
        return False

def test_task_converter():
    """測試任務轉換器"""
    try:
        from plugins.aiva_converters.converters.task_converter import TaskConverter
        print("✅ TaskConverter 導入成功")
        tc = TaskConverter()
        print(f"   - 類別: {tc.__class__.__name__}")
        return True
    except Exception as e:
        print(f"❌ TaskConverter 導入失敗: {e}")
        return False

def test_docx_converter():
    """測試 DOCX 轉換器"""
    try:
        from plugins.aiva_converters.converters.docx_to_md_converter import DocxToMarkdownConverter
        print("✅ DocxToMarkdownConverter 導入成功")
        return True
    except ImportError as e:
        if "docx" in str(e):
            print("⚠️  DocxToMarkdownConverter 需要安裝: pip install python-docx")
        else:
            print(f"❌ DocxToMarkdownConverter 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ DocxToMarkdownConverter 導入失敗: {e}")
        return False

def test_schema_codegen():
    """測試 Schema 代碼生成器"""
    try:
        from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodeGenerator
        print("✅ SchemaCodeGenerator 導入成功")
        return True
    except ImportError as e:
        if "jinja2" in str(e):
            print("⚠️  SchemaCodeGenerator 需要安裝: pip install jinja2")
        else:
            print(f"❌ SchemaCodeGenerator 導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ SchemaCodeGenerator 導入失敗: {e}")
        return False

def test_typescript_generator():
    """測試 TypeScript 生成器"""
    try:
        from plugins.aiva_converters.core.typescript_generator import TypeScriptGenerator
        print("✅ TypeScriptGenerator 導入成功")
        return True
    except Exception as e:
        print(f"❌ TypeScriptGenerator 導入失敗: {e}")
        return False

def test_cross_language_validator():
    """測試跨語言驗證器"""
    try:
        from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator
        print("✅ CrossLanguageValidator 導入成功")
        return True
    except Exception as e:
        print(f"❌ CrossLanguageValidator 導入失敗: {e}")
        return False

def main():
    print("=" * 60)
    print("AIVA Plugins 導入測試")
    print("=" * 60)
    
    tests = [
        ("SARIF 轉換器", test_sarif_converter),
        ("任務轉換器", test_task_converter),
        ("DOCX 轉換器", test_docx_converter),
        ("Schema 代碼生成", test_schema_codegen),
        ("TypeScript 生成器", test_typescript_generator),
        ("跨語言驗證器", test_cross_language_validator),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n測試 {name}...")
        results.append(test_func())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"測試結果: {passed}/{total} 通過 ({passed/total*100:.0f}%)")
    print("=" * 60)
    
    if passed < total:
        print("\n建議安裝缺失的依賴:")
        print("  pip install python-docx jinja2 pyyaml")

if __name__ == "__main__":
    main()
