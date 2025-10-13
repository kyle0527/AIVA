#!/usr/bin/env python
"""
test_tools.py
-------------
測試腳本 - 驗證所有分析工具正常運作
"""

import sys
from pathlib import Path

# 加入工具路徑
sys.path.insert(0, str(Path(__file__).parent))

from analyze_codebase import CodeAnalyzer


def test_code_analyzer_simple():
    """測試 CodeAnalyzer 簡單模式（向後兼容）."""
    print("測試 CodeAnalyzer - 簡單模式...")
    
    analyzer = CodeAnalyzer("/home/runner/work/AIVA/AIVA/services")
    result = analyzer.execute(path="core/aiva_core/__init__.py", detailed=False)
    
    assert result["status"] == "success", "分析應該成功"
    assert "total_lines" in result, "應包含總行數"
    assert "imports" in result, "應包含導入數"
    assert "functions" in result, "應包含函數數"
    assert "classes" in result, "應包含類別數"
    
    print("✓ 簡單模式測試通過")
    return True


def test_code_analyzer_detailed():
    """測試 CodeAnalyzer 詳細模式."""
    print("測試 CodeAnalyzer - 詳細模式...")
    
    analyzer = CodeAnalyzer("/home/runner/work/AIVA/AIVA/services")
    result = analyzer.execute(path="core/aiva_core/ai_engine/tools.py", detailed=True)
    
    assert result["status"] == "success", "分析應該成功"
    assert "total_lines" in result, "應包含總行數"
    assert "code_lines" in result, "應包含程式碼行數"
    assert "comment_lines" in result, "應包含註解行數"
    assert "blank_lines" in result, "應包含空白行數"
    assert "function_count" in result, "應包含函數計數"
    assert "class_count" in result, "應包含類別計數"
    assert "cyclomatic_complexity" in result, "應包含複雜度"
    assert "has_type_hints" in result, "應檢查類型提示"
    assert "has_docstrings" in result, "應檢查文檔字串"
    
    # 驗證實際數據
    assert result["total_lines"] > 0, "應該有行數"
    assert result["function_count"] > 0, "應該有函數"
    assert result["class_count"] > 0, "應該有類別"
    assert result["cyclomatic_complexity"] > 0, "應該有複雜度"
    
    print("✓ 詳細模式測試通過")
    print(f"  - 總行數: {result['total_lines']}")
    print(f"  - 函數數: {result['function_count']}")
    print(f"  - 類別數: {result['class_count']}")
    print(f"  - 複雜度: {result['cyclomatic_complexity']}")
    return True


def test_py2mermaid():
    """測試 py2mermaid 工具."""
    print("測試 py2mermaid...")
    
    from py2mermaid import build_for_file
    
    test_file = Path("/home/runner/work/AIVA/AIVA/services/core/aiva_core/__init__.py")
    charts = build_for_file(test_file)
    
    assert len(charts) > 0, "應該生成至少一個圖表"
    
    # 檢查第一個圖表
    chart_name, mermaid_code = charts[0]
    assert "flowchart" in mermaid_code, "應包含 flowchart 聲明"
    assert "開始" in mermaid_code, "應包含開始節點"
    assert "結束" in mermaid_code, "應包含結束節點"
    
    print("✓ py2mermaid 測試通過")
    print(f"  - 生成圖表數: {len(charts)}")
    return True


def test_error_handling():
    """測試錯誤處理."""
    print("測試錯誤處理...")
    
    analyzer = CodeAnalyzer("/home/runner/work/AIVA/AIVA/services")
    
    # 測試不存在的檔案
    result = analyzer.execute(path="nonexistent/file.py", detailed=True)
    assert result["status"] == "error", "應返回錯誤狀態"
    assert "error" in result, "應包含錯誤訊息"
    
    # 測試缺少參數
    result = analyzer.execute(detailed=True)
    assert result["status"] == "error", "應返回錯誤狀態"
    assert "缺少必需參數" in result["error"], "應提示缺少參數"
    
    print("✓ 錯誤處理測試通過")
    return True


def main():
    """執行所有測試."""
    print("=" * 80)
    print("AIVA 分析工具測試套件")
    print("=" * 80)
    print()
    
    tests = [
        test_code_analyzer_simple,
        test_code_analyzer_detailed,
        test_py2mermaid,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} 失敗")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} 失敗: {e}")
        print()
    
    print("=" * 80)
    print(f"測試結果: {passed} 通過, {failed} 失敗")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
