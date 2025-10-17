#!/usr/bin/env python3
"""測試簡單匹配器"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.core.aiva_core.ai_engine.simple_matcher import SimpleTaskMatcher
from services.core.aiva_core.ai_engine.cli_tools import get_all_tools

# 創建匹配器
cli_tools = get_all_tools()
tools = [{"name": n, "instance": o} for n, o in cli_tools.items()]
matcher = SimpleTaskMatcher(tools)

# 測試案例
tests = [
    ("掃描目標網站 example.com", "ScanTrigger"),
    ("檢測 SQL 注入漏洞", "SQLiDetector"),
    ("檢測 XSS 漏洞", "XSSDetector"),
    ("分析代碼結構", "CodeAnalyzer"),
    ("讀取 README.md 文件", "CodeReader"),
    ("寫入配置文件", "CodeWriter"),
    ("生成掃描報告", "ReportGenerator"),
]

print("="*70)
print("簡單匹配器測試")
print("="*70)

print(f"\n工具數量: {len(tools)}")
print(f"測試案例: {len(tests)}\n")

correct = 0
for i, (task, expected) in enumerate(tests, 1):
    matched, confidence = matcher.match(task)
    is_correct = matched == expected
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    print(f"[{i}] {status}")
    print(f"  任務: {task}")
    print(f"  預期: {expected}")
    print(f"  匹配: {matched}")
    print(f"  信心度: {confidence:.1%}")
    print()

accuracy = correct / len(tests) * 100
print(f"準確度: {correct}/{len(tests)} = {accuracy:.1f}%")
print("="*70)
