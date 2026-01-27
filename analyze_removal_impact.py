#!/usr/bin/env python3
"""分析移除問題字元的影響"""

file_path = r"C:\D\fold7\AIVA-git\services\features\function_info_leak\sensitive_info_detector1.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=" * 70)
print("移除問題字元（PUA）的影響分析")
print("=" * 70)

# 1. 找出所有問題行
problem_lines = []
for line_no, line in enumerate(lines, start=1):
    has_pua = any(0xE000 <= ord(c) <= 0xF8FF for c in line)
    if has_pua:
        problem_lines.append((line_no, line))

print(f"\n📊 統計:")
print(f"  包含問題字元的行數: {len(problem_lines)}")

# 2. 分析這些行的類型
docstring_lines = []
comment_lines = []
code_lines = []

for line_no, line in problem_lines:
    stripped = line.strip()
    if '"""' in stripped or "'''" in stripped:
        docstring_lines.append(line_no)
    elif stripped.startswith('#'):
        comment_lines.append(line_no)
    else:
        code_lines.append(line_no)

print(f"\n📍 問題字元位置:")
print(f"  Docstring 中: {len(docstring_lines)} 行")
print(f"  註解中: {len(comment_lines)} 行")
print(f"  代碼行: {len(code_lines)} 行")

# 3. 移除問題字元後的效果
print(f"\n✂️ 移除後效果示例:")
print(f"\n【示例 1】Docstring (Line 34)")
line_34 = lines[33]
print(f"  原始: {repr(line_34.rstrip())}")
cleaned_34 = ''.join(c for c in line_34 if not (0xE000 <= ord(c) <= 0xF8FF))
print(f"  移除後: {repr(cleaned_34.rstrip())}")

print(f"\n【示例 2】Docstring (Line 112)")
line_112 = lines[111]
print(f"  原始: {repr(line_112.rstrip())}")
cleaned_112 = ''.join(c for c in line_112 if not (0xE000 <= ord(c) <= 0xF8FF))
print(f"  移除後: {repr(cleaned_112.rstrip())}")

print(f"\n【示例 3】註解/代碼 (Line 122)")
line_122 = lines[121]
print(f"  原始: {repr(line_122.rstrip())}")
cleaned_122 = ''.join(c for c in line_122 if not (0xE000 <= ord(c) <= 0xF8FF))
print(f"  移除後: {repr(cleaned_122.rstrip())}")

# 4. 檢查移除後是否能通過語法檢查
print(f"\n🐍 語法檢查預測:")
cleaned_content = ''.join(
    ''.join(c for c in line if not (0xE000 <= ord(c) <= 0xF8FF))
    for line in lines
)

import ast
try:
    ast.parse(cleaned_content)
    print("  ✅ 預測移除後可以通過 AST 解析")
    syntax_ok = True
except SyntaxError as e:
    print(f"  ❌ 預測仍有語法錯誤: Line {e.lineno}")
    syntax_ok = False

# 5. 總結影響
print(f"\n" + "=" * 70)
print("📋 影響總結")
print("=" * 70)

print(f"\n✅ 正面影響:")
if syntax_ok:
    print("  1. Python 語法檢查會通過")
    print("  2. AST 解析器可以正確處理文件")
    print("  3. 代碼可以正常執行")
    print("  4. 不會影響功能性代碼邏輯")

print(f"\n⚠️ 負面影響:")
print("  1. Docstring 會留下不完整的中文字元")
print("  2. 註解內容可能不易閱讀")
print("  3. 文檔的中文部分會變成亂碼")
print("  4. 需要後續手動修復這些文檔字串")

print(f"\n💡 建議:")
if syntax_ok:
    print("  方案 A: 先移除問題字元 → 使代碼可執行")
    print("          → 再逐步修復 docstring（用英文或正確中文）")
    print()
    print("  方案 B: 直接從原始文件 (sensitive_info_detector.py)")
    print("          複製正確的中文內容進行替換")
    print()
    print("  方案 C: 將所有中文 docstring 改為英文")
else:
    print("  ⚠️ 單純移除可能不夠，需要更深入的修復")

print()
