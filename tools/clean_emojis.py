"""批量清理 Python 文件中的 emoji"""
import os
import re

# Emoji 映射表
EMOJI_REPLACEMENTS = {
    "✅": "[SUCCESS]",
    "❌": "[ERROR]",
    "⚠️": "[WARNING]",
    "🎯": "[TARGET]",
    "📊": "[REPORT]",
    "🔧": "[FIX]",
    "⏱️": "[TIME]",
    "📝": "[NOTE]",
    "🚀": "[START]",
    "💬": "[CHAT]",
    "🤖": "[AI]",
    "⏰": "[CLOCK]",
    "🔄": "[RELOAD]",
}

def clean_emojis(filepath):
    """清理文件中的 emoji"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        content = content.replace(emoji, replacement)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# 要處理的文件
files = [
    r"c:\D\fold7\AIVA-git\services\scan\coordinators\multi_engine_coordinator.py",
]

for filepath in files:
    if os.path.exists(filepath):
        if clean_emojis(filepath):
            print(f"✓ Cleaned: {filepath}")
        else:
            print(f"- No emoji found: {filepath}")
    else:
        print(f"✗ Not found: {filepath}")
