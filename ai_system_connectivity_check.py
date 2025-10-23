#!/usr/bin/env python3
"""
⚠️  重定向通知: ai_system_connectivity_check.py 已移動

新位置: scripts/testing/ai_system_connectivity_check.py
請使用: python scripts/testing/ai_system_connectivity_check.py

此重定向檔案將在 2026年4月移除
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("正在重定向到新位置...")
    print("WARNING: ai_system_connectivity_check.py 已移動到 scripts/testing/")
    print("   新指令: python scripts/testing/ai_system_connectivity_check.py")
    print("   此重定向將在 2026年4月移除")
    print("-" * 50)
    
    script_path = Path(__file__).parent / "scripts" / "testing" / "ai_system_connectivity_check.py"
    
    if script_path.exists():
        try:
            result = subprocess.run([sys.executable, str(script_path)] + sys.argv[1:])
            sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ 執行錯誤: {e}")
            sys.exit(1)
    else:
        print("❌ 找不到目標檔案")
        sys.exit(1)

if __name__ == "__main__":
    main()