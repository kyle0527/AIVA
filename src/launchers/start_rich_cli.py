#!/usr/bin/env python3
"""
AIVA Rich CLI 啟動腳本

此腳本啟動整合了 HackingTool Rich UI 框架的 AIVA 命令行界面。
提供現代化的互動式命令行體驗。

使用方式:
    python start_rich_cli.py

需求:
- Python 3.8+
- Rich 庫
- AIVA 核心模組
"""

import sys
import os
import asyncio
from pathlib import Path

# 確保在專案根目錄
project_root = Path(__file__).parent
if not (project_root / "pyproject.toml").exists():
    print("❌ 請在 AIVA 專案根目錄執行此腳本")
    sys.exit(1)

# 添加專案路徑到 Python 路徑
sys.path.insert(0, str(project_root))

def check_dependencies():
    """檢查必要的依賴"""
    try:
        import rich
        print(f"✓ Rich UI 庫版本: {rich.__version__}")
    except ImportError:
        print("❌ 未安裝 Rich 庫，請執行: pip install rich")
        return False
    
    try:
        from services.core.aiva_core.ui_panel.rich_cli import AIVARichCLI
        print("✓ AIVA Rich CLI 模組已就緒")
    except ImportError as e:
        print(f"❌ 無法導入 AIVA Rich CLI: {e}")
        return False
    
    return True

async def main():
    """主函數"""
    print("🚀 正在啟動 AIVA Rich CLI...")
    print("=" * 50)
    
    # 檢查依賴
    if not check_dependencies():
        sys.exit(1)
    
    print("✓ 所有依賴檢查通過")
    print("=" * 50)
    
    # 導入並啟動 CLI
    try:
        from services.core.aiva_core.ui_panel.rich_cli import AIVARichCLI
        
        cli = AIVARichCLI()
        await cli.run()
        
    except KeyboardInterrupt:
        print("\n⚠ 用戶中斷程式")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 在 Windows 上設定正確的事件循環策略
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())