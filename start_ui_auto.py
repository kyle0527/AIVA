#!/usr/bin/env python3
"""
AIVA UI 自動端口啟動腳本

功能:
- 自動選擇可用端口
- 智能重試機制  
- 支援多種運行模式
"""

import sys
import os
from pathlib import Path

# 確保在專案根目錄
project_root = Path(__file__).parent
if not (project_root / "pyproject.toml").exists():
    print("❌ 請在 AIVA 專案根目錄執行此腳本")
    sys.exit(1)

# 添加專案路徑到 Python 路徑
sys.path.insert(0, str(project_root))

try:
    from services.core.aiva_core.ui_panel.auto_server import start_auto_server
    
    print("🚀 啟動 AIVA UI 自動端口伺服器...")
    
    # 使用預設設定啟動
    start_auto_server(
        mode="hybrid",
        host="127.0.0.1", 
        preferred_ports=[8080, 8081, 3000, 5000, 9000]
    )
    
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    print("請確保所有依賴已安裝")
    sys.exit(1)
except Exception as e:
    print(f"❌ 啟動失敗: {e}")
    sys.exit(1)