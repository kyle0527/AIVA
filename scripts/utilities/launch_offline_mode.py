#!/usr/bin/env python3
"""
AIVA 離線模式啟動器
"""
import os
import sys
from pathlib import Path

# 設置項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_offline_env():
    """設置離線環境"""
    env_vars = {
        "AIVA_RABBITMQ_URL": "memory://localhost",
        "AIVA_RABBITMQ_USER": "offline",
        "AIVA_RABBITMQ_PASSWORD": "offline",
        "AIVA_OFFLINE_MODE": "true",
        "AIVA_LOG_LEVEL": "INFO",
        "AIVA_ENVIRONMENT": "offline"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("🔧 離線環境已設置")

def main():
    setup_offline_env()
    
    print("🚀 AIVA 離線模式啟動")
    print("=" * 40)
    print("✅ 環境變數已設置")
    print("📋 可用功能:")
    print("  - AI 組件探索")
    print("  - 學習成效分析")  
    print("  - 基礎安全掃描")
    print("  - 系統健康檢查")
    print()
    print("🔧 建議的測試命令:")
    print("  python health_check.py")
    print("  python ai_component_explorer.py")
    print("  python ai_system_explorer_v3.py --help")
    print()

if __name__ == "__main__":
    main()
