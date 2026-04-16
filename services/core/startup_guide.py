#!/usr/bin/env python3
"""
AIVA Core 啟動指南

系統完整啟動步驟和故障排除指南
"""

from pathlib import Path
import sys


def print_startup_guide():
    """打印啟動指南"""
    
    guide = f"""
[START] AIVA Core 系統完整啟動步驟
=====================================

[1] 1. 前置檢查
--------------
[OK] Python 版本: 3.11+ (當前: {sys.version.split()[0]})
[OK] 工作目錄: C:\\D\\fold7\\AIVA-git\\services\\core
[OK] 核心模組: aiva_core v4.1.1

[2] 2. 依賴安裝
--------------
cd C:\\D\\fold7\\AIVA-git\\services\\core
pip install -r requirements.txt

[3] 3. 環境配置
--------------
# 設定環境變數（可選）
set AIVA_ENV=production
set AIVA_LOG_LEVEL=INFO

[4] 4. 啟動方式
--------------

方式 1: 完整系統啟動 (推薦)
---------------------------
python main.py

特點: HTTP API + 安全防護層
端口: 預設 8000
URL:  http://localhost:8000

方式 2: 核心應用啟動
------------------
python app.py

特點: FastAPI 應用
端口: 預設 8000
API:  POST /scan

方式 3: 模組化啟動
-----------------
python -m aiva_core

特點: 直接啟動核心引擎

方式 4: 互動式 CLI
------------------
python -m aiva_core.core_capabilities.cli.aiva_cli

特點: 命令行界面

🏗️ 5. 系統架構
--------------
外部請求
    ↓
main.py (安全防護層)
    ↓
app.py (FastAPI 核心)
    ↓
aiva_core (五大模組)
    ├─ 🧠 cognitive_core (認知核心)
    ├─ 🧭 internal_exploration (內部探索)
    ├─ 📋 task_planning (任務規劃)
    ├─ 🎯 core_capabilities (核心能力)
    └─ 🏗️ service_backbone (服務骨幹)

🔍 6. 健康檢查
--------------
啟動後檢查:
curl http://localhost:8000/health
curl http://localhost:8000/capabilities

📊 7. 監控端點
--------------
- /health - 系統健康狀態
- /metrics - Prometheus 指標
- /capabilities - 能力列表
- /scan - 掃描接口 (POST)

❌ 8. 故障排除
--------------
問題: 啟動失敗
解決: 檢查 requirements.txt 依賴安裝

問題: 端口佔用
解決: 修改端口設定或停止衝突服務

問題: 模組載入錯誤
解決: 檢查 PYTHONPATH 設定

📚 9. 更多資訊
--------------
- README: services/core/aiva_core/README.md
- API 文檔: 啟動後訪問 /docs
- 日誌位置: ./logs/aiva_core.log

=======================================
[QUICK] 快速啟動: python main.py
🌐 Web UI: http://localhost:8000
📖 API 文檔: http://localhost:8000/docs
=======================================
    """
    
    print(guide)

def check_requirements():
    """檢查系統需求"""
    print("[CHECK] 系統需求檢查...")
    
    # 檢查 Python 版本
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"[Error] Python 版本過舊: {version.major}.{version.minor}.{version.micro}")
        print("   需要 Python 3.11+")
        return False
    
    # 檢查工作目錄
    core_path = Path("aiva_core")
    if core_path.exists():
        print("[OK] aiva_core 目錄存在")
    else:
        print("[Error] 找不到 aiva_core 目錄")
        print("   請確認在正確的工作目錄: services/core")
        return False
    
    # 檢查主要文件
    files_to_check = ["main.py", "app.py", "requirements.txt"]
    for file_name in files_to_check:
        if Path(file_name).exists():
            print(f"[OK] {file_name}")
        else:
            print(f"[Missing] 缺少文件: {file_name}")
            return False
    
    return True

def main():
    """主函數"""
    print("🤖 AIVA Core 啟動助手\n")
    
    if check_requirements():
        print("\n" + "="*50)
        print("✅ 系統需求檢查通過!")
        print("="*50)
        print_startup_guide()
    else:
        print("\n" + "="*50)
        print("❌ 系統需求檢查失敗!")
        print("   請解決上述問題後重新運行")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()