"""AIVA Dashboard 啟動腳本

此腳本用於一鍵啟動 AIVA Core API + Dashboard。

使用方式：
    python scripts/start_dashboard.py
    
    或者分別啟動：
    python services/core/aiva_core/service_backbone/api/app.py &
    streamlit run services/dashboard/streamlit_app.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

print("="*60)
print("🚀 AIVA Dashboard 啟動器")
print("="*60)
print()

# ==================== 檢查依賴 ====================

print("📦 檢查依賴套件...")

required_packages = [
    "streamlit",
    "requests",
    "pyyaml",
    "pandas",
    "sse-starlette"
]

missing_packages = []

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package} - 未安裝")
        missing_packages.append(package)

if missing_packages:
    print()
    print("⚠️  缺少依賴套件，請執行：")
    print("   pip install -r requirements-dashboard.txt")
    print()
    sys.exit(1)

print()

# ==================== 啟動 AIVA Core API ====================

print("🔧 啟動 AIVA Core API...")

app_path = PROJECT_ROOT / "services" / "core" / "aiva_core" / "service_backbone" / "api" / "app.py"

if not app_path.exists():
    print(f"❌ app.py 不存在: {app_path}")
    sys.exit(1)

# 啟動 FastAPI（背景執行）
api_process = subprocess.Popen(
    [sys.executable, str(app_path)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

print(f"  ✅ API 已啟動 (PID: {api_process.pid})")
print("  📍 API URL: http://localhost:8000")
print("  📚 API 文檔: http://localhost:8000/docs")
print()

# 等待 API 啟動
print("⏳ 等待 API 就緒...")
time.sleep(3)

# 檢查 API 是否正常運行
try:
    import requests
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("  ✅ API 健康檢查通過")
    else:
        print(f"  ⚠️  API 回應異常: {response.status_code}")
except Exception as e:
    print(f"  ⚠️  API 連線失敗: {e}")
    print("  提示: API 可能仍在啟動中，Dashboard 會持續嘗試連線")

print()

# ==================== 啟動 Streamlit Dashboard ====================

print("🎨 啟動 Streamlit Dashboard...")

dashboard_path = PROJECT_ROOT / "services" / "dashboard" / "streamlit_app.py"

if not dashboard_path.exists():
    print(f"❌ streamlit_app.py 不存在: {dashboard_path}")
    api_process.terminate()
    sys.exit(1)

print("  📍 Dashboard URL: http://localhost:8501")
print()

# 啟動 Streamlit（前台執行）
try:
    subprocess.run([
        "streamlit", "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])
except KeyboardInterrupt:
    print()
    print("⚠️  收到中斷信號，正在關閉...")

# ==================== 清理 ====================

print()
print("🛑 關閉 AIVA Core API...")
api_process.terminate()
api_process.wait(timeout=5)
print("  ✅ API 已關閉")

print()
print("👋 AIVA Dashboard 已停止")
