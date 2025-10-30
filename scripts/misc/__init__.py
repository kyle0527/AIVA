"""
AIVA Platform - AI-Assisted Vulnerability Analysis Platform
統一的四大模組架構系統
"""

# 確保 services 包能被正確找到
from pathlib import Path
import sys

# 將當前目錄添加到 Python 路徑
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
