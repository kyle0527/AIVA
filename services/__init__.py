"""
AIVA Services Module
統一的服務模組包
根據 SCHEMA_MANAGEMENT_SOP.md 的硬導入問題排除指南實作
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

def setup_aiva_paths():
    """
    設置 AIVA 系統的 Python 路徑
    確保所有服務模組都能正確導入
    """
    # 獲取當前目錄（services）
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    
    # 添加必要的路徑到 sys.path
    paths_to_add = [
        str(current_dir),  # services 目錄
        str(project_root), # 專案根目錄
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# 自動執行路徑設置
setup_aiva_paths()

# 版本資訊
__version__ = "1.0.0"
__author__ = "AIVA Development Team"

# 匯出常用模組
try:
    import aiva_common
    # 不使用 * 導入以避免屬性衝突
    __all__ = ['aiva_common']
except ImportError:
    # 如果還是無法導入，提供詳細錯誤信息
    import warnings
    warnings.warn(
        "aiva_common 模組導入失敗。請檢查 services/aiva_common/ 目錄是否存在",
        ImportWarning
    )
