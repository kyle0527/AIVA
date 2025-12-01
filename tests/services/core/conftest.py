"""
Pytest 配置文件 - 設置測試環境

此文件確保測試可以正確導入 aiva_common 和 aiva_core 模組。
根據 DEVELOPMENT_STANDARDS.md：
- aiva_common 應該可以直接導入（from aiva_common import ...）
- models.py 在 services/core/ 目錄，需要在 Python 路徑中
"""

import sys
from pathlib import Path

# 將 services/ 目錄添加到 Python 路徑
# 這樣可以直接 import aiva_common
services_dir = Path(__file__).parent.parent.parent
if str(services_dir) not in sys.path:
    sys.path.insert(0, str(services_dir))

# 將 services/core/ 目錄添加到 Python 路徑
# 這樣可以直接 import aiva_core, import models, import ai_models
core_dir = Path(__file__).parent.parent
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))
