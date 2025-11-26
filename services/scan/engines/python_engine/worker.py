"""
⚠️ DEPRECATED - 此檔案已廢棄

AIVA v2.0 架構變更：
- 舊架構: 使用 RabbitMQ + async worker 模式
- 新架構: 直接通過 command_handler.py 調用 MultiEngineCoordinator

執行流程 (v2.0):
1. AI Core → command_handler.py → MultiEngineCoordinator
2. MultiEngineCoordinator → ScanOrchestrator (Python 引擎)
3. 同步數據合約直接通信，無需 MQ

遷移說明:
- 請使用 services/scan/command_handler.py (同步模式)
- 多引擎協調使用 coordinators/multi_engine_coordinator.py
- 數據模型使用 services/aiva_common/schemas

相關文件:
- services/scan/command_handler.py
- services/scan/coordinators/multi_engine_coordinator.py
- services/aiva_common/README.md

最後更新: 2025-01-12
"""

# 保留導入以防向後兼容需求
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

# ⚠️ 所有舊代碼已移除
# 請使用 services/scan/command_handler.py (v2.0 架構)

