"""
⚠️ DEPRECATED - 此檔案已廢棄

AIVA v2.0 架構變更：
- 舊架構: 使用 RabbitMQ + async worker 模式
- 新架構: MultiEngineCoordinator 直接調用 TypeScript 掃描器

執行流程 (v2.0):
1. AI Core → command_handler.py → MultiEngineCoordinator
2. MultiEngineCoordinator → TypeScript 掃描器 (Node.js subprocess)
3. 同步數據合約直接通信，無需 MQ

遷移說明:
- TypeScript 引擎整合在 MultiEngineCoordinator 中
- 數據模型使用 services/aiva_common/schemas
- TypeScript 掃描器通過 subprocess 調用

相關文件:
- services/scan/command_handler.py
- services/scan/coordinators/multi_engine_coordinator.py
- services/scan/engines/typescript_engine/src (Node.js 掃描器)

最後更新: 2025-01-12
"""

# 保留函數供測試使用（已移除 MQ 依賴）
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

# ⚠️ 所有 MQ 相關代碼已移除
# TypeScript 引擎整合在 MultiEngineCoordinator 中
# 請參考 services/scan/coordinators/multi_engine_coordinator.py

