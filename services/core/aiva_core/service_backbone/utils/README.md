# Utils 工具模組

> **路徑**: `service_backbone/utils/`  
> **狀態**: ✅ 正常 | **文件數**: 3 | **最後更新**: 2026-01-21  
> **父模組**: [Service Backbone](../README.md)

## 概述

通用工具集，提供跨語言日誌管理、系統配置、AI 識別和修復工具。

## 核心組件

### repair_tool.py ⭐ 新增
- `AIVASystemRepair` - AIVA 系統修復工具
  - Go 依賴修復
  - 編譯錯誤修復
  - 系統配置修復

### config.py
- `AIVAConfig` - AIVA 配置管理
- `ConfigManager` - 配置管理器

### ai_identifier.py
- `AIIdentifier` - AI 識別器
- `AISignature` - AI 簽名

### logging_formatter.py
- `AIVALogFormatter` - AIVA 日誌格式化器（繼承 logging.Formatter）
  - 統一的日誌格式
  - 彩色輸出支援
  - 結構化日誌
  - 上下文信息附加

- `CrossLanguageLogManager` - 跨語言日誌管理器
  - Python/Rust/Node.js 日誌統一
  - 日誌級別映射
  - 日誌聚合和路由
  - 多目標輸出（控制台、文件、遠端）

## 日誌格式

```
[2026-01-07 10:30:45.123] [INFO] [core.task_executor] 任務執行開始
[2026-01-07 10:30:45.456] [DEBUG] [core.task_executor] 參數: {"target": "http://..."}
[2026-01-07 10:30:46.789] [INFO] [core.task_executor] 任務執行完成
```

## 使用方式

```python
import logging
from service_backbone.utils.logging_formatter import AIVALogFormatter

# 配置日誌格式
handler = logging.StreamHandler()
handler.setFormatter(AIVALogFormatter())

logger = logging.getLogger("aiva")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# 使用
logger.info("操作完成", extra={"task_id": "task_123"})
```

## 跨語言支援

遵循 `CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md` 規範，確保：
- Python、Rust、Node.js 日誌格式一致
- 時間戳統一使用 ISO 8601 格式
- 日誌級別映射標準化

## 依賴關係

- `logging` - Python 標準日誌庫
- 無外部套件依賴
