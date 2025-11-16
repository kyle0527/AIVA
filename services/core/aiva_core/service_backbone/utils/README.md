# Utils - 工具函數集

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [logging_formatter.py](#logging_formatterpy-252-行)
- [📝 日誌最佳實踐](#-日誌最佳實踐)
- [🔧 配置示例](#-配置示例)
- [📊 日誌監控](#-日誌監控)
- [🎨 輸出示例](#-輸出示例)
- [📚 相關模組](#-相關模組)
- [💡 實用函數 (待擴展)](#-實用函數-待擴展)

---

## 📋 概述

**定位**: 通用工具函數和輔助模組  
**狀態**: ✅ 已實現  
**文件數**: 1 個 Python 文件 (252 行)

## 📂 文件結構

```
utils/
├── logging_formatter.py (252 行) - 日誌格式化器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### logging_formatter.py (252 行)

**職責**: 統一日誌格式化和輸出管理

**主要類/函數**:
- `LoggingFormatter` - 自定義日誌格式化器
- `ColoredFormatter` - 彩色終端輸出
- `JSONFormatter` - JSON 格式日誌
- `StructuredFormatter` - 結構化日誌

**日誌級別**:
```python
import logging

# 標準級別
logging.DEBUG     # 詳細調試信息
logging.INFO      # 一般信息
logging.WARNING   # 警告信息
logging.ERROR     # 錯誤信息
logging.CRITICAL  # 嚴重錯誤
```

**使用範例**:
```python
from aiva_core.service_backbone.utils import LoggingFormatter

# 配置彩色日誌
import logging

logger = logging.getLogger("aiva")
handler = logging.StreamHandler()
handler.setFormatter(LoggingFormatter.colored())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 使用日誌
logger.info("掃描已啟動")
logger.warning("發現潛在漏洞")
logger.error("掃描失敗", exc_info=True)
```

**彩色輸出**:
```python
formatter = LoggingFormatter.colored()

# 輸出示例 (帶顏色):
# 2025-11-16 10:00:00 [INFO   ] 掃描已啟動
# 2025-11-16 10:01:00 [WARNING] 發現潛在漏洞
# 2025-11-16 10:02:00 [ERROR  ] 掃描失敗
```

---

### JSON 格式日誌

**適用場景**: 
- 日誌聚合系統 (ELK, Splunk)
- 機器解析
- 結構化分析

**使用範例**:
```python
# JSON 格式化器
formatter = LoggingFormatter.json()

logger.info("掃描完成", extra={
    "scan_id": "123",
    "target": "example.com",
    "findings_count": 15
})

# 輸出:
# {
#   "timestamp": "2025-11-16T10:00:00Z",
#   "level": "INFO",
#   "message": "掃描完成",
#   "scan_id": "123",
#   "target": "example.com",
#   "findings_count": 15
# }
```

---

### 結構化日誌

**特點**:
- 易於查詢
- 支持索引
- 自動添加上下文

**使用範例**:
```python
from aiva_core.service_backbone.utils import StructuredLogger

logger = StructuredLogger("aiva.scan")

# 自動添加上下文
with logger.context(scan_id="123", user="alice"):
    logger.info("開始掃描")
    logger.info("掃描完成")
    # 所有日誌自動包含 scan_id 和 user
```

## 📝 日誌最佳實踐

### 1. 日誌級別選擇

```python
# ✅ 正確使用
logger.debug(f"處理參數: {params}")  # 詳細調試
logger.info("掃描已啟動")            # 關鍵操作
logger.warning("目標響應慢")         # 可能的問題
logger.error("連接失敗", exc_info=True)  # 錯誤 + 堆棧
logger.critical("數據庫不可用")      # 嚴重錯誤

# ❌ 錯誤使用
logger.info(f"變量值: {x}")          # 應使用 debug
logger.error("用戶未登錄")           # 應使用 warning
```

### 2. 結構化日誌

```python
# ✅ 結構化 (易於查詢)
logger.info("掃描完成", extra={
    "scan_id": "123",
    "duration_ms": 5000,
    "findings_count": 15
})

# ❌ 非結構化 (難以查詢)
logger.info(f"掃描 123 完成,耗時 5000ms,發現 15 個問題")
```

### 3. 敏感信息保護

```python
# ✅ 脫敏處理
logger.info(f"用戶登錄: {mask_email(email)}")

# ❌ 直接記錄敏感信息
logger.info(f"用戶登錄: {email}")  # 不要這樣做!
logger.info(f"密碼: {password}")   # 絕對不要!
```

## 🔧 配置示例

### 完整日誌配置

```python
import logging
from aiva_core.service_backbone.utils import LoggingFormatter

# 日誌配置
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "colored": {
            "()": "aiva_core.service_backbone.utils.LoggingFormatter.colored"
        },
        "json": {
            "()": "aiva_core.service_backbone.utils.LoggingFormatter.json"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "aiva.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "aiva": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

import logging.config
logging.config.dictConfig(logging_config)
```

### 日誌輪轉

```python
from logging.handlers import TimedRotatingFileHandler

# 按時間輪轉 (每天)
handler = TimedRotatingFileHandler(
    filename="aiva.log",
    when="midnight",
    interval=1,
    backupCount=30  # 保留 30 天
)
```

## 📊 日誌監控

### 日誌聚合

```python
# 發送到 ELK Stack
from logging.handlers import SysLogHandler

elk_handler = SysLogHandler(address=("elk-server", 514))
elk_handler.setFormatter(LoggingFormatter.json())
logger.addHandler(elk_handler)
```

### 日誌告警

```python
# 錯誤日誌觸發告警
class AlertHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            send_alert(record.getMessage())

logger.addHandler(AlertHandler())
```

## 🎨 輸出示例

### 彩色終端輸出

```
2025-11-16 10:00:00 [INFO   ] 🚀 AIVA 系統啟動
2025-11-16 10:00:01 [INFO   ] ✅ 數據庫連接成功
2025-11-16 10:00:02 [WARNING] ⚠️  Redis 連接緩慢
2025-11-16 10:00:03 [ERROR  ] ❌ 掃描服務啟動失敗
```

### JSON 日誌輸出

```json
{"timestamp": "2025-11-16T10:00:00Z", "level": "INFO", "message": "系統啟動", "module": "main"}
{"timestamp": "2025-11-16T10:00:01Z", "level": "INFO", "message": "數據庫連接", "status": "success"}
{"timestamp": "2025-11-16T10:00:02Z", "level": "WARNING", "message": "Redis 慢", "latency_ms": 500}
```

## 📚 相關模組

- [monitoring](../monitoring/README.md) - 性能監控
- [coordination](../coordination/README.md) - 服務協調

## 💡 實用函數 (待擴展)

未來可添加到此目錄的工具函數:

- `validators.py` - 數據驗證工具
- `converters.py` - 數據轉換工具
- `crypto.py` - 加密解密工具
- `network.py` - 網絡工具函數
- `file_utils.py` - 文件操作工具

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

**完整規範**: [aiva_common 開發指南](../../../../aiva_common/README.md#-開發指南)

### 工具函數特別注意

```python
# ✅ 正確：使用標準類型
from aiva_common import ModuleName, TaskStatus
from typing import Optional, Dict, Any

def format_log(module: ModuleName, status: TaskStatus, message: str) -> str:
    """格式化日誌消息"""
    return f"[{module.value}] {status.value}: {message}"

# ✅ 正確：工具函數應該是純函數
def validate_config(config: Dict[str, Any]) -> bool:
    """驗證配置"""
    # 不依賴全局狀態
    return all(key in config for key in ["host", "port"])

# ❌ 禁止：在工具函數中定義枚舉
class LogLevel(str, Enum):  # 錯誤！這應該在 aiva_common
    INFO = "info"
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
