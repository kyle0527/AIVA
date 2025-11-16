# Storage - 存儲管理子系統

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [backends.py](#backendspy-560-行-)
  - [storage_manager.py](#storage_managerpy-240-行)
  - [models.py](#modelspy-185-行)
  - [config.py](#configpy-97-行)
- [💾 數據存儲策略](#-數據存儲策略)
- [🔄 數據生命週期](#-數據生命週期)
- [📊 查詢優化](#-查詢優化)
- [🔒 數據安全](#-數據安全)
- [📚 相關模組](#-相關模組)
- [🔧 配置最佳實踐](#-配置最佳實踐)

---

## 📋 概述

**定位**: 統一存儲接口和數據持久化  
**狀態**: ✅ 已實現  
**文件數**: 4 個 Python 文件 (1,082 行)

## 📂 文件結構

```
storage/
├── backends.py (560 行) ⭐ - 存儲後端實現
├── storage_manager.py (240 行) - 存儲管理器
├── models.py (185 行) - 數據模型
├── config.py (97 行) - 配置管理
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### backends.py (560 行) ⭐

**職責**: 多種存儲後端的統一實現

**支援的後端**:
| 後端類型 | 類名 | 特點 | 適用場景 |
|---------|------|------|---------|
| **SQLite** | `SQLiteBackend` | 輕量,單文件 | 開發測試,小型部署 |
| **PostgreSQL** | `PostgreSQLBackend` | 強大,ACID | 生產環境 |
| **MongoDB** | `MongoDBBackend` | NoSQL,靈活 | 非結構化數據 |
| **Redis** | `RedisBackend` | 快速,內存 | 快取,會話 |
| **文件系統** | `FileSystemBackend` | 簡單,本地 | 文件存儲 |

**使用範例**:
```python
from aiva_core.service_backbone.storage import backends

# PostgreSQL 後端
db = backends.PostgreSQLBackend(
    host="localhost",
    port=5432,
    database="aiva",
    user="admin",
    password="***"
)

# 保存數據
db.save("scan_results", {
    "scan_id": "123",
    "target": "example.com",
    "findings": [...]
})

# 查詢數據
results = db.query("scan_results", {
    "target": "example.com",
    "status": "completed"
})
```

**後端接口** (統一 API):
```python
class StorageBackend(ABC):
    def save(self, collection: str, data: dict) -> str: pass
    def query(self, collection: str, filters: dict) -> list: pass
    def update(self, collection: str, id: str, data: dict): pass
    def delete(self, collection: str, id: str): pass
    def get_by_id(self, collection: str, id: str) -> dict: pass
```

---

### storage_manager.py (240 行)

**職責**: 高層存儲管理和操作封裝

**主要功能**:
- 自動選擇存儲後端
- 連接池管理
- 事務支持
- 數據遷移

**使用範例**:
```python
from aiva_core.service_backbone.storage import StorageManager

# 初始化 (自動選擇後端)
storage = StorageManager.from_config({
    "backend": "postgresql",
    "connection_string": "postgresql://localhost/aiva"
})

# 使用事務
with storage.transaction():
    storage.save("scans", scan_data)
    storage.save("findings", findings_data)
    # 自動提交或回滾

# 批量操作
storage.bulk_save("scans", [scan1, scan2, scan3])
```

**高級功能**:
```python
# 數據遷移
storage.migrate_data(
    from_backend="sqlite",
    to_backend="postgresql",
    collections=["scans", "findings"]
)

# 數據備份
storage.backup(
    path="/backups/aiva_backup_20251116.sql",
    format="sql"
)
```

---

### models.py (185 行)

**職責**: 數據模型定義和 ORM 映射

**主要模型**:
```python
from aiva_core.service_backbone.storage import models

# 掃描結果模型
class ScanResult(models.Model):
    scan_id = models.StringField(primary_key=True)
    target = models.StringField(required=True)
    status = models.EnumField(["pending", "running", "completed"])
    findings = models.ListField()
    created_at = models.DateTimeField(auto_now_add=True)

# 漏洞發現模型
class Finding(models.Model):
    finding_id = models.StringField(primary_key=True)
    scan_id = models.ForeignKey(ScanResult)
    severity = models.EnumField(["low", "medium", "high", "critical"])
    title = models.StringField()
    description = models.TextField()
```

**使用範例**:
```python
# 創建記錄
scan = ScanResult(
    scan_id="scan_123",
    target="example.com",
    status="completed"
)
scan.save()

# 查詢記錄
scans = ScanResult.query(target="example.com")

# 更新記錄
scan.status = "completed"
scan.save()

# 關聯查詢
findings = Finding.query(scan_id=scan.scan_id)
```

---

### config.py (97 行)

**職責**: 存儲配置管理

**配置項**:
```python
from aiva_core.service_backbone.storage import StorageConfig

config = StorageConfig(
    backend="postgresql",
    host="localhost",
    port=5432,
    database="aiva",
    pool_size=20,
    max_overflow=10,
    echo=False,  # SQL 日誌
    timeout=30
)
```

## 💾 數據存儲策略

### 1. 掃描結果存儲

```python
# 關係型數據庫 (PostgreSQL)
storage.save("scan_results", {
    "scan_id": "123",
    "target": "example.com",
    "scan_type": "full",
    "status": "completed",
    "findings_count": 15
})

# 詳細發現存儲在 MongoDB (靈活結構)
mongo_storage.save("finding_details", {
    "finding_id": "f456",
    "raw_data": {...},  # 任意結構
    "metadata": {...}
})
```

### 2. 快取層

```python
# Redis 快取熱門數據
redis_backend = backends.RedisBackend()
redis_backend.save("cache:scan:123", scan_data, ttl=3600)

# 快取命中邏輯
def get_scan_result(scan_id):
    # 先查快取
    cached = redis_backend.get(f"cache:scan:{scan_id}")
    if cached:
        return cached
    
    # 快取未命中,查數據庫
    result = db.get_by_id("scan_results", scan_id)
    redis_backend.save(f"cache:scan:{scan_id}", result, ttl=3600)
    return result
```

### 3. 文件存儲

```python
# 大文件存儲在文件系統
fs_backend = backends.FileSystemBackend(base_path="/data/aiva")
fs_backend.save_file("reports/scan_123.pdf", pdf_content)

# 元數據存儲在數據庫
db.save("reports", {
    "report_id": "r789",
    "scan_id": "123",
    "file_path": "/data/aiva/reports/scan_123.pdf",
    "size_bytes": 1024000
})
```

## 🔄 數據生命週期

```
創建
  ↓
存儲到主數據庫
  ↓
同步到快取 (熱數據)
  ↓
定期歸檔 (舊數據)
  ↓
自動清理 (過期數據)
```

## 📊 查詢優化

### 索引策略

```python
# 創建索引
storage.create_index("scan_results", ["target", "created_at"])
storage.create_index("findings", ["severity", "scan_id"])

# 複合索引
storage.create_index("scans", [
    ("target", "asc"),
    ("status", "asc"),
    ("created_at", "desc")
])
```

### 查詢優化

```python
# 分頁查詢
results = storage.query_paginated(
    collection="scan_results",
    filters={"status": "completed"},
    page=1,
    page_size=50
)

# 投影查詢 (只返回需要的字段)
results = storage.query(
    collection="findings",
    filters={"severity": "critical"},
    projection=["finding_id", "title", "severity"]
)
```

## 🔒 數據安全

### 加密存儲

```python
# 啟用靜態加密
storage = StorageManager(
    backend="postgresql",
    encryption_key="your_encryption_key",
    encrypt_at_rest=True
)

# 敏感字段加密
storage.encrypt_fields("users", ["password", "api_key"])
```

### 備份策略

```python
# 定期備份
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(
    func=storage.backup,
    trigger="cron",
    hour=2,  # 每天凌晨 2 點
    args=["/backups/daily_backup.sql"]
)
scheduler.start()
```

## 📚 相關模組

- [state](../state/README.md) - 狀態存儲
- [messaging](../messaging/README.md) - 消息持久化
- [cognitive_core/rag](../../cognitive_core/rag/README.md) - 向量存儲

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md) 的修復規範。

```python
# ✅ 正確：使用標準配置和類型
from aiva_common import UnifiedConfig, ModuleName, Environment

# 獲取存儲配置
config = UnifiedConfig.get_instance()
storage_config = config.get_module_config(ModuleName.STORAGE)

# 使用標準環境
if config.environment == Environment.PRODUCTION:
    backend = "postgresql"
else:
    backend = "sqlite"

# ❌ 禁止：自定義存儲配置類
class StorageConfig:
    def __init__(self, backend):
        self.backend = backend  # 使用 UnifiedConfig

# ❌ 禁止：硬編碼環境檢查
if os.getenv("ENV") == "prod":  # 使用 Environment 枚舉
    backend = "postgresql"
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md)

---

## 🔧 配置最佳實踐

### 生產環境

```python
# PostgreSQL 主數據庫 + Redis 快取 + MongoDB 文檔存儲
storage_config = {
    "primary": {
        "backend": "postgresql",
        "host": "prod-db.example.com",
        "pool_size": 50,
        "max_overflow": 20
    },
    "cache": {
        "backend": "redis",
        "host": "prod-cache.example.com",
        "ttl": 3600
    },
    "documents": {
        "backend": "mongodb",
        "host": "prod-mongo.example.com"
    }
}
```

### 開發環境

```python
# SQLite 單文件數據庫
storage_config = {
    "backend": "sqlite",
    "database": "aiva_dev.db",
    "echo": True  # 顯示 SQL 日誌
}
```

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
