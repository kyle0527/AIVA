# Storage 存儲模組

> **路徑**: `services/core/aiva_core/service_backbone/storage`  
> **狀態**: ✅ 正常 | **文件數**: 7 | **最後更新**: 2026-01-07

## 概述

多後端數據存儲系統，支援 SQLite、PostgreSQL、JSONL 和混合存儲，提供統一的數據持久化接口。

## 核心組件

### backends.py
**抽象基類：**
- `StorageBackend` (ABC) - 存儲後端抽象基類

**具體實現：**
- `SQLiteBackend` - SQLite 後端（輕量級，適合開發和小規模部署）
- `PostgreSQLBackend` - PostgreSQL 後端（生產級，適合大規模部署）
- `JSONLBackend` - JSONL 文件後端（適合導出和分析）
- `HybridBackend` - 混合後端（數據庫 + 文件）

### models.py
**ORM 模型：**
- `ExperienceSampleModel` - 經驗樣本模型
- `TraceRecordModel` - 追蹤記錄模型
- `TrainingSessionModel` - 訓練會話模型
- `ModelCheckpointModel` - 模型檢查點模型
- `KnowledgeEntryModel` - 知識條目模型
- `ScenarioModel` - 場景模型
- `CommandExecutionModel` - 命令執行模型

### storage_manager.py
- `StorageManager` - 存儲管理器
  - 統一的數據存取接口
  - 後端切換
  - 數據遷移

### command_repository.py
- `CommandRepository` - 命令倉儲
  - 命令歷史記錄
  - 命令執行結果存取

### __init__.py
- 模組初始化和導出

## 後端選擇指南

| 後端 | 適用場景 | 特點 |
|------|----------|------|
| SQLite | 開發、小規模 | 零配置、嵌入式 |
| PostgreSQL | 生產、大規模 | 高性能、支援並發 |
| JSONL | 導出、分析 | 人類可讀、易處理 |
| Hybrid | 綜合需求 | 結構數據入庫、大文件存文件系統 |

## 依賴關係

- `sqlalchemy` - ORM 框架
- `aiva_common.schemas` - ExperienceSample, TraceRecord
- `pathlib`, `json` - 文件處理
