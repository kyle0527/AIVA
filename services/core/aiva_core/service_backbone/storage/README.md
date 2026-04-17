# Storage 存儲模組

> **路徑**: `services/core/aiva_core/service_backbone/storage`  
> **狀態**: ✅ 正常 | **Python 文件數**: 6 | **最後更新**: 2026-04-05

## 概述

多後端數據存儲系統，支援 SQLite、PostgreSQL、JSONL 和混合存儲，提供統一的數據持久化接口。

## 📄 檔案詳細資訊 (Files Details)

### `backends.py`
**說明**: 數據存儲後端實現

**類別 (Classes)**:
- `StorageBackend` - 存儲後端抽象基類
- `SQLiteBackend` - SQLite 存儲後端
- `PostgreSQLBackend` - PostgreSQL 存儲後端（繼承 SQLite 的實現）
- `JSONLBackend` - JSONL 文件存儲後端
- `HybridBackend` - 混合存儲後端（數據庫 + JSONL）

### `command_repository.py`
**說明**: 指令儲存庫

**類別 (Classes)**:
- `CommandRepository` - 指令儲存庫

### `config.py`
**說明**: AIVA 數據存儲配置

**函式 (Functions)**:
- `get_storage_config()` - 獲取存儲配置

### `models.py`
**說明**: SQLAlchemy ORM 模型定義

**類別 (Classes)**:
- `ExperienceSampleModel` - 經驗樣本模型
- `TraceRecordModel` - 執行追蹤記錄模型
- `TrainingSessionModel` - 訓練會話模型
- `ModelCheckpointModel` - 模型檢查點模型
- `KnowledgeEntryModel` - 知識條目模型（RAG）
- `ScenarioModel` - 靶場場景模型
- `CommandExecutionModel` - CLI 指令執行歷史模型

### `storage_manager.py`
**說明**: 存儲管理器

**類別 (Classes)**:
- `StorageManager` - 存儲管理器

