# Performance 效能監控模組

> **路徑**: `service_backbone/performance/`  
> **狀態**: ✅ 正常 | **文件數**: 6 | **最後更新**: 2026-01-21  
> **父模組**: [Service Backbone](../README.md)

## 概述

系統效能監控和資源管理模組，提供指標收集、健康狀態監控、並行處理、系統診斷和統一記憶體管理功能。

## 📄 檔案詳細資訊 (Files Details)

### `diagnose.py`
**說明**: AIVA 系統診斷工具

**函式 (Functions)**:
- `print_header()`
- `check_docker()` - 檢查 Docker 靶場狀態

### `health_check.py`
**說明**: AIVA 系統健康檢查器

**函式 (Functions)**:
- `check_schemas()` - 檢查 AIVA Common Schemas 可用性
- `check_tools()` - 檢查專業分析工具可用性
- `check_ai_explorer()` - 檢查 AI 系統探索器可用性
- `check_directories()` - 檢查關鍵目錄結構

### `monitoring.py`
**說明**: 系統整合監控模組

**類別 (Classes)**:
- `ComponentHealth` - 組件健康狀態
- `Metric` - 監控指標
- `MetricsCollector` - 效能指標收集器 - 中央監控服務
**函式 (Functions)**:
- `monitor_performance()` - 效能監控裝飾器

### `parallel_processor.py`
**說明**: 並行訊息處理器模組

**類別 (Classes)**:
- `ParallelMessageProcessor` - 並行訊息處理器 - 替代原本的單線程處理

### `unified_memory_manager.py`
**說明**: 統一記憶體管理器 - 整合AI專用與通用記憶體管理功能

**類別 (Classes)**:
- `UnifiedMemoryManager` - 統一記憶體管理器 - 整合AI快取與系統記憶體管理
- `ComponentPool` - 組件對象池 - 避免頻繁建立/銷毀對象

