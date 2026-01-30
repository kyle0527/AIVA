# Performance 效能監控模組

> **路徑**: `service_backbone/performance/`  
> **狀態**: ✅ 正常 | **文件數**: 6 | **最後更新**: 2026-01-21  
> **父模組**: [Service Backbone](../README.md)

## 概述

系統效能監控和資源管理模組，提供指標收集、健康狀態監控、並行處理、系統診斷和統一記憶體管理功能。

## 核心組件

### health_check.py ⭐ 新增
- `check_schemas()` - Schema 完整性檢查
- `check_tools()` - 工具可用性檢查
- `check_directories()` - 目錄結構檢查

### diagnose.py ⭐ 新增
- `check_engines()` - 引擎狀態檢查 (Go/Rust/Node)
- `check_docker()` - Docker 環境檢查
- `check_http()` - HTTP 服務檢查

### monitoring.py
- `ComponentHealth` - 組件健康狀態枚舉（str, Enum）
- `Metric` - 指標結構
- `MetricsCollector` - 指標收集器
  - 收集系統指標
  - 效能數據聚合
  - 健康狀態追蹤

### parallel_processor.py
- `ParallelMessageProcessor` - 並行消息處理器
  - 多線程消息處理
  - 任務隊列管理
  - 併發控制

### unified_memory_manager.py
- `UnifiedMemoryManager` - 統一記憶體管理器
  - 記憶體分配和釋放
  - 記憶體池管理
  - 垃圾回收優化
  - 記憶體使用統計

- `ComponentPool` - 組件池
  - 組件實例池化
  - 資源重用
  - 生命週期管理

### __init__.py
- 模組初始化和導出

## 監控指標類型

| 指標類型 | 描述 |
|----------|------|
| CPU 使用率 | 處理器負載 |
| 記憶體使用 | RAM 佔用 |
| 處理延遲 | 請求回應時間 |
| 隊列深度 | 待處理任務數 |
| 錯誤率 | 失敗請求比例 |

## 依賴關係

- `enum` - 枚舉類型支援
- `threading` - 多線程支援
- `collections` - 數據結構
- 無外部套件依賴（純 Python 實現）
