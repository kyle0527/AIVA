# Ingestion 資料接收模組

> **路徑**: `services/core/aiva_core/core_capabilities/ingestion`  
> **狀態**: ✅ 正常 | **文件數**: 2 | **最後更新**: 2026-01-07

## 概述

掃描模組介面層，負責接收掃描模組的原始數據並進行標準化處理，包含格式檢測、資料清理、去重、豐富化等功能。

## 核心組件

### scan_module_interface.py
- `ScanModuleInterface` - 掃描模組介面
  - `process_scan_data()` - 處理掃描模組回傳的原始數據
  - `_process_assets()` - 處理資產清單，進行分類與標準化
  - `_process_fingerprints()` - 處理指紋識別結果
  - 支援 Phase0/Phase1 兩階段掃描流程

### __init__.py
- 模組初始化和導出

## 數據處理流程

```
掃描模組原始數據
       ↓
ScanModuleInterface.process_scan_data()
       ↓
標準化處理後的資料結構
       ↓
進入七階段處理流程
```

## 輸入格式

- `ScanCompletedPayload` - 掃描完成負載
- `Phase0StartPayload` / `Phase0CompletedPayload` - Phase0 相關
- `Phase1StartPayload` - Phase1 開始負載

## 依賴關係

- `aiva_common.enums` - Topic 枚舉
- `aiva_common.mq` - AbstractBroker 消息代理
- `aiva_common.schemas` - 標準化數據結構
- `aiva_common.utils` - 日誌工具
