# Adapters 適配器模組

> **路徑**: `services/core/aiva_core/service_backbone/adapters`  
> **狀態**: ✅ 正常 | **文件數**: 2 | **最後更新**: 2026-01-07

## 概述

協議適配器層，提供抽象的協議適配接口和具體實現，用於統一不同通訊協議的處理方式。

## 📄 檔案詳細資訊 (Files Details)

### `protocol_adapter.py`
**說明**: 協議適配器 - Gang of Four Adapter 設計模式實現

**類別 (Classes)**:
- `ProtocolAdapter` - 協議適配器抽象基類
- `HttpProtocolAdapter` - HTTP 協議適配器
**函式 (Functions)**:
- `create_http_adapter()` - 創建 HTTP 協議適配器實例

