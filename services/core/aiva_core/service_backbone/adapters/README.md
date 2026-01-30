# Adapters 適配器模組

> **路徑**: `services/core/aiva_core/service_backbone/adapters`  
> **狀態**: ✅ 正常 | **文件數**: 2 | **最後更新**: 2026-01-07

## 概述

協議適配器層，提供抽象的協議適配接口和具體實現，用於統一不同通訊協議的處理方式。

## 核心組件

### protocol_adapter.py
- `ProtocolAdapter` (ABC) - 協議適配器抽象基類
  - 定義協議適配的標準接口
  - 支援請求/回應處理
  - 錯誤處理機制

- `HttpProtocolAdapter` - HTTP 協議適配器
  - 實現 HTTP 請求/回應處理
  - 支援 GET、POST、PUT、DELETE 等方法
  - 處理 HTTP 標頭和狀態碼

### __init__.py
- 模組初始化和導出

## 適配器模式

```
外部請求
    ↓
ProtocolAdapter (抽象)
    ↓
HttpProtocolAdapter (具體實現)
    ↓
內部服務調用
```

## 擴展指南

若需新增協議支援，繼承 `ProtocolAdapter` 並實現所有抽象方法：

```python
class WebSocketAdapter(ProtocolAdapter):
    async def handle_request(self, request):
        # 實現 WebSocket 請求處理
        pass
```

## 依賴關係

- `abc` - 抽象基類支援
- 無外部套件依賴（純 Python 實現）
