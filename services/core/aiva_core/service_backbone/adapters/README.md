# Adapters - 協議適配器

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
- [🔄 協議轉換流程](#-協議轉換流程)
- [💡 使用場景](#-使用場景)
- [📚 相關模組](#-相關模組)

---

## 📋 概述

**定位**: 協議轉換和適配層
**狀態**: ✅ 已實現  
**文件數**: 1 個 Python 文件 (200 行)

## 📂 文件結構

```
adapters/
├── protocol_adapter.py (200 行) - 協議適配器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### protocol_adapter.py

**職責**: 提供不同協議間的轉換和適配

**主要類/函數**:
- `ProtocolAdapter` - 協議適配器基類
- 支援 HTTP, WebSocket, gRPC 等協議轉換

**使用場景**:
- 統一不同服務的通信協議
- 外部系統集成適配
- 遺留系統協議橋接

**使用範例**:
```python
from aiva_core.service_backbone.adapters import ProtocolAdapter

adapter = ProtocolAdapter()
converted_data = adapter.convert(data, from_protocol="http", to_protocol="grpc")
```

## 📚 相關模組

- [messaging](../messaging/README.md) - 消息傳遞
- [api](../api/README.md) - API 服務

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準消息格式
from aiva_common import AivaMessage, MessageHeader

# ❌ 禁止：自創消息格式
class CustomMessage(BaseModel): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
