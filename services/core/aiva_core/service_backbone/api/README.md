# API - 統一 API 服務層

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [app.py](#apppy-282-行)
  - [unified_function_caller.py](#unified_function_callerpy-476-行-)
  - [enhanced_unified_caller.py](#enhanced_unified_callerpy-304-行)
- [🔄 API 調用流程](#-api-調用流程)
- [🔒 安全機制](#-安全機制)
- [📚 相關模組](#-相關模組)
- [🔧 配置示例](#-配置示例)

---

## 📋 概述

**定位**: 統一函數調用和 API 接口層  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (1,062 行)

## 📂 文件結構

```
api/
├── app.py (282 行) - FastAPI 應用主體
├── enhanced_unified_caller.py (304 行) - 增強統一調用器
├── unified_function_caller.py (476 行) - 統一函數調用器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### app.py

**職責**: FastAPI 應用初始化和路由配置

**主要類/函數**:
- `create_app()` - 創建 FastAPI 應用實例
- 健康檢查端點
- API 路由註冊

**使用範例**:
```python
from aiva_core.service_backbone.api import create_app

app = create_app()
# 應用已配置完整的路由和中間件
```

---

### unified_function_caller.py (476 行)

**職責**: 統一的函數調用接口,支援動態能力調用

**主要類/函數**:
- `UnifiedFunctionCaller` - 統一調用器類
- `call_capability(name, params)` - 調用註冊的能力
- `list_capabilities()` - 列出所有可用能力

**使用場景**:
- 動態能力調用
- 跨模組函數調用
- API 層到能力層的橋接

**使用範例**:
```python
from aiva_core.service_backbone.api import UnifiedFunctionCaller

caller = UnifiedFunctionCaller()
result = await caller.call_capability(
    name="scan_sql_injection",
    params={"target": "https://example.com"}
)
```

---

### enhanced_unified_caller.py (304 行)

**職責**: 增強版統一調用器,支援更多特性

**增強功能**:
- ✅ 調用追蹤和日誌記錄
- ✅ 參數驗證和轉換
- ✅ 錯誤處理和重試機制
- ✅ 性能監控和指標收集

**使用範例**:
```python
from aiva_core.service_backbone.api import EnhancedUnifiedCaller

caller = EnhancedUnifiedCaller(
    enable_tracing=True,
    retry_on_failure=True
)

result = await caller.call_with_validation(
    capability="xss_scanner",
    params={"url": "https://target.com"}
)
```

## 🔗 整合關係

```
API 層架構:
    FastAPI App (app.py)
        ↓
    EnhancedUnifiedCaller
        ↓
    UnifiedFunctionCaller
        ↓
    CapabilityRegistry (core_capabilities)
        ↓
    實際能力函數執行
```

## 📚 相關模組

- [core_capabilities](../../core_capabilities/README.md) - 能力註冊表
- [messaging](../messaging/README.md) - 消息傳遞
- [coordination](../coordination/README.md) - 服務協調

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準類型
from aiva_common import ModuleName, TaskStatus, AivaMessage

# ❌ 禁止：自定義狀態
class APIStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
