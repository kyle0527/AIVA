# AuthZ 授權管理模組

> **路徑**: `services/core/aiva_core/service_backbone/authz`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-07

## 概述

授權和權限管理系統，提供權限矩陣、風險評估和視覺化功能。

## 核心組件

### permission_matrix.py
- `PermissionMatrix` - 權限矩陣管理器
  - 定義和管理操作權限
  - 權限檢查和驗證
  - 權限變更追蹤

- `RiskLevel` - 風險等級枚舉（str, Enum）
- `OperationContext` - 操作上下文結構
- `RiskGuard` - 風險守衛
  - 評估操作風險
  - 阻止高風險操作
  - 風險緩解建議

### authz_mapper.py
- `AuthZMapper` - 授權映射器
  - 映射用戶角色到權限
  - 動態權限計算
  - 繼承權限處理

### matrix_visualizer.py
- `MatrixVisualizer` - 矩陣視覺化器
  - 生成權限矩陣圖表
  - 支援多種輸出格式
  - 差異對比視覺化

### __init__.py
- 模組初始化和導出

## 權限檢查流程

```
操作請求
    ↓
AuthZMapper（角色→權限映射）
    ↓
PermissionMatrix（權限檢查）
    ↓
RiskGuard（風險評估）
    ↓
允許/拒絕
```

## 依賴關係

- `enum` - 枚舉類型支援
- 無外部套件依賴
