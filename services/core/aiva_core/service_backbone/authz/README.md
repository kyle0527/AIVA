# AuthZ 授權管理模組

> **路徑**: `services/core/aiva_core/service_backbone/authz`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-07

## 概述

授權和權限管理系統，提供權限矩陣、風險評估和視覺化功能。

## 📄 檔案詳細資訊 (Files Details)

### `authz_mapper.py`
**說明**: AuthZ Mapper - 權限映射器

**類別 (Classes)**:
- `AuthZMapper` - 權限映射器
**函式 (Functions)**:
- `main()` - 測試範例

### `matrix_visualizer.py`
**說明**: Matrix Visualizer - 權限矩陣視覺化

**類別 (Classes)**:
- `MatrixVisualizer` - 權限矩陣視覺化器
**函式 (Functions)**:
- `main()` - 測試範例

### `permission_matrix.py`
**說明**: 無特定描述。

**類別 (Classes)**:
- `PermissionMatrix` - 權限矩陣
- `RiskLevel` - 風險等級枚舉 (整合自 aiva_core_v1)
- `OperationContext` - 操作上下文
- `RiskGuard` - 風險控制守衛 (整合自 aiva_core_v1 Guard)
**函式 (Functions)**:
- `main()` - 測試範例
- `get_risk_guard()` - 獲取全域風險守衛實例
- `authorize_operation()` - 便捷的操作授權函數

