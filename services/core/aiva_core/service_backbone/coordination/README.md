# Coordination 協調模組

> **路徑**: `service_backbone/coordination/`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-21  
> **父模組**: [Service Backbone](../README.md)

## 概述

核心服務協調層，負責狀態管理、服務工廠、命令路由和配置管理。注意：這是被動的狀態管理器，不是系統主線程。

## 核心組件

### ai_manager.py ⭐ 新增
- `AIComponentManager` - AI 組件管理器
  - 遵循 SOT (Single Source of Truth) 原則
  - 組件生命週期管理
  - 系統指標收集
- `ComponentStatus` - 組件狀態枚舉
- `ComponentHealth` - 組件健康資訊
- `SystemMetrics` - 系統指標數據類

### core_service_coordinator.py
- `AIVACoreServiceCoordinator` - 核心服務協調器
  - **職責**:
    - 狀態管理 - 管理服務實例和狀態
    - 服務工廠 - 提供服務創建和初始化
    - 命令路由 - 協調命令處理流程
    - 上下文管理 - 管理執行上下文
    - 配置管理 - 處理配置變更
  - **不負責**:
    - ❌ 系統主線程
    - ❌ 主動循環運行
    - ❌ 作為系統入口點

### ai_controller.py
- `AISubsystemController` - AI 子系統控制器
  - 管理 AI 相關子系統
  - 協調神經網路和決策引擎

### optimized_core.py
- `OptimizedBioNet` - 優化的生物神經網路
- `AIVAAutonomyProof` - AIVA 自主性證明

## 架構位置

```
app.py (FastAPI)          ← 系統唯一入口
    ↓ 持有和調用
CoreServiceCoordinator    ← 狀態管理器（本模組）
    ↓ 管理和提供
各功能服務 (Decision, Planning, Execution...)
```

## 依賴關係

- `aiva_common.config_manager` - ConfigChangeEvent, get_config_manager
- `aiva_common.cross_language` - error_handler, get_cross_language_service
- `aiva_common.error_handling` - AIVAError, ErrorSeverity, ErrorType
- `aiva_common.monitoring` - MetricType, get_monitoring_service, trace_operation
