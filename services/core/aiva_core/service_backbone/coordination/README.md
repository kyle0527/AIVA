# Coordination 協調模組

> **路徑**: `service_backbone/coordination/`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-21  
> **父模組**: [Service Backbone](../README.md)

## 概述

核心服務協調層，負責狀態管理、服務工廠、命令路由和配置管理。注意：這是被動的狀態管理器，不是系統主線程。

## 📄 檔案詳細資訊 (Files Details)

### `ai_controller.py`
**說明**: AIVA AI 子系統控制器 - 5M Decision Engine 的專門模組

**類別 (Classes)**:
- `AISummaryPluginProtocol` - 摘要插件協議
- `AISubsystemController` - AIVA AI 子系統控制器 - 避免與主控制器衝突

### `ai_manager.py`
**說明**: AIVA AI 組件管理器

**類別 (Classes)**:
- `ComponentStatus` - 組件狀態枚舉
- `ComponentHealth` - 組件健康狀態
- `SystemMetrics` - 系統指標
- `AIComponentManager` - AI 組件持續運作管理器

### `core_service_coordinator.py`
**說明**: AIVA Core Service Coordinator - 核心服務協調器

**類別 (Classes)**:
- `AIVACoreServiceCoordinator` - AIVA 核心服務協調器（狀態管理器模式）
**函式 (Functions)**:
- `get_core_service_coordinator()` - 獲取核心服務協調器實例

