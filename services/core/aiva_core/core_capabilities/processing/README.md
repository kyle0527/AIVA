# Processing 處理模組

> **路徑**: `services/core/aiva_core/core_capabilities/processing`  
> **狀態**: ✅ 正常 | **Python 文件數**: 2 | **最後更新**: 2026-04-05

## 概述

掃描結果處理器，封裝核心引擎處理掃描結果的完整七階段流程，支援兩階段掃描（Phase0/Phase1）。

## 核心組件

### scan_result_processor.py
- `ScanResultProcessor` - 掃描結果處理器
  - 執行七階段處理流程
  - 協調各子系統協作
  - 統一的處理入口點

### __init__.py
- 模組初始化和導出

## 七階段處理流程

| 階段 | 名稱 | 描述 |
|------|------|------|
| 1 | 資料接收與預處理 | Data Ingestion |
| 2 | 初步攻擊面分析 | Initial Attack Surface Analysis |
| 3 | 測試策略生成 | Test Strategy Generation |
| 4 | 動態策略調整 | Dynamic Strategy Adjustment |
| 5 | 任務生成 | Task Generation |
| 6 | 任務佇列管理與分發 | Task Queue Management & Distribution |
| 7 | 執行狀態監控 | Execution Status Monitoring |

## 兩階段掃描支援

- **Phase0**: 快速偵察（5-10 分鐘）
- **Phase1**: 深度掃描（10-30 分鐘）

## 依賴關係

- `core_capabilities.ingestion.ScanModuleInterface` - 資料接收
- `core_capabilities.analysis.InitialAttackSurface` - 攻擊面分析
- `cognitive_core.learning_system.analysis.StrategyAdjuster` - 策略調整
- `task_planning.planner.TaskGenerator` - 任務生成
- `task_planning.executor.TaskQueueManager` - 任務佇列管理
- `service_backbone.state.SessionStateManager` - 會話狀態管理
- `aiva_common.schemas` - ScanCompletedPayload, Phase0CompletedPayload
