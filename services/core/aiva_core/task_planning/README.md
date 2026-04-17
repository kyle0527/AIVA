# 📋 Task Planning - 任務規劃系統

> **路徑**: `task_planning/`  
> **狀態**: ✅ Production Ready | **最後更新**: 2026-04-05  
> **子模組**: 3 個 | **總文件數**: 30 | **Bug Bounty 整合**: ✅ 已完成  
> **父模組**: [AIVA Core](../README.md)

## 概述

**Task Planning** 是 AIVA 五大核心模組之一，作為任務規劃和執行系統。負責將高層次目標分解為可執行的子任務，並通過 CLI 命令協調執行過程。採用 CLI 命令執行架構（subprocess）。

**核心職責**：
- 📋 **智能規劃** - 將複雜任務分解為可執行步驟和編排流程
- ⚡ **CLI 執行** - 使用 subprocess 直接執行 CLI 命令
- 🎯 **Bug Bounty 決策** - 智慧掃描工具選擇，HackerOne 實戰優化
- 🔄 **動態調整** - 根據 AI 分析結果動態調整計劃
- 📊 **進度追蹤** - 實時監控任務編排狀態和結果收集
- 🔗 **Internal Exploration 整合** - 與 internal_exploration 分析引擎深度整合

---

## 架構

### 子模組結構

| 子模組 | 功能 | 文件數 | 文檔 |
|--------|------|--------|------|
| [commander/](commander/README.md) | AI 指揮協調器、Bug Bounty 決策整合 | 9 | [README](commander/README.md) |
| [executor/](executor/README.md) | 計劃執行器、任務執行、狀態監控 | 7 | [README](executor/README.md) |
| [planner/](planner/README.md) | 執行計劃生成、任務生成、工具選擇 | 8 | [README](planner/README.md) |

---

## 🎯 Bug Bounty 整合

### 根目錄組件 (5 個文件)

- `unified_executor.py` - 統一攻擊執行器，靶場與實戰統一 (841 行)
- `command_builder.py` - AI 決策到 CLI 命令生成器
- `command_router.py` - 智能命令路由系統
- `dispatcher.py` - 任務規劃發送器，跨模組通信，整合 internal_exploration
- `__init__.py` - 模組初始化

> **注意**: `mode_manager.py` 已被徹底歸檔移除，攻擊強度現完全由 `target_sensitivity` (0.0-1.0) 參數控制。

---

## 主要類別

| 類別 | 文件 | 說明 |
|------|------|------|
| **`AttackCoordinator`** | **commander/attack_coordinator.py** | **攻擊協調器 (含 Bug Bounty 決策)** ⭐ |
| `UnifiedAttackExecutor` | unified_executor.py | 統一攻擊執行器，持續學習 |
| `CommandBuilder` | command_builder.py | AI 決策到 CLI 命令生成 |
| `CommandRouter` | command_router.py | 智能命令路由器 |
| `PlanningDispatcher` | dispatcher.py | 任務規劃統一發送器 |
| `StrategyEngine` | commander/strategy_engine.py | 策略引擎 |
| `PlanExecutor` | executor/plan_executor.py | 計劃執行器 |
| `TaskExecutor` | executor/task_executor.py | 任務執行器 |
| `ExecutionPlanner` | planner/execution_planner.py | 執行計劃生成器 |
| `TaskGenerator` | planner/task_generator.py | 任務生成器 |

---

## 依賴關係

**外部依賴**：
- `subprocess` - CLI 命令執行
- `asyncio` - 異步執行
- `pydantic` - 數據驗證

**內部依賴**：
- `aiva_common.utils` - 通用工具
- `aiva_common.error_handling` - 錯誤處理
- `service_backbone.messaging` - 消息代理
- `services.integration.capability` - 能力註冊
- `internal_exploration` - 分析引擎和 Python 工具 ⭐

---

**導航**: [← 返回 AIVA Core](../README.md)

---

## 📂 子模組 (Submodules)

- [commander](./commander/README.md)
- [executor](./executor/README.md)
- [planner](./planner/README.md)

## 📄 檔案概覽 (Files Overview)

- `command_builder.py` - Command Builder - AI 決策到 CLI 命令生成器
- `command_router.py` - AIVA Command Router - 智能命令路由系統
- `dispatcher.py` - Task Planning Dispatcher - 任務規劃發送器
- `strategy_profiles.py` - 攻擊策略配置檔案 - 統一策略架構 v2.0
- `unified_executor.py` - 統一攻擊執行器 - 靶場與實戰統一，持續學習

