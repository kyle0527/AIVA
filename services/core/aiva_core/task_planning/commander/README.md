# 🎯 Commander - AI 指揮協調器

> **版本**: v2.5.0  
> **狀態**: ✅ 生產就緒（重構完成）  
> **最後更新**: 2026-01-07  
> **父模組**: [Task Planning](../README.md)  
> **符合規範**: [aiva_common](../../../../aiva_common/README.md)  
> **檔案數**: 8 個 Python 模組  
> **代碼行數**: 2,029 行

---

## 📋 目錄

- [模組概述](#-模組概述)
- [重構說明](#-重構說明)
- [核心組件](#-核心組件)
- [使用範例](#-使用範例)
- [API 參考](#-api-參考)

---

## 🎯 模組概述

Commander 是 Task Planning 的 AI 指揮協調子模組，負責協調 AI 決策引擎和攻擊計劃執行。

**核心職責**：
- 🎯 **策略決策** - 基於 5M 架構的攻擊策略選擇
- 📋 **計劃建構** - RAG 增強的攻擊計劃生成
- ⚡ **攻擊協調** - 協調多步驟攻擊執行
- 📚 **學習適配** - 與學習系統整合

---

## 🔄 重構說明

**重構日期**: 2026-01-06  
**方案**: 方案 B - 完全替換（無 facade 保留）

### 重構前
- **檔案**: `ai_commander.py`
- **大小**: 2,114 行 / 73 KB
- **問題**: 單一巨型類，違反單一責任原則

### 重構後
- **結構**: `commander/` 子模組
- **檔案數**: 7 個專職模組
- **最大檔案**: 380 行

詳細驗證報告：[COMMANDER_REFACTOR_VERIFICATION.md](../../../../../COMMANDER_REFACTOR_VERIFICATION.md)

---

## 🏗️ 核心組件

### 1. CommanderCoordinator (`__init__.py`)

主協調器，提供統一的命令入口。

**主要功能**：
```python
from services.core.aiva_core.task_planning.commander import (
    CommanderCoordinator,
    AICommander,  # 向後兼容別名
    AITaskType,
)

# 初始化協調器
coordinator = CommanderCoordinator(data_directory=Path("./data"))

# 執行命令
result = await coordinator.execute_command(
    task_type=AITaskType.ATTACK_PLANNING,
    context={"target": "example.com"},
)
```

### 2. Types (`types.py`)

模組專屬類型定義。

**主要枚舉**：
- `AITaskType` - AI 任務類型（attack_planning, strategy_decision 等）
- `AIComponent` - AI 組件類型（decision_engine_5m, rag_engine 等）

### 3. CapabilityManager (`capability_manager.py`)

能力選單管理器。

```python
from services.core.aiva_core.task_planning.commander import CapabilityManager

manager = CapabilityManager()
capabilities = manager.get_available_capabilities()
```

### 4. PlanBuilder (`plan_builder.py`)

攻擊計劃建構器，支持 RAG 增強。

```python
from services.core.aiva_core.task_planning.commander import PlanBuilder

builder = PlanBuilder(data_directory=Path("./data/plans"))
plan = await builder.build_attack_plan(context)
```

### 5. StrategyEngine (`strategy_engine.py`)

策略決策引擎。

```python
from services.core.aiva_core.task_planning.commander import StrategyEngine

engine = StrategyEngine()
strategy = await engine.decide_strategy(context)
```

### 6. AttackCoordinator (`attack_coordinator.py`)

攻擊執行協調器。

```python
from services.core.aiva_core.task_planning.commander import AttackCoordinator

coordinator = AttackCoordinator()
result = await coordinator.coordinate_attack(plan)
```

### 7. LearningAdapter (`learning_adapter.py`)

學習系統適配器。

```python
from services.core.aiva_core.task_planning.commander import LearningAdapter

adapter = LearningAdapter()
await adapter.record_experience(execution_result)
```

---

## 📊 架構圖

```
CommanderCoordinator
        │
        ├── CapabilityManager   (能力管理)
        ├── PlanBuilder         (計劃建構)
        ├── StrategyEngine      (策略決策)
        ├── AttackCoordinator   (攻擊協調)
        └── LearningAdapter     (學習適配)
```

---

## 🔗 依賴關係

**內部依賴**：
- `services.aiva_common.schemas` - 標準化數據結構
- `services.aiva_common.enums` - 標準枚舉

**外部依賴**：
- `cognitive_core` - AI 決策支持
- `learning_system` - 學習系統整合

---

## 📝 符合規範

本模組符合 aiva_common 規範：
- ✅ 無重複定義枚舉（僅定義模組專屬的 AITaskType, AIComponent）
- ✅ 正確使用相對路徑 import
- ✅ 模組專屬枚舉合理分離
- ✅ 提供向後兼容別名（AICommander = CommanderCoordinator）
