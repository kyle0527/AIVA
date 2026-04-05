# 📋 Planner - 任務規劃器

> **版本**: v2.5.0  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2026-04-05  
> **父模組**: [Task Planning](../README.md)  
> **符合規範**: [aiva_common](../../../../aiva_common/README.md)  
> **檔案數**: 7 個 Python 模組  
> **代碼行數**: 1,869 行

---

## 📋 目錄

- [模組概述](#-模組概述)
- [核心組件](#-核心組件)
- [使用範例](#-使用範例)
- [API 參考](#-api-參考)

---

## 🎯 模組概述

Planner 是 Task Planning 的任務規劃子模組，負責將高層次目標分解為可執行的任務序列。

**核心職責**：
- 📋 **任務分解** - 將複雜目標分解為原子任務
- 🔧 **工具選擇** - 為每個任務選擇最適合的工具
- 📊 **計劃生成** - 生成完整的執行計劃
- 🔄 **AST 解析** - 解析和處理任務抽象語法樹

---

## 🏗️ 核心組件

### 1. ExecutionPlanner (`execution_planner.py`)

執行計劃生成器，創建完整的執行計劃。

**主要功能**：
```python
from services.core.aiva_core.task_planning.planner import ExecutionPlanner

planner = ExecutionPlanner()

# 生成執行計劃
plan = await planner.create_plan(
    goal="掃描目標並發現漏洞",
    constraints={"timeout": 3600},
)
```

### 2. TaskGenerator (`task_generator.py`)

任務生成器，從目標生成具體任務。

**主要功能**：
```python
from services.core.aiva_core.task_planning.planner import TaskGenerator

generator = TaskGenerator()

# 生成任務列表
tasks = await generator.generate_tasks(
    objective=objective,
    context=scan_context,
)
```

### 3. ToolSelector (`tool_selector.py`)

工具選擇器，為任務選擇最適合的工具。

**主要類別**：
- `ToolDecision` - 工具決策結果
- `ToolSelector` - 工具選擇邏輯

```python
from services.core.aiva_core.task_planning.planner import ToolSelector

selector = ToolSelector()

# 選擇工具
decision = selector.select_tool(task)
```

### 4. TaskConverter (`task_converter.py`)

任務轉換器，將抽象任務轉換為可執行任務。

**主要類別**：
- `ExecutableTask` - 可執行任務定義

### 5. ASTParser (`ast_parser.py`)

抽象語法樹解析器，解析任務定義。

### 6. PlanComparator (`plan_comparator.py`)

計劃比較器，比較和評估不同的執行計劃。

---

## 📊 數據流

```
Goal → TaskGenerator → TaskConverter → ExecutableTask
         ↓                 ↓
   ASTParser         ToolSelector
         ↓                 ↓
 ExecutionPlanner  →   ExecutionPlan
```

---

## 🔗 依賴關係

**內部依賴**：
- `services.aiva_common.schemas` - 標準化數據結構
- `services.aiva_common.enums` - 標準枚舉

**外部依賴**：
- `cognitive_core` - AI 決策支持

---

## 📝 符合規範

本模組符合 aiva_common 規範：
- ✅ 使用標準化任務結構
- ✅ 正確的數據傳輸格式
- ✅ 無重複定義
- ✅ 模組專屬類型合理分離（如 ExecutableTask, ToolDecision）
