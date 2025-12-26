# 📋 Task Planning - 任務規劃系統

> **版本**: v2.2.0  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2025-12-21  
> **角色**: AIVA 的智能任務編排和協調引擎  
> **架構**: 純編排決策，不直接執行具體任務

**導航**: [← 返回 AIVA Core](../README.md)

---

## 📋 目錄

- [模組概述](#-模組概述)
- [子系統架構](#-子系統架構)
  - [Planner - 規劃器](#1-planner---規劃器)
  - [Executor - 執行器](#2-executor---執行器)  
  - [Coordinators - 協調器](#3-coordinators---協調器)
- [完整工作流程](#-完整工作流程)
- [性能指標](#-性能指標)

---

## 🎯 模組概述

Task Planning 是 AIVA 的任務規劃和執行系統，負責將高層次目標分解為可執行的子任務，並協調執行過程。

**核心職責**：
- 📋 **智能規劃** - 將複雜任務分解為可執行步驟和編排流程
- ⚡ **任務編排** - 支援多任務並行和依賴管理編排
- 🔄 **動態調整** - 根據AI分析結果動態調整計劃
- 📊 **進度追蹤** - 實時監控任務編排狀態和結果收集
- 🎯 **攻擊計劃映射** - AI決策映射為具體Features模組任務

---

## 🏗️ 子系統架構

### 1. Planner - 規劃器

**位置**: `task_planning/planner/`

**核心組件**：
- `enhanced_planner.py` - AI 增強任務規劃器（500+ 行）
- `task_decomposer.py` - 任務分解引擎（300+ 行）

**主要功能**：
```python
from aiva_core.task_planning.planner import EnhancedPlanner

# 初始化規劃器
planner = EnhancedPlanner(neural_core, decision_agent)

# 生成任務計劃
plan = await planner.create_plan(
    goal="掃描目標網站並發現漏洞",
    constraints={"timeout": 3600, "max_depth": 3}
)

# 計劃包含
# - 任務列表 (tasks)
# - 依賴關係 (dependencies)
# - 資源需求 (resources)
# - 預估時間 (estimated_time)
```

**特性**：
- ✅ AI 驅動的任務分解
- ✅ 依賴關係自動識別
- ✅ 資源需求評估
- ✅ 多策略規劃（BFS/DFS/啟發式）

---

### 2. Executor - 執行器

**位置**: `task_planning/executor/`

**核心組件**：
- `task_executor.py` - 任務執行引擎（600+ 行）
- `parallel_executor.py` - 並行執行管理器（400+ 行）
- `execution_monitor.py` - 執行監控器（250+ 行）

**主要功能**：
```python
from aiva_core.task_planning.executor import TaskExecutor, ParallelExecutor

# 初始化執行器
executor = TaskExecutor(features_invoker)
parallel_exec = ParallelExecutor(max_workers=5)

# 執行單個任務
result = await executor.execute_task(task)

# 並行執行多個任務
results = await parallel_exec.execute_batch(tasks)

# 監控執行進度
status = executor.get_execution_status(task_id)
```

**特性**：
- ✅ 異步並行執行
- ✅ 依賴順序管理
- ✅ 錯誤處理和重試
- ✅ 資源限制和隊列管理
- ✅ 實時進度報告

---

### 3. ⭐ UnifiedAttackExecutor - 統一攻擊執行器（新增 2025-12-17）

**位置**: `task_planning/unified_executor.py`

**核心功能**：
- 🎯 **統一執行路徑** - 靶場與實戰統一流程（消除雙重邏輯）
- 📊 **自動經驗收集** - 每次執行自動記錄經驗到 ExperienceManager
- 🎓 **自動觸發訓練** - 累積到閾值（默認 100 樣本）自動訓練
- 🔧 **可配置學習** - 可禁用學習模式（純執行）

**主要接口**：
```python
from task_planning.unified_executor import UnifiedAttackExecutor
from external_learning import ExperienceManager
from external_learning.learning import ModelTrainer

# 初始化（自動學習已內建）
executor = UnifiedAttackExecutor(
    plan_executor=plan_executor,
    experience_manager=experience_manager,
    model_trainer=model_trainer,
    rag_engine=rag_engine,
    auto_learn=True,  # 啟用自動學習
    learn_threshold=100  # 訓練閾值（樣本數）
)

# 執行攻擊（自動學習）
result = await executor.execute_with_learning(
    plan=attack_plan,
    context=task_context
)

# 純執行（不學習）
result = await executor.execute_without_learning(
    plan=attack_plan,
    context=task_context
)

# 查看學習統計
stats = executor.get_learning_stats()
print(f"已收集樣本: {stats['samples_collected']}")
print(f"訓練次數: {stats['training_runs']}")
```

**特性**：
- ✅ 代碼量減少 47%（800 行 vs 舊架構 1500 行）
- ✅ 學習覆蓋 100%（vs 舊架構 50%）
- ✅ 數據利用率提升 10x（靶場 = 實戰）
- ✅ 單一執行路徑，消除雙重邏輯

**架構優勢**：
- 消除 TrainingOrchestrator 雙重執行路徑
- 靶場和實戰數據統一收集
- 學習過程透明化和可配置
- 減少代碼重複和維護成本

**替代說明**：
> ⚠️ 此組件取代了原有的 `TrainingOrchestrator`（external_learning 模組）  
> TrainingOrchestrator 包含 40+ 錯誤且與 AI Commander 存在雙重執行邏輯  
> 詳見：[架構簡化報告](../_ARCHITECTURE_SIMPLIFICATION_REPORT_2025-12-17.md)

---

### 4. TaskContext - 標準任務參數包

**位置**: `core_capabilities/task_context.py`

**核心功能**：
- 📦 **標準化參數** - 統一任務參數結構
- 🎯 **類型特化** - ScanTaskContext, AttackTaskContext
- 🔄 **解析器** - parse_user_input_to_context()

**主要數據結構**：
```python
from core_capabilities.task_context import (
    TaskContext,
    ScanTaskContext,
    AttackTaskContext,
    parse_user_input_to_context
)

# 基礎上下文
context = TaskContext(
    objective="SQL 注入測試",
    target_info={"url": "https://target.com", "port": 443}
)

# 掃描專用上下文
scan_context = ScanTaskContext(
    objective="全面掃描",
    target_info={"url": "https://target.com"},
    scan_depth=3,
    scan_types=["ports", "web", "vulns"]
)

# 攻擊專用上下文
attack_context = AttackTaskContext(
    objective="SQL 注入",
    target_info={"url": "https://target.com/login"},
    attack_type="sqli",
    payload_config={"injection_point": "username"}
)

# 從用戶輸入解析
context = parse_user_input_to_context(
    user_input="掃描 192.168.1.1",
    context_type="scan"
)
```

**特性**：
- ✅ 類型安全的參數傳遞
- ✅ 驗證和默認值
- ✅ 可擴展的上下文類型
- ✅ 自然語言解析

**整合說明**：
TaskContext 用於標準化 AI Commander → UnifiedExecutor → Core Capabilities 的參數傳遞，確保指揮鏈集成的一致性。

---

### 5. Coordinators - 協調器

**位置**: `task_planning/coordinators/`

**核心組件**：
- `multi_scanner_coordinator.py` - 多掃描器協調（350+ 行）

**主要功能**：
```python
from aiva_core.task_planning.coordinators import MultiScannerCoordinator

# 初始化協調器
coordinator = MultiScannerCoordinator(
    scanners=["network_scanner", "web_scanner", "api_scanner"]
)

# 協調多個掃描器
results = await coordinator.coordinate_scan(
    target="https://example.com",
    scan_types=["full", "quick", "focused"]
)

# 結果合併和去重
merged_findings = coordinator.merge_findings(results)
```

**特性**：
- ✅ 多掃描器協調
- ✅ 結果合併和去重
- ✅ 衝突解決
- ✅ 優先級調度

---

## 🔄 完整工作流程

### 典型任務規劃和執行流程

```python
from aiva_core.task_planning import EnhancedPlanner, TaskExecutor, MultiScannerCoordinator
from aiva_core.cognitive_core import RealNeuralCore, EnhancedDecisionAgent

# 1. 初始化組件
neural_core = RealNeuralCore(use_5m_model=True)
decision_agent = EnhancedDecisionAgent(neural_core)
planner = EnhancedPlanner(neural_core, decision_agent)
executor = TaskExecutor(features_invoker)
coordinator = MultiScannerCoordinator()

# 2. 創建任務計劃
plan = await planner.create_plan(
    goal="全面安全評估",
    target="https://target.com",
    constraints={
        "max_time": 7200,  # 2小時
        "max_concurrent": 5,
        "scan_depth": 3
    }
)

# 3. 執行計劃
execution_id = await executor.start_execution(plan)

# 4. 監控執行
while not executor.is_complete(execution_id):
    status = executor.get_status(execution_id)
    print(f"進度: {status.progress}%, 已完成: {status.completed_tasks}/{status.total_tasks}")
    await asyncio.sleep(5)

# 5. 獲取結果
results = await executor.get_results(execution_id)
summary = coordinator.generate_summary(results)
```

### 動態調整示例

```python
# 根據執行結果動態調整計劃
async def adaptive_execution(planner, executor, initial_plan):
    execution_id = await executor.start_execution(initial_plan)
    
    while not executor.is_complete(execution_id):
        status = executor.get_status(execution_id)
        
        # 檢查是否需要調整
        if status.should_adjust:
            # 重新規劃
            new_plan = await planner.replan(
                current_state=status,
                findings=status.intermediate_results
            )
            
            # 更新執行計劃
            await executor.update_plan(execution_id, new_plan)
        
        await asyncio.sleep(10)
    
    return await executor.get_results(execution_id)
```

---

## 📊 性能指標

### 規劃性能
- **規劃速度**: ~200ms (簡單任務), ~1s (複雜任務)
- **分解深度**: 最多 5 層嵌套
- **任務數量**: 單次規劃支援 100+ 子任務

### 執行性能
- **並行度**: 最多 20 個並行任務
- **執行延遲**: ~10ms (任務調度)
- **監控開銷**: <5% CPU

### 協調性能
- **掃描器數量**: 支援 10+ 個掃描器
- **結果合併**: ~100ms (1000條發現)
- **去重效率**: >98%

---

## 🔗 相關模組

- [Cognitive Core](../cognitive_core/README.md) - 提供 AI 規劃能力
- [Integration](../integration/README.md) - Features 調用和外部整合
- [Persistence](../persistence/README.md) - 任務狀態持久化

---

**最後更新**: 2025-12-01 | **維護者**: AIVA Team
