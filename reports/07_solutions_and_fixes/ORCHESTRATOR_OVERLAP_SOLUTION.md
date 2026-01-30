# AIVA 職責重疊解決方案

**分析日期**: 2026-01-03  
**架構約束**: 必須維持六大模組邊界  
**問題**: 11 個 Orchestrator/Dispatcher 類別職責重疊

---

## 🔍 問題分析

### 當前 11 個編排器

| 編號 | 類別名稱 | 位置 | 主要職責 | 重疊度 |
|------|---------|------|---------|--------|
| 1 | **CapabilityOrchestrator** | cognitive_core/ | Flow 編排、能力路由 | 🔴 高 |
| 2 | **NeuralIntegrationOrchestrator** | cognitive_core/neural/ | 神經網路統一介面 | 🟡 中 |
| 3 | **TaskOrchestrator** | task_planning/ | 任務分解、規劃 | 🟡 中 |
| 4 | **LearningOrchestrator** | external_learning/ | 學習策略編排 | 🟢 低 |
| 5 | **ExplorationOrchestrator** | internal_exploration/ | 自我探索、記憶管理 | 🟢 低 |
| 6 | **ServiceOrchestrator** | service_backbone/ | 服務生命週期管理 | 🟡 中 |
| 7 | **CapabilityDispatcher** | cognitive_core/ | 能力調度 | 🔴 高 |
| 8 | **TaskDispatcher** | task_planning/ | 任務分發 | 🟡 中 |
| 9 | **LearningDispatcher** | external_learning/ | 學習任務分發 | 🟢 低 |
| 10 | **ReasoningEngine** | cognitive_core/ | 推理邏輯執行 | 🟢 低 |
| 11 | **DecisionEngine** | cognitive_core/ | 決策執行 | 🟢 低 |

### 重疊類型分析

#### 🔴 類型 1: 能力路由重疊
```
CapabilityOrchestrator (cognitive_core)
    ├── 負責: Flow 編排、能力發現、動態加載
    └── 功能: 接收 flow_id → 查找 → 加載模組 → 執行

CapabilityDispatcher (cognitive_core)
    ├── 負責: 能力調度、參數傳遞
    └── 功能: 接收能力請求 → 參數驗證 → 調度執行

❌ 問題: 兩者都做"找能力→執行"，邊界模糊
```

#### 🟡 類型 2: 任務管理重疊
```
TaskOrchestrator (task_planning)
    ├── 負責: 任務分解、依賴分析、執行規劃
    └── 功能: 複雜任務 → 子任務圖 → 執行順序

TaskDispatcher (task_planning)
    ├── 負責: 子任務分發、負載均衡
    └── 功能: 子任務 → 工作隊列 → 分配執行

ServiceOrchestrator (service_backbone)
    ├── 負責: 服務協調、資源管理
    └── 功能: 服務發現 → 健康檢查 → 調度

❌ 問題: 三者都涉及"任務→執行單元"的映射
```

#### 🟢 類型 3: 專業功能（無重疊）
```
LearningOrchestrator (external_learning)
    └── 獨特職責: 學習策略選擇、訓練流程編排

ExplorationOrchestrator (internal_exploration)
    └── 獨特職責: 自我反思、經驗存檔

NeuralIntegrationOrchestrator (cognitive_core/neural)
    └── 獨特職責: 神經網路模型統一介面

ReasoningEngine / DecisionEngine (cognitive_core)
    └── 獨特職責: 推理和決策邏輯
```

---

## 🎯 解決方案設計

### 原則
✅ 維持六大模組邊界  
✅ 單一職責原則（SRP）  
✅ 向後兼容（漸進式重構）  
✅ 清晰的層次結構

### 建議架構: 三層編排模型

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 1: 統一調度層                        │
│               (位於 cognitive_core)                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  UnifiedOrchestrator (新增)                                   │
│    ├── 職責: 全局路由、模組協調、生命週期管理                   │
│    ├── 介面: orchestrate(request_type, params)                │
│    └── 路由邏輯:                                               │
│         - capability_execution → CapabilityOrchestrator       │
│         - task_planning → TaskOrchestrator                    │
│         - learning_strategy → LearningOrchestrator            │
│         - exploration → ExplorationOrchestrator                │
│         - service_management → ServiceOrchestrator            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓ 路由到
┌─────────────────────────────────────────────────────────────┐
│                 Layer 2: 模組編排層                           │
│              (各模組內部的 Orchestrator)                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  CapabilityOrchestrator (cognitive_core)                      │
│    ├── 合併: CapabilityDispatcher 的調度功能                   │
│    ├── 職責: Flow 編排、能力組合                               │
│    └── 介面: execute_flow(flow_id, context)                   │
│                                                               │
│  TaskOrchestrator (task_planning)                             │
│    ├── 合併: TaskDispatcher 的分發功能                         │
│    ├── 職責: 任務分解、執行規劃                                │
│    └── 介面: plan_and_execute(task_spec)                      │
│                                                               │
│  LearningOrchestrator (external_learning)                     │
│    ├── 合併: LearningDispatcher 的調度功能                     │
│    ├── 職責: 學習策略選擇、訓練編排                            │
│    └── 介面: orchestrate_learning(strategy, data)             │
│                                                               │
│  ExplorationOrchestrator (internal_exploration)               │
│    ├── 職責: 自我反思、經驗管理                                │
│    └── 介面: explore(context, depth)                          │
│                                                               │
│  ServiceOrchestrator (service_backbone)                       │
│    ├── 職責: 服務生命週期、資源管理                            │
│    └── 介面: manage_service(service_id, operation)            │
│                                                               │
│  NeuralIntegrationOrchestrator (cognitive_core/neural)        │
│    ├── 職責: 神經網路模型統一介面                              │
│    └── 介面: integrate_neural(model_type, data)               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓ 調用
┌─────────────────────────────────────────────────────────────┐
│                  Layer 3: 執行引擎層                          │
│                (具體功能執行)                                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ReasoningEngine (cognitive_core)                             │
│    └── 職責: 推理邏輯執行                                       │
│                                                               │
│  DecisionEngine (cognitive_core)                              │
│    └── 職責: 決策邏輯執行                                       │
│                                                               │
│  各模組的執行類別                                               │
│    └── ScalableBioTrainer, AttackPathFinder, ...             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 實施計畫

### Phase 1: 標記與整合（1-2 週）

#### 1.1 合併重複功能
```python
# cognitive_core/capability_orchestrator.py
class CapabilityOrchestrator:
    """統一的能力編排器 - 合併 CapabilityDispatcher 功能"""
    
    def __init__(self):
        self.capability_registry = {}
        self.dispatcher = self._integrated_dispatcher()  # 內建調度邏輯
        
    def execute_flow(self, flow_id: int, context: dict) -> dict:
        """主要介面: Flow 編排與執行"""
        # 1. 發現能力
        capability = self._discover_capability(flow_id)
        
        # 2. 參數驗證（原 Dispatcher 功能）
        validated_params = self._validate_params(capability, context)
        
        # 3. 調度執行（原 Dispatcher 功能）
        result = self._dispatch_and_execute(capability, validated_params)
        
        return result
        
    def _integrated_dispatcher(self):
        """內部調度器 - 整合 CapabilityDispatcher 邏輯"""
        # 將 CapabilityDispatcher 的核心邏輯內建到此處
        pass
```

#### 1.2 標記棄用
```python
# cognitive_core/capability_dispatcher.py
class CapabilityDispatcher:
    """
    ⚠️ DEPRECATED - 此類別已整合進 CapabilityOrchestrator
    
    遷移指南:
        舊代碼: dispatcher.dispatch(capability_id, params)
        新代碼: orchestrator.execute_flow(flow_id, context)
    
    保留至: v7.1.0
    預計移除: v7.2.0
    """
    
    def __init__(self):
        warnings.warn(
            "CapabilityDispatcher 已棄用，請使用 CapabilityOrchestrator",
            DeprecationWarning,
            stacklevel=2
        )
```

### Phase 2: 創建統一介面（2-3 週）

#### 2.1 定義 UnifiedOrchestrator
```python
# cognitive_core/unified_orchestrator.py
from typing import Literal, Any
from .capability_orchestrator import CapabilityOrchestrator
from ..task_planning.task_orchestrator import TaskOrchestrator
from ..external_learning.learning_orchestrator import LearningOrchestrator
# ...

RequestType = Literal[
    "capability_execution",
    "task_planning",
    "learning_strategy",
    "exploration",
    "service_management",
    "neural_integration"
]

class UnifiedOrchestrator:
    """
    統一編排器 - Layer 1 全局路由
    
    職責:
        1. 請求路由到對應模組編排器
        2. 模組間協調（如需多模組協作）
        3. 全局上下文管理
        4. 生命週期追蹤
    """
    
    def __init__(self):
        # Layer 2 編排器註冊
        self.orchestrators = {
            "capability_execution": CapabilityOrchestrator(),
            "task_planning": TaskOrchestrator(),
            "learning_strategy": LearningOrchestrator(),
            "exploration": ExplorationOrchestrator(),
            "service_management": ServiceOrchestrator(),
            "neural_integration": NeuralIntegrationOrchestrator(),
        }
        
    def orchestrate(
        self,
        request_type: RequestType,
        params: dict[str, Any],
        context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        統一編排介面
        
        Args:
            request_type: 請求類型（決定路由到哪個模組）
            params: 請求參數
            context: 全局上下文（跨模組共享）
        
        Returns:
            執行結果
        """
        # 1. 驗證請求類型
        if request_type not in self.orchestrators:
            raise ValueError(f"Unknown request type: {request_type}")
        
        # 2. 獲取對應編排器
        orchestrator = self.orchestrators[request_type]
        
        # 3. 路由執行
        result = orchestrator.execute(params, context)
        
        # 4. 記錄追蹤
        self._log_execution(request_type, params, result)
        
        return result
        
    def orchestrate_multi_module(
        self,
        workflow: list[dict]
    ) -> list[dict]:
        """
        多模組協作編排
        
        Example:
            workflow = [
                {"type": "task_planning", "params": {...}},
                {"type": "capability_execution", "params": {...}},
                {"type": "learning_strategy", "params": {...}},
            ]
        """
        results = []
        shared_context = {}
        
        for step in workflow:
            result = self.orchestrate(
                step["type"],
                step["params"],
                shared_context
            )
            results.append(result)
            
            # 更新共享上下文
            shared_context.update(result.get("context_updates", {}))
        
        return results
```

#### 2.2 重構各模組編排器

**統一介面規範**:
```python
# cognitive_core/orchestrator_interface.py
from abc import ABC, abstractmethod
from typing import Any

class OrchestratorInterface(ABC):
    """所有 Layer 2 編排器的統一介面"""
    
    @abstractmethod
    def execute(
        self,
        params: dict[str, Any],
        context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        執行編排邏輯
        
        Args:
            params: 模組特定參數
            context: 全局上下文
        
        Returns:
            {
                "status": "success" | "failure",
                "result": Any,
                "context_updates": dict,  # 需要傳遞給後續步驟的上下文
                "metadata": dict
            }
        """
        pass
```

**更新 CapabilityOrchestrator**:
```python
# cognitive_core/capability_orchestrator.py
from .orchestrator_interface import OrchestratorInterface

class CapabilityOrchestrator(OrchestratorInterface):
    """能力編排器 - 實現統一介面"""
    
    def execute(
        self,
        params: dict[str, Any],
        context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        執行 Flow
        
        Params:
            - flow_id: int
            - inputs: dict
            - dry_run: bool (optional)
        """
        flow_id = params["flow_id"]
        inputs = params.get("inputs", {})
        dry_run = params.get("dry_run", False)
        
        # 原有的 execute_flow 邏輯
        result = self.execute_flow(flow_id, inputs, dry_run)
        
        return {
            "status": "success" if result["success"] else "failure",
            "result": result,
            "context_updates": {"last_flow": flow_id},
            "metadata": {"execution_time": result["time"]}
        }
```

### Phase 3: 漸進式遷移（4-6 週）

#### 3.1 CLI 工具更新
```python
# internal_exploration/python_tools/aiva_cli_v2.py
from aiva_core.cognitive_core.unified_orchestrator import UnifiedOrchestrator

def main():
    orchestrator = UnifiedOrchestrator()
    
    if args.flow:
        # 新方式: 透過統一編排器
        result = orchestrator.orchestrate(
            request_type="capability_execution",
            params={
                "flow_id": args.flow,
                "inputs": {},
                "dry_run": args.dry_run
            }
        )
    elif args.task:
        # 任務規劃也走統一介面
        result = orchestrator.orchestrate(
            request_type="task_planning",
            params={"task_spec": args.task}
        )
```

#### 3.2 向後兼容層
```python
# cognitive_core/capability_orchestrator.py
class CapabilityOrchestrator(OrchestratorInterface):
    """提供舊介面的向後兼容"""
    
    def execute_flow(self, flow_id: int, context: dict) -> dict:
        """舊介面 - 向後兼容"""
        return self.execute(
            params={"flow_id": flow_id, "inputs": context},
            context={}
        )["result"]
```

### Phase 4: 清理與優化（2-3 週）

#### 4.1 移除棄用代碼
```bash
# 確認無引用後刪除
rm cognitive_core/capability_dispatcher.py
rm task_planning/task_dispatcher.py
rm external_learning/learning_dispatcher.py
```

#### 4.2 更新文檔
- 更新所有模組 README
- 編寫 UnifiedOrchestrator 使用指南
- 提供遷移示例

---

## 📊 預期效果

### 重構前 vs 重構後

| 指標 | 重構前 | 重構後 | 改進 |
|------|--------|--------|------|
| 編排器數量 | 11 個 | 7 個 | ⬇️ 36% |
| 職責重疊 | 3 組 | 0 組 | ✅ 消除 |
| 調用路徑複雜度 | 高 | 低 | ✅ 簡化 |
| 模組邊界清晰度 | 模糊 | 清晰 | ✅ 改善 |
| 向後兼容 | N/A | 100% | ✅ 保證 |

### 合併計畫

```
原 11 個編排器:
  ✅ CapabilityOrchestrator + CapabilityDispatcher → CapabilityOrchestrator
  ✅ TaskOrchestrator + TaskDispatcher → TaskOrchestrator
  ✅ LearningOrchestrator + LearningDispatcher → LearningOrchestrator
  ✅ 保留: ExplorationOrchestrator (無重疊)
  ✅ 保留: ServiceOrchestrator (職責明確)
  ✅ 保留: NeuralIntegrationOrchestrator (專業功能)
  ✅ 保留: ReasoningEngine (執行層)
  ✅ 保留: DecisionEngine (執行層)
  ✨ 新增: UnifiedOrchestrator (全局路由)

重構後 7 個核心編排器:
  1. UnifiedOrchestrator (Layer 1 - cognitive_core)
  2. CapabilityOrchestrator (Layer 2 - cognitive_core)
  3. TaskOrchestrator (Layer 2 - task_planning)
  4. LearningOrchestrator (Layer 2 - external_learning)
  5. ExplorationOrchestrator (Layer 2 - internal_exploration)
  6. ServiceOrchestrator (Layer 2 - service_backbone)
  7. NeuralIntegrationOrchestrator (Layer 2 - cognitive_core/neural)

執行層 (Layer 3):
  - ReasoningEngine
  - DecisionEngine
  - 各模組執行類別
```

---

## ⚠️ 風險與緩解

### 風險 1: 破壞現有功能
**緩解措施**:
- 所有變更必須有單元測試
- 保留向後兼容介面至少 2 個版本
- 使用 DeprecationWarning 提前告知

### 風險 2: 效能影響
**緩解措施**:
- 額外的路由層開銷小於 1ms
- 使用快取優化路由決策
- 效能測試對比（重構前後）

### 風險 3: 學習曲線
**緩解措施**:
- 詳細的遷移文檔
- 提供程式碼範例
- CLI 工具同時支援新舊兩種方式

---

## 🚀 立即可行動項

### 不需等待重構的改進

1. **標記棄用** (立即執行)
   ```python
   # 在重複的 Dispatcher 類別加上警告
   @deprecated(version="7.1.0", reason="Use Orchestrator instead")
   class CapabilityDispatcher:
       pass
   ```

2. **文檔更新** (本週完成)
   - 在各編排器 README 說明職責邊界
   - 提供調用關係圖
   - 標註推薦使用的編排器

3. **介面統一** (下週開始)
   - 定義 OrchestratorInterface
   - 讓現有編排器實現統一介面
   - 不破壞現有調用方式

---

## 📌 總結

### 核心設計理念
✅ **三層架構**: 全局路由 → 模組編排 → 執行引擎  
✅ **單一職責**: 每個編排器專注於其模組內部編排  
✅ **清晰邊界**: 六大模組邊界不變，職責明確  
✅ **向後兼容**: 舊代碼繼續運作，漸進式遷移

### 實施優先級
🔴 **P0**: 標記重複代碼（已完成）  
🟡 **P1**: 合併 Orchestrator+Dispatcher（2-3週）  
🟢 **P2**: 創建 UnifiedOrchestrator（1個月）  
⚪ **P3**: 清理棄用代碼（3個月後）

### 最大收益
- 減少 36% 的編排器數量
- 消除職責重疊
- 提升代碼可維護性
- 保持六大模組架構完整

---

*分析完成 - 2026-01-03*
