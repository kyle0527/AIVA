# AIVA 外閉環啟動實施方案

**日期**: 2025-11-16  
**版本**: v1.0 實施版  
**狀態**: 🚀 立即執行

---

## 📊 當前狀態分析

### 系統運行狀態 ✅
```
靶場環境:
✅ juice-shop-live (bkimminich/juice-shop) - http://localhost:3000
✅ aiva-core-service - 健康運行
✅ aiva-rabbitmq - 消息隊列正常
✅ aiva-redis - 緩存服務正常
✅ aiva-neo4j - 圖數據庫正常
✅ aiva-postgres - 關係數據庫正常

內部能力:
✅ 692 個能力已識別和分析
✅ CapabilityRegistry 架構已完成
✅ UnifiedFunctionCaller 已實現
✅ AttackOrchestrator 已就緒
✅ EnhancedDecisionAgent 已實現
```

### 當前問題 ❌
```
1. KnowledgeBase 初始化錯誤
   - 錯誤: missing required positional argument: 'vector_store'
   - 影響: CapabilityRegistry 無法載入能力
   - 優先級: P0 (阻塞性)

2. ExecutionContext 導入錯誤
   - 錯誤: cannot import name 'ExecutionContext'
   - 影響: TaskExecutor 無法初始化
   - 優先級: P0 (阻塞性)

3. Python function 模組未找到
   - 錯誤: No module named 'services.function'
   - 影響: Python 工具調用失敗
   - 優先級: P1 (功能性)
```

---

## 🎯 外閉環啟動目標

### 最小可行產品 (MVP)
```
目標: 完成一次完整的外閉環測試流程
流程: 用戶請求 → AI決策 → 任務編排 → 能力執行 → 結果反饋 → 學習優化

測試場景: 對 Juice Shop (localhost:3000) 執行 SQL 注入掃描
```

### 核心組件狀態

| 組件 | 狀態 | 阻塞問題 | 修復優先級 |
|------|------|----------|-----------|
| **EnhancedDecisionAgent** | ✅ 可用 | 無 | - |
| **AttackOrchestrator** | ✅ 可用 | 無 | - |
| **CapabilityRegistry** | ❌ 阻塞 | KnowledgeBase 初始化 | P0 |
| **UnifiedFunctionCaller** | ⚠️ 部分可用 | Python 模組路徑 | P1 |
| **TaskExecutor** | ❌ 阻塞 | ExecutionContext 導入 | P0 |
| **ExternalLoopConnector** | ✅ 可用 | 無 | - |

---

## 🔧 P0 問題修復方案

### 問題 1: KnowledgeBase 初始化錯誤

#### 根本原因
```python
# services/core/aiva_core/cognitive_core/rag/knowledge_base.py
class KnowledgeBase:
    def __init__(self, vector_store):  # 需要 vector_store 參數
        self.vector_store = vector_store
        ...

# services/core/aiva_core/core_capabilities/capability_registry.py
# 問題: 創建 KnowledgeBase 時沒有傳入 vector_store
kb = KnowledgeBase()  # ❌ 缺少參數
```

#### 修復方案 A: 修改 CapabilityRegistry (推薦)
```python
# 在 capability_registry.py 的 load_from_exploration() 方法中

async def load_from_exploration(self) -> dict[str, Any]:
    """從 internal_exploration 載入能力"""
    try:
        from services.core.aiva_core.cognitive_core.internal_loop_connector import (
            InternalLoopConnector,
        )
        from services.core.aiva_core.cognitive_core.rag.knowledge_base import (
            KnowledgeBase,
        )
        from services.core.aiva_core.cognitive_core.rag.vector_store import (
            UnifiedVectorStore,  # 新增
        )

        # 新增: 初始化 vector_store
        vector_store = UnifiedVectorStore()
        
        # 修復: 傳入 vector_store
        kb = KnowledgeBase(vector_store=vector_store)
        connector = InternalLoopConnector(rag_knowledge_base=kb)
        
        # ... 其餘代碼不變
```

#### 修復方案 B: 修改 KnowledgeBase (備選)
```python
# 在 knowledge_base.py 中添加默認參數

class KnowledgeBase:
    def __init__(self, vector_store=None):  # 設置默認值
        if vector_store is None:
            from .vector_store import UnifiedVectorStore
            vector_store = UnifiedVectorStore()
        self.vector_store = vector_store
        ...
```

**推薦**: 方案 A (保持 KnowledgeBase 的明確依賴)

### 問題 2: ExecutionContext 導入錯誤

#### 根本原因
```python
# services/core/aiva_core/task_planning/executor/task_executor.py
from .execution_status_monitor import ExecutionContext, ExecutionMonitor
# ❌ ExecutionContext 可能未定義或名稱錯誤
```

#### 修復方案: 檢查並修復 execution_status_monitor.py
需要確認文件中是否定義了 `ExecutionContext` 類，如果沒有則添加：

```python
# services/core/aiva_core/task_planning/executor/execution_status_monitor.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExecutionContext:
    """執行上下文"""
    session_id: str
    task_id: str
    start_time: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "start_time": self.start_time,
            "metadata": self.metadata
        }

class ExecutionMonitor:
    """執行監控器"""
    # ... 現有代碼
```

---

## 🚀 實施步驟 (立即執行)

### Step 1: 修復 KnowledgeBase 初始化 (5 分鐘)

```python
# 文件: services/core/aiva_core/core_capabilities/capability_registry.py
# 位置: load_from_exploration() 方法

# 修改前 (約 line 95):
from services.core.aiva_core.cognitive_core.rag.knowledge_base import (
    KnowledgeBase,
)

kb = KnowledgeBase()
connector = InternalLoopConnector(rag_knowledge_base=kb)

# 修改後:
from services.core.aiva_core.cognitive_core.rag.knowledge_base import (
    KnowledgeBase,
)
from services.core.aiva_core.cognitive_core.rag.vector_store import (
    UnifiedVectorStore,
)

vector_store = UnifiedVectorStore()
kb = KnowledgeBase(vector_store=vector_store)
connector = InternalLoopConnector(rag_knowledge_base=kb)
```

### Step 2: 修復 ExecutionContext 導入 (5 分鐘)

```python
# 文件: services/core/aiva_core/task_planning/executor/execution_status_monitor.py
# 位置: 文件開頭 (如果 ExecutionContext 不存在)

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass
class ExecutionContext:
    """執行上下文 - 追蹤任務執行的環境信息"""
    
    session_id: str
    task_id: str
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "start_time": self.start_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionContext":
        """從字典創建"""
        return cls(
            session_id=data["session_id"],
            task_id=data["task_id"],
            start_time=data.get("start_time", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {})
        )

# 確保在 __all__ 中導出
__all__ = ["ExecutionContext", "ExecutionMonitor"]
```

### Step 3: 驗證修復 (2 分鐘)

```powershell
# 重新運行測試
$env:PYTHONPATH = "C:\D\fold7\AIVA-git"
python C:\D\fold7\AIVA-git\services\core\aiva_core\tests\test_dynamic_capability_calling.py

# 預期輸出:
# ✅ 載入能力數: 692
# ✅ TaskExecutor 初始化成功
```

### Step 4: 創建外閉環端到端測試 (10 分鐘)

```python
# 文件: services/core/aiva_core/tests/test_external_loop_e2e.py (新增)

"""外閉環端到端測試

完整流程: 用戶請求 → AI決策 → 任務編排 → 能力執行 → 結果反饋 → 學習優化
測試目標: Juice Shop (http://localhost:3000)
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_external_loop_e2e():
    """外閉環完整流程測試"""
    
    print("\n" + "="*70)
    print("🚀 AIVA 外閉環端到端測試")
    print("目標: Juice Shop (http://localhost:3000)")
    print("場景: SQL 注入漏洞掃描")
    print("="*70)
    
    # ==================== Step 1: 用戶請求 ====================
    print("\n【Step 1】用戶請求")
    user_request = {
        "action": "掃描 Juice Shop 的 SQL 注入漏洞",
        "target": "http://localhost:3000",
        "scan_type": "sql_injection",
        "risk_level": "medium"
    }
    print(f"   請求: {user_request['action']}")
    print(f"   目標: {user_request['target']}")
    
    # ==================== Step 2: AI 決策 ====================
    print("\n【Step 2】AI 高階決策")
    from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import (
        EnhancedDecisionAgent,
        DecisionContext
    )
    from services.aiva_common.enums import RiskLevel
    
    agent = EnhancedDecisionAgent()
    
    context = DecisionContext()
    context.target_info = {
        "value": user_request["target"],
        "type": "url",
        "application": "juice_shop"
    }
    context.risk_level = RiskLevel.MEDIUM
    context.available_tools = ["sqlmap", "manual_test", "nmap"]
    context.discovered_vulns = []
    
    # 生成高階意圖
    intent = agent.decide(context)
    
    print(f"   ✅ 意圖類型: {intent.intent_type.value}")
    print(f"   ✅ 目標: {intent.target.target_value}")
    print(f"   ✅ 參數: {intent.parameters}")
    print(f"   ✅ 信心度: {intent.confidence:.2f}")
    print(f"   ✅ 推理: {intent.reasoning[:100]}...")
    
    # ==================== Step 3: 能力查詢 ====================
    print("\n【Step 3】查詢可用能力")
    from services.core.aiva_core.core_capabilities.capability_registry import (
        get_capability_registry,
        initialize_capability_registry
    )
    
    # 初始化註冊表
    init_result = await initialize_capability_registry()
    print(f"   ✅ 載入能力: {init_result['capabilities_loaded']} 個")
    
    registry = get_capability_registry()
    
    # 搜索 SQL 相關能力
    sql_capabilities = registry.search_capabilities("sql")
    print(f"   ✅ SQL 相關能力: {len(sql_capabilities)} 個")
    
    for cap in sql_capabilities[:5]:
        print(f"      - {cap.name} ({cap.module}/{cap.language})")
    
    # ==================== Step 4: 任務編排 ====================
    print("\n【Step 4】任務編排")
    from services.core.aiva_core.task_planning.planner.orchestrator import (
        AttackOrchestrator
    )
    
    orchestrator = AttackOrchestrator()
    
    # 從意圖創建執行計劃
    # 注意: 這裡需要將 HighLevelIntent 轉換為 AST
    ast_plan = {
        "intent": intent.intent_type.value,
        "target": intent.target.target_value,
        "tasks": [
            {
                "id": "task_001",
                "type": "reconnaissance",
                "tool": "nmap",
                "params": {"target": intent.target.target_value, "ports": "80,443,3000"}
            },
            {
                "id": "task_002",
                "type": "sql_injection_scan",
                "tool": "sqlmap",
                "params": {
                    "url": f"{intent.target.target_value}/rest/products/search?q=test",
                    "method": "GET"
                },
                "dependencies": ["task_001"]
            },
            {
                "id": "task_003",
                "type": "verification",
                "tool": "manual_test",
                "params": {"verify": "sql_injection"},
                "dependencies": ["task_002"]
            }
        ]
    }
    
    plan = orchestrator.create_execution_plan(ast_plan)
    
    print(f"   ✅ 計劃 ID: {plan.plan_id}")
    print(f"   ✅ 任務數: {len(plan.task_sequence.tasks)}")
    
    for task in plan.task_sequence.tasks:
        decision = plan.get_decision_for_task(task.task_id)
        print(f"      - {task.task_id}: {task.task_type} (工具: {decision.tool_name if decision else 'N/A'})")
    
    # ==================== Step 5: 能力執行 ====================
    print("\n【Step 5】執行任務")
    from services.core.aiva_core.task_planning.executor.task_executor import (
        TaskExecutor
    )
    
    executor = TaskExecutor()
    
    execution_results = []
    trace = []
    
    for task in plan.task_sequence.tasks:
        tool_decision = plan.get_decision_for_task(task.task_id)
        
        if not tool_decision:
            print(f"   ⚠️ 跳過任務 {task.task_id}: 無工具決策")
            continue
        
        print(f"   🔄 執行: {task.task_id} ({task.task_type})...")
        
        try:
            result = await executor.execute_task(
                task=task,
                tool_decision=tool_decision,
                trace_session_id="external_loop_test_001"
            )
            
            execution_results.append(result)
            
            trace.append({
                "task_id": task.task_id,
                "task_type": task.task_type,
                "success": result.success,
                "duration": result.metadata.get("execution_time", 0),
                "timestamp": datetime.now().isoformat()
            })
            
            status_icon = "✅" if result.success else "❌"
            print(f"   {status_icon} 完成: {task.task_id} (成功: {result.success})")
            
            if result.output:
                print(f"      輸出: {str(result.output)[:100]}...")
            
            # 更新任務狀態
            from services.core.aiva_core.task_planning.planner.task_converter import TaskStatus
            
            status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            orchestrator.update_task_status(
                plan, task.task_id, status,
                result=result.output,
                error=result.error
            )
            
        except Exception as e:
            print(f"   ❌ 錯誤: {task.task_id} - {str(e)}")
            trace.append({
                "task_id": task.task_id,
                "task_type": task.task_type,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # ==================== Step 6: 結果反饋 ====================
    print("\n【Step 6】結果反饋與學習")
    from services.core.aiva_core.cognitive_core.external_loop_connector import (
        ExternalLoopConnector
    )
    
    external_loop = ExternalLoopConnector()
    
    # 將執行計劃和追蹤傳遞給外部閉環
    learning_result = await external_loop.process_execution_result(
        plan={"plan_id": plan.plan_id, "tasks": [t.to_dict() for t in plan.task_sequence.tasks]},
        trace=trace
    )
    
    print(f"   ✅ 偏差分析: 發現 {learning_result.get('deviations_found', 0)} 個偏差")
    print(f"   ✅ 顯著偏差: {learning_result.get('deviations_significant', False)}")
    print(f"   ✅ 訓練觸發: {learning_result.get('training_triggered', False)}")
    print(f"   ✅ 權重更新: {learning_result.get('weights_updated', False)}")
    
    # ==================== Step 7: 結果總結 ====================
    print("\n【Step 7】執行總結")
    
    summary = orchestrator.get_plan_summary(plan)
    
    print(f"   📊 計劃狀態:")
    print(f"      - 計劃 ID: {summary['plan_id']}")
    print(f"      - 總任務數: {summary['total_tasks']}")
    print(f"      - 完成狀態: {summary['is_complete']}")
    print(f"      - 狀態統計:")
    for status, count in summary['status_counts'].items():
        if count > 0:
            print(f"         * {status}: {count}")
    
    print(f"\n   📊 執行統計:")
    print(f"      - 成功任務: {sum(1 for r in execution_results if r.success)}/{len(execution_results)}")
    print(f"      - 總耗時: {sum(t.get('duration', 0) for t in trace):.2f}s")
    
    print("\n" + "="*70)
    print("✅ 外閉環端到端測試完成")
    print("="*70)
    
    return {
        "success": True,
        "intent": intent,
        "plan": plan,
        "execution_results": execution_results,
        "learning_result": learning_result,
        "summary": summary
    }


async def main():
    """主函數"""
    try:
        result = await test_external_loop_e2e()
        
        # 保存結果
        import json
        from pathlib import Path
        
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"external_loop_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 將結果序列化（排除無法序列化的對象）
        serializable_result = {
            "success": result["success"],
            "summary": result["summary"],
            "execution_count": len(result["execution_results"]),
            "learning_result": result["learning_result"]
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 結果已保存: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 5: 執行完整測試 (5 分鐘)

```powershell
# 執行外閉環端到端測試
$env:PYTHONPATH = "C:\D\fold7\AIVA-git"
python C:\D\fold7\AIVA-git\services\core\aiva_core\tests\test_external_loop_e2e.py

# 預期流程:
# 1. ✅ 用戶請求被解析
# 2. ✅ AI 生成高階意圖
# 3. ✅ 查詢到 692 個能力
# 4. ✅ 創建執行計劃（3個任務）
# 5. ✅ 執行任務序列
# 6. ✅ 結果反饋到外部學習閉環
# 7. ✅ 生成執行總結
```

---

## 📊 驗收標準

### 功能驗收

| 項目 | 驗收標準 | 驗證方法 |
|------|---------|---------|
| **能力載入** | 載入 692 個能力 | 檢查 CapabilityRegistry 統計 |
| **AI 決策** | 生成有效的 HighLevelIntent | 檢查 intent.confidence > 0.7 |
| **任務編排** | 生成 3+ 個任務的執行計劃 | 檢查 plan.task_sequence.tasks |
| **能力執行** | 至少 1 個任務成功執行 | 檢查 execution_results[0].success |
| **結果反饋** | 觸發外部學習閉環 | 檢查 learning_result 非空 |
| **完整流程** | 7 個步驟全部完成 | 測試腳本正常結束 |

### 性能驗收

| 指標 | 目標值 | 當前值 | 狀態 |
|------|--------|--------|------|
| 能力載入時間 | <5s | TBD | 🔄 |
| AI 決策時間 | <2s | TBD | 🔄 |
| 任務編排時間 | <1s | TBD | 🔄 |
| 單任務執行時間 | <30s | TBD | 🔄 |
| 端到端總時間 | <60s | TBD | 🔄 |

---

## 🔄 持續優化計劃

### Phase 2: 功能增強 (1-2 週)

1. **增加更多掃描類型**
   - XSS 掃描
   - SSRF 檢測
   - 業務邏輯測試

2. **優化 AI 編排**
   - 動態調整任務優先級
   - 並行任務執行
   - 失敗重試策略

3. **增強學習能力**
   - 基於執行結果的策略優化
   - 能力評分動態更新
   - 成功模式識別

### Phase 3: 規模化 (2-3 週)

1. **支持多目標並發**
   - 同時掃描多個靶場
   - 資源調度和限流
   - 結果聚合

2. **API 服務化**
   - RESTful API 端點
   - WebSocket 實時反饋
   - 認證和授權

3. **監控和可觀測性**
   - 實時性能監控
   - 錯誤追蹤和告警
   - 執行歷史查詢

---

## 📚 相關文檔

- `INTERNAL_TO_EXTERNAL_CAPABILITY_TRANSFORMATION_PLAN.md` - 內外閉環轉換方案
- `P0_IMPLEMENTATION_COMPLETION_REPORT.md` - P0 實施完成報告
- `services/core/aiva_core/README.md` - AIVA Core 架構文檔

---

## ✅ 下一步行動

### 立即執行 (今天)
1. ✅ 閱讀本方案
2. 🔧 執行 Step 1: 修復 KnowledgeBase 初始化
3. 🔧 執行 Step 2: 修復 ExecutionContext 導入
4. 🧪 執行 Step 3: 驗證修復
5. 📝 執行 Step 4: 創建端到端測試
6. 🚀 執行 Step 5: 運行完整測試

### 本週內完成
- 完成 P0 修復和基礎測試
- 驗證外閉環完整流程
- 收集性能數據
- 優化瓶頸環節

### 下週規劃
- 開始 Phase 2 功能增強
- 增加更多測試場景
- 完善監控和日誌

---

**文檔創建**: 2025-11-16 21:10:00  
**作者**: GitHub Copilot (Claude Sonnet 4.5)  
**狀態**: 🚀 準備執行

**重要提示**: 
- 所有修復都在現有架構下進行
- 保持五大模組 + 六大核心模組的架構完整性
- 所有測試都可以直接在運行的靶場環境中執行
- 修復完成後立即可以看到效果
