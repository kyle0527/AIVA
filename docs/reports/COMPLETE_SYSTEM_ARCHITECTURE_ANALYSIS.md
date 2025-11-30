# AIVA 完整程式架構深度分析

> **分析日期**: 2025-11-29  
> **分析範圍**: 完整系統架構、數據流、執行流程  
> **實戰驗證**: 對 Docker 容器實際目標進行掃描

---

## 🏗️ 系統整體架構

```
AIVA (AI-driven Vulnerability Assessment)
├── 🧠 AI Core (services/core/aiva_core/)          - AI 大腦
│   ├── cognitive_core/                            - 認知核心
│   │   ├── internal_loop_connector.py             - 內閉環連接器
│   │   ├── external_loop_connector.py             - 外閉環連接器
│   │   ├── ai_capability_query.py                 - AI 能力查詢
│   │   └── decision/                              - 決策引擎
│   │
│   ├── core_capabilities/                         - 核心能力
│   │   └── dialog/
│   │       └── assistant.py                       - AI 對話助理 ⭐
│   │
│   ├── task_planning/                             - 任務規劃
│   │   ├── planner/                               - 攻擊計畫生成
│   │   └── executor/
│   │       └── plan_executor.py                   - 計畫執行器
│   │
│   ├── internal_exploration/                      - 內部探索
│   │   └── capability_registry.py                 - 能力註冊表
│   │
│   ├── external_learning/                         - 外部學習
│   │   ├── analysis/                              - 偏差分析
│   │   └── learning/                              - 模型訓練
│   │
│   └── service_backbone/                          - 服務骨幹
│       └── api/
│           └── unified_function_caller.py         - 統一功能調用器
│
├── 🔍 Scan Engine (services/scan/)                - 掃描引擎
│   ├── coordinators/
│   │   └── multi_engine_coordinator.py            - 多引擎協調器 ⭐
│   │
│   └── engines/                                   - 各語言掃描引擎
│       ├── python_engine/                         - Python 引擎
│       ├── typescript_engine/                     - TypeScript 引擎
│       ├── rust_engine/                           - Rust 引擎
│       └── go_engine/                             - Go 引擎
│
├── 🔗 Integration (services/integration/)         - 整合層
│   └── capability/
│       └── registry.py                            - 全局能力註冊表 (782個)
│
├── ⚡ Features (services/features/)               - 功能模組
│   └── function_*/                                - 各種攻擊能力 (XSS, SQLi, SSRF...)
│
├── 📡 API (api/)                                  - REST API 介面
│   ├── main.py                                    - FastAPI 主程式
│   └── routers/                                   - 路由模組
│
├── 🖥️ CLI (/)                                     - 命令行介面
│   ├── aiva_cli.py                                - 主 CLI 工具 ⭐
│   └── aiva_ai_menu.py                            - AI 智能選單
│
└── 🚀 Scripts (scripts/startup/)                  - 啟動腳本
    └── start_ai_service.py                        - 服務啟動器
```

---

## 🔄 雙閉環架構詳解

### 內閉環 (Internal Loop) - "認識自己"

```
目的: AI 知道自己有哪些能力、如何使用

┌─────────────────────────────────────────────────────────┐
│                    內閉環循環流程                        │
└─────────────────────────────────────────────────────────┘

Step 1: 能力發現
  internal_exploration/capability_registry.py
  ↓ 掃描所有模組 (scan, core, integration, features)
  ↓ 發現 782 個能力 (Python/Rust/TypeScript/Go)
  ↓ 提取元數據 (名稱、參數、返回值、用法)

Step 2: 能力分類
  InternalLoopConnector.classify_capability()
  ↓ 分類: Scanning/Attacking/Analysis/Utility/Reporting
  ↓ 子分類: PortScan/WebScan/SQLi/XSS/SSRF...
  ↓ 複雜度: Low/Medium/High/Critical

Step 3: RAG 知識庫同步
  InternalLoopConnector.sync_capabilities_to_rag()
  ↓ 轉換為向量嵌入
  ↓ 寫入 ChromaDB 向量數據庫
  ↓ data/vector_db/chroma/

Step 4: AI 查詢能力
  AICapabilityQuery.query("SQL 注入")
  ↓ 語義搜索向量數據庫
  ↓ 返回最相關的能力
  ↓ 提供使用方法和範例

循環閉合:
  使用經驗 → 更新能力描述 → 重新同步 RAG → 下次查詢更準確
```

**當前狀態**: 
- ✅ 能力註冊表完成 (782 個能力)
- ⚠️ RAG 同步未自動執行
- ⚠️ 向量數據庫可能為空

---

### 外閉環 (External Loop) - "從經驗學習"

```
目的: AI 從實戰結果中學習，優化決策

┌─────────────────────────────────────────────────────────┐
│                    外閉環循環流程                        │
└─────────────────────────────────────────────────────────┘

Step 1: 執行任務
  用戶: "掃描 http://target.com"
  ↓ AIVADialogAssistant 識別意圖
  ↓ MultiEngineCoordinator 執行掃描
  ↓ 記錄執行軌跡 (ExecutionTrace)

Step 2: 記錄調用
  CapabilityRegistry.record_invocation()
  ↓ 記錄: 能力ID, 成功/失敗, 執行時間, 錯誤訊息
  ↓ 累積統計: 成功率, 平均耗時, 可用性

Step 3: 偏差分析
  ExternalLoopConnector.process_execution_result()
  ↓ 比較: 計劃 vs 實際執行
  ↓ 識別: 超時、失敗、意外結果
  ↓ 生成: DeviationRecord

Step 4: 模型訓練
  ModelTrainer.train_from_experience()
  ↓ 偏差數據 → 訓練樣本
  ↓ 更新神經網路權重
  ↓ 生成新版本權重

Step 5: 權重更新
  AIWeightManager.register_weights()
  ↓ 註冊新權重版本
  ↓ 通知 DecisionEngine
  ↓ 下次決策使用新權重

循環閉合:
  實戰結果 → 偏差分析 → 模型訓練 → 更新權重 → 決策更優 → 實戰結果更好
```

**當前狀態**:
- ✅ ExecutionTrace 記錄機制存在
- ⚠️ 偏差分析未實際觸發
- ⚠️ 模型訓練流程未完整實現

---

## 🎯 實際執行流程分析

### 用戶命令: "掃描 http://localhost:3000"

```
【完整執行鏈】

1. CLI 入口
   aiva_cli.py --attack "掃描 http://localhost:3000"
   ↓

2. AI 對話助理初始化
   AIVADialogAssistant.__init__()
   ├── 初始化 CapabilityRegistry (782 個能力)
   ├── 準備 RAG 知識庫連接
   └── 初始化 UnifiedFunctionCaller
   ↓

3. 處理用戶輸入
   assistant.process_user_input("掃描 http://localhost:3000")
   ↓
   DialogIntent.identify_intent()
   ├── 正則匹配: r"(掃描|scan).*(https?://\S+)" ✅
   ├── 提取: target = "http://localhost:3000"
   └── 返回: intent = "run_scan"
   ↓

4. 執行掃描處理
   _handle_run_scan(scan_type="", target="http://localhost:3000", ...)
   ↓
   【關鍵決策點】不使用 RAG + UnifiedFunctionCaller
                 直接使用 MultiEngineCoordinator ✅
   ↓
   from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
   coordinator = MultiEngineCoordinator()
   ↓

5. 生成掃描 ID
   scan_id = f"ai_scan_{uuid4().hex[:8]}"
   例如: "ai_scan_948a0484"
   ⚠️ 注意: 應為 "scan_xxx" 格式
   ↓

6. 執行快速掃描策略
   coordinator.execute_strategy_fast(
       scan_id="ai_scan_948a0484",
       targets=["http://localhost:3000"]
   )
   ↓

7. MultiEngineCoordinator 內部流程
   ┌─────────────────────────────────────┐
   │ Phase 0: Rust 快速發現 (可選)       │
   │ Phase 1: 多引擎並行掃描 ⭐          │
   │ Phase 2: 結果聚合                   │
   └─────────────────────────────────────┘
   ↓
   Phase 1 啟動:
   ├── 選擇引擎: Python (預設快速策略)
   ├── 創建適配器: PythonAdapter
   └── 執行掃描: adapter.scan(scan_id, targets)
   ↓

8. Python 引擎執行
   PythonAdapter.scan()
   ├── 驗證 scan_id 格式 ⚠️ 
   │   期望: "scan_xxx"
   │   實際: "ai_scan_xxx" ❌
   │   錯誤: Pydantic validation error
   │
   ├── 發送 ScanStartPayload
   ├── 執行掃描邏輯
   │   └── 爬取 HTML/JS/CSS
   │       解析 DOM
   │       識別資產 (URL, Script, API...)
   │
   └── 發送 ScanCompletedPayload
   ↓

9. 結果聚合
   coordinator._aggregate_results()
   ├── 收集所有引擎結果
   ├── 去重資產
   ├── 生成 Summary
   └── 返回 Phase1CompletedPayload
   ↓

10. 構建回應
    _handle_run_scan() 返回:
    {
        "intent": "run_scan",
        "executable": True,
        "message": "✅ 掃描完成！\n目標: http://localhost:3000\n發現資產: X 個",
        "data": {
            "scan_id": "ai_scan_948a0484",
            "status": "completed",
            "assets": [...]
        }
    }
    ↓

11. 顯示給用戶
    CLI 輸出結果
```

---

## 🔧 關鍵組件深度分析

### 1. AIVADialogAssistant (核心對話引擎)

**位置**: `services/core/aiva_core/core_capabilities/dialog/assistant.py`

**職責**:
```python
class AIVADialogAssistant:
    """AI 對話助理 - 用戶與系統的橋梁"""
    
    # 初始化
    def __init__(self):
        self.capability_registry = global_registry  # 782 個能力
        self._function_caller = None                # 跨語言調用器
        self._rag_kb = None                         # RAG 知識庫
    
    # 核心方法
    async def process_user_input(user_input: str):
        """處理用戶自然語言輸入"""
        # 1. 意圖識別
        intent, params = DialogIntent.identify_intent(user_input)
        
        # 2. 路由到對應處理器
        if intent == "run_scan":
            return await self._handle_run_scan(...)
        elif intent == "list_capabilities":
            return await self._handle_list_capabilities()
        ...
    
    # 掃描處理 (我剛修復的)
    async def _handle_run_scan(scan_type, target, original_input):
        """執行實際掃描"""
        # 提取 URL
        url_match = re.search(r"https?://[^\s]+", original_input)
        target = url_match.group(0)
        
        # 直接調用 MultiEngineCoordinator (不走 RAG)
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        coordinator = MultiEngineCoordinator()
        
        # 執行掃描
        result = await coordinator.execute_strategy_fast(
            scan_id=f"ai_scan_{uuid4().hex[:8]}",
            targets=[target]
        )
        
        # 返回結果
        return {
            "intent": "run_scan",
            "executable": True,
            "message": f"✅ 掃描完成！發現資產: {len(result.assets)} 個",
            "data": {...}
        }
```

**問題**:
- ❌ scan_id 格式錯誤: `ai_scan_xxx` 應為 `scan_xxx`
- ⚠️ 未使用 RAG 查詢最佳能力
- ⚠️ 未記錄到 ExternalLoopConnector

---

### 2. MultiEngineCoordinator (掃描大腦)

**位置**: `services/scan/coordinators/multi_engine_coordinator.py`

**架構**:
```python
class MultiEngineCoordinator:
    """多引擎協調器 - 統籌 4 種語言的掃描引擎"""
    
    # 支援的引擎
    ENGINES = {
        "python": PythonAdapter,      # Python 爬蟲 + 靜態分析
        "typescript": TypeScriptAdapter,  # Playwright 動態渲染
        "rust": RustAdapter,          # 高性能掃描 + 敏感資訊
        "go": GoAdapter               # 並發服務發現
    }
    
    # 掃描階段
    PHASES = {
        "Phase 0": "Rust 快速發現",
        "Phase 1": "多引擎並行掃描",
        "Phase 2": "結果聚合分析"
    }
    
    # 執行策略
    async def execute_strategy_fast(scan_id, targets):
        """快速掃描策略 - 只用 Python 引擎"""
        return await self._execute_phase_1(
            scan_id, targets, engines=["python"]
        )
    
    async def execute_strategy_balanced(scan_id, targets):
        """平衡策略 - Python + TypeScript"""
        return await self._execute_phase_1(
            scan_id, targets, engines=["python", "typescript"]
        )
    
    async def execute_strategy_deep(scan_id, targets):
        """深度掃描 - 全部引擎 + Phase 0"""
        # Phase 0: Rust 快速發現
        phase0_result = await self._execute_phase_0(scan_id, targets)
        
        # Phase 1: 根據 Phase 0 結果選擇引擎
        engines = self._decide_engines(phase0_result)
        phase1_result = await self._execute_phase_1(
            scan_id, targets, engines=engines
        )
        
        return phase1_result
```

**執行流程**:
```
execute_strategy_fast()
    ↓
_execute_phase_1(engines=["python"])
    ↓
對每個引擎:
    ├── 創建適配器: adapter = PythonAdapter()
    ├── 發送任務: await adapter.scan(scan_id, targets)
    └── 收集結果
    ↓
_aggregate_results()
    ├── 合併所有引擎的資產
    ├── 去重 (by asset.value)
    └── 生成摘要
    ↓
返回 Phase1CompletedPayload
```

---

### 3. CapabilityRegistry (能力中樞)

**位置**: `services/integration/capability/registry.py`

**數據結構**:
```python
# 全局單例
registry = CapabilityRegistry()

# 782 個能力存儲
capabilities: Dict[str, CapabilityMetadata] = {
    "capability_001": {
        "id": "capability_001",
        "name": "detect_sqli",
        "module": "features/function_sqli",
        "language": "Python",
        "entrypoint": "services/features/function_sqli/main.py",
        "status": "healthy",
        "tags": ["sqli", "injection", "database"],
        "invocation_metadata": {
            "protocol": "unified_caller",
            "module_arg": "function_sqli",
            "function_arg": "detect_sqli"
        }
    },
    ...
}

# 統計數據
stats: Dict[str, Any] = {
    "total_capabilities": 782,
    "by_module": {
        "scan": 286,        # 36.6%
        "core": 207,        # 26.5%
        "integration": 111, # 14.2%
        "features": 98      # 12.5%
    },
    "by_language": {
        "Python": 495,      # 63.3%
        "Rust": 123,        # 15.7%
        "TypeScript": 84,   # 10.7%
        "Go": 80            # 10.2%
    }
}

# 調用記錄 (外閉環數據)
invocation_records: List[InvocationRecord] = [
    {
        "capability_id": "capability_001",
        "timestamp": "2025-11-29T14:00:00",
        "success": True,
        "execution_time_ms": 234.5,
        "error_message": None
    },
    ...
]
```

**關鍵方法**:
```python
async def discover_capabilities():
    """掃描所有模組，發現能力"""
    
async def search_capabilities(query: str):
    """搜索能力 (基於名稱/標籤)"""
    
async def record_invocation(capability_id, success, execution_time_ms, ...):
    """記錄能力調用 (外閉環數據收集)"""
    
async def get_capability_stats():
    """獲取統計資訊"""
```

---

## 📊 數據流分析

### 用戶輸入 → AI 響應

```
【數據流向】

用戶輸入 (字符串)
  "掃描 http://localhost:3000"
        ↓
DialogIntent (正則匹配)
  intent = "run_scan"
  params = {}
  original_input = "掃描 http://localhost:3000"
        ↓
URL 提取 (正則)
  target = "http://localhost:3000"
        ↓
MultiEngineCoordinator
  scan_id = "ai_scan_948a0484"
  targets = ["http://localhost:3000"]
        ↓
PythonAdapter
  ScanStartPayload {
    scan_id: "ai_scan_948a0484",  # ❌ 格式錯誤
    targets: ["http://localhost:3000"],
    timestamp: "2025-11-29T14:00:00"
  }
        ↓
Python 掃描引擎
  HTTP GET http://localhost:3000
  解析 HTML
  提取資產:
    - URL: http://localhost:3000/api/products
    - Script: http://localhost:3000/main.js
    - API: /rest/products/search
        ↓
ScanCompletedPayload
  scan_id: "ai_scan_948a0484"
  assets: [Asset, Asset, ...]
  summary: Summary {
    total_assets: 15,
    urls: 8,
    scripts: 3,
    apis: 4
  }
        ↓
Phase1CompletedPayload
  scan_id: "ai_scan_948a0484"
  status: "completed"
  assets: [...] (15個)
  engines_used: ["python"]
        ↓
AI 響應 (dict)
  {
    "intent": "run_scan",
    "executable": True,
    "message": "✅ 掃描完成！\n發現資產: 15 個",
    "data": {
      "scan_id": "ai_scan_948a0484",
      "status": "completed",
      "assets": [...]
    }
  }
        ↓
顯示給用戶
  ✅ 掃描完成！
  🎯 目標: http://localhost:3000
  📊 掃描 ID: ai_scan_948a0484
  📈 狀態: completed
  🔍 發現資產: 15 個
```

---

## 🐛 當前問題匯總

### 已發現的問題

1. **scan_id 格式錯誤** 🔴
   ```python
   # 當前: ai_scan_948a0484
   # 期望: scan_948a0484
   # 影響: Pydantic 驗證失敗
   ```

2. **RAG 知識庫未初始化** 🟡
   ```python
   # 問題: data/vector_db/chroma/ 可能為空
   # 影響: AI 無法語義搜索能力
   # 解決: 需要執行 sync_capabilities_to_rag()
   ```

3. **外閉環未實際運作** 🟡
   ```python
   # 問題: 沒有調用 ExternalLoopConnector.process_execution_result()
   # 影響: 無法從實戰中學習優化
   # 數據: record_invocation() 有記錄，但未觸發分析
   ```

4. **部分模組導入失敗** 🟡
   ```
   ⚠️ cannot import name 'FunctionTaskSchema'
   ⚠️ No module named 'target_environment_detector'
   ```

---

## 🎯 實戰掃描結果分析

等待實際掃描完成...
