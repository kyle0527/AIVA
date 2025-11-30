# AIVA 雙閉環完整規劃與數據流分析

**分析日期**: 2025年11月28日  
**目的**: 深入理解雙閉環系統的運作機制、數據流向和實際用途

---

## 📋 雙閉環系統核心概念

### 設計哲學

AIVA 採用**雙閉環自我優化架構**,模擬人類學習過程:

1. **內閉環 (對內探索)**: 相當於「自我認知」
   - 問題: 我有什麼能力?
   - 方法: 掃描自身代碼,分析功能
   - 結果: 建立能力知識庫

2. **外閉環 (對外實戰)**: 相當於「實踐學習」
   - 問題: 這些能力在實戰中效果如何?
   - 方法: 執行滲透測試任務,記錄結果
   - 結果: 從成功/失敗中學習優化

### 關鍵創新點

```
傳統系統: 人工維護能力列表 → 容易過時、不完整
AIVA 系統: AI 自動發現能力 → 永遠同步、完整準確

傳統系統: 固定測試策略 → 無法適應新場景
AIVA 系統: 基於實戰反饋優化 → 持續進化、越用越強
```

---

## 🔄 內閉環 (對內探索) 詳細分析

### 執行流程

```
┌─────────────────────────────────────────────────────┐
│  步驟 1: 模組掃描 (ModuleExplorer)                    │
│  ─────────────────────────────────────────────────  │
│  掃描目標: services/ 目錄下的所有模組                 │
│  └─ scan/          (Python 掃描引擎)                │
│  └─ integration/   (Rust 工具)                      │
│  └─ features/      (Go 專項工具、TypeScript SPA)    │
│  └─ core/aiva_core (核心 AI 模組)                   │
│                                                       │
│  輸出: 模組列表 + 文件路徑                            │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 2: 能力分析 (CapabilityAnalyzer)               │
│  ─────────────────────────────────────────────────  │
│  多語言提取:                                          │
│  • Python: AST 解析 @capability 裝飾器              │
│  • Go: 正則提取 func 函數                            │
│  • Rust: 正則提取 pub fn                            │
│  • TypeScript: 正則提取 export function            │
│                                                       │
│  提取信息:                                            │
│  - 函數名稱 (capability_name)                       │
│  - 所屬模組 (module)                                 │
│  - 參數列表 (parameters)                            │
│  - 返回類型 (return_type)                           │
│  - 文檔字串 (description)                           │
│  - 是否異步 (is_async)                              │
│                                                       │
│  輸出: 800 個原始能力記錄                             │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 3: 能力增強 (InternalLoopConnector)            │
│  ─────────────────────────────────────────────────  │
│  自動分類:                                            │
│  • 按功能: scanning/attacking/analysis/utility/     │
│            reporting/integration                    │
│  • 按子類: port_scan/sql_injection/xss 等         │
│  • 複雜度: 1-5 級 (trivial → advanced)             │
│                                                       │
│  添加元數據:                                          │
│  • 標籤 (tags): async, security, web, network      │
│  • 健康度 (health_score): 0.0-1.0                  │
│  • 可用性 (availability): 0.0-1.0                  │
│  • 錯誤率 (error_rate): 0.0-1.0                    │
│                                                       │
│  輸出: 800 個增強能力記錄 (ModuleCapability)         │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 4: 數據驗證 (Pydantic)                         │
│  ─────────────────────────────────────────────────  │
│  使用 ModuleCapability Pydantic 模型驗證:            │
│  • 字段類型正確性                                     │
│  • 必需字段完整性                                     │
│  • 數值範圍合法性 (0-1, 1-5)                        │
│                                                       │
│  輸出: 通過驗證的能力對象                             │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 5: 轉換為 RAG 文檔                              │
│  ─────────────────────────────────────────────────  │
│  將能力對象轉換為向量文檔:                            │
│                                                       │
│  document = {                                        │
│    "content": f"{cap.name}: {cap.description}...",  │
│    "metadata": {                                     │
│      "capability_name": cap.name,                   │
│      "module": cap.module,                          │
│      "language": cap.language,                      │
│      "category": cap.category,                      │
│      "complexity": cap.complexity,                  │
│      "tags": cap.tags,                              │
│      ...                                             │
│    }                                                 │
│  }                                                   │
│                                                       │
│  輸出: 800 個 RAG 文檔                                │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 6: 注入 RAG 知識庫                              │
│  ─────────────────────────────────────────────────  │
│  向量化存儲 (ChromaDB):                              │
│  • 使用 all-MiniLM-L6-v2 模型嵌入                   │
│  • 384 維向量表示                                    │
│  • 使用 SHA256 生成穩定 ID                           │
│  • 自動去重 (800 → 782 份)                          │
│                                                       │
│  持久化位置: data/vector_db/chroma/                  │
│  數據庫大小: 7.50 MB                                  │
│                                                       │
│  輸出: 782 份向量文檔 (18 份合理重複已去重)           │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 7: 生成能力摘要                                 │
│  ─────────────────────────────────────────────────  │
│  統計信息 (CapabilitySummary):                       │
│  • 總能力數: 782                                     │
│  • 按模組: scan(286), core(207), integration(111),  │
│           features(98), 其他(80)                    │
│  • 按語言: Python(495), Rust(123), TypeScript(84),  │
│           Go(80)                                    │
│  • 健康度: 平均 health_score, 可用性統計            │
│                                                       │
│  輸出: InternalLoopSyncResult (完整同步結果)          │
└─────────────────────────────────────────────────────┘
```

### 預期得到的數據 (內閉環產出)

#### 1. 能力清單 (ModuleCapability[])

每個能力包含:

```python
{
  # 基本信息
  "capability_id": "cap-scan-run_nmap_scan",
  "name": "run_nmap_scan",
  "module": "scan",
  "function": "run_nmap_scan",
  "description": "使用 Nmap 執行端口掃描",
  
  # 分類信息
  "category": "scanning",           # scanning/attacking/analysis/utility/reporting/integration
  "sub_category": "port_scan",      # 詳細子類別
  "complexity": 3,                  # 1-5 (Moderate)
  "tags": ["async", "network", "security"],
  
  # 使用方法
  "parameters": [
    {
      "name": "target",
      "type": "str",
      "required": true,
      "description": "目標 IP 或域名",
      "example": "192.168.1.1"
    },
    {
      "name": "ports",
      "type": "str",
      "required": false,
      "default": "1-1000",
      "description": "端口範圍"
    }
  ],
  "return_info": {
    "type": "ScanResult",
    "description": "掃描結果對象",
    "structure": {"open_ports": [], "services": []}
  },
  
  # 健康狀態
  "health_score": 0.95,      # 0.0-1.0
  "availability": 0.98,      # 0.0-1.0
  "error_rate": 0.02,        # 0.0-1.0
  "last_used": "2025-11-28T12:00:00Z",
  
  # 元數據
  "version": "1.0.0",
  "created_at": "2025-11-28T04:07:20Z"
}
```

#### 2. 能力摘要 (CapabilitySummary)

```python
{
  "total_capabilities": 782,
  "by_category": {
    "scanning": 286,
    "analysis": 207,
    "integration": 111,
    "attacking": 98,
    "utility": 60,
    "reporting": 20
  },
  "by_complexity": {
    "1": 150,  # Trivial
    "2": 280,  # Simple
    "3": 200,  # Moderate
    "4": 100,  # Complex
    "5": 52    # Advanced
  },
  "healthy_count": 750,      # health_score >= 0.7
  "unhealthy_count": 32,     # health_score < 0.7
  "avg_health_score": 0.92
}
```

#### 3. RAG 向量數據庫

```
位置: data/vector_db/chroma/
大小: 7.50 MB
文檔數: 782
向量維度: 384 (all-MiniLM-L6-v2)
嵌入模型: sentence-transformers/all-MiniLM-L6-v2

支援操作:
- 語義搜索: kb.search("掃描能力")
- 相似度排序: 返回最相關的 top_k 結果
- 元數據過濾: 按模組、語言、類別過濾
```

---

## 🎯 這些數據的用途 (關鍵價值)

### 用途 1: AI 任務規劃時選擇能力

**場景**: 用戶要求「對 example.com 進行 SQL 注入測試」

```python
# AI 核心模組的決策過程
async def plan_sql_injection_task(target: str):
    # 1. 查詢 RAG 知識庫: 有哪些 SQL 注入相關能力?
    capabilities = await kb.search(
        query="SQL injection attack testing",
        top_k=10
    )
    
    # 2. 從搜索結果中找到可用能力
    for cap in capabilities:
        print(f"找到能力: {cap.metadata['capability_name']}")
        print(f"  模組: {cap.metadata['module']}")
        print(f"  健康度: {cap.metadata['health_score']}")
        print(f"  複雜度: {cap.metadata['complexity']}")
    
    # 3. 選擇最佳能力 (高健康度 + 適當複雜度)
    best_cap = select_best_capability(capabilities)
    
    # 4. 構建執行計劃
    plan = ExecutionPlan(
        steps=[
            ExecutionStep(
                capability=best_cap.metadata['capability_name'],
                module=best_cap.metadata['module'],
                parameters={"target": target, "payload": "' OR 1=1 --"}
            )
        ]
    )
    
    return plan
```

**價值**: AI 自動找到合適工具,無需人工硬編碼

---

### 用途 2: 實時健康監控

**場景**: 某個能力執行頻繁失敗,需要告警

```python
# 定期檢查能力健康度
async def monitor_capabilities():
    # 查詢所有不健康的能力
    unhealthy = await kb.search(
        query="",  # 空查詢返回所有
        filters={"health_score": {"$lt": 0.7}}
    )
    
    for cap in unhealthy:
        logger.warning(
            f"⚠️ 不健康能力: {cap.metadata['capability_name']}\n"
            f"   健康度: {cap.metadata['health_score']}\n"
            f"   錯誤率: {cap.metadata['error_rate']}\n"
            f"   建議: 檢查模組或暫時停用"
        )
```

**價值**: 主動發現問題,避免使用失效工具

---

### 用途 3: 新手引導 (類似 Copilot)

**場景**: 用戶詢問「我該如何掃描端口?」

```python
# AI 助手回答
async def answer_user_question(question: str):
    # 查詢相關能力
    results = await kb.search(query=question, top_k=3)
    
    response = f"根據您的問題,我找到以下能力:\n\n"
    
    for i, cap in enumerate(results, 1):
        name = cap.metadata['capability_name']
        desc = cap.metadata['description']
        params = cap.metadata.get('parameters', [])
        
        response += f"{i}. **{name}**\n"
        response += f"   描述: {desc}\n"
        response += f"   參數: {', '.join([p['name'] for p in params])}\n"
        response += f"   示例: python aiva_cli.py execute {name} --target 192.168.1.1\n\n"
    
    return response
```

**價值**: 智能文檔,自動生成使用指南

---

### 用途 4: 跨語言能力協調

**場景**: 需要組合 Python、Rust、Go 工具完成複雜任務

```python
# 多階段掃描任務
async def multi_stage_scan(target: str):
    # 階段 1: Rust 快速偵察
    rust_caps = await kb.search(
        query="fast reconnaissance",
        filters={"language": "rust"}
    )
    rust_result = await execute_capability(rust_caps[0], target)
    
    # 階段 2: Python 深度爬取
    python_caps = await kb.search(
        query="web crawling form extraction",
        filters={"language": "python"}
    )
    python_result = await execute_capability(python_caps[0], target)
    
    # 階段 3: Go SSRF 測試
    go_caps = await kb.search(
        query="SSRF testing",
        filters={"language": "go"}
    )
    go_result = await execute_capability(go_caps[0], target)
    
    return integrate_results([rust_result, python_result, go_result])
```

**價值**: 自動選擇最佳語言/工具組合

---

### 用途 5: 能力依賴分析

**場景**: 某個能力需要其他能力作為前置條件

```python
# 檢查依賴
async def check_dependencies(capability_name: str):
    cap_info = await kb.search(
        query=capability_name,
        top_k=1
    )[0]
    
    dependencies = cap_info.metadata.get('dependencies', [])
    
    for dep in dependencies:
        dep_cap = await kb.search(query=dep, top_k=1)[0]
        
        if dep_cap.metadata['health_score'] < 0.7:
            raise Exception(
                f"依賴能力 {dep} 不健康,無法執行 {capability_name}"
            )
    
    return True
```

**價值**: 確保執行環境完整

---

## 🔄 外閉環 (對外實戰) 詳細分析

### 執行流程

```
┌─────────────────────────────────────────────────────┐
│  步驟 1: 任務執行 (TaskExecutor)                      │
│  ─────────────────────────────────────────────────  │
│  使用內閉環發現的能力執行任務:                         │
│  • 從 RAG 查詢合適能力                                │
│  • 構建執行計劃 (ExecutionPlan)                      │
│  • 執行各個步驟                                       │
│  • 記錄執行軌跡 (ExecutionTrace)                     │
│                                                       │
│  輸入: ExecutionPlan (計劃 AST)                       │
│  輸出: list[ExecutionTrace] (實際執行軌跡)            │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 2: 偏差分析 (ASTTraceComparator)               │
│  ─────────────────────────────────────────────────  │
│  對比計劃 vs 實際:                                    │
│  • 步驟順序是否一致?                                  │
│  • 預期結果 vs 實際結果?                              │
│  • 是否有異常或錯誤?                                  │
│  • 執行時間差異?                                      │
│                                                       │
│  偏差類型:                                            │
│  • STEP_SKIPPED: 跳過了計劃步驟                      │
│  • STEP_ADDED: 新增了未計劃步驟                      │
│  • RESULT_MISMATCH: 結果與預期不符                   │
│  • ERROR: 執行錯誤                                    │
│                                                       │
│  輸出: list[DeviationRecord] (偏差記錄)               │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 3: 判斷是否需要訓練                             │
│  ─────────────────────────────────────────────────  │
│  顯著偏差標準:                                        │
│  • 偏差數量 > 閾值 (如 3 個)                         │
│  • 偏差嚴重程度 > 閾值 (如 high)                     │
│  • 重複出現的偏差模式                                 │
│                                                       │
│  決策: 是否觸發模型訓練?                              │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 4: 模型訓練 (ModelTrainer)                      │
│  ─────────────────────────────────────────────────  │
│  使用偏差數據訓練 AI 模型:                            │
│  • 準備訓練樣本 (輸入 → 預期輸出 → 實際偏差)        │
│  • 調整模型權重                                       │
│  • 生成新版本權重文件                                 │
│                                                       │
│  輸出: ModelTrainingResult (新權重版本)               │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 5: 權重註冊 (AIWeightManager)                  │
│  ─────────────────────────────────────────────────  │
│  將新權重註冊到系統:                                  │
│  • 保存權重文件到 weights/ 目錄                      │
│  • 更新權重版本記錄                                   │
│  • 通知相關模組可用新版本                             │
│                                                       │
│  輸出: new_weights_version (如 "v1.2.3")             │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  步驟 6: 回饋到內閉環 (可選)                          │
│  ─────────────────────────────────────────────────  │
│  根據實戰結果更新能力健康度:                          │
│  • 成功執行 → 提升 health_score                      │
│  • 失敗執行 → 降低 health_score, 提升 error_rate    │
│  • 更新 last_used 時間戳                             │
│                                                       │
│  回饋: 更新 RAG 知識庫中的能力元數據                  │
└─────────────────────────────────────────────────────┘
```

### 預期得到的數據 (外閉環產出)

#### 1. 執行軌跡 (ExecutionTrace[])

```python
[
  {
    "step_id": "step_1",
    "capability": "run_nmap_scan",
    "module": "scan",
    "input": {"target": "192.168.1.1", "ports": "1-1000"},
    "output": {"open_ports": [22, 80, 443], "services": [...]},
    "status": "success",
    "start_time": "2025-11-28T12:00:00Z",
    "end_time": "2025-11-28T12:00:15Z",
    "duration_ms": 15000,
    "error": null
  },
  {
    "step_id": "step_2",
    "capability": "test_sql_injection",
    "module": "features/function_attacks",
    "input": {"url": "http://192.168.1.1/login", "payload": "' OR 1=1 --"},
    "output": {"vulnerable": true, "confidence": 0.95},
    "status": "success",
    "start_time": "2025-11-28T12:00:15Z",
    "end_time": "2025-11-28T12:00:20Z",
    "duration_ms": 5000,
    "error": null
  }
]
```

#### 2. 偏差記錄 (DeviationRecord[])

```python
[
  {
    "deviation_id": "dev_001",
    "type": "RESULT_MISMATCH",
    "severity": "medium",
    "step_id": "step_2",
    "expected": {"vulnerable": false},
    "actual": {"vulnerable": true},
    "description": "預期不會發現漏洞,但實際發現 SQL 注入",
    "timestamp": "2025-11-28T12:00:20Z"
  }
]
```

#### 3. 訓練結果 (ModelTrainingResult)

```python
{
  "training_id": "train_001",
  "samples_used": 50,
  "training_duration_ms": 120000,
  "new_weights_version": "v1.2.3",
  "performance_metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91
  },
  "improvement": "+3.5%",  # 相比上個版本
  "timestamp": "2025-11-28T12:05:00Z"
}
```

---

## 🔗 雙閉環協同運作

### 完整循環示例

```python
# 第一次運行 (冷啟動)
# 1. 內閉環: 掃描能力 → 發現 782 個能力 → 注入 RAG
result_internal = await internal_loop.sync_capabilities_to_rag()

# 2. 用戶發起任務
task = "對 example.com 進行安全測試"

# 3. AI 規劃: 查詢 RAG 選擇能力
plan = await ai_core.create_execution_plan(task)

# 4. 執行任務 (外閉環開始)
trace = await task_executor.execute(plan)

# 5. 偏差分析
deviations = await external_loop.process_execution_result(plan, trace)

# 6. 如果發現顯著偏差 → 訓練模型
if deviations.deviations_significant:
    training_result = await external_loop.train_model(deviations)
    
# 7. 回饋內閉環: 更新能力健康度
await internal_loop.update_capability_health(trace)

# --- 第二次運行 (已優化) ---
# 8. AI 使用新權重做出更好的決策
plan2 = await ai_core.create_execution_plan(task)  # 使用 v1.2.3 權重

# 9. 執行效果更好,偏差減少
trace2 = await task_executor.execute(plan2)

# 持續循環...
```

---

## 📊 數據統計與價值總結

### 內閉環產出

| 數據類型 | 數量/大小 | 用途 |
|---------|----------|------|
| 能力記錄 | 782 個 | AI 任務規劃選擇工具 |
| 向量文檔 | 7.50 MB | 語義搜索、相似度匹配 |
| 模組統計 | 4 大模組 | 模組健康監控 |
| 語言統計 | 4 種語言 | 跨語言協調 |
| 分類統計 | 6 大類別 | 功能分類查詢 |

### 外閉環產出

| 數據類型 | 用途 |
|---------|------|
| 執行軌跡 | 實戰經驗記錄 |
| 偏差記錄 | 模型訓練樣本 |
| 訓練結果 | 模型權重更新 |
| 性能指標 | 效果評估 |

### 核心價值

1. **自動化**: AI 自動發現和管理能力,無需人工維護
2. **智能化**: 基於 RAG 語義搜索,選擇最佳工具組合
3. **自適應**: 從實戰中學習,持續優化決策策略
4. **可視化**: 健康度監控、偏差分析、訓練進度可視化
5. **可擴展**: 新增工具自動被發現和整合

---

## 🎯 實際應用場景

### 場景 1: 智能滲透測試

```
用戶: "測試 example.com 的 SQL 注入漏洞"

內閉環: 
  → 查詢 RAG: "SQL injection testing"
  → 找到 5 個相關能力
  → 選擇健康度最高的 test_sql_injection

外閉環:
  → 執行測試
  → 記錄結果 (成功/失敗)
  → 如果失敗,分析原因
  → 調整策略,再次嘗試
  → 學習經驗,優化未來測試
```

### 場景 2: 系統健康監控

```
定時任務 (每小時):
  → 查詢所有能力健康度
  → 發現 test_xss 錯誤率 > 30%
  → 告警: "XSS 測試工具異常"
  → 建議: "檢查依賴庫或回退到舊版本"
  → 自動標記為 "不推薦使用"
```

### 場景 3: 新手培訓

```
新手: "我該如何掃描目標?"

AI 助手:
  → 查詢 RAG: "scanning reconnaissance"
  → 找到 3 個新手友好的能力 (complexity=1-2)
  → 提供使用範例和參數說明
  → 引導逐步執行
  → 解釋每步結果
```

---

## 🚀 結論

AIVA 雙閉環系統通過**內閉環自我認知**和**外閉環實戰學習**,實現了:

1. ✅ **完整能力庫**: 782 個跨語言能力自動發現
2. ✅ **智能任務規劃**: RAG 語義搜索選擇最佳工具
3. ✅ **實時健康監控**: 主動發現和預警問題能力
4. ✅ **持續自我優化**: 從實戰經驗中學習進化
5. ✅ **跨語言協調**: Python/Rust/TypeScript/Go 自動組合

這些數據不是靜態文檔,而是**活的知識庫**,隨著系統運行持續更新和優化,使 AIVA 越用越強。
