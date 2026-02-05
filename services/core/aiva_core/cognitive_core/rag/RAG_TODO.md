# RAG 系統待辦清單

## 當前狀態（2026-02-04 更新）
- ✅ 已完成 RAG 架構設計分析
- ✅ 已決定使用 JSONL（不使用 SQLite）
- ✅ 已設計 CLI 指令庫結構
- ✅ **重大發現**: Internal Exploration 已有 286 個數據流！
- ✅ 已識別攻擊相關流程：attack_coordinator, unified_attack_executor
- ✅ **P0 完成**: RAG 決策層整合 External Exploration（525 flows）
- ✅ **CLI Decision Engine 已實現**
- ✅ **FlowExecutorAdapter 已實現**
- ✅ **AttackCoordinator 整合完成**

---

## 🎉 P0 階段完成總結

### 已實現組件

1. **CLIDecisionEngine** (`services/core/aiva_core/cognitive_core/learning_system/cli_decision_engine.py`)
   - ✅ 直接讀取 `external_classification.json`
   - ✅ 載入 525 個攻擊 flows（287 個可操作）
   - ✅ 支援能力檢索（XSS, SQLi, SSRF, Scanner, PostEx）
   - ✅ 根據掃描上下文推薦攻擊流程
   - ✅ 關鍵字搜索和優先級排序

2. **FlowExecutorAdapter** (`services/core/aiva_core/cognitive_core/learning_system/flow_executor_adapter.py`)
   - ✅ 橋接 CLIDecisionEngine 和 FlowExecutor
   - ✅ 參數轉換（AttackFlow → 執行參數）
   - ✅ 執行協調和結果標準化
   - ✅ 移除 Dry Run，直接真實執行

3. **AttackCoordinator 整合** (`services/core/aiva_core/task_planning/commander/attack_coordinator.py`)
   - ✅ 新增 `rag_smart_attack()` 方法
   - ✅ 新增 `rag_targeted_attack()` 方法
   - ✅ 根據掃描結果自動選擇攻擊流程
   - ✅ 支援多種攻擊模式（auto, aggressive, cautious）
   - ✅ 移除 dry_run 模式

### 測試結果

執行 `test_rag_smart_attack.py` 完成 5 個測試：
- ✅ CLI 決策引擎基本功能
- ✅ FlowExecutorAdapter 參數轉換
- ✅ 定向攻擊（指定能力）
- ✅ 多能力攻擊（XSS + SQLi + SSRF）
- ✅ 完整整合流程

**關鍵數據**:
- 載入 525 個攻擊 flows
- 287 個可操作flows（54.7%）
- XSS: 48/97 可操作
- SQLi: 68/115 可操作
- SSRF: 28/64 可操作

---

## 📋 後續優化方向（P1-P3）

### P1: 實際執行與驗證
- [x] 移除 Dry Run 模式，執行真實攻擊
- [ ] 靶場驗證（testphp.vulnweb.com）
- [ ] 收集執行結果和錯誤處理
- [ ] 優化超時和重試策略

### P2: 優化決策算法
- [ ] 根據掃描結果調整優先級算法
- [ ] 考慮目標指紋（技術棧、WAF、防護等級）
- [ ] 學習成功率，動態調整推薦
- [ ] 實現 Vector Store 語義搜索（可選）

### P3: 擴展功能
- [ ] 支援更多攻擊能力（XXE, File Upload, Deserialization）
- [ ] 實現攻擊鏈組合（Phase1 → Phase2 → PostEx）
- [ ] 結果反饋循環（成功率統計）
- [ ] AI 自然語言查詢接口

---

## 🎯 整合策略變更

**原計劃**: 從零創建 CLI 指令庫和執行器  
**新發現**: Internal Exploration 已有完整的執行框架（286 flows）

**新策略**: 
1. 復用 `FlowExecutor` 執行現有的 286 個數據流
2. 創建 RAG 決策層，映射能力到 Flow ID
3. 根據掃描結果智能選擇 Flow 和參數

**參考文檔**: `RAG_INTERNAL_EXPLORATION_INTEGRATION.md`

---

## 一、對內搜索：CLI 指令決策系統（基於 Internal Exploration）

### 1.1 映射現有數據流到攻擊能力 ⏳ 高優先級

**目標**: 將 Internal Exploration 的 286 個流程映射到 RAG 決策系統

**已識別的攻擊相關流程**:
```
Flow 96-99:  attack_coordinator 系列
Flow 103:    attack_coordinator -> two_phase_scan_orchestrator
Flow 8,11,12: unified_executor -> unified_attack_executor
更多流程...  (需要完整分析 latest_classification.json)
```

**任務**:
- [ ] 1.1.1 完整分析 latest_classification.json（286 flows）
- [ ] 1.1.2 識別攻擊能力相關的 Flow ID（xss, sqli, ssrf, lfi, rce）
- [ ] 1.1.3 提取每個 Flow 的入口方法和參數
- [ ] 1.1.4 創建能力→Flow ID 映射表

**命令**:
```bash
# 查看所有流程
python -m services.core.aiva_core.internal_exploration.aiva_internal_executor --list

# 查看特定流程
python -m services.core.aiva_core.internal_exploration.aiva_internal_executor --flow 96 --dry-run
```

---

### 1.2 創建 CLI 指令庫數據 ⏳ 高優先級（新格式）

**目標**: 創建指令庫，每條指令映射到 Internal Exploration 的 Flow ID

**新的文件結構**:
```
services/integration/data/cli_commands/
├── xss_commands.jsonl          # XSS → Flow ID 映射
├── sqli_commands.jsonl         # SQL 注入 → Flow ID 映射
├── ssrf_commands.jsonl         # SSRF → Flow ID 映射
├── phase0_commands.jsonl       # 偵察 → Flow ID 映射
└── attack_commands.jsonl       # 通用攻擊流程
```

**新的內容格式**（整合 Internal Exploration）:
```json
{
  "tool_name": "unified_attack_executor",
  "capability": "sqli",
  "flow_id": 8,  // ← 映射到 Internal Exploration 的 Flow ID
  "flow_path": "unified_executor -> unified_attack_executor",
  "entry_method": "execute",
  "適用場景": {
    "技術棧": ["PHP", "ASP", "JSP"],
    "發現端口": [80, 443, 8080],
    "前置條件": ["發現參數", "發現表單"]
  },
  "參數調整規則": {
    "如果檢測到 MySQL": {
      "context_data": {"dbms": "MySQL", "攻擊強度": "高"}
    },
    "如果有 WAF": {
      "context_data": {"繞過技術": "編碼混淆", "延遲時間": 2}
    }
  },
  "context_template": {
    "target": "{target_url}",
    "capability": "sqli",
    "parameters": {}
  },
  "priority": 5,
  "succe2.1 準備 attack_commands.jsonl（基於 attack_coordinator, Flow 96-103）
- [ ] 1.2.2 準備 xss_commands.jsonl（基於 xss 相關 flows）
- [ ] 1.2.3 準備 sqli_commands.jsonl（基於 sqli 相關 flows）
- [ ] 1.2.4 準備 phase0_commands.jsonl（基於偵察相關 flows）
- [ ] 1.2.5 驗證每個 Flow ID 的執行能力

---

### 1.3 實現 CLIDecisionEngine ⏳ 高優先級（新名稱）
- [ ] 1.1.5 準備 phase0/phase1 偵察工具

---

### 1.2 實現 CLICommandManager ⏳

**目標**: 創建 CLI 決策引擎（整合 Internal Exploration）

**文件**: `services/core/aiva_core/cognitive_core/learning_system/cli_decision_engine.py`

**核心功能**:
```python
class CLIDecisionEngine:
    def __init__(self):
        self.cli_commands = self._load_cli_commands()
        self.flow_executor = FlowExecutor()  # ← 復用 Internal Exploration
    
    def search_commands(capability, scan_results, top_k=5):
        """根據掃描結果搜索合適的 Flow"""
        # 搜索指令庫，返回 Flow ID 列表
        pass
    
    def adjust_context_data(command, scan_results):
        """根據掃描結果自動調整 context_data"""
        # 應用參數調整規則
        pass
    
    def 3.1 實現 JSONL 載入（讀取 cli_commands/*.jsonl）
- [ ] 1.3.2 實現 search_commands（適用場景匹配，返回 Flow ID）
- [ ] 1.3.3 實現 adjust_context_data（參數調整規則）
- [ ] 1.3.4 整合 FlowExecutor（from internal_exploration）
- [ ] 1.3.5 實現 execute_flow（調用 FlowExecutor）
- [ ] 1.3.6 添加單元測試

---

### 1.4.2.2 實現 search_commands（適用場景匹配）
- [ ] 1.2.3 實現 adjust_parameters（參數調整規則）
- [ ] 1.2.4 實現 build_command（命令構建）
- [ ] 1.2.5 添加單元測試

---

### 1.3 整合到 RAG Trigger ⏳

**目標**: 將 CLICommdecision_engine = CLIDecisionEngine()  # 新增（整合 FlowExecutor）
        self.vector_store = VectorStore()
    
    async def _decide_normal_flow(self, current_phase, scan_results):
        """正常流程：使用 CLI 決策引擎執行攻擊流程"""
        # 1. 搜索合適的 Flow
        commands = self.cli_decision_engine.search_commands(
            capability=current_phase,
            scan_results=scan_results
        )
        
        # 2. 選擇最佳 Flow
        best_command = commands[0]
        flow_id = best_command["flow_id"]
        
        # 3. 調整參數
        context_data = self.cli_decision_engine.adjust_context_data(
            command=best_command,
            scan_results=scan_results
        )
        
        # 4. 執行 Flow（復用 Internal Exploration）
        result = self.cli_decision_engine.execute_flow(flow_id, context_data)
        
        return {"action": "executed", "result": result, ...}
```

**子任務**:
- [ ] 1.4.1 更新 RAGTrigger 初始化（添加 CLIDecisionEngine）
- [ ] 1.4.2 實現 _decide_normal_flow 方法（調用 CLIDecisionEngine）
- [ ] 1.4.3 定義掃描結果的標準格式
- [ ] 1.4.4 測試整合流程（Dry Run 模式）
- [ ] 1.4.5 實際執行測試（選擇安全的 Flow）self.cli_manager.build_command(...)
        return {"action": "execute_command", "command": command, ...}
```

**子任務**:
- [ ] 1.3.1 更新 RAGTrigger 初始化
- [ ] 1.3.2 實現 _decide_normal_flow 方法
- [ ] 1.3.3 定義掃描結果的標準格式
- [ ] 1.3.4 測試整合流程

---

## 二、對外搜索：關鍵字提取系統

### 2.1 實現 KeywordExtractor ⏳

**目標**: 從錯誤響應中提取搜索關鍵字

**文件**: `services/core/aiva_core/cognitive_core/learning_system/keyword_extractor.py`

**核心功能**:
```python
class KeywordExtractor:
    def extract_keywords(error_data):
        """從錯誤中提取關鍵字"""
        # 1. 提取技術棧（ModSecurity, Cloudflare, Apache, etc.）
        # 2. 提取錯誤類型（403, Access Denied, etc.）
        # 3. 提取規則 ID（981242, etc.）
        # 4. 提取攻擊類型（SQL injection, XSS, etc.）
        return keywords
```

**子任務**:
- [ ] 2.1.1 實現技術棧提取（從 headers 和 error_message）
- [ ] 2.1.2 實現錯誤類型提取
- [ ] 2.1.3 實現規則 ID 提取（正則表達式）
- [ ] 2.1.4 實現攻擊類型推測
- [ ] 2.1.5 添加單元測試（各種錯誤場景）

---

### 2.2 實現外部搜索 API ⏳

**目標**: 使用關鍵字搜索外部資源

**文件**: `services/core/aiva_core/cognitive_core/learning_system/external_search.py`

**核心功能**:
```python
async def search_modsecurity_rules(keywords):
    """搜索 ModSecurity 規則庫"""
    pass

async def search_waf_bypass_techniques(keywords):
    """搜索 WAF 繞過技術"""
    pass

async def search_cve_database(keywords):
    """搜索 CVE 資料庫"""
    pass

async def search_google(query):
    """Google 技術搜索"""
    pass
```

**子任務**:
- [ ] 2.2.1 實現 ModSecurity 規則搜索（GitHub API）
- [ ] 2.2.2 實現 WAF 繞過技術搜索（Exploit-DB）
- [ ] 2.2.3 實現 CVE 搜索（NVD API）
- [ ] 2.2.4 實現 Google 自定義搜索
- [ ] 2.2.5 添加錯誤處理和重試邏輯

---

### 2.3 整合錯誤處理流程 ⏳

**目標**: 將關鍵字提取和外部搜索整合到 RAG

**文件**: `services/core/aiva_core/cognitive_core/learning_system/rag_trigger.py`

**整合點**:
```python
class RAGTrigger:
    async def _handle_error_with_rag(self, error_data, scan_results):
        """錯誤處理：提取關鍵字並對外搜索"""
        # 1. 提取關鍵字
        keywords = self.keyword_extractor.extract_keywords(error_data)
        
        # 2. 對內搜索（歷史解決方案）
        internal_results = await self._search_internal_solutions(keywords)
        
        # 3. 對外搜索（如果內部沒找到）
        if not internal_results or internal_results[0]["score"] < 0.8:
            external_results = await self._search_external_resources(keywords)
        
        # 4. 生成新策略
        return new_strategy
```

**子任務**:
- [ ] 2.3.1 實現 _handle_error_with_rag 方法
- [ ] 2.3.2 實現 _search_internal_solutions（向量搜索）
- [ ] 2.3.3 實現 _search_external_resources（外部 API）
- [ ] 2.3.4 實現策略生成邏輯
- [ ] 2.3.5 測試完整錯誤處理流程

---

## 三、向量存儲同步

### 3.1 同步整合模組數據 ⏳

**目標**: 將 experiences/*.jsonl 數據同步到向量存儲

**文件**: `sync_experiences_to_vector_store.py`（已存在）

**任務**:
- [ ] 3.1.1 執行同步腳本
  ```bash
  python sync_experiences_to_vector_store.py --sync-knowledge-base
  ```
- [ ] 3.1.2 驗證向量存儲記錄數（應從 782 增加到 6000+）
- [ ] 3.1.3 測試內部向量搜索功能

---

### 3.2 實現實時向量更新 ⏳

**目標**: 在執行任務時自動更新向量存儲

**文件**: `services/integration/app.py`

**整合點**:
```python
# 在保存任務數據後，同時更新向量存儲
data_manager.save_task_data(...)
vector_store.add_document(...)  # 新增
```

**子任務**:
- [ ] 3.2.1 在 app.py 中添加向量存儲更新邏輯
- [ ] 3.2.2 確保數據格式一致
- [ ] 3.2.3 添加錯誤處理（向量更新失敗不影響主流程）

---

## 四、測試和驗證

### 4.1 單元測試 ⏳

**待測試的模組**:
- [ ] CLICommandManager
  - [ ] 測試指令搜索
  - [ ] 測試參數調整
  - [ ] 測試命令構建
  
- [ ] KeywordExtractor
  - [ ] 測試各種錯誤場景
  - [ ] 測試關鍵字提取準確性
  
- [ ] ExternalSearch
  - [ ] 測試 API 調用（mock）
  - [ ] 測試錯誤處理

---

### 4.2 集成測試 ⏳

**測試場景**:
- [ ] 4.2.1 正常流程：Phase 0 → 選擇偵察工具
- [ ] 4.2.2 正常流程：檢測到 PHP+MySQL → 選擇 sqlmap 並調整參數
- [ ] 4.2.3 錯誤處理：遇到 ModSecurity 403 → 提取關鍵字 → 搜索繞過技術
- [ ] 4.2.4 錯誤處理：遇到未知錯誤 → 對外搜索 → 生成新策略

---

## 五、優化和擴展

### 5.1 性能優化 ⏳
- [ ] 添加 CLI 指令緩存
- [ ] 優化外部 API 調用（並發、超時）
- [ ] 添加結果緩存（相同關鍵字不重複搜索）

### 5.2 功能擴展 ⏳
- [ ] 添加成功率統計（更新 CLI 指令的 success_rate）
- [ ] 添加執行歷史分析（學習最佳參數組合）
- [ ] 添加自動化繞過策略生成

---

## 六、未來升級點

### 6.1 資料庫升級條件
當以下條件觸發時，考慮從 JSONL 升級到 SQLite：
- [ ] experiences/*.jsonl 任一文件 > 10,000 條記錄
- [ ] cli_commands/*.jsonl 總記錄 > 1,000 條
- [ ] 需要複雜統計分析（工具成功率、參數效果分析）
- [ ] 需要多進程並發寫入

### 6.2 升級到 PostgreSQL 條件
- [ ] 總數據量 > 100,000 條
- [ ] 需要分布式部署
- [ ] 需要實時統計儀表板

---

## 當前優先級排序

### 🔥 高優先級（立即開始）
1. **創建 CLI 指令庫數據**（xss_commands.jsonl, sqli_commands.jsonl）
2. **實現 CLICommandManager**（基本的搜索和參數調整）
3. **整合到 RAG Trigger**（對內搜索功能）

### 🟡 中優先級（後續實現）
4. **實現 KeywordExtractor**（關鍵字提取）
5. **實現外部搜索 API**（對外搜索功能）
6. **同步向量存儲**（執行 sync 腳本）

### 🔵 低優先級（有時間再做）
7. **實時向量更新**
8. **性能優化**
9. **功能擴展**

---

## 檢查點

完成每個階段後，檢查以下項目：
- [ ] 代碼是否符合現有架構？
- [ ] 是否添加了足夠的錯誤處理？
- [ ] 是否添加了日誌記錄？
- [ ] 是否可以獨立測試？
- [ ] 是否更新了相關文檔？

---

**最後更新**: 2026-01-20
**狀態**: 規劃完成，等待實現
