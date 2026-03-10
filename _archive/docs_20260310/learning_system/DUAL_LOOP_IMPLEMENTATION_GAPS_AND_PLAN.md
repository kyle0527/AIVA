# 🔄 雙閉環實施缺口分析與 CLI 整合計劃

**分析日期**: 2026年1月20日  
**參考文檔**: [DUAL_LOOP_DESIGN_GUIDE.md](guides/DUAL_LOOP_DESIGN_GUIDE.md)

---

## 📊 當前系統狀態評估

### ✅ 已實現的組件

#### 1. **多引擎掃描系統** (外部閉環 Phase 0-2)
- 📍 位置: `services/scan/coordinators/multi_engine_coordinator.py`
- 狀態: ✅ **已實現**
- 功能:
  - ✅ Rust 快速偵察引擎配置
  - ✅ Python 深度掃描引擎配置
  - ✅ 引擎健康檢查
  - ✅ 任務分發機制
  - ⚠️ **缺少 AI 決策層整合**

```python
# 已有代碼
class MultiEngineCoordinator:
    ENGINES = {
        "rust_scanner": EngineConfig(...),
        "python_passive": EngineConfig(...),
        "python_xxe": EngineConfig(...),
    }
    # ❌ 缺少: AI 決策接口
    # ❌ 缺少: RAG 知識庫查詢
    # ❌ 缺少: 歷史數據庫整合
```

#### 2. **Internal Exploration CLI 系統** (內部閉環基礎)
- 📍 位置: `services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py`
- 狀態: ✅ **已實現**
- 功能:
  - ✅ 286 個數據流定義
  - ✅ 動態流程執行器
  - ✅ CLI 命令生成
  - ✅ Dry-run 模式
  - ⚠️ **只有執行能力,無探索能力**

```bash
# 已有 CLI 命令
python -m aiva_core.internal_exploration.aiva_cli_implementation --list
python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11
```

#### 3. **RAG 系統** (知識增強)
- 📍 位置: `services/core/aiva_core/cognitive_core/rag/`
- 狀態: ✅ **已實現**
- 功能:
  - ✅ Vector Store (782+ 記錄)
  - ✅ RAG Trigger (內部/外部搜索)
  - ✅ Knowledge Base
  - ⚠️ **未整合到決策流程**

#### 4. **Task Planning 系統** (任務規劃)
- 📍 位置: `services/core/aiva_core/task_planning/`
- 狀態: ✅ **已實現**
- 功能:
  - ✅ Unified Executor
  - ✅ Execution Planner
  - ✅ Task Queue Manager
  - ⚠️ **未整合雙閉環決策**

---

## ❌ 缺失的核心組件

### 🔴 高優先級 (阻塞雙閉環運作)

#### 1. **AI 決策核心層** (Critical)
```
當前狀況: 完全缺失
阻塞影響: 無法進行智能引擎選擇和策略調整
```

**需要實現**:
```python
# services/core/aiva_core/cognitive_core/ai_decision_core.py

class AIDecisionCore:
    """AI 決策核心 - 雙閉環的大腦"""
    
    def __init__(self):
        self.internal_explorer = SystemSelfExplorer()  # ❌ 不存在
        self.rag_agent = RAGTrigger()                  # ✅ 存在
        self.history_db = ExternalFeedbackDB()         # ❌ 不存在
        
    async def decide_scan_strategy(
        self,
        target: str,
        user_constraints: dict
    ) -> dict:
        """
        整合三個數據源做決策:
        1. 內部探索: 我有哪些能力?
        2. RAG 知識庫: 類似目標的最佳實踐
        3. 歷史數據: 過去的成功率統計
        """
        # 1. 查詢內部能力
        available_engines = await self.internal_explorer.get_available_engines()
        available_attacks = await self.internal_explorer.get_available_attacks()
        
        # 2. RAG 檢索建議
        rag_suggestions = await self.rag_agent.search_similar_targets(target)
        
        # 3. 查詢歷史數據
        historical_stats = await self.history_db.query_success_rates(target)
        
        # 4. 整合決策
        return {
            "selected_engines": [...],
            "attack_strategy": {...},
            "reasoning": "基於..."
        }
```

**缺失原因**: 設計中存在,實現時未優先處理

---

#### 2. **SystemSelfExplorer (內部探索器)** (Critical)
```
當前狀況: 完全缺失
阻塞影響: AI 不知道自己有什麼能力
```

**需要實現**:
```python
# services/core/aiva_core/internal_exploration/system_self_explorer.py

class SystemSelfExplorer:
    """系統自我探索器 - 內部閉環的眼睛"""
    
    async def explore_system(self) -> dict:
        """掃描所有模組,返回能力清單"""
        return {
            "available_engines": {
                "rust_scanner": {"status": "healthy", "capabilities": [...]},
                "python_passive": {"status": "healthy", "capabilities": [...]},
            },
            "available_attacks": {
                "sqli": {"module": "...", "confidence": 0.8},
                "xss": {"module": "...", "confidence": 0.9},
            },
            "broken_modules": ["nmap_integration"],  # 有問題的模組
            "capability_gaps": ["xxe", "ssrf_advanced"]  # 缺失的能力
        }
    
    async def get_available_engines(self) -> list[str]:
        """快速查詢可用引擎"""
        # 讀取 latest_classification.json 的 286 flows
        # 解析哪些 flow 是引擎相關
        pass
```

**缺失原因**: 設計階段認為是後期優化,實際上是核心依賴

---

#### 3. **ExternalFeedbackLoop (外部反饋循環)** (High)
```
當前狀況: 完全缺失
阻塞影響: 無法記錄實戰數據,無法學習優化
```

**需要實現**:
```python
# services/core/aiva_core/cognitive_core/learning_system/external_feedback_loop.py

class ExternalFeedbackLoop:
    """外部反饋閉環 - 從實戰中學習"""
    
    def __init__(self):
        self.db = SQLiteDatabase("external_feedback.db")
        self.vector_store = VectorStore()
        
    async def record_scan_result(self, scan_result: ScanResult):
        """記錄掃描結果到數據庫"""
        await self.db.insert("scan_history", {
            "target": scan_result.target,
            "tech_stack": scan_result.technologies,
            "vulnerabilities": scan_result.vulnerabilities,
            "success_rate": scan_result.success_rate,
            "timestamp": datetime.now()
        })
        
        # 更新向量庫
        await self.vector_store.add_documents([{
            "content": f"Target {scan_result.target} with {scan_result.tech_stack}",
            "metadata": scan_result.to_dict()
        }])
    
    async def query_success_rates(self, target: str) -> dict:
        """查詢歷史成功率"""
        similar_targets = await self.db.query(
            "SELECT * FROM scan_history WHERE target LIKE ?",
            f"%{target}%"
        )
        
        return {
            "sqli_success_rate": 0.78,
            "xss_success_rate": 0.62,
            "common_defenses": ["cloudflare_waf"]
        }
```

**缺失原因**: 依賴數據庫設計,尚未優先實現

---

### 🟡 中優先級 (影響優化效果)

#### 4. **InternalFeedbackLoop (內部反饋循環)** (Medium)
```
當前狀況: 部分實現 (有 RAG 但無整合)
影響: 無法生成優化建議
```

**需要補充**:
```python
# services/core/aiva_core/internal_exploration/internal_feedback_loop.py

class InternalFeedbackLoop:
    """內部反饋閉環 - 自我優化建議"""
    
    async def generate_optimization_plan(self) -> dict:
        """生成優化計劃"""
        exploration_data = await self.explorer.explore_system()
        analysis_data = await self.analyzer.analyze_code_quality()
        rag_knowledge = await self.rag.retrieve_best_practices()
        
        return {
            "capability_gaps": [...],  # 缺失的能力
            "refactoring_targets": [...],  # 需要重構的代碼
            "optimization_priorities": [...]  # 優化優先級
        }
```

---

#### 5. **Visualization Layer (視覺化層)** (Low)
```
當前狀況: 完全缺失
影響: 無法展示優化方案給用戶審核
```

**可延後實現**: 初期可用簡單 JSON 報告替代

---

## 🎯 實施計劃與 CLI 整合方案

### Phase 1: 基礎建設 (Week 1-2) 🔴 **當前階段**

#### 1.1 實現 SystemSelfExplorer
```bash
# 創建文件
services/core/aiva_core/internal_exploration/system_self_explorer.py

# 核心功能
- [x] 讀取 latest_classification.json (286 flows)
- [ ] 解析 flow → 能力映射
- [ ] 檢查引擎健康狀態
- [ ] 生成能力清單
```

**與現有 CLI 整合**:
```python
# 現有 CLI (aiva_cli_implementation.py) 提供:
# - 執行能力: execute_flow(flow_id)
# - 列表能力: list_flows()

# SystemSelfExplorer 補充:
# - 探索能力: explore_system() -> 分析 286 flows
# - 映射能力: map_flows_to_capabilities()

# 整合方式:
class SystemSelfExplorer:
    def __init__(self):
        self.cli_executor = CLIImplementation()  # 復用現有 CLI
        
    async def get_available_attacks(self) -> dict:
        """解析攻擊相關 flows"""
        all_flows = self.cli_executor.load_flows()
        attack_flows = [f for f in all_flows if "attack" in f["path"]]
        
        return {
            "sqli": {"flow_ids": [11, 96], "status": "ready"},
            "xss": {"flow_ids": [97, 98], "status": "ready"},
        }
```

---

#### 1.2 實現 AIDecisionCore
```bash
# 創建文件
services/core/aiva_core/cognitive_core/ai_decision_core.py

# 核心功能
- [ ] 整合 SystemSelfExplorer
- [ ] 整合 RAGTrigger (已存在)
- [ ] 決策引擎選擇邏輯
- [ ] 生成攻擊策略
```

**與 MultiEngineCoordinator 整合**:
```python
# 修改 multi_engine_coordinator.py

class MultiEngineCoordinator:
    def __init__(self):
        self.ai_decision = AIDecisionCore()  # 新增
        
    async def execute_smart_strategy(self, target: str, constraints: dict):
        """智能策略 - AI 決策"""
        # 1. AI 決策
        plan = await self.ai_decision.decide_scan_strategy(target, constraints)
        
        # 2. 執行計劃 (復用現有引擎調用邏輯)
        if "rust_scanner" in plan["selected_engines"]:
            rust_result = await self._call_rust_engine(...)
            
        # 3. 返回結果
        return rust_result
```

---

### Phase 2: 外部閉環構建 (Week 3-4)

#### 2.1 實現 ExternalFeedbackLoop
```bash
# 創建文件
services/core/aiva_core/cognitive_core/learning_system/external_feedback_loop.py

# 數據庫設計
CREATE TABLE scan_history (
    id INTEGER PRIMARY KEY,
    target TEXT,
    tech_stack TEXT,
    vulnerabilities TEXT,
    success_rate REAL,
    timestamp DATETIME
);

# 核心功能
- [ ] 記錄掃描結果
- [ ] 查詢歷史數據
- [ ] 統計成功率
- [ ] 更新向量庫
```

**與現有系統整合**:
```python
# 在 MultiEngineCoordinator 添加回調
class MultiEngineCoordinator:
    def __init__(self):
        self.feedback_loop = ExternalFeedbackLoop()
        
    async def execute_strategy_balanced(self, ...):
        result = await self._execute_engines(...)
        
        # 記錄到外部閉環
        await self.feedback_loop.record_scan_result(result)
        
        return result
```

---

#### 2.2 完善 AI 決策 (整合歷史數據)
```python
class AIDecisionCore:
    async def decide_scan_strategy(self, target, constraints):
        # 新增: 查詢歷史數據
        history = await self.external_feedback.query_success_rates(target)
        
        # 決策時考慮歷史成功率
        if history["sqli_success_rate"] < 0.2:
            # 跳過 SQLi,嘗試其他攻擊
            pass
```

---

### Phase 3: 內部閉環構建 (Week 5-6)

#### 3.1 實現 InternalFeedbackLoop
```bash
# 創建文件
services/core/aiva_core/internal_exploration/internal_feedback_loop.py

# 核心功能
- [ ] 整合 SystemSelfExplorer
- [ ] 整合 AnalysisEngine (如有)
- [ ] 生成優化建議
```

---

### Phase 4: 雙閉環整合 (Week 7-8)

#### 4.1 創建 DualLoopOrchestrator
```python
# services/core/aiva_core/cognitive_core/dual_loop_orchestrator.py

class DualLoopOrchestrator:
    """雙閉環協調器"""
    
    def __init__(self):
        self.internal_loop = InternalFeedbackLoop()
        self.external_loop = ExternalFeedbackLoop()
        self.ai_decision = AIDecisionCore()
        
    async def run_optimization_cycle(self):
        """執行一次完整優化週期"""
        # 1. 內部閉環: 探索能力
        internal_insights = await self.internal_loop.generate_insights()
        
        # 2. 外部閉環: 實戰反饋
        external_insights = await self.external_loop.collect_insights()
        
        # 3. AI 決策: 生成優化方案
        optimization_plan = await self.ai_decision.generate_optimization(
            internal_insights, external_insights
        )
        
        # 4. 返回方案 (等待人工審核)
        return optimization_plan
```

---

## 🔗 CLI 既有任務整合方案

### 當前 CLI 系統架構
```
aiva_cli_implementation.py (286 flows)
    ├── --list              # 列出所有 flows
    ├── --flow <id>         # 執行指定 flow
    ├── --dry-run           # 預覽執行計劃
    └── --generate-doc      # 生成文檔
```

### 整合到雙閉環的方式

#### 方案 1: CLI 作為執行層 (推薦)
```
雙閉環架構:
    AI 決策層 (AIDecisionCore)
         ↓ 決定執行哪個 flow
    CLI 執行層 (aiva_cli_implementation.py)
         ↓ 執行 flow 11 (SQLi attack)
    反饋層 (ExternalFeedbackLoop)
         ↓ 記錄結果
```

**實現**:
```python
# 在 AIDecisionCore 中
class AIDecisionCore:
    def __init__(self):
        self.cli_executor = CLIImplementation()
        
    async def execute_attack(self, attack_type: str, target: str):
        # 1. 查找對應的 flow ID
        flow_mapping = {
            "sqli": 11,
            "xss": 97,
            "unified_attack": 8
        }
        flow_id = flow_mapping.get(attack_type)
        
        # 2. 使用 CLI 執行
        result = await self.cli_executor.execute_flow(
            flow_id=flow_id,
            params={"target": target}
        )
        
        return result
```

---

#### 方案 2: CLI 與多引擎並行 (擴展)
```
MultiEngineCoordinator
    ├── Rust Engine (Phase 0: 快速偵察)
    ├── Python Engine (Phase 1: 深度掃描)
    ├── Go Engine (Phase 2: 專項測試)
    └── CLI Flows (Phase 3: 複雜攻擊鏈)  ← 新增
         └── 調用 aiva_cli_implementation.py
```

**實現**:
```python
# 在 MultiEngineCoordinator 添加
class MultiEngineCoordinator:
    ENGINES = {
        # ... 現有引擎 ...
        "cli_flow_executor": EngineConfig(
            name="CLI Flow Executor",
            language="python",
            cli_command="python -m aiva_core.internal_exploration.aiva_cli_implementation",
            capabilities=["complex_attack_chain", "unified_attack"]
        )
    }
    
    async def _call_cli_flow(self, flow_id: int, params: dict):
        """調用 CLI flow"""
        cmd = [
            "python", "-m",
            "aiva_core.internal_exploration.aiva_cli_implementation",
            "--flow", str(flow_id),
            "--params", json.dumps(params)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return json.loads(result.stdout)
```

---

## 📋 實施檢查清單

### Week 1-2: 基礎建設
- [ ] 創建 `system_self_explorer.py`
  - [ ] 讀取 latest_classification.json
  - [ ] 解析 286 flows 能力映射
  - [ ] 實現 `get_available_engines()`
  - [ ] 實現 `get_available_attacks()`
  
- [ ] 創建 `ai_decision_core.py`
  - [ ] 整合 SystemSelfExplorer
  - [ ] 整合 RAGTrigger
  - [ ] 實現 `decide_scan_strategy()`
  
- [ ] 修改 `multi_engine_coordinator.py`
  - [ ] 添加 `ai_decision` 屬性
  - [ ] 實現 `execute_smart_strategy()`
  - [ ] 整合 AI 決策到現有策略

### Week 3-4: 外部閉環
- [ ] 創建 `external_feedback_loop.py`
  - [ ] 設計數據庫 schema
  - [ ] 實現 `record_scan_result()`
  - [ ] 實現 `query_success_rates()`
  
- [ ] 整合到 MultiEngineCoordinator
  - [ ] 添加結果記錄回調
  - [ ] 更新向量庫

### Week 5-6: 內部閉環
- [ ] 創建 `internal_feedback_loop.py`
  - [ ] 整合現有分析工具
  - [ ] 實現 `generate_optimization_plan()`

### Week 7-8: 雙閉環整合
- [ ] 創建 `dual_loop_orchestrator.py`
  - [ ] 整合內外閉環
  - [ ] 實現完整優化週期

---

## 🚨 當前最緊迫的問題

### 問題 1: AI 決策層缺失
**影響**: 無法實現智能策略選擇  
**優先級**: 🔴 Critical  
**預計時間**: 3-5 天  
**依賴**: SystemSelfExplorer

### 問題 2: SystemSelfExplorer 缺失
**影響**: AI 不知道自己有什麼能力  
**優先級**: 🔴 Critical  
**預計時間**: 2-3 天  
**依賴**: latest_classification.json (已有)

### 問題 3: ExternalFeedbackLoop 缺失
**影響**: 無法學習和優化  
**優先級**: 🟡 High  
**預計時間**: 4-6 天  
**依賴**: 數據庫設計

---

## 💡 快速啟動建議

### 最小可行產品 (MVP) - 2 週內
```
目標: 實現基本的 AI 決策 + CLI 執行

必須實現:
1. SystemSelfExplorer (簡化版)
   - 只需解析 latest_classification.json
   - 返回可用 flow 列表
   
2. AIDecisionCore (基礎版)
   - 基於規則的引擎選擇 (不依賴歷史數據)
   - 整合 RAGTrigger 搜索建議
   
3. 整合到 MultiEngineCoordinator
   - 實現 execute_smart_strategy()
   - 調用 CLI flows

可延後:
- 外部反饋循環 (先用簡單 JSON 記錄)
- 內部優化建議 (先手動分析)
- 視覺化界面 (用 JSON 報告)
```

### 測試場景
```bash
# 用戶輸入
python start_scan.py \
  --target "http://test.com" \
  --allowed-attacks "xss,sqli" \
  --strategy "smart"

# 系統執行流程:
1. AIDecisionCore.decide_scan_strategy()
   - 查詢 SystemSelfExplorer: 有哪些 XSS/SQLi flows?
   - RAG 搜索: test.com 類似案例
   - 決策: 使用 flow 11 (SQLi) + flow 97 (XSS)

2. MultiEngineCoordinator.execute_smart_strategy()
   - 調用 Rust Engine (快速偵察)
   - 調用 CLI flow 11 (SQLi 測試)
   - 調用 CLI flow 97 (XSS 測試)

3. 返回結果
   - 生成報告
   - (可選) 記錄到反饋數據庫
```

---

## 📝 總結

### 核心問題
1. **AI 決策層完全缺失** - 無法智能選擇策略
2. **SystemSelfExplorer 缺失** - AI 不知道自己的能力
3. **反饋循環缺失** - 無法學習和優化

### CLI 整合方案
- **現有 CLI (286 flows)** 作為執行層
- **AI 決策層** 決定調用哪個 flow
- **MultiEngineCoordinator** 統一協調 CLI 與其他引擎

### 建議優先級
1. 🔴 Week 1-2: SystemSelfExplorer + AIDecisionCore (MVP)
2. 🟡 Week 3-4: ExternalFeedbackLoop (學習能力)
3. 🟢 Week 5-8: InternalFeedbackLoop + 完整整合

### 可立即開始的任務
- [ ] 創建 `system_self_explorer.py` 骨架
- [ ] 解析 `latest_classification.json` 找出攻擊相關 flows
- [ ] 在 `multi_engine_coordinator.py` 添加 `execute_smart_strategy()` 方法
- [ ] 設計 AI 決策的輸入輸出格式

---

**下一步行動**: 
建議從 SystemSelfExplorer 開始實現,因為它是 AI 決策層的基礎依賴。
