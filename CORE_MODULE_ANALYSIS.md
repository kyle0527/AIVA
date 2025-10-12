# AIVA Core 模組完整分析與建議報告
**Core Module Analysis & Recommendations Report**

生成時間：2025-10-13  
分析範圍：Core 核心模組（services/core/aiva_core）  
維護狀態：四大模組架構 ✅

---

## 📋 **目錄**

1. [執行摘要](#執行摘要)
2. [架構分析](#架構分析)
3. [代碼質量評估](#代碼質量評估)
4. [問題與建議](#問題與建議)
5. [改進計劃](#改進計劃)
6. [實施優先級](#實施優先級)

---

## 🎯 **執行摘要**

### 總體評估

| 項目 | 評分 | 狀態 |
|------|------|------|
| **架構設計** | ⭐⭐⭐⭐⭐ 9/10 | ✅ 優秀 |
| **代碼質量** | ⭐⭐⭐⭐ 7/10 | ⚠️ 良好 |
| **數據合約** | ⭐⭐⭐ 6/10 | ⚠️ 需改進 |
| **錯誤處理** | ⭐⭐⭐⭐ 7/10 | ⚠️ 良好 |
| **測試覆蓋** | ⭐⭐ 3/10 | ❌ 缺失 |
| **文檔完整性** | ⭐⭐⭐⭐ 8/10 | ✅ 良好 |
| **性能優化** | ⭐⭐⭐⭐ 7/10 | ⚠️ 良好 |
| **可維護性** | ⭐⭐⭐⭐ 8/10 | ✅ 良好 |

### 關鍵發現

#### ✅ **優勢**
1. **清晰的架構分層**：五大子系統分工明確
2. **異步處理機制**：使用 asyncio 實現高效並發
3. **狀態管理完善**：SessionStateManager 設計良好
4. **日誌系統完整**：詳細的分階段日誌記錄

#### ⚠️ **需改進**
1. **缺少專用數據合約**：未創建 core 模組專用 schemas.py
2. **策略生成器被移除**：test_strategy_generation.py 被註釋
3. **測試覆蓋不足**：沒有單元測試和集成測試
4. **硬編碼配置**：缺少配置管理機制

#### ❌ **嚴重問題**
1. **任務生成邏輯簡化**：TaskGenerator 只生成基本任務
2. **攻擊面分析不完整**：IDOR 候選檢測缺失
3. **學習機制未啟用**：StrategyAdjuster 的學習功能未被調用

---

## 🏗️ **架構分析**

### 模組結構

```
services/core/aiva_core/
├── __init__.py                 ✅ 模組初始化
├── app.py                      ✅ 主應用入口（FastAPI）
│
├── ingestion/                  ✅ 1. 資料接收與預處理
│   ├── __init__.py
│   └── scan_module_interface.py   ✅ 掃描數據處理
│
├── analysis/                   ⚠️ 2. 分析與策略引擎
│   ├── __init__.py
│   ├── initial_surface.py         ✅ 攻擊面分析
│   ├── dynamic_strategy_adjustment.py  ✅ 動態策略調整
│   └── test_strategy_generation.py     ❌ 已被移除
│
├── execution/                  ✅ 3. 任務協調與執行
│   ├── __init__.py
│   ├── task_generator.py           ⚠️ 任務生成（簡化）
│   ├── task_queue_manager.py       ✅ 任務隊列管理
│   └── execution_status_monitor.py ✅ 執行監控
│
├── state/                      ✅ 4. 狀態與知識庫管理
│   ├── __init__.py
│   └── session_state_manager.py    ✅ 會話狀態管理
│
└── output/                     ✅ 5. 輸出與通訊
    ├── __init__.py
    └── to_functions.py             ✅ 功能模組消息封裝
```

### 子系統分析

#### 1. 資料接收與預處理（Ingestion）

**職責**：
- 接收掃描模組數據
- 數據標準化和清理
- 資產分類和風險評分

**現狀**：
```python
class ScanModuleInterface:
    async def process_scan_data(self, payload: ScanCompletedPayload) -> dict[str, Any]
    def _process_assets(self, assets: list[Any]) -> list[dict[str, Any]]
    def _process_fingerprints(self, fingerprints: Any) -> dict[str, Any]
    def _calculate_risk_score(self, asset: Any) -> int
    def _categorize_asset(self, asset: Any) -> list[str]
```

**評估**：✅ **良好**
- 清晰的數據處理流程
- 完善的資產分類邏輯
- 風險評分機制合理

**建議**：
1. 添加輸入驗證（使用 Pydantic）
2. 提取風險評分規則到配置文件
3. 添加數據清理和去重邏輯

---

#### 2. 分析與策略引擎（Analysis）

**職責**：
- 攻擊面分析
- 測試策略生成
- 動態策略調整

**現狀**：

##### InitialAttackSurface ✅

```python
class InitialAttackSurface:
    def analyze(self, payload: ScanCompletedPayload) -> dict[str, Any]
    def _summarize_asset(self, asset: Asset) -> dict[str, Any]
    def _detect_ssrf_candidates(self, asset: Asset) -> Iterable[dict[str, Any]]
```

**優點**：
- SSRF 候選檢測邏輯清晰
- 參數語義分析機制

**缺點**：
- ❌ 只檢測 SSRF，缺少 XSS/SQLi/IDOR 候選檢測
- ❌ 沒有風險優先級排序
- ❌ 返回格式未使用 Pydantic 驗證

##### StrategyAdjuster ✅

```python
class StrategyAdjuster:
    def adjust(self, plan: dict, context: dict) -> dict
    def learn_from_result(self, feedback_data: dict) -> None
    def _adjust_for_waf(self, plan: dict, context: dict) -> dict
    def _adjust_based_on_success_rate(self, plan: dict, context: dict) -> dict
```

**優點**：
- 完整的調整邏輯（WAF、成功率、技術棧）
- 學習機制設計良好

**缺點**：
- ❌ `learn_from_result()` 從未被調用
- ❌ 學習數據未持久化
- ⚠️ 調整規則硬編碼

##### TestStrategyGeneration ❌

**狀態**：**已被移除/註釋**

```python
# from services.core.aiva_core.analysis.test_strategy_generation import StrategyGenerator  # noqa: E501
```

**問題**：
- 核心策略生成功能缺失
- 目前使用空策略：`{"test_plans": [], "strategy_type": "default"}`
- 導致任務生成不完整

**建議**：⚠️ **高優先級修復**

---

#### 3. 任務協調與執行（Execution）

##### TaskGenerator ⚠️

```python
class TaskGenerator:
    def from_strategy(self, plan: dict, payload: ScanCompletedPayload) 
        -> Iterable[tuple[Topic, FunctionTaskPayload]]
```

**問題**：
1. **過度簡化**：只從 plan 中的 xss/sqli/ssrf 列表生成任務
2. **缺少智能**：沒有基於攻擊面分析的任務生成
3. **缺少 IDOR**：沒有 IDOR 任務生成邏輯
4. **缺少優先級**：priority 直接從 plan 獲取，沒有動態計算

**當前流程**：
```
plan (空的) -> TaskGenerator -> 0 個任務
```

**預期流程**：
```
AttackSurface -> StrategyGenerator -> Plan -> TaskGenerator -> N 個任務
```

##### TaskQueueManager ✅

```python
class TaskQueueManager:
    def enqueue_task(self, topic: Topic, task_payload: FunctionTaskPayload)
    def get_pending_tasks(self, scan_id: str) -> list[dict]
    def mark_task_running(self, task_id: str)
    def mark_task_completed(self, task_id: str, result: dict | None)
```

**評估**：✅ **優秀**
- 完整的任務狀態追蹤
- 優先級隊列管理
- 統計數據收集

##### ExecutionStatusMonitor ✅

```python
class ExecutionStatusMonitor:
    def record_worker_heartbeat(self, worker_id: str, status: str)
    def record_task_start(self, task_id: str, worker_id: str)
    def record_task_completion(self, task_id: str, success: bool, duration: float)
    def get_system_health(self) -> dict
    def check_sla_violations(self) -> list[dict]
```

**評估**：✅ **優秀**
- 完整的健康監控
- SLA 違規檢測
- 系統指標追蹤

---

#### 4. 狀態與知識庫管理（State）

##### SessionStateManager ✅

```python
class SessionStateManager:
    async def record_scan_result(self, payload: ScanCompletedPayload)
    async def record_task_update(self, payload: TaskUpdatePayload)
    def get_session_status(self, scan_id: str) -> dict[str, str]
    def get_session_context(self, scan_id: str) -> dict[str, Any]
    def update_context(self, scan_id: str, context_data: dict)
```

**評估**：✅ **優秀**
- 完整的會話管理
- 歷史記錄追蹤
- 上下文豐富化

**建議**：
1. 添加持久化機制（數據庫/Redis）
2. 添加會話過期和清理邏輯
3. 使用 Pydantic 模型替代 dict

---

#### 5. 輸出與通訊（Output）

##### to_functions.py ✅

```python
def to_function_message(
    topic: Topic,
    payload: FunctionTaskPayload,
    trace_id: str,
    correlation_id: str,
) -> AivaMessage
```

**評估**：✅ **完美**
- 簡潔明確
- 符合數據合約

---

## 📊 **代碼質量評估**

### 優點

1. **類型提示完整**：
   ```python
   async def process_scan_data(self, payload: ScanCompletedPayload) -> dict[str, Any]:
   ```

2. **異步處理**：
   ```python
   asyncio.create_task(process_scan_results())
   asyncio.create_task(process_function_results())
   asyncio.create_task(monitor_execution_status())
   ```

3. **清晰的日誌**：
   ```python
   logger.info(f"📥 [Stage 1/7] Data ingested - Assets: {len(payload.assets)}")
   logger.info(f"🔍 [Stage 2/7] Analyzing attack surface for {scan_id}")
   ```

4. **錯誤處理**：
   ```python
   except Exception as e:
       logger.error(f"❌ Error processing scan results: {e}")
   ```

### 缺點

1. **字典驅動**：大量使用 `dict[str, Any]` 而非 Pydantic 模型
   ```python
   # ❌ 不佳
   def analyze(self, payload: ScanCompletedPayload) -> dict[str, Any]:
   
   # ✅ 建議
   def analyze(self, payload: ScanCompletedPayload) -> AttackSurfaceAnalysis:
   ```

2. **硬編碼配置**：
   ```python
   # ❌ 硬編碼
   if runtime > 600:  # 10分鐘
   
   # ✅ 建議
   if runtime > self.config.sla_timeout_seconds:
   ```

3. **魔術數字**：
   ```python
   # ❌ 魔術數字
   if success_rate > 0.7:
       task["priority"] = min(task.get("priority", 5) + 1, 10)
   
   # ✅ 建議
   SUCCESS_RATE_THRESHOLD_HIGH = 0.7
   PRIORITY_BOOST = 1
   MAX_PRIORITY = 10
   ```

4. **未使用的導入**：
   ```python
   # ❌ 被註釋但未刪除
   # from services.core.aiva_core.analysis.test_strategy_generation import StrategyGenerator
   ```

---

## ⚠️ **問題與建議**

### 嚴重問題（P0 - 立即修復）

#### 問題 1：策略生成器缺失

**現狀**：
```python
# Legacy strategy generator removed - using direct strategy
base_strategy = {"test_plans": [], "strategy_type": "default"}
```

**影響**：
- ❌ 無法生成測試任務
- ❌ 攻擊面分析結果未被使用
- ❌ Core 模組核心功能缺失

**建議**：

**選項 A：修復現有策略生成器**
```python
# services/core/aiva_core/analysis/test_strategy_generation.py
class StrategyGenerator:
    def generate(
        self, attack_surface: AttackSurfaceAnalysis
    ) -> TestStrategy:
        """從攻擊面生成測試策略"""
        tasks = []
        
        # XSS 任務
        for asset in attack_surface.high_risk_assets:
            if asset.has_parameters:
                tasks.append({
                    "type": "xss",
                    "asset": asset.url,
                    "parameter": param,
                    "priority": 8,  # 高風險優先
                })
        
        # SQLi 任務
        # SSRF 任務
        # IDOR 任務
        
        return TestStrategy(tasks=tasks)
```

**選項 B：使用基於規則的簡化生成器**
```python
class RuleBasedStrategyGenerator:
    def from_attack_surface(
        self, surface: dict[str, Any], payload: ScanCompletedPayload
    ) -> dict[str, Any]:
        """基於規則生成測試策略"""
        plan = {"xss": [], "sqli": [], "ssrf": [], "idor": []}
        
        # 對每個資產生成任務
        for asset in payload.assets:
            if asset.parameters:
                for param in asset.parameters:
                    # 生成 XSS 任務
                    plan["xss"].append({
                        "asset": asset.value,
                        "parameter": param,
                        "priority": self._calculate_priority(asset),
                    })
                    # ... 其他類型
        
        return plan
```

---

#### 問題 2：缺少 Core 模組專用 schemas.py

**現狀**：
- ❌ 大量使用 `dict[str, Any]`
- ❌ 缺少類型安全
- ❌ 難以維護和擴展

**建議**：創建 `services/core/aiva_core/schemas.py`

```python
"""
Core 模組專用數據合約
"""
from pydantic import BaseModel, Field

class AttackSurfaceAnalysis(BaseModel):
    """攻擊面分析結果"""
    forms: int = 0
    parameters: int = 0
    waf_detected: bool = False
    high_risk_assets: list[AssetAnalysis] = Field(default_factory=list)
    medium_risk_assets: list[AssetAnalysis] = Field(default_factory=list)
    low_risk_assets: list[AssetAnalysis] = Field(default_factory=list)
    ssrf_candidates: list[SsrfCandidate] = Field(default_factory=list)
    xss_candidates: list[XssCandidate] = Field(default_factory=list)
    sqli_candidates: list[SqliCandidate] = Field(default_factory=list)

class AssetAnalysis(BaseModel):
    """資產分析結果"""
    asset_id: str
    url: str
    type: str
    risk_score: int = Field(ge=0, le=100)
    categories: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    has_form: bool = False

class SsrfCandidate(BaseModel):
    """SSRF 候選"""
    asset_url: str
    parameter: str
    location: str  # "query" | "body"
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

class TestStrategy(BaseModel):
    """測試策略"""
    strategy_type: str = "comprehensive"
    xss_tasks: list[TestTask] = Field(default_factory=list)
    sqli_tasks: list[TestTask] = Field(default_factory=list)
    ssrf_tasks: list[TestTask] = Field(default_factory=list)
    idor_tasks: list[TestTask] = Field(default_factory=list)
    total_tasks: int = 0
    estimated_duration_seconds: int = 0

class TestTask(BaseModel):
    """測試任務"""
    asset: str
    parameter: str | None = None
    priority: int = Field(ge=1, le=10)
    method: str = "GET"
    location: str = "query"
    confidence: float = Field(ge=0.0, le=1.0)

class StrategyAdjustment(BaseModel):
    """策略調整記錄"""
    adjustment_type: str  # "waf", "success_rate", "tech_stack", "findings"
    applied_rules: list[str] = Field(default_factory=list)
    priority_changes: dict[str, int] = Field(default_factory=dict)
    timing_adjustments: dict[str, float] = Field(default_factory=dict)
```

---

#### 問題 3：學習機制未啟用

**現狀**：
```python
def learn_from_result(self, feedback_data: dict[str, Any]) -> None:
    """從測試結果中學習，更新策略知識庫"""
    # 實現完整，但從未被調用！
```

**影響**：
- ❌ 無法根據測試結果優化策略
- ❌ 浪費了設計良好的學習機制
- ❌ 無法實現自適應測試

**建議**：在 `process_function_results()` 中啟用

```python
async def process_function_results() -> None:
    """處理功能模組回傳的結果"""
    broker = await get_broker()
    aiterator = broker.subscribe(Topic.FINDING_DETECTED)
    
    async for mqmsg in aiterator:
        msg = AivaMessage.model_validate_json(mqmsg.body)
        finding = FindingPayload(**msg.payload)
        
        # 🆕 啟用學習機制
        feedback_data = {
            "scan_id": finding.scan_id,
            "task_id": finding.task_id,
            "module": _get_module_from_finding(finding),
            "success": finding.status == "confirmed",
            "vulnerability_type": finding.vulnerability.name,
            "confidence": finding.vulnerability.confidence,
        }
        strategy_adjuster.learn_from_result(feedback_data)
        
        # 記錄到會話
        session_state_manager.update_context(
            finding.scan_id,
            {"findings_count": session_state_manager.get_session_context(
                finding.scan_id
            ).get("findings_count", 0) + 1}
        )
```

---

### 重要問題（P1 - 盡快修復）

#### 問題 4：TaskGenerator 功能不完整

**現狀**：
```python
def from_strategy(self, plan: dict, payload: ScanCompletedPayload):
    tasks = []
    # 只從 plan 的 xss/sqli/ssrf 列表生成任務
    for index, x in enumerate(plan.get("xss", [])):
        # ...
```

**問題**：
1. 依賴於 plan 中已有的任務列表
2. 無法自動從攻擊面生成任務
3. 缺少 IDOR 任務生成

**建議**：擴展為智能任務生成器

```python
class EnhancedTaskGenerator:
    def from_attack_surface(
        self, 
        surface: AttackSurfaceAnalysis,
        scan_id: str
    ) -> Iterable[tuple[Topic, FunctionTaskPayload]]:
        """從攻擊面直接生成任務"""
        tasks = []
        
        # XSS 任務
        for candidate in surface.xss_candidates:
            tasks.append((
                Topic.TASK_FUNCTION_XSS,
                FunctionTaskPayload(
                    task_id=new_id("task"),
                    scan_id=scan_id,
                    priority=self._calculate_priority(candidate),
                    target=FunctionTaskTarget(
                        url=candidate.asset_url,
                        parameter=candidate.parameter,
                        parameter_location=candidate.location,
                    ),
                ),
            ))
        
        # SQLi 任務
        for candidate in surface.sqli_candidates:
            # ...
        
        # SSRF 任務
        for candidate in surface.ssrf_candidates:
            # ...
        
        # IDOR 任務
        for asset in surface.high_risk_assets:
            if self._is_idor_candidate(asset):
                tasks.append((
                    Topic.FUNCTION_IDOR_TASK,
                    FunctionTaskPayload(
                        task_id=new_id("task"),
                        scan_id=scan_id,
                        priority=8,
                        target=FunctionTaskTarget(url=asset.url),
                    ),
                ))
        
        return tasks
    
    def _calculate_priority(self, candidate: Any) -> int:
        """動態計算優先級"""
        base_priority = 5
        if candidate.confidence > 0.8:
            base_priority += 3
        elif candidate.confidence > 0.6:
            base_priority += 2
        return min(base_priority, 10)
```

---

#### 問題 5：攻擊面分析不完整

**現狀**：
```python
class InitialAttackSurface:
    def _detect_ssrf_candidates(self, asset: Asset):
        # 只檢測 SSRF 候選
```

**缺失**：
- ❌ XSS 候選檢測
- ❌ SQLi 候選檢測  
- ❌ IDOR 候選檢測
- ❌ 風險優先級排序

**建議**：擴展分析能力

```python
class EnhancedAttackSurfaceAnalyzer:
    def analyze(self, payload: ScanCompletedPayload) -> AttackSurfaceAnalysis:
        """完整的攻擊面分析"""
        assets = []
        xss_candidates = []
        sqli_candidates = []
        ssrf_candidates = []
        idor_candidates = []
        
        for asset in payload.assets:
            # 資產分析
            analyzed_asset = self._analyze_asset(asset)
            assets.append(analyzed_asset)
            
            # 檢測各類候選
            xss_candidates.extend(self._detect_xss_candidates(asset))
            sqli_candidates.extend(self._detect_sqli_candidates(asset))
            ssrf_candidates.extend(self._detect_ssrf_candidates(asset))
            idor_candidates.extend(self._detect_idor_candidates(asset))
        
        # 風險分層
        high_risk = [a for a in assets if a.risk_score >= 70]
        medium_risk = [a for a in assets if 40 <= a.risk_score < 70]
        low_risk = [a for a in assets if a.risk_score < 40]
        
        return AttackSurfaceAnalysis(
            high_risk_assets=high_risk,
            medium_risk_assets=medium_risk,
            low_risk_assets=low_risk,
            xss_candidates=xss_candidates,
            sqli_candidates=sqli_candidates,
            ssrf_candidates=ssrf_candidates,
            idor_candidates=idor_candidates,
        )
    
    def _detect_xss_candidates(self, asset: Asset) -> list[XssCandidate]:
        """檢測 XSS 候選"""
        candidates = []
        if not asset.parameters:
            return candidates
        
        for param in asset.parameters:
            confidence = 0.5
            reasons = []
            
            # 反射型 XSS 檢測邏輯
            if any(hint in param.lower() for hint in ["search", "q", "query", "name"]):
                confidence += 0.2
                reasons.append("Parameter name suggests user input")
            
            if asset.has_form:
                confidence += 0.1
                reasons.append("Form input detected")
            
            candidates.append(XssCandidate(
                asset_url=asset.value,
                parameter=param,
                location="body" if asset.has_form else "query",
                confidence=min(confidence, 1.0),
                reasons=reasons,
            ))
        
        return candidates
```

---

### 次要問題（P2 - 計劃修復）

#### 問題 6：缺少配置管理

**建議**：創建配置類

```python
# services/core/aiva_core/config.py
from pydantic import BaseModel, Field

class CoreEngineConfig(BaseModel):
    """Core 引擎配置"""
    
    # SLA 配置
    task_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    worker_heartbeat_interval: int = Field(default=30, ge=10, le=300)
    
    # 策略調整配置
    waf_delay_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    success_rate_threshold_high: float = Field(default=0.7, ge=0.0, le=1.0)
    success_rate_threshold_low: float = Field(default=0.3, ge=0.0, le=1.0)
    priority_boost: int = Field(default=1, ge=1, le=5)
    
    # 風險評分配置
    high_risk_threshold: int = Field(default=70, ge=0, le=100)
    medium_risk_threshold: int = Field(default=40, ge=0, le=100)
    
    # 任務生成配置
    max_tasks_per_scan: int = Field(default=1000, ge=1, le=10000)
    default_task_priority: int = Field(default=5, ge=1, le=10)
```

---

#### 問題 7：缺少持久化機制

**建議**：集成數據庫

```python
# services/core/aiva_core/state/persistent_state_manager.py
from sqlalchemy.ext.asyncio import AsyncSession

class PersistentStateManager(SessionStateManager):
    def __init__(self, db_session: AsyncSession):
        super().__init__()
        self.db = db_session
    
    async def save_session(self, scan_id: str) -> None:
        """持久化會話狀態"""
        session_data = self._sessions.get(scan_id)
        # 保存到數據庫
    
    async def load_session(self, scan_id: str) -> dict:
        """從數據庫加載會話"""
        # 從數據庫查詢
```

---

## 📅 **改進計劃**

### Week 1-2：關鍵功能修復

**任務 1：創建 Core 模組 schemas.py**
- [ ] 定義 AttackSurfaceAnalysis, TestStrategy, TestTask 等模型
- [ ] 更新所有組件使用 Pydantic 模型
- [ ] 添加完整的 field_validator

**任務 2：修復策略生成器**
- [ ] 選擇實現方案（修復現有 vs. 新建簡化版）
- [ ] 實現從攻擊面到測試策略的轉換
- [ ] 集成到主流程

**任務 3：啟用學習機制**
- [ ] 在 process_function_results() 中調用 learn_from_result()
- [ ] 添加學習數據持久化
- [ ] 實現學習效果監控

### Week 2-3：功能擴展

**任務 4：擴展攻擊面分析**
- [ ] 實現 XSS 候選檢測
- [ ] 實現 SQLi 候選檢測
- [ ] 實現 IDOR 候選檢測
- [ ] 添加風險評分和排序

**任務 5：增強任務生成**
- [ ] 實現從攻擊面直接生成任務
- [ ] 添加智能優先級計算
- [ ] 支持 IDOR 任務生成

### Week 3-4：質量提升

**任務 6：配置管理**
- [ ] 創建 CoreEngineConfig
- [ ] 提取所有硬編碼值
- [ ] 支持環境變量和配置文件

**任務 7：測試覆蓋**
- [ ] 單元測試：所有分析器和生成器
- [ ] 集成測試：完整的處理流程
- [ ] 性能測試：高負載場景

---

## 🎯 **實施優先級**

### P0 - 立即執行（本週）

1. **創建 Core 模組 schemas.py**
   - 影響：高
   - 工作量：中
   - 依賴：無

2. **修復策略生成器**
   - 影響：極高
   - 工作量：高
   - 依賴：schemas.py

3. **啟用學習機制**
   - 影響：高
   - 工作量：低
   - 依賴：無

### P1 - 盡快執行（下週）

4. **擴展攻擊面分析**
   - 影響：高
   - 工作量：中
   - 依賴：schemas.py

5. **增強任務生成**
   - 影響：高
   - 工作量：中
   - 依賴：攻擊面分析、策略生成器

### P2 - 計劃執行（2-4週）

6. **配置管理**
   - 影響：中
   - 工作量：低
   - 依賴：無

7. **測試覆蓋**
   - 影響：中
   - 工作量：高
   - 依賴：所有功能穩定

8. **持久化機制**
   - 影響：中
   - 工作量：中
   - 依賴：schemas.py

---

## 📋 **總結**

### 核心模組現狀

**優勢**：
✅ 清晰的五層架構  
✅ 完善的狀態管理  
✅ 良好的監控機制  
✅ 異步處理設計

**亟需改進**：
❌ 策略生成器缺失  
❌ 缺少專用數據合約  
❌ 學習機制未啟用  
❌ 任務生成過於簡化

### 建議行動

1. **第一優先級**：修復策略生成器，恢復核心功能
2. **第二優先級**：創建 Pydantic 數據模型，提升類型安全
3. **第三優先級**：啟用學習機制，實現自適應優化
4. **長期計劃**：完善測試、配置、持久化

### 預期成果

完成改進後，Core 模組將具備：
- ✅ 完整的測試策略生成能力
- ✅ 類型安全的數據處理
- ✅ 自適應學習和優化
- ✅ 完整的漏洞檢測覆蓋
- ✅ 高質量的代碼和文檔

---

**分析完成 - AIVA Core Module v1.0**
