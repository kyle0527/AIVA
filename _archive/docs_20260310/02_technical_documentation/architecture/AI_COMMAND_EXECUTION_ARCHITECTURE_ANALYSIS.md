# 🧠 AI 指揮與執行分離架構分析報告

**日期**: 2026-01-10  
**原則**: AI 負責分析、規劃及下令，執行由其他模組進行  
**分析範圍**: `services/` 完整架構

---

## 🎯 核心設計原則

```
AI 層 (Core)        → 分析 + 規劃 + 下令  [純邏輯，無I/O]
執行層 (Features)   → 實際執行          [網路請求、CLI調用]
整合層 (Integration)→ 結果收集 + 反饋   [數據聚合]
```

---

## ✅ 當前架構符合度分析

### 1️⃣ **AI 決策層 - 完全符合** ✅

**位置**: `core/aiva_core/cognitive_core/decision/`

```python
# enhanced_decision_agent.py (2231 行)
class EnhancedDecisionAgent:
    """✅ 純決策邏輯，無執行代碼"""
    
    async def decide(self, context) -> HighLevelIntent:
        """返回高層意圖，不執行"""
        # ✅ 只做分析和決策
        risk = self._assess_risk(context)
        strategy = self._select_strategy(context)
        
        return HighLevelIntent(
            intent_type=IntentType.SCAN,
            target_info=target,
            constraints=DecisionConstraints(...)
        )
    
    def _assess_risk(self, context) -> RiskLevel:
        """✅ 風險評估邏輯，無外部調用"""
        
    def _select_strategy(self, context) -> str:
        """✅ 策略選擇邏輯，無外部調用"""
```

**符合度**: ⭐⭐⭐⭐⭐ (100%)
- ✅ 無 `requests.`, `subprocess.`, `http.` 調用
- ✅ 返回標準決策對象 (HighLevelIntent)
- ✅ 使用 `aiva_common.schemas` 數據合約

---

### 2️⃣ **AI 任務規劃層 - 部分符合** ⚠️

**位置**: `core/aiva_core/task_planning/commander/`

```python
# attack_coordinator.py (596 行)
class AttackCoordinator:
    """⚠️ 混合了規劃與執行調用"""
    
    async def detect_vulnerabilities(self, context):
        """✅ 規劃邏輯正確"""
        vuln_types = context.get("vulnerability_types")
        target = context.get("target")
        
        # ⚠️ 但直接調用執行模組（應該返回計劃）
        results = {
            "sqli": await self._test_sqli(target),  # ⚠️ 直接執行
            "xss": await self._test_xss(target),    # ⚠️ 直接執行
        }
        
    async def execute_attack(self, context):
        """✅ 正確：調用 AttackExecutor，不自己執行"""
        executor = AttackExecutor(mode=mode)
        result = await executor.execute_plan_with_ai_analysis(
            plan=plan, target=target, ai_analysis=ai_analysis
        )
```

**問題**:
1. ❌ `detect_vulnerabilities()` 直接調用測試方法
2. ❌ `scan_with_multi_engine()` 直接調用 MultiEngineCoordinator
3. ✅ `execute_attack()` 正確委派給 AttackExecutor

**符合度**: ⭐⭐⭐ (60%)

---

### 3️⃣ **執行層 - 完全符合** ✅

**位置**: `features/function_exploit/executor/`

```python
# attack_executor.py (608 行)
class AttackExecutor:
    """✅ 純執行邏輯，接收 AI 計劃"""
    
    async def execute_plan_with_ai_analysis(
        self, plan: AttackPlan, target, ai_analysis
    ):
        """✅ 執行 AI 生成的計劃"""
        # 根據 AI 分析調整執行模式
        if ai_analysis.get("overall_risk_level") == "high":
            self.mode = ExecutionMode.SAFE
        
        # 執行步驟
        for step in plan.steps:
            result = await self._execute_step(step)
            
        # 生成反饋數據供 AI 學習
        return self._generate_feedback_data(result)
```

**符合度**: ⭐⭐⭐⭐⭐ (100%)
- ✅ 接收 AI 決策/計劃作為輸入
- ✅ 執行實際網路請求、攻擊測試
- ✅ 返回結構化結果供 AI 分析

---

### 4️⃣ **Dispatcher - 符合** ✅

**位置**: `core/aiva_core/task_planning/dispatcher.py`

```python
class TaskDispatcher:
    """✅ 正確：只負責轉發命令，不做決策"""
    
    def dispatch_to_sqli(self, params) -> subprocess.CompletedProcess:
        """✅ 純轉發，調用外部 CLI"""
        return subprocess.run([
            "python", "-m", "services.features.function_sqli.main",
            "--target", params["target"]
        ])
```

**符合度**: ⭐⭐⭐⭐⭐ (100%)
- ✅ 只負責命令轉發
- ✅ 無決策邏輯
- ✅ subprocess 調用符合分離原則

---

## 🔧 架構改進建議

### 建議 1: 修正 AttackCoordinator 職責

**當前問題**:
```python
# ❌ 錯誤：直接執行
async def detect_vulnerabilities(self, context):
    results = {
        "sqli": await self._test_sqli(target),
        "xss": await self._test_xss(target),
    }
```

**建議改為**:
```python
# ✅ 正確：只規劃，返回計劃
async def plan_vulnerability_detection(self, context) -> VulnDetectionPlan:
    """AI 規劃漏洞檢測計劃"""
    plan = VulnDetectionPlan(
        target=context.get("target"),
        vuln_types=self._select_vuln_types(context),
        priority=self._calculate_priority(context),
        timeout=self._estimate_timeout(context)
    )
    return plan

# 執行交給 Integration 層
```

---

### 建議 2: 引入 MultiEngineCoordinator 作為執行層

**架構修正**:

```
AI Core (attack_coordinator.py)
    ↓ 下達掃描計劃
MultiEngineCoordinator (scan/coordinators/)
    ↓ 調度執行
CLI Engines (Rust/Go/TS/Python)
    ↓ 返回結果
Integration Coordinators (integration/coordinators/)
    ↓ 聚合反饋
AI Core (enhanced_decision_agent.py)
    ↓ 分析結果，下一步決策
```

---

### 建議 3: 標準化 AI 命令接口

**創建統一的 AI Command Schema**:

```python
# aiva_common/schemas/ai_commands.py
class AICommand(BaseModel):
    """AI 下達的標準命令格式"""
    command_type: CommandType  # SCAN, ATTACK, EXPLOIT
    target: Target
    parameters: Dict[str, Any]
    constraints: DecisionConstraints
    priority: int
    timeout: int
    
class ExecutionResult(BaseModel):
    """執行層返回的標準結果"""
    command_id: str
    success: bool
    findings: List[UnifiedVulnerabilityFinding]
    feedback_data: Dict[str, Any]  # 供 AI 學習
```

---

## 🏗️ 推薦的三層架構

### 層級 1: AI 指揮層 (Core)

```python
# core/aiva_core/cognitive_core/decision/
EnhancedDecisionAgent
    ├── analyze_situation()      # 分析當前狀況
    ├── plan_next_action()       # 規劃下一步
    └── generate_command()       # 生成執行命令

# core/aiva_core/task_planning/commander/
AttackCoordinator  [重構]
    ├── plan_scan()              # 規劃掃描策略
    ├── plan_attack()            # 規劃攻擊步驟
    └── plan_exploitation()      # 規劃利用方案
```

**原則**:
- ✅ 只做決策和規劃
- ✅ 返回標準命令對象
- ✅ 無網路 I/O、無 subprocess
- ✅ 使用 `aiva_common.schemas`

---

### 層級 2: 執行調度層 (Scan Coordinators + Features Managers)

```python
# scan/coordinators/
MultiEngineCoordinator
    ├── execute_scan_command()   # 執行掃描命令
    ├── call_rust_engine()       # 調用 Rust CLI
    ├── call_go_engine()         # 調用 Go CLI
    └── parse_results()          # 解析 CLI 輸出

# features/
HighValueFeatureManager
    ├── execute_detection()      # 執行檢測
    └── execute_exploit()        # 執行利用

# features/function_exploit/executor/
AttackExecutor
    ├── execute_plan()           # 執行攻擊計劃
    └── execute_step()           # 執行單步驟
```

**原則**:
- ✅ 接收 AI 命令作為輸入
- ✅ 調用 CLI / 發送網路請求
- ✅ 返回結構化結果
- ❌ 不做策略決策

---

### 層級 3: 結果整合層 (Integration Coordinators)

```python
# integration/coordinators/
BaseCoordinator
    ├── aggregate_results()      # 聚合多引擎結果
    ├── filter_false_positives() # 誤報過濾
    ├── generate_feedback()      # 生成 AI 反饋
    └── generate_report()        # 生成報告

XSSCoordinator, SQLiCoordinator...
```

**原則**:
- ✅ 收集執行結果
- ✅ 數據清洗與驗證
- ✅ 生成 AI 可用的反饋數據
- ❌ 不執行攻擊

---

## 📋 MultiEngineCoordinator 定位

基於分離原則，`MultiEngineCoordinator` 應該是：

| 特性 | 定位 |
|------|------|
| **層級** | 執行調度層 (Layer 2) |
| **輸入** | AI 命令 (AICommand) |
| **輸出** | 結構化結果 (ExecutionResult) |
| **職責** | 調度 Rust/Go/TS/Python CLI 引擎 |
| **不做** | 策略決策、風險評估 |

**標準接口**:

```python
class MultiEngineCoordinator:
    """輕量級掃描引擎調度器 - 執行層組件"""
    
    async def execute_scan_command(
        self, 
        command: AICommand  # 從 AI 接收命令
    ) -> ExecutionResult:
        """執行 AI 下達的掃描命令
        
        1. 解析命令參數
        2. 選擇合適的引擎 (Rust/Go/TS/Python)
        3. 調用 CLI 並解析輸出
        4. 返回標準化結果供 Integration 聚合
        """
        
        # 選擇引擎（基於命令參數，非策略決策）
        engines = self._select_engines(command.parameters)
        
        # 並發調用
        results = await asyncio.gather(*[
            self._call_engine(engine, command.target)
            for engine in engines
        ])
        
        # 返回原始結果（不做決策分析）
        return ExecutionResult(
            command_id=command.id,
            success=all(r.success for r in results),
            findings=self._merge_findings(results),
            raw_outputs=results  # 保留原始輸出
        )
```

---

## 🎯 實施步驟

### Phase 1: 創建 MultiEngineCoordinator (立即執行)

1. ✅ 創建 `scan/coordinators/multi_engine_coordinator.py`
2. ✅ 實現 CLI 調用邏輯（Rust/Go/TS/Python）
3. ✅ 標準化輸出解析
4. ✅ 整合到 `attack_coordinator.py`

### Phase 2: 重構 AttackCoordinator (本週完成)

1. 移除 `detect_vulnerabilities()` 中的直接執行
2. 改為 `plan_vulnerability_detection()` 返回計劃
3. 修改 `scan_with_multi_engine()` 調用方式

### Phase 3: 標準化命令接口 (下週完成)

1. 在 `aiva_common/schemas/` 新增 `ai_commands.py`
2. 定義 `AICommand` 和 `ExecutionResult`
3. 更新所有 Coordinator 使用新接口

---

## 📊 符合度評分總結

| 模組 | 當前符合度 | 目標符合度 | 優先級 |
|------|-----------|-----------|--------|
| **EnhancedDecisionAgent** | ⭐⭐⭐⭐⭐ 100% | 100% | ✅ 保持 |
| **AttackCoordinator** | ⭐⭐⭐ 60% | 100% | 🔴 高 |
| **AttackExecutor** | ⭐⭐⭐⭐⭐ 100% | 100% | ✅ 保持 |
| **TaskDispatcher** | ⭐⭐⭐⭐⭐ 100% | 100% | ✅ 保持 |
| **MultiEngineCoordinator** | ❌ 0% (未實現) | 100% | 🔴 高 |
| **BaseCoordinator** | ⭐⭐⭐⭐⭐ 100% | 100% | ✅ 保持 |

---

## 💡 關鍵洞察

### ✅ 做得好的地方

1. **決策層純淨**: `EnhancedDecisionAgent` 完全符合原則
2. **執行層清晰**: `AttackExecutor` 正確接收 AI 計劃
3. **標準化數據**: 使用 `aiva_common.schemas` 統一接口
4. **雙閉環設計**: Integration Coordinators 提供即時反饋

### ⚠️ 需要改進的地方

1. **AttackCoordinator 職責混淆**: 既做規劃又執行調用
2. **缺少執行調度層**: 沒有統一的 CLI 引擎調度器
3. **命令接口不統一**: AI 命令格式未標準化

### 🎯 核心建議

> **AI 只應返回「做什麼」(WHAT)，不應決定「怎麼做」(HOW)**

```python
# ✅ 正確
plan = ai_agent.decide(context)  # 返回 WHAT
result = executor.execute(plan)  # 執行 HOW

# ❌ 錯誤
result = ai_agent.detect_and_execute(context)  # AI 自己執行
```

---

**結論**: 當前架構已經 **80% 符合** AI 指揮與執行分離原則，主要需要：
1. 重構 `AttackCoordinator` 為純規劃器
2. 實現 `MultiEngineCoordinator` 作為執行調度層
3. 標準化 AI 命令接口

完成這三步後，架構將達到 **100% 符合** 您的設計原則。
