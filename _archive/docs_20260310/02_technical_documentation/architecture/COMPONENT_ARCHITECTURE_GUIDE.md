# 🏗️ 現有組件架構定位與處理指南

**日期**: 2026-01-10  
**原則**: AI 分析、規劃、下令 | 其他模組執行

---

## 📊 架構三層總覽

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 Layer 1: AI 指揮層 (Core)                                 │
│  ─────────────────────────────────────────────────────────  │
│  職責: 分析 + 規劃 + 下令                                      │
│  禁止: 網路請求、subprocess、直接執行                          │
│                                                              │
│  ✅ EnhancedDecisionAgent (2231 行)  - 純決策邏輯             │
│  ⚠️ AttackCoordinator (596 行)      - 需重構（規劃+執行混合）  │
└───────────────────────┬──────────────────────────────────────┘
                        │ 下達 AICommand
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  ⚡ Layer 2: 執行調度層 (Features + Scan Coordinators)        │
│  ─────────────────────────────────────────────────────────  │
│  職責: 接收命令 + 調度執行 + 返回結果                           │
│  允許: CLI 調用、網路請求、實際測試                             │
│                                                              │
│  ✅ AttackExecutor (608 行)          - 攻擊執行               │
│  ✅ HighValueFeatureManager (366 行) - 功能管理               │
│  ✅ SmartDetectionManager (273 行)   - 檢測管理               │
│  ⚠️ ExploitManager (968 行)         - 需移動到 Features       │
│  ❌ MultiEngineCoordinator           - 需新建（掃描調度）      │
└───────────────────────┬──────────────────────────────────────┘
                        │ 返回 ExecutionResult
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  🎯 Layer 3: 結果整合層 (Integration Coordinators)            │
│  ─────────────────────────────────────────────────────────  │
│  職責: 聚合結果 + 驗證 + 生成反饋                               │
│  輸出: AI 可用的學習數據                                       │
│                                                              │
│  ✅ BaseCoordinator (548 行)   - 雙閉環基類                   │
│  ✅ XSSCoordinator (439 行)    - XSS 特化協調器               │
└───────────────────────┬──────────────────────────────────────┘
                        │ 反饋 Feedback
                        ↓
                  回到 Layer 1 (AI 分析)
```

---

## 📁 組件逐一處理建議

### 🎯 Integration Coordinators (Layer 3) - ✅ 完全符合

#### `base_coordinator.py` (548 行)

**當前狀態**: ✅ 完美符合架構  
**處理建議**: **無需修改**

**角色定位**:
```python
class BaseCoordinator:
    """結果整合層 - 雙閉環架構"""
    
    async def collect_result(self, result_dict):
        """✅ 正確：收集 Features 執行結果"""
        # ✅ 只做數據處理，不執行攻擊
        optimization_data = self._extract_optimization_data(result)  # 內循環
        report_data = self._extract_report_data(result)              # 外循環
        feedback = self._generate_core_feedback(result)              # AI 反饋
        return feedback
```

**在架構中的位置**:
```
AttackExecutor (Layer 2) 執行攻擊
    ↓ 返回結果
BaseCoordinator (Layer 3) ✅
    ↓ 聚合驗證
AI Core (Layer 1) 收到反饋
```

**行動項**: 
- ✅ 無需修改
- ✅ 可選：添加更多特化協調器（SQLiCoordinator, SSRFCoordinator）

---

#### `xss_coordinator.py` (439 行)

**當前狀態**: ✅ 完全符合架構  
**處理建議**: **無需修改**

**角色定位**:
```python
class XSSCoordinator(BaseCoordinator):
    """✅ XSS 特化的結果處理器"""
    
    async def _analyze_payload_efficiency(self, result):
        """✅ 分析 XSS Payload 效率 → AI 學習"""
    
    async def _verify_findings(self, result):
        """✅ 驗證 XSS 漏洞真實性 → 誤報過濾"""
```

**行動項**: 
- ✅ 無需修改
- 💡 可參考創建其他特化協調器

---

### ⚡ Features Executors (Layer 2) - 大部分符合

#### `attack_executor.py` (608 行) ✅

**當前狀態**: ✅ 完美符合架構  
**處理建議**: **無需修改**

**角色定位**:
```python
class AttackExecutor:
    """執行層 - 接收 AI 計劃，執行攻擊"""
    
    async def execute_plan_with_ai_analysis(
        self, plan: AttackPlan, target, ai_analysis
    ):
        """✅ 完美流程：
        1. 接收 AI 規劃的攻擊計劃
        2. 根據 AI 風險評估調整執行模式
        3. 執行實際攻擊步驟（網路請求、Payload 測試）
        4. 生成反饋數據供 AI 學習
        """
        if ai_analysis.get("overall_risk_level") == "high":
            self.mode = ExecutionMode.SAFE  # ✅ 根據 AI 決策調整
        
        for step in plan.steps:
            result = await self._execute_step(step)  # ✅ 實際執行
        
        return self._generate_feedback_data(result)  # ✅ 反饋給 AI
```

**為什麼符合**:
- ✅ 接收 AI 決策（`plan`, `ai_analysis`）作為輸入
- ✅ 執行實際測試（允許網路 I/O）
- ✅ 不做策略決策（只按計劃執行）
- ✅ 返回結構化結果

**行動項**: 
- ✅ 無需修改
- 💡 可選：標準化 `AttackPlan` schema

---

#### `high_value_manager.py` (366 行) ✅

**當前狀態**: ✅ 符合架構  
**處理建議**: **保持現狀**

**角色定位**:
```python
class HighValueFeatureManager:
    """功能管理器 - 簡化 Bug Bounty 功能調用"""
    
    def run_mass_assignment_test(self, target, params):
        """✅ 執行測試，不做決策"""
        result = self._execute_feature("mass_assignment", params)  # ✅ 執行
        return result  # ✅ 返回結果
```

**行動項**: 
- ✅ 保持現狀
- 💡 可選：標準化參數接口，使用 `aiva_common.schemas`

---

#### `smart_detection_manager.py` (273 行) ✅

**當前狀態**: ✅ 完全符合架構  
**處理建議**: **無需修改**

**角色定位**:
```python
class SmartDetectionManager:
    """檢測器管理 - 註冊與執行"""
    
    def register(self, name, detector_func):
        """✅ 註冊檢測器"""
    
    def execute_all(self, params):
        """✅ 執行所有註冊的檢測器"""
        for name, func in self._detectors.items():
            result = func(params)  # ✅ 執行，不決策
```

**行動項**: ✅ 無需修改

---

#### `exploit_manager.py` (968 行) ⚠️

**當前狀態**: ⚠️ 位置錯誤（在 Core，應在 Features）  
**處理建議**: **移動到 Features 或重構**

**問題**:
```python
# 當前位置：services/core/aiva_core/.../exploit_manager.py  ❌ 錯誤
# 包含實際執行代碼（違反 Core 原則）
```

**檔案自己的註釋**:
```python
"""⚠️ **架構警告 - 需要重構** ⚠️
違反 AIVA 五大模組架構原則：
- Core 模組應該負責**決策和編排**，不應執行實際測試

建議方案：
1. 移動到 Features 模組 - services/features/function_exploit/
2. 或重構為 ExploitOrchestrator（純編排器）
"""
```

**行動項**:
```bash
# 選項 1: 移動到 Features（推薦）
mkdir -p services/features/function_exploit/managers/
mv services/core/.../exploit_manager.py \
   services/features/function_exploit/managers/exploit_manager.py

# 選項 2: 保留在 Core，但移除所有執行代碼，只做 Exploit 庫管理
# 實際執行交給 AttackExecutor
```

**重構後**:
```python
# services/features/function_exploit/managers/exploit_manager.py
class ExploitManager:
    """✅ 執行層 - Exploit 庫管理與執行"""
    
    def select_exploit(self, vuln_type) -> Exploit:
        """✅ 選擇 Exploit（基於 AI 規劃）"""
    
    async def execute_exploit(self, exploit, target):
        """✅ 執行 Exploit（實際測試）"""
```

---

### 🔧 Core Commander (Layer 1) - 需要重構

#### `attack_coordinator.py` (596 行) 🔴

**當前狀態**: ⚠️ 60% 符合（混合規劃與執行）  
**處理建議**: **重構為純規劃器**

**問題代碼**:
```python
async def detect_vulnerabilities(self, context):
    """❌ 問題：直接執行 Worker"""
    for vuln_type in vuln_types:
        module = __import__(module_path)          # ❌ Core 不應 import 執行模組
        worker = worker_class()                    # ❌ Core 不應實例化 Worker
        result = await worker.process_task(task)   # ❌ Core 不應執行任務
```

**重構方案**: 請參考 [AI_COORDINATOR_REFACTOR_PLAN.md](AI_COORDINATOR_REFACTOR_PLAN.md)

**重構後**:
```python
class AttackCoordinator:
    """✅ AI 規劃器 - 只規劃，不執行"""
    
    async def plan_vulnerability_detection(self, context) -> VulnDetectionPlan:
        """✅ AI 規劃：選擇檢測類型、優先級、策略"""
        selected_types = self._select_vuln_types_by_ai(target)  # ✅ AI 邏輯
        priorities = self._calculate_priorities(selected_types)  # ✅ AI 決策
        
        return VulnDetectionPlan(  # ✅ 返回計劃，不執行
            target=target,
            vuln_types=selected_types,
            priorities=priorities
        )
    
    # ⚠️ 保留舊方法向後兼容（內部調用新方法）
    async def detect_vulnerabilities(self, context):
        """已棄用，內部調用新架構"""
        plan = await self.plan_vulnerability_detection(context)
        executor = VulnDetectionExecutor()  # 執行層
        return await executor.execute_plan(plan)
```

**行動項**:
1. 🔴 高優先級：新增 `plan_vulnerability_detection()`
2. 🔴 高優先級：創建 `VulnDetectionExecutor`（Integration 層）
3. 🟡 中優先級：修改舊方法內部調用新架構
4. 🟢 低優先級：完全移除舊方法（長期目標）

**詳細重構計劃**: 
→ [AI_COORDINATOR_REFACTOR_PLAN.md](AI_COORDINATOR_REFACTOR_PLAN.md)

---

### 🔄 Scan Engines (CLI Layer) - 需要協調器

#### `scan/` 目錄結構

**當前狀態**: ❌ 缺少 Python 協調器  
**處理建議**: **創建 MultiEngineCoordinator**

**問題**:
```
scan/
├── rust_engine/      # ✅ CLI 可執行檔
├── go_engine/        # ✅ CLI 可執行檔
├── typescript_engine/# ✅ CLI 可執行檔
├── python_engine/    # ✅ Python 模組
└── coordinators/     # ❌ 只有 __init__.py，缺 multi_engine_coordinator.py
```

**需要**:
```python
# scan/coordinators/multi_engine_coordinator.py（需新建）
class MultiEngineCoordinator:
    """執行層 - 掃描引擎調度器"""
    
    async def execute_scan_command(self, command: AICommand) -> ExecutionResult:
        """執行 AI 下達的掃描命令
        
        1. 接收 AI 命令（不做決策）
        2. 選擇引擎（基於命令參數）
        3. 調用 CLI 並解析輸出
        4. 返回標準化結果
        """
        engines = self._select_engines(command.parameters)  # ✅ 基於參數，非決策
        
        results = await asyncio.gather(*[
            self._call_rust_cli(command.target) if e == "rust" else
            self._call_go_cli(command.target) if e == "go" else
            self._call_ts_cli(command.target)
            for e in engines
        ])
        
        return ExecutionResult(
            command_id=command.id,
            findings=self._merge_findings(results),
            raw_outputs=results
        )
    
    async def _call_rust_cli(self, target) -> dict:
        """✅ 調用 Rust CLI"""
        proc = await asyncio.create_subprocess_exec(
            "./rust_engine/target/release/aiva-info-gatherer.exe",
            "--target", target,
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return self._parse_rust_output(stdout.decode())
```

**行動項**:
1. 🔴 高優先級：創建 `multi_engine_coordinator.py`
2. 🔴 高優先級：實現 CLI 調用邏輯
3. 🟡 中優先級：標準化輸出解析
4. 🟡 中優先級：整合到 `AttackCoordinator`

---

## 🎯 總結：每個組件的處理方式

| 組件 | 當前狀態 | 處理方式 | 優先級 |
|------|----------|----------|--------|
| **Layer 3: Integration** |
| `base_coordinator.py` | ✅ 完美 | 保持不變 | - |
| `xss_coordinator.py` | ✅ 完美 | 保持不變 | - |
| **Layer 2: Executors** |
| `attack_executor.py` | ✅ 完美 | 保持不變 | - |
| `high_value_manager.py` | ✅ 良好 | 保持，可選優化 | 🟢 低 |
| `smart_detection_manager.py` | ✅ 完美 | 保持不變 | - |
| `exploit_manager.py` | ⚠️ 位置錯誤 | 移動到 Features | 🟡 中 |
| `multi_engine_coordinator.py` | ❌ 缺失 | 新建 | 🔴 高 |
| **Layer 1: Core** |
| `attack_coordinator.py` | ⚠️ 混合 | 重構為純規劃器 | 🔴 高 |
| `enhanced_decision_agent.py` | ✅ 完美 | 保持不變 | - |

---

## 📋 實施順序

### Phase 1: 立即執行（本週）

1. **創建 MultiEngineCoordinator** 🔴
   - 新建 `scan/coordinators/multi_engine_coordinator.py`
   - 實現 CLI 調用邏輯
   - 整合到 `AttackCoordinator`

2. **重構 AttackCoordinator** 🔴
   - 新增 `plan_vulnerability_detection()`
   - 創建 `VulnDetectionExecutor`（Integration 層）
   - 修改舊方法保持向後兼容

### Phase 2: 優化清理（下週）

3. **移動 ExploitManager** 🟡
   ```bash
   mv services/core/.../exploit_manager.py \
      services/features/function_exploit/managers/
   ```

4. **標準化接口** 🟡
   - 創建 `aiva_common/schemas/ai_commands.py`
   - 定義 `AICommand`, `ExecutionResult`
   - 更新所有 Coordinator 使用標準接口

### Phase 3: 完整遷移（長期）

5. **完全分離 Core** 🟢
   - 移除所有執行代碼
   - Core 只保留規劃方法
   - 所有執行交給 Layer 2

---

## ✅ 驗收標準

### 架構合規檢查

```python
# ✅ Layer 1 (Core) 不應出現
assert "requests." not in core_source_code
assert "subprocess.run" not in core_source_code
assert "worker.process_task" not in core_source_code

# ✅ Layer 1 應返回計劃對象
plan = await coordinator.plan_xxx(context)
assert isinstance(plan, (VulnDetectionPlan, AttackPlan, ScanPlan))

# ✅ Layer 2 應接收計劃作為輸入
result = await executor.execute_plan(plan)
assert isinstance(result, ExecutionResult)

# ✅ Layer 3 應生成 AI 反饋
feedback = await coordinator.collect_result(result)
assert "optimization_data" in feedback
assert "learning_data" in feedback
```

---

## 🔗 相關文檔

- [AI_COMMAND_EXECUTION_ARCHITECTURE_ANALYSIS.md](AI_COMMAND_EXECUTION_ARCHITECTURE_ANALYSIS.md) - 完整架構分析
- [AI_COORDINATOR_REFACTOR_PLAN.md](AI_COORDINATOR_REFACTOR_PLAN.md) - AttackCoordinator 重構計劃
- [features/README.md](features/README.md) - Features 模組架構
- [integration/README.md](integration/README.md) - Integration 模組架構

---

## 💡 關鍵原則

> **AI 只應知道「做什麼」(WHAT)，不應決定「怎麼做」(HOW)**

```python
# ✅ 正確：AI 規劃 → 執行層執行
plan = ai.plan_attack(context)      # AI 決定「做什麼」
result = executor.execute(plan)     # 執行層決定「怎麼做」

# ❌ 錯誤：AI 直接執行
result = ai.detect_and_execute(context)  # AI 自己執行
```

**符合度**: 當前 80% → 目標 100%
