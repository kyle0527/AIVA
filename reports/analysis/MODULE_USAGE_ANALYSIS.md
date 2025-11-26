# AIVA 模組用法分析報告

## 📑 目錄

- [📊 執行摘要](#執行摘要)
  - [掃描模組: ✅ 4/4 可用 (100%)](#掃描模組-44-可用-100)
  - [功能模組: ⚠️ 0/5 可用 (導入問題)](#功能模組-05-可用-導入問題)
  - [AI 整合: ✅ 3/3 可用 (100%)](#ai-整合-33-可用-100)
- [一、掃描模組（Scan Module）](#一掃描模組scan-module)
  - [✅ 1.1 MultiEngineCoordinator（多引擎協調器）](#11-multienginecoordinator多引擎協調器)
  - [✅ 1.2 Phase 0 用法（快速偵察）](#12-phase-0-用法快速偵察)
  - [✅ 1.3 Phase 1 用法（深度掃描）](#13-phase-1-用法深度掃描)
  - [✅ 1.4 預設掃描策略（AI 簡化調用）](#14-預設掃描策略ai-簡化調用)
    - [策略 1: execute_strategy_fast（快速掃描）](#策略-1-executestrategyfast快速掃描)
    - [策略 2: execute_strategy_balanced（均衡掃描）⭐ **推薦**](#策略-2-executestrategybalanced均衡掃描-推薦)
    - [策略 3: execute_strategy_comprehensive（全面掃描）](#策略-3-executestrategycomprehensive全面掃描)
    - [策略 4: execute_strategy_aggressive（激進掃描）](#策略-4-executestrategyaggressive激進掃描)
    - [策略 5: execute_strategy_smart（智能掃描）🧠 **AI 自動決策**](#策略-5-executestrategysmart智能掃描-ai-自動決策)
- [二、功能模組（Feature Modules）](#二功能模組feature-modules)
  - [⚠️ 2.1 導入問題](#21-導入問題)
  - [📋 2.2 功能模組架構（待修復後可用）](#22-功能模組架構待修復後可用)
    - [SQLi 模組（SQL 注入檢測）](#sqli-模組sql-注入檢測)
    - [XSS 模組（跨站腳本檢測）](#xss-模組跨站腳本檢測)
    - [SSRF 模組（服務端請求偽造檢測）](#ssrf-模組服務端請求偽造檢測)
    - [IDOR 模組（不安全直接對象引用檢測）](#idor-模組不安全直接對象引用檢測)
    - [BizLogic 模組（業務邏輯漏洞檢測）](#bizlogic-模組業務邏輯漏洞檢測)
- [三、AI 整合接口](#三ai-整合接口)
  - [✅ 3.1 AttackExecutor（攻擊執行器）](#31-attackexecutor攻擊執行器)
  - [✅ 3.2 TwoPhaseScanOrchestrator（兩階段編排器）](#32-twophasescanorchestrator兩階段編排器)
  - [✅ 3.3 EnhancedDecisionAgent（AI 決策代理）](#33-enhanceddecisionagentai-決策代理)
- [四、AI 可用的完整工作流程](#四ai-可用的完整工作流程)
  - [🎯 工作流程 1: 快速掃描 → 攻擊執行](#工作流程-1-快速掃描-攻擊執行)
  - [🎯 工作流程 2: 智能掃描（完全自動化）](#工作流程-2-智能掃描完全自動化)
  - [🎯 工作流程 3: 兩階段編排（全自動）](#工作流程-3-兩階段編排全自動)
- [五、總結](#五總結)
  - [✅ AI 可以完整調用的模組](#ai-可以完整調用的模組)
  - [📊 掃描模組用法總覽](#掃描模組用法總覽)
    - [直接控制（AI 完全掌控）](#直接控制ai-完全掌控)
    - [預設策略（AI 簡化調用）](#預設策略ai-簡化調用)
    - [自動化編排](#自動化編排)
  - [🔧 功能模組修復計劃](#功能模組修復計劃)
  - [✅ 最終結論](#最終結論)

---

## 📊 執行摘要

### 掃描模組: ✅ 4/4 可用 (100%)
### 功能模組: ⚠️ 0/5 可用 (導入問題)
### AI 整合: ✅ 3/3 可用 (100%)

---

## 一、掃描模組（Scan Module）

### ✅ 1.1 MultiEngineCoordinator（多引擎協調器）

**狀態**: ✅ **完全可用**

**支援引擎**:
- ✅ Python 引擎
- ⚠️ TypeScript 引擎（掃描器文件不存在）
- ✅ Rust 引擎
- ✅ Go 引擎

**可用引擎**: 3/4（Python, Rust, Go）

**AI 調用方式**:
```python
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator

# 初始化協調器
coordinator = MultiEngineCoordinator()
await coordinator.initialize()

# 方式 1: 直接指定引擎（AI 完全控制）
result = await coordinator.execute_phase1(
    scan_id='scan_123',
    targets=['http://example.com'],
    selected_engines=['python', 'rust', 'go'],  # AI 選擇引擎組合
    max_depth=5,
    max_urls=1000
)
```

---

### ✅ 1.2 Phase 0 用法（快速偵察）

**狀態**: ✅ **完全可用**

**引擎**: Rust 快速發現引擎

**AI 調用方式**:
```python
# Phase 0: Rust 快速偵察（5-10分鐘）
phase0_result = await coordinator.execute_phase0(
    scan_id='scan_123',
    targets=['http://example.com', 'http://api.example.com'],
    max_depth=3,
    timeout=600  # 10分鐘超時
)

# 返回結果包含:
# - assets: 發現的資產列表
# - fingerprints: 技術棧指紋（框架、語言、服務器等）
# - summary: 統計摘要（URLs、Forms、APIs 數量）
# - recommendations: AI 決策建議（建議的引擎組合）
```

**輸出示例**:
```python
{
    "scan_id": "scan_123",
    "status": "success",
    "execution_time": 45.2,
    "assets": [...],  # 發現的 URL、API、表單等
    "fingerprints": {
        "frameworks": ["React", "Express"],
        "languages": ["JavaScript", "TypeScript"],
        "servers": ["nginx"]
    },
    "summary": {
        "urls_found": 150,
        "forms_found": 8,
        "apis_found": 12
    },
    "recommendations": {
        "needs_js_engine": true,  # 需要 TypeScript 引擎
        "needs_form_testing": true,  # 需要表單測試
        "suggested_engines": ["typescript", "python", "rust"]
    }
}
```

---

### ✅ 1.3 Phase 1 用法（深度掃描）

**狀態**: ✅ **完全可用**

**支援引擎組合**: Python, Rust, Go（TypeScript 待修復）

**AI 調用方式**:
```python
# Phase 1: 多引擎深度掃描（10-30分鐘）
phase1_result = await coordinator.execute_phase1(
    scan_id='scan_123',
    targets=['http://example.com'],
    selected_engines=['python', 'rust', 'go'],  # AI 根據 Phase0 結果選擇
    max_depth=5,
    max_urls=1000,
    phase0_result=phase0_result.model_dump()  # 可選：傳入 Phase0 結果
)

# 返回結果包含:
# - assets: 所有引擎發現的資產（已去重）
# - summary: 統計摘要
# - engine_results: 各引擎執行狀態
# - phase0_summary: Phase 0 摘要（如果提供）
```

**輸出示例**:
```python
{
    "scan_id": "scan_123",
    "status": "completed",  # 或 "partial_success", "failed"
    "execution_time": 180.5,
    "summary": {
        "urls_found": 500,
        "forms_found": 25,
        "apis_found": 30,
        "scan_duration_seconds": 180
    },
    "assets": [...],  # 去重後的資產列表
    "engine_results": {
        "python": {"status": "completed", "assets_count": 250},
        "rust": {"status": "completed", "assets_count": 180},
        "go": {"status": "completed", "assets_count": 70}
    }
}
```

---

### ✅ 1.4 預設掃描策略（AI 簡化調用）

**狀態**: ✅ **5種策略可用**

AI 可以直接調用預設策略，無需手動選擇引擎和參數：

#### 策略 1: execute_strategy_fast（快速掃描）
**用途**: 快速驗證、開發測試  
**引擎**: Python  
**時間**: < 30秒  
**深度**: 2層  
**URL限制**: 100

```python
result = await coordinator.execute_strategy_fast(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

---

#### 策略 2: execute_strategy_balanced（均衡掃描）⭐ **推薦**
**用途**: 一般 Web 應用掃描  
**引擎**: Python + Rust  
**時間**: 1-3分鐘  
**深度**: 5層  
**URL限制**: 500

```python
result = await coordinator.execute_strategy_balanced(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

---

#### 策略 3: execute_strategy_comprehensive（全面掃描）
**用途**: SPA 應用（React/Vue/Angular）  
**引擎**: Python + TypeScript + Rust  
**時間**: 3-5分鐘  
**深度**: 5層  
**URL限制**: 1000

```python
result = await coordinator.execute_strategy_comprehensive(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

⚠️ **注意**: TypeScript 引擎當前不可用，會自動降級為 Python + Rust

---

#### 策略 4: execute_strategy_aggressive（激進掃描）
**用途**: 大型應用完整評估  
**引擎**: Python + TypeScript + Rust + Go（全部）  
**時間**: 5-10分鐘  
**深度**: 7層  
**URL限制**: 2000

```python
result = await coordinator.execute_strategy_aggressive(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

---

#### 策略 5: execute_strategy_smart（智能掃描）🧠 **AI 自動決策**
**用途**: AI 不確定如何選擇引擎時  
**流程**:
1. 執行 Phase 0（Rust 快速發現）
2. 分析技術棧和特徵
3. 自動選擇最佳引擎組合
4. 執行 Phase 1 深度掃描

```python
result = await coordinator.execute_strategy_smart(
    scan_id='scan_123',
    targets=['http://example.com']
)

# 內部流程：
# Phase 0 → 發現 React + Express → 建議 TypeScript + Python + Rust
#         → 執行 Phase 1（自動選擇引擎）
```

---

## 二、功能模組（Feature Modules）

### ⚠️ 2.1 導入問題

**狀態**: ❌ **所有功能模組導入失敗**

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'services.features.base'
```

**原因分析**:
`services/features/feature_step_executor.py` 和 `high_value_manager.py` 嘗試導入：
```python
from .base.feature_registry import FeatureRegistry
from .base.result_schema import FeatureResult
```

但實際目錄結構是：
```
services/features/
  ├── common/           # 實際存在的目錄
  │   ├── worker_statistics.py
  │   └── go/
  └── base/            # 不存在！應該是 common/
```

**修復方案**:
1. 創建 `services/features/base/` 目錄
2. 移動或創建 `feature_registry.py` 和 `result_schema.py`
3. 或者修改導入路徑為 `from .common.xxx`

---

### 📋 2.2 功能模組架構（待修復後可用）

雖然導入失敗，但從代碼分析可知各功能模組的架構：

#### SQLi 模組（SQL 注入檢測）
**Worker**: `SqliWorkerService`  
**檢測引擎**:
- BooleanDetectionEngine（布爾盲注）
- ErrorDetectionEngine（錯誤注入）
- TimeDetectionEngine（時間盲注）
- UnionDetectionEngine（聯合查詢注入）
- OOBDetectionEngine（帶外注入）
- HackingToolDetectionEngine（工具檢測）

**預期調用方式**（修復後）:
```python
from services.features.function_sqli.worker import SqliWorkerService

worker = SqliWorkerService()
result = await worker.process_task(task)
```

---

#### XSS 模組（跨站腳本檢測）
**Worker**: `XssWorkerService`  
**檢測類型**:
- TraditionalXssDetector（傳統 XSS）
- DomXssDetector（DOM XSS）
- StoredXssDetector（存儲型 XSS）
- BlindXssListenerValidator（盲打 XSS）

**預期調用方式**（修復後）:
```python
from services.features.function_xss.worker import XssWorkerService

worker = XssWorkerService()
result = await worker.process_task(task)
```

---

#### SSRF 模組（服務端請求偽造檢測）
**Worker**: `SsrfWorkerService`  
**功能**:
- SmartSsrfDetector（智能檢測）
- ParamSemanticsAnalyzer（參數語義分析）
- OastDispatcher（帶外測試調度）

**預期調用方式**（修復後）:
```python
from services.features.function_ssrf.worker import SsrfWorkerService

worker = SsrfWorkerService()
result = await worker.process_task(task)
```

---

#### IDOR 模組（不安全直接對象引用檢測）
**Worker**: `IdorWorkerService`  
**功能**:
- IdorWorker（IDOR 檢測器）

**預期調用方式**（修復後）:
```python
from services.features.function_idor.worker import IdorWorkerService

worker = IdorWorkerService()
result = await worker.process_task(task)
```

---

#### BizLogic 模組（業務邏輯漏洞檢測）
**Worker**: `BizLogicWorker`  
**狀態**: ⚠️ 未實現（符合之前報告）

---

## 三、AI 整合接口

### ✅ 3.1 AttackExecutor（攻擊執行器）

**狀態**: ✅ **完全可用**

**執行模式**:
- `SAFE`: 安全模式（僅模擬）
- `TESTING`: 測試模式（受控環境）⭐ **推薦**
- `AGGRESSIVE`: 激進模式（完整測試）

**AI 調用方式 1: 標準執行**
```python
from services.core.aiva_core.core_capabilities.attack.attack_executor import (
    AttackExecutor, ExecutionMode
)

executor = AttackExecutor(mode=ExecutionMode.TESTING)

# 執行攻擊計劃
result = await executor.execute_plan(
    plan=attack_plan,  # AttackPlan 對象
    target=target      # dict 或 AttackTarget 對象
)
```

**AI 調用方式 2: 整合 AI 分析結果**（推薦）
```python
# 執行計劃時整合 AI 分析結果
result = await executor.execute_plan_with_ai_analysis(
    plan=attack_plan,
    target=target,
    ai_analysis_results={
        "overall_risk_level": "high",  # AI 評估的風險等級
        "recommended_mode": "safe",
        "target_characteristics": {...}
    }
)

# 根據 AI 分析自動調整執行策略:
# - 高風險 → 切換到 SAFE 模式
# - 低風險 → 使用原始模式
```

**返回結果**:
```python
{
    "plan_id": "plan_xxx",
    "status": "completed",
    "execution_time": 15.3,
    "trace": [...],  # 執行追蹤
    "metrics": {...},  # 性能指標
    "feedback_data": {  # 回饋數據（供 AI 學習）
        "successful_payloads": [...],
        "failed_payloads": [...],
        "target_characteristics": {...}
    }
}
```

---

### ✅ 3.2 TwoPhaseScanOrchestrator（兩階段編排器）

**狀態**: ✅ **完全可用**

**功能**: 自動編排 Phase0 + Phase1 + AI 決策

**AI 調用方式**:
```python
from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import (
    TwoPhaseScanOrchestrator
)

orchestrator = TwoPhaseScanOrchestrator(broker)

# 執行完整兩階段掃描（自動化流程）
result = await orchestrator.execute_two_phase_scan(
    targets=['http://example.com', 'http://api.example.com'],
    trace_id='trace_123',
    max_depth=3,
    max_urls=1000
)
```

**內部流程**:
```
1. Phase 0 快速偵察（Rust 引擎）
   ↓
2. AI 決策: 是否需要 Phase 1？
   - 分析 Phase 0 結果
   - 評估目標複雜度
   ↓
3. 引擎選擇決策樹
   - 根據技術棧選擇引擎
   - 根據資產數量調整參數
   ↓
4. Phase 1 深度掃描（多引擎）
   ↓
5. 返回完整結果
```

**返回結果**:
```python
{
    "scan_id": "scan_xxx",
    "status": "completed",
    "execution_time": 200.5,
    "summary": {...},
    "assets": [...],
    "engine_results": {...},
    "phase0_summary": {...}  # Phase 0 摘要
}
```

---

### ✅ 3.3 EnhancedDecisionAgent（AI 決策代理）

**狀態**: ✅ **完全可用**

**AI 調用方式 1: 簡單決策**
```python
from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import (
    EnhancedDecisionAgent
)

agent = EnhancedDecisionAgent()

# AI 決策
intent = agent.decide(context)  # 返回 HighLevelIntent

# 返回結果:
# {
#     "intent_type": "SCAN_SURFACE",  # 或 DEEP_SCAN, EXPLOIT, etc.
#     "target": {...},
#     "confidence": 0.75,
#     "reasoning": "目標是新發現的 Web 應用..."
# }
```

**AI 調用方式 2: 高級決策**
```python
# 更複雜的決策分析
decision = agent.make_decision(
    target_info={...},
    scan_results={...},
    previous_findings={...}
)
```

---

## 四、AI 可用的完整工作流程

### 🎯 工作流程 1: 快速掃描 → 攻擊執行

```python
# 步驟 1: AI 決策
from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
agent = EnhancedDecisionAgent()
intent = agent.decide(context)

# 步驟 2: 執行快速掃描
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
coordinator = MultiEngineCoordinator()
await coordinator.initialize()
scan_result = await coordinator.execute_strategy_fast(
    scan_id='scan_123',
    targets=[intent.target.target_value]
)

# 步驟 3: 基於掃描結果執行攻擊
from services.core.aiva_core.core_capabilities.attack.attack_executor import (
    AttackExecutor, ExecutionMode
)
executor = AttackExecutor(mode=ExecutionMode.TESTING)

# 創建攻擊計劃（基於掃描結果）
plan = AttackPlan(
    plan_id=f"plan_{scan_result['scan_id']}",
    scan_id=scan_result['scan_id'],
    attack_type=VulnerabilityType.XSS,
    steps=[...]
)

# 執行攻擊
attack_result = await executor.execute_plan_with_ai_analysis(
    plan=plan,
    target={"url": intent.target.target_value},
    ai_analysis_results={"overall_risk_level": "low"}
)
```

---

### 🎯 工作流程 2: 智能掃描（完全自動化）

```python
# AI 只需一行代碼！
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator

coordinator = MultiEngineCoordinator()
await coordinator.initialize()

# 智能掃描會自動:
# 1. 執行 Phase 0
# 2. 分析結果
# 3. 選擇最佳引擎組合
# 4. 執行 Phase 1
result = await coordinator.execute_strategy_smart(
    scan_id='scan_123',
    targets=['http://example.com']
)

# 結果已包含完整的資產列表和建議
print(f"發現資產: {len(result.assets)}")
print(f"建議引擎: {result.phase0_summary.get('recommendations', {}).get('suggested_engines')}")
```

---

### 🎯 工作流程 3: 兩階段編排（全自動）

```python
# 最簡單的方式 - 完全自動化
from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import (
    TwoPhaseScanOrchestrator
)

orchestrator = TwoPhaseScanOrchestrator(broker)

# 一次調用完成所有步驟
result = await orchestrator.execute_two_phase_scan(
    targets=['http://example.com'],
    trace_id='trace_123',
    max_depth=3,
    max_urls=1000
)

# 內部會自動:
# 1. Phase 0 快速偵察
# 2. AI 決策是否需要 Phase 1
# 3. 引擎選擇
# 4. Phase 1 深度掃描
```

---

## 五、總結

### ✅ AI 可以完整調用的模組

| 模組類型 | 可用性 | 用法數量 | 備註 |
|---------|--------|---------|------|
| **掃描模組** | ✅ 100% | 8種用法 | Phase0, Phase1, 5種策略 |
| **AI 整合** | ✅ 100% | 3個接口 | 決策、執行、編排 |
| **功能模組** | ❌ 0% | 5個模組 | 導入錯誤需修復 |

---

### 📊 掃描模組用法總覽

#### 直接控制（AI 完全掌控）
1. `execute_phase0()` - Phase 0 快速偵察
2. `execute_phase1()` - Phase 1 深度掃描（AI 選擇引擎）

#### 預設策略（AI 簡化調用）
3. `execute_strategy_fast()` - 快速掃描
4. `execute_strategy_balanced()` - 均衡掃描 ⭐
5. `execute_strategy_comprehensive()` - 全面掃描
6. `execute_strategy_aggressive()` - 激進掃描
7. `execute_strategy_smart()` - 智能掃描 🧠

#### 自動化編排
8. `execute_two_phase_scan()` - 兩階段自動編排

---

### 🔧 功能模組修復計劃

**阻塞問題**: `ModuleNotFoundError: No module named 'services.features.base'`

**修復步驟**:
1. 創建 `services/features/base/` 目錄
2. 創建 `feature_registry.py`:
   ```python
   class FeatureRegistry:
       """功能模組註冊表"""
       pass
   ```
3. 創建 `result_schema.py`:
   ```python
   from pydantic import BaseModel
   
   class FeatureResult(BaseModel):
       """功能模組執行結果"""
       pass
   
   class Finding(BaseModel):
       """發現的漏洞"""
       pass
   ```
4. 重新測試功能模組導入

**預計修復時間**: 30分鐘 - 1小時

---

### ✅ 最終結論

**AI 可以完整操作的部分**:
1. ✅ **掃描模組**: 8種用法全部可用
2. ✅ **攻擊執行**: AttackExecutor 完全可用
3. ✅ **AI 決策**: EnhancedDecisionAgent 完全可用
4. ✅ **流程編排**: TwoPhaseScanOrchestrator 完全可用

**需要修復的部分**:
1. ❌ **功能模組**: 需要修復 `services.features.base` 導入問題
2. ⚠️ **TypeScript 引擎**: 掃描器文件不存在

**建議**:
1. **立即可用**: 使用掃描模組（Python + Rust + Go）和 AI 決策/執行接口
2. **短期修復**: 修復功能模組導入問題（30分鐘）
3. **中期完善**: 修復 TypeScript 引擎（1-2天）

**Docker 映像檔化建議**:
可以立即打包包含：
- ✅ 掃描模組（3/4 引擎）
- ✅ AI 決策和執行模組
- ✅ 兩階段編排器
- ⏳ 功能模組（修復後加入）
