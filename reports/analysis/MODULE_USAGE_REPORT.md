# AIVA 模組用法完整報告
## 📑 目錄

- [一、掃描模組（Scan Module）](#一掃描模組scan-module)
  - [1.1 MultiEngineCoordinator 用法](#11-multienginecoordinator-用法)
  - [1.2 Phase 0 用法（快速偵察）](#12-phase-0-用法快速偵察)
  - [1.3 Phase 1 用法（深度掃描）](#13-phase-1-用法深度掃描)
  - [1.4 預設掃描策略](#14-預設掃描策略)
    - [execute_strategy_fast](#executestrategyfast)
    - [execute_strategy_balanced](#executestrategybalanced)
    - [execute_strategy_comprehensive](#executestrategycomprehensive)
    - [execute_strategy_aggressive](#executestrategyaggressive)
    - [execute_strategy_smart](#executestrategysmart)
- [二、功能模組（Feature Modules）](#二功能模組feature-modules)
  - [2.1 SQLI 模組](#21-sqli-模組)
  - [2.2 XSS 模組](#22-xss-模組)
  - [2.3 SSRF 模組](#23-ssrf-模組)
  - [2.4 IDOR 模組](#24-idor-模組)
  - [2.5 BIZLOGIC 模組](#25-bizlogic-模組)
- [三、AI 整合接口](#三ai-整合接口)
  - [3.1 AttackExecutor（攻擊執行器）](#31-attackexecutor攻擊執行器)
  - [3.2 TwoPhaseScanOrchestrator（兩階段編排器）](#32-twophasescanorchestrator兩階段編排器)
  - [3.3 EnhancedDecisionAgent（AI 決策代理）](#33-enhanceddecisionagentai-決策代理)
- [四、總結](#四總結)
  - [掃描模組: 4/4 可用](#掃描模組-44-可用)
  - [功能模組: 4/5 可用](#功能模組-45-可用)
  - [AI 整合: 3/3 可用](#ai-整合-33-可用)

---
## 一、掃描模組（Scan Module）
### 1.1 MultiEngineCoordinator 用法
**狀態**: ✅ 可用
**支援引擎**:
- python
- typescript
- rust
- go

**用法**:
```python
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator

coordinator = MultiEngineCoordinator()
await coordinator.initialize()
```

### 1.2 Phase 0 用法（快速偵察）
**狀態**: ✅ 可用
**描述**: Rust 引擎快速發現
**用法**:
```python
result = await coordinator.execute_phase0(
    scan_id='scan_123',
    targets=['http://example.com'],
    max_depth=3,
    timeout=600
)
```

### 1.3 Phase 1 用法（深度掃描）
**狀態**: ✅ 可用
**描述**: 多引擎深度掃描
**用法**:
```python
result = await coordinator.execute_phase1(
    scan_id='scan_123',
    targets=['http://example.com'],
    selected_engines=['python', 'rust', 'typescript'],  # AI 選擇
    max_depth=5,
    max_urls=1000
)
```

### 1.4 預設掃描策略
**狀態**: ✅ 可用
**AI 可用策略**:

#### execute_strategy_fast
- **描述**: 快速掃描 (Python)
- **用法**:
```python
result = await coordinator.execute_strategy_fast(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

#### execute_strategy_balanced
- **描述**: 均衡掃描 (Python + Rust)
- **用法**:
```python
result = await coordinator.execute_strategy_balanced(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

#### execute_strategy_comprehensive
- **描述**: 全面掃描 (Python + TypeScript + Rust)
- **用法**:
```python
result = await coordinator.execute_strategy_comprehensive(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

#### execute_strategy_aggressive
- **描述**: 激進掃描 (四引擎全開)
- **用法**:
```python
result = await coordinator.execute_strategy_aggressive(
    scan_id='scan_123',
    targets=['http://example.com']
)
```

#### execute_strategy_smart
- **描述**: 智能掃描 (AI自動選擇引擎)
- **用法**:
```python
result = await coordinator.execute_strategy_smart(
    scan_id='scan_123',
    targets=['http://example.com']
)
```


## 二、功能模組（Feature Modules）

### 2.1 SQLI 模組
**狀態**: ✅ 可用
**Worker**: `SqliWorkerService`
**用法**:
```python
from services.features.function_sqli.worker import SqliWorkerService

worker = SqliWorkerService()
result = await worker.process_task(task)
```

### 2.2 XSS 模組
**狀態**: ✅ 可用
**Worker**: `XssWorkerService`
**用法**:
```python
from services.features.function_xss.worker import XssWorkerService

worker = XssWorkerService()
result = await worker.process_task(task)
```

### 2.3 SSRF 模組
**狀態**: ✅ 可用
**Worker**: `SsrfWorkerService`
**用法**:
```python
from services.features.function_ssrf.worker import SsrfWorkerService

worker = SsrfWorkerService()
result = await worker.process_task(task)
```

### 2.4 IDOR 模組
**狀態**: ✅ 可用
**Worker**: `IdorWorkerService`
**用法**:
```python
from services.features.function_idor.worker import IdorWorkerService

worker = IdorWorkerService()
result = await worker.process_task(task)
```

### 2.5 BIZLOGIC 模組
**狀態**: ⚠️ 未實現（符合預期）

## 三、AI 整合接口

### 3.1 AttackExecutor（攻擊執行器）
**狀態**: ✅ 可用
**執行模式**:
- safe
- testing
- aggressive

**用法**:
```python
from services.core.aiva_core.core_capabilities.attack.attack_executor import (
    AttackExecutor, ExecutionMode
)

executor = AttackExecutor(mode=ExecutionMode.TESTING)
result = await executor.execute_plan(plan, target)
```

**AI 整合方法**:
```python
# 執行計劃時整合 AI 分析結果
result = await executor.execute_plan_with_ai_analysis(
    plan=attack_plan,
    target=target,
    ai_analysis_results=ai_results  # AI 決策結果
)
```

### 3.2 TwoPhaseScanOrchestrator（兩階段編排器）
**狀態**: ✅ 可用
**描述**: Phase0 + Phase1 流程控制
**用法**:
```python
from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import (
    TwoPhaseScanOrchestrator
)

orchestrator = TwoPhaseScanOrchestrator(broker)
result = await orchestrator.execute_two_phase_scan(
    targets=['http://example.com'],
    trace_id='trace_123',
    max_depth=3,
    max_urls=1000
)
```

### 3.3 EnhancedDecisionAgent（AI 決策代理）
**狀態**: ✅ 可用
**用法**:
```python
from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import (
    EnhancedDecisionAgent
)

agent = EnhancedDecisionAgent()
intent = agent.decide(context)  # 返回 HighLevelIntent
```

## 四、總結
### 掃描模組: 4/4 可用
### 功能模組: 4/5 可用
### AI 整合: 3/3 可用

✅ **結論**: AI 可以完整調用掃描和功能模組！

**AI 可用的操作方式**:
1. 使用 `MultiEngineCoordinator` 執行掃描（5種預設策略）
2. 使用 `AttackExecutor` 執行攻擊計劃
3. 使用 `EnhancedDecisionAgent` 進行決策
4. 透過 Worker 調用各功能模組（SQLi/XSS/SSRF/IDOR）
