# AIVA AI 指令流架構 — 基於 13 步驟工作流

## 📑 目錄

- [📋 13 步驟工作流概覽](#-13-步驟工作流概覽)
- [🔄 13 步驟數據流詳圖](#-13-步驟數據流詳圖)
- [🎯 核心架構原則: AI 分析 → 規劃 → 下令 → 協調 → 執行](#-核心架構原則-ai-分析-規劃-下令-協調-執行)
  - [對應程式碼](#對應程式碼)
- [📁 目錄結構與 13 步驟對應](#-目錄結構與-13-步驟對應)
- [🧠 三大 AI 決策點詳解](#-三大-ai-決策點詳解)
  - [決策點 ① ─ 步驟 6: decide_phase1_strategy()](#決策點-①-步驟-6-decide_phase1_strategy)
  - [決策點 ② ─ 步驟 9: decide_phase2_targets()](#決策點-②-步驟-9-decide_phase2_targets)
  - [決策點 ③ ─ 步驟 11: evaluate_phase2_results()](#決策點-③-步驟-11-evaluate_phase2_results)
- [🔗 CLI 執行模式 (步驟 0)](#-cli-執行模式-步驟-0)
  - [執行方式](#執行方式)
  - [資料來源](#資料來源)
  - [執行流程](#執行流程)
- [📊 數據合約 (aiva_common.schemas)](#-數據合約-aiva_commonschemas)
  - [核心數據結構](#核心數據結構)
  - [示例](#示例)
- [📈 實現狀態總表 (2026-01-10)](#-實現狀態總表-2026-01-10)
- [🔧 待改進項目](#-待改進項目)
  - [P0 (緊急)](#p0-緊急)
  - [P1 (短期)](#p1-短期)
  - [P2 (中期)](#p2-中期)
- [📝 總結](#-總結)
  - [核心實現](#核心實現)
  - [數據流完整性](#數據流完整性)
  - [設計原則](#設計原則)

---


**更新時間**: 2026-01-10  
**依據**: `13_STEPS_DATAFLOW_STATIC_ANALYSIS.md` + 實際程式碼驗證  
**目標**: AI 分析 → 規劃 → 下令 → 協調 → 執行

---

## 📋 13 步驟工作流概覽

本架構嚴格對應 AIVA 13 步驟執行流程，分為四大階段：

| 階段 | 步驟範圍 | 說明 | 核心模組 |
|------|----------|------|---------|
| **Phase 0** | 步驟 0-4 | 任務接收 & 快速偵察 | `task_planning/commander` |
| **Phase 1** | 步驟 5-6 | 深度掃描 + AI 決策 1 | `cognitive_core/decision` |
| **Phase 2** | 步驟 7-9 | 攻擊測試 + AI 決策 2 | `features/function_exploit` |
| **Phase 3** | 步驟 10-13 | 結果評估 + 學習回饋 | `integration/coordinators` |

---

## 🔄 13 步驟數據流詳圖

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         AIVA 13 步驟 AI 指令流架構                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 0: 任務接收與快速偵察 (步驟 0-4)                                        ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                               ║  │
│  ║  [步驟 0] CLI/API 入口                                                        ║  │
│  ║     │    aiva_cli_implementation.py --flow <ID>                               ║  │
│  ║     │    讀取 latest_classification.json (276 flows)                          ║  │
│  ║     ▼                                                                         ║  │
│  ║  [步驟 1] CommanderCoordinator (task_planning/commander/__init__.py)          ║  │
│  ║     │    接收指令 → 創建 Session                                              ║  │
│  ║     ▼                                                                         ║  │
│  ║  [步驟 2] PlanBuilder → AttackPlan (task_planning/planner/plan_builder.py)    ║  │
│  ║     │    任務分解 → 生成 AttackStep[]                                         ║  │
│  ║     ▼                                                                         ║  │
│  ║  [步驟 3-4] Phase 0 執行 → 快速偵察結果                                        ║  │
│  ║     │    PlanExecutor.execute_plan() → MessageBroker                          ║  │
│  ║     │    (task_planning/executor/plan_executor.py:794行)                      ║  │
│  ║     ▼                                                                         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                               │ phase_0_results                                     │
│                               ▼                                                     │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 1: 深度掃描 (步驟 5-6) + AI 決策點 ①                                   ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                               ║  │
│  ║  [步驟 5a] Integration 歷史查詢 (⚠️ 待完善)                                   ║  │
│  ║     │    查詢類似目標歷史數據                                                 ║  │
│  ║     │                                                                         ║  │
│  ║  [步驟 6] 🧠 AI 決策點 ① ─ decide_phase1_strategy()                          ║  │
│  ║     │    ┌────────────────────────────────────────┐                          ║  │
│  ║     │    │ cognitive_core/decision/               │                          ║  │
│  ║     │    │   enhanced_decision_agent.py (2231行)  │                          ║  │
│  ║     │    │                                        │                          ║  │
│  ║     │    │  ┌─────────────────────┐               │                          ║  │
│  ║     │    │  │ RealDecisionEngine  │               │                          ║  │
│  ║     │    │  │ 5M 神經網路 (1077行)│               │                          ║  │
│  ║     │    │  └─────────────────────┘               │                          ║  │
│  ║     │    │                                        │                          ║  │
│  ║     │    │  輸入: phase_0_results                 │                          ║  │
│  ║     │    │  輸出: HighLevelIntent (深度掃描策略)   │                          ║  │
│  ║     │    │  策略: api_focused, spa_optimized,     │                          ║  │
│  ║     │    │        web_comprehensive, stealth,     │                          ║  │
│  ║     │    │        waf_evasion, fast_discovery     │                          ║  │
│  ║     │    └────────────────────────────────────────┘                          ║  │
│  ║     ▼                                                                         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                               │ HighLevelIntent                                     │
│                               ▼                                                     │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 2: 攻擊測試 (步驟 7-9) + AI 決策點 ②                                   ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                               ║  │
│  ║  [步驟 7] Phase 1 深度掃描執行                                                ║  │
│  ║     │    AttackCoordinator → Features 模組調用                                ║  │
│  ║     │    (task_planning/commander/attack_coordinator.py:596行)                ║  │
│  ║     │                                                                         ║  │
│  ║     │    ┌──────────────────────────────────────┐                            ║  │
│  ║     │    │ services/features/function_exploit/  │                            ║  │
│  ║     │    │   executor/attack_executor.py (608行)│                            ║  │
│  ║     │    │   exploit_manager.py (968行)         │                            ║  │
│  ║     │    └──────────────────────────────────────┘                            ║  │
│  ║     ▼                                                                         ║  │
│  ║  [步驟 8a] Integration 歷史查詢 (⚠️ 待完善)                                   ║  │
│  ║     │                                                                         ║  │
│  ║  [步驟 9] 🧠 AI 決策點 ② ─ decide_phase2_targets()                           ║  │
│  ║     │    輸入: phase_1_results (漏洞清單)                                     ║  │
│  ║     │    輸出: 攻擊目標優先順序 + exploit_strategy                            ║  │
│  ║     │    排序: SQL_INJECTION > XSS > SSRF > IDOR                             ║  │
│  ║     ▼                                                                         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                               │ attack_targets                                      │
│                               ▼                                                     │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 3: 攻擊執行與結果評估 (步驟 10-11) + AI 決策點 ③                       ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                               ║  │
│  ║  [步驟 10] Phase 2 攻擊執行                                                   ║  │
│  ║     │    ┌──────────────────────────────────────────┐                        ║  │
│  ║     │    │ services/features/ (功能模組群)          │                        ║  │
│  ║     │    │   function_exploit/  → 漏洞利用          │                        ║  │
│  ║     │    │   smart_detection_manager.py (273行)     │                        ║  │
│  ║     │    │   high_value_manager.py (366行)          │                        ║  │
│  ║     │    └──────────────────────────────────────────┘                        ║  │
│  ║     │    ┌──────────────────────────────────────────┐                        ║  │
│  ║     │    │ services/integration/coordinators/       │                        ║  │
│  ║     │    │   base_coordinator.py (548行) - 雙閉環   │                        ║  │
│  ║     │    │   xss_coordinator.py (439行) - XSS特化   │                        ║  │
│  ║     │    │                                          │                        ║  │
│  ║     │    │   內循環: OptimizationData (Payload優化) │                        ║  │
│  ║     │    │   外循環: ReportData (Bug Bounty 報告)   │                        ║  │
│  ║     │    └──────────────────────────────────────────┘                        ║  │
│  ║     ▼                                                                         ║  │
│  ║  [步驟 11] 🧠 AI 決策點 ③ ─ evaluate_phase2_results()                        ║  │
│  ║     │    輸入: phase_2_results (攻擊結果)                                     ║  │
│  ║     │    輸出: RiskLevel + Impact Score + 建議                               ║  │
│  ║     │    評估: CRITICAL(≥3 exploits) / HIGH(≥1) / MEDIUM / LOW               ║  │
│  ║     ▼                                                                         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                               │ evaluation_result                                   │
│                               ▼                                                     │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  PHASE 4: 學習與回報 (步驟 12-13)                                             ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                               ║  │
│  ║  [步驟 12] 經驗學習                                                           ║  │
│  ║     │    cognitive_core/learning_system/                                      ║  │
│  ║     │    儲存成功模式 → RAG 知識庫                                            ║  │
│  ║     ▼                                                                         ║  │
│  ║  [步驟 13] 結果返回                                                           ║  │
│  ║          PlanExecutionResult → CommandCenter → User                           ║  │
│  ║          (使用 aiva_common.schemas 標準合約)                                  ║  │
│  ║                                                                               ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 核心架構原則: AI 分析 → 規劃 → 下令 → 協調 → 執行

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   AI 分析    │ ──► │    規劃      │ ──► │    下令      │ ──► │   協調執行   │   │
│  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                   │                    │                    │             │
│         ▼                   ▼                    ▼                    ▼             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │ 步驟 6/9/11  │     │  步驟 2      │     │  步驟 3      │     │ 步驟 4/7/10  │   │
│  │              │     │              │     │              │     │              │   │
│  │ Enhanced-    │     │ PlanBuilder  │     │ PlanExecutor │     │ Features/    │   │
│  │ Decision-    │     │ → AttackPlan │     │ → Message-   │     │ Coordinators │   │
│  │ Agent        │     │ → AttackStep │     │    Broker    │     │              │   │
│  │              │     │              │     │              │     │              │   │
│  │ cognitive_   │     │ task_        │     │ task_        │     │ integration/ │   │
│  │ core/        │     │ planning/    │     │ planning/    │     │ features/    │   │
│  │ decision/    │     │ planner/     │     │ executor/    │     │              │   │
│  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 對應程式碼

| 概念 | 步驟 | 模組 | 核心類/方法 |
|------|------|------|------------|
| **AI 分析** | 6, 9, 11 | `cognitive_core/decision/` | `EnhancedDecisionAgent.decide_phase1_strategy()` |
| | | | `EnhancedDecisionAgent.decide_phase2_targets()` |
| | | | `EnhancedDecisionAgent.evaluate_phase2_results()` |
| **規劃** | 2 | `task_planning/planner/` | `PlanBuilder.build_plan() → AttackPlan` |
| **下令** | 3 | `task_planning/executor/` | `PlanExecutor.execute_plan() → MessageBroker` |
| **協調** | 4, 7, 10 | `integration/coordinators/` | `BaseCoordinator.execute()` |
| **執行** | 4, 7, 10 | `features/function_exploit/` | `AttackExecutor.execute_plan_with_ai_analysis()` |

---

## 📁 目錄結構與 13 步驟對應

```
services/
├── core/aiva_core/
│   ├── cognitive_core/          ← AI 分析 (步驟 6/9/11)
│   │   ├── decision/
│   │   │   └── enhanced_decision_agent.py  (2231行) ─ 三大決策方法
│   │   ├── neural/
│   │   │   └── real_neural_core.py (1077行) ─ 5M 神經網路
│   │   ├── learning_system/     ← 步驟 12 經驗學習
│   │   └── rag/                 ← RAG 知識檢索
│   │
│   ├── task_planning/           ← 規劃 + 下令 (步驟 1-3)
│   │   ├── commander/
│   │   │   ├── __init__.py      ─ CommanderCoordinator (步驟 1)
│   │   │   ├── plan_builder.py  ─ 生成 AttackPlan (步驟 2)
│   │   │   └── attack_coordinator.py (596行) ─ 攻擊協調
│   │   ├── executor/
│   │   │   └── plan_executor.py (794行) ─ 執行計畫 (步驟 3)
│   │   └── planner/
│   │
│   ├── internal_exploration/
│   │   └── python_tools/
│   │       └── aiva_cli_implementation.py (841行) ─ CLI 入口 (步驟 0)
│   │
│   └── service_backbone/
│       └── messaging/
│           └── message_broker.py ─ 異步消息發送
│
├── integration/                  ← 協調 (步驟 4/7/10)
│   └── coordinators/
│       ├── base_coordinator.py (548行) ─ 雙閉環協調基類
│       └── xss_coordinator.py (439行)  ─ XSS 特化協調
│
├── features/                     ← 執行 (步驟 4/7/10)
│   ├── function_exploit/
│   │   ├── executor/
│   │   │   └── attack_executor.py (608行) ─ 攻擊執行
│   │   └── exploit_manager.py (968行)    ─ Exploit 管理
│   ├── smart_detection_manager.py (273行)
│   ├── high_value_manager.py (366行)
│   └── features_ready/          ← Workers (實際執行)
│       ├── xss_worker.py
│       ├── sqli_worker.py
│       ├── ssrf_worker.py
│       └── idor_worker.py
│
└── aiva_common/                  ← 標準數據合約
    └── schemas/
        ├── AttackPlan, AttackStep
        ├── HighLevelIntent
        ├── FunctionTaskPayload
        └── PlanExecutionResult
```

---

## 🧠 三大 AI 決策點詳解

### 決策點 ① ─ 步驟 6: decide_phase1_strategy()

```python
# cognitive_core/decision/enhanced_decision_agent.py

async def decide_phase1_strategy(self, phase_0_results: dict) -> HighLevelIntent:
    """
    輸入: Phase 0 快速偵察結果
        {
            "open_ports": [80, 443, 3306],
            "services": {"80": "HTTP", "443": "HTTPS"},
            "os_fingerprint": "Linux 5.x",
            "response_time": 0.05
        }
    
    輸出: Phase 1 深度掃描策略 (HighLevelIntent)
        - intent_type: DEEP_SCAN
        - parameters.focus_ports: [80, 443]
        - parameters.tools: ["nikto", "sqlmap"]
        - parameters.scan_depth: "intensive"
    
    策略選擇 (6 種):
        - api_focused: API 端點密集目標
        - spa_optimized: 前端應用優化
        - web_comprehensive: 完整 Web 掃描
        - stealth: 隱蔽模式 (低速)
        - waf_evasion: WAF 繞過
        - fast_discovery: 快速發現
    """
```

### 決策點 ② ─ 步驟 9: decide_phase2_targets()

```python
async def decide_phase2_targets(self, phase_1_results: dict) -> dict:
    """
    輸入: Phase 1 深度掃描結果 (漏洞清單)
        {
            "vulnerabilities": [
                {"type": "SQL_INJECTION", "severity": "HIGH", "location": "/api/user"},
                {"type": "XSS", "severity": "MEDIUM", "location": "/search"}
            ]
        }
    
    輸出: 攻擊目標排序
        {
            "targets": [
                {"vulnerability_id": "...", "type": "SQL_INJECTION", "priority": 1},
                {"vulnerability_id": "...", "type": "XSS", "priority": 2}
            ],
            "reasoning": "SQL注入風險最高，優先測試"
        }
    
    排序邏輯:
        1. SQL_INJECTION (CRITICAL)
        2. XSS (HIGH)
        3. SSRF (HIGH)
        4. IDOR (MEDIUM)
    """
```

### 決策點 ③ ─ 步驟 11: evaluate_phase2_results()

```python
async def evaluate_phase2_results(self, phase_2_results: dict) -> dict:
    """
    輸入: Phase 2 攻擊測試結果
        {
            "successful_exploits": [...],
            "failed_attempts": [...],
            "risk_indicators": [...]
        }
    
    輸出: 風險評估報告
        {
            "risk_level": RiskLevel.CRITICAL,  # ≥3 成功
            "impact_score": 0.95,
            "recommendations": ["立即修復 SQL 注入"],
            "next_actions": ["深入測試", "生成報告"]
        }
    
    風險等級:
        - CRITICAL: 成功利用 ≥ 3 個漏洞
        - HIGH: 成功利用 ≥ 1 個漏洞
        - MEDIUM: 有風險指標但無成功利用
        - LOW: 無明顯風險
    """
```

---

## 🔗 CLI 執行模式 (步驟 0)

### 執行方式

```powershell
# 列出可用 Flow
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --list

# 預覽執行計畫 (dry-run)
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 11 --dry-run

# 實際執行
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 11
```

### 資料來源

- **主要**: `latest_classification.json` (276 flows)
- **備用**: `services_classification_v9_new/classification_data.json`

### 執行流程

```
CLI 參數 (--flow <ID>)
    │
    ▼
讀取 latest_classification.json
    │
    ▼
動態導入模組 (importlib)
    │
    ▼
實例化類別 (智能推斷 CamelCase)
    │
    ▼
偵測入口方法 (train, execute, run, process, analyze)
    │
    ▼
Pipeline 數據傳遞 (步驟間串接)
    │
    ▼
返回執行結果
```

---

## 📊 數據合約 (aiva_common.schemas)

### 核心數據結構

| 結構 | 用途 | 傳遞階段 |
|------|------|----------|
| `HighLevelIntent` | AI 決策輸出 | 步驟 6 → 步驟 7 |
| `AttackPlan` | 攻擊計畫 | 步驟 2 → 步驟 3 |
| `AttackStep` | 單一攻擊步驟 | 步驟 3 → 步驟 4/7/10 |
| `FunctionTaskPayload` | Features 任務 | 步驟 10 → Workers |
| `PlanExecutionResult` | 執行結果 | 步驟 13 → 用戶 |

### 示例

```python
# AttackPlan (步驟 2 輸出)
AttackPlan(
    plan_id="plan_20260110_001",
    target=AttackTarget(url="https://target.com"),
    steps=[
        AttackStep(step_id="1", action="port_scan"),
        AttackStep(step_id="2", action="vuln_detect"),
        AttackStep(step_id="3", action="exploit"),
    ],
    estimated_duration=300,
    risk_level=RiskLevel.MEDIUM
)

# FunctionTaskPayload (步驟 10 輸入)
FunctionTaskPayload(
    task_id="task_ai_sqli_20260110",
    scan_id="scan_20260110",
    target=FunctionTaskTarget(url="https://target.com/api/user", method="GET"),
    priority=8  # 深度掃描優先級較高
)

# PlanExecutionResult (步驟 13 輸出)
PlanExecutionResult(
    session_id="sess_001",
    plan_id="plan_20260110_001",
    success=True,
    steps_completed=3,
    findings=[FindingPayload(...)],
    metrics=PlanExecutionMetrics(...)
)
```

---

## 📈 實現狀態總表 (2026-01-10)

| 步驟 | 名稱 | 模組 | 狀態 | 備註 |
|------|------|------|------|------|
| 0 | CLI 入口 | `aiva_cli_implementation.py` | ✅ 100% | 276 flows |
| 1 | CommanderCoordinator | `commander/__init__.py` | ✅ 100% | |
| 2 | PlanBuilder | `planner/plan_builder.py` | ✅ 100% | |
| 3 | PlanExecutor | `executor/plan_executor.py` | ✅ 100% | 794 行 |
| 4 | Phase 0 執行 | `features/` + `coordinators/` | ✅ 95% | |
| 5a | Integration 查詢 | ─ | ⚠️ 待完善 | 歷史查詢 |
| **6** | **AI 決策 ①** | `enhanced_decision_agent.py` | ✅ 100% | decide_phase1_strategy |
| 7 | Phase 1 執行 | `attack_executor.py` | ✅ 100% | 608 行 |
| 8a | Integration 查詢 | ─ | ⚠️ 待完善 | 歷史查詢 |
| **9** | **AI 決策 ②** | `enhanced_decision_agent.py` | ✅ 100% | decide_phase2_targets |
| 10 | Phase 2 執行 | `features/function_exploit/` | ✅ 90% | |
| **11** | **AI 決策 ③** | `enhanced_decision_agent.py` | ✅ 100% | evaluate_phase2_results |
| 12 | 經驗學習 | `learning_system/` | ✅ 100% | |
| 13 | 結果返回 | `PlanExecutionResult` | ✅ 100% | |

**整體完成度**: 95%

---

## 🔧 待改進項目

### P0 (緊急)
- 無

### P1 (短期)
| 項目 | 說明 | 預估工時 |
|------|------|----------|
| Integration 歷史查詢 | 步驟 5a/8a | 3-5 天 |
| AttackCoordinator 重構 | 移除直接 Worker 調用 | 2-3 天 |

### P2 (中期)
| 項目 | 說明 | 預估工時 |
|------|------|----------|
| 靶場實戰驗證 | 端到端測試 | 1 週 |
| 多語言引擎編譯 | Go/Rust/TS | 2 週 |

---

## 📝 總結

### 核心實現
- ✅ **AI 分析**: EnhancedDecisionAgent (2231 行) + 5M 神經網路 (1077 行)
- ✅ **規劃**: PlanBuilder + StrategyEngine
- ✅ **下令**: PlanExecutor + MessageBroker
- ✅ **協調**: BaseCoordinator (雙閉環) + XSSCoordinator
- ✅ **執行**: AttackExecutor + Workers (XSS/SQLi/SSRF/IDOR)

### 數據流完整性
13 步驟中：
- **Phase 0-1**: 100% 實現
- **Phase 2**: 95% 實現 (Integration 待完善)
- **Phase 3**: 100% 實現

### 設計原則
1. **AI 只決策，不執行** ─ cognitive_core 輸出意圖
2. **規劃層編排步驟** ─ task_planning 生成可執行計畫
3. **Dispatcher 下令** ─ 透過 MessageBroker 或 CLI
4. **Coordinator 協調** ─ 雙閉環優化
5. **Features 執行** ─ Workers 實際操作
