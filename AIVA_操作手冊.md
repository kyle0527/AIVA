# 🚀 AIVA 系統使用者手冊

> **版本**: v2.1.2  
> **最後更新**: 2025年12月1日  
> **狀態**: ✅ 經實際測試驗證  
> **系統狀態**: MigrationPhase.TRANSITION（過渡期）

---

## 📋 目錄

### 第一章：整體運作流程
- [1.1 核心架構原則](#11-核心架構原則)
- [1.2 完整自動化操作流程](#12-完整自動化操作流程)
  - [步驟 0: 用戶輸入（結構化參數）](#步驟-0-用戶輸入結構化參數)
  - [步驟 1: Core 模組接收並分析](#步驟-1-core-模組接收並分析)
  - [步驟 2: AnalysisCoordinator 分解任務](#步驟-2-analysiscoordinator-分解任務)
  - [步驟 3: ScannerPlugin 創建掃描命令](#步驟-3-scannerplugin-創建掃描命令)
  - [步驟 4: Phase 0 執行（Rust 引擎快速偵察）](#步驟-4-phase-0-執行rust-引擎快速偵察)
  - [步驟 5a: Core → Integration 並行請求歷史數據](#步驟-5a-core--integration-並行請求歷史數據)
  - [步驟 6: AI 決策階段 1（Core 模組）](#步驟-6-ai-決策階段-1core-模組)
  - [步驟 7: Phase 1 執行（多引擎並行掃描）](#步驟-7-phase-1-執行多引擎並行掃描)
  - [步驟 8a: Core → Integration 再次並行比對](#步驟-8a-core--integration-再次並行比對)
  - [步驟 9: AI 決策階段 2（Core 模組）](#步驟-9-ai-決策階段-2core-模組)
  - [步驟 10: Phase 2 執行（Features 模組攻擊測試）](#步驟-10-phase-2-執行features-模組攻擊測試)
  - [步驟 11: AI 決策階段 3（最終評估）](#步驟-11-ai-決策階段-3最終評估)
  - [步驟 12: Integration 整合階段（生成報告）](#步驟-12-integration-整合階段生成報告)
  - [步驟 13: 完成（返回結果）](#步驟-13-完成返回結果)
- [1.3 暫停/恢復機制](#13-暫停恢復機制)
- [1.4 流程總結](#14-流程總結)

### 第二章：系統安裝與配置
- [2.1 系統概述](#21-系統概述)
- [2.2 環境需求](#22-環境需求)
- [2.3 啟動方式](#23-啟動方式)
  - [方式1: 快速啟動（推薦）](#方式1-快速啟動推薦)
  - [方式2: 命令行啟動](#方式2-命令行啟動)
  - [方式3: CLI 交互模式](#方式3-cli-交互模式)

### 第三章：核心功能使用
- [3.1 四種運作模式詳解](#31-四種運作模式詳解)
- [3.2 核心功能使用](#32-核心功能使用)
- [3.3 系統架構狀態](#33-系統架構狀態)

### 第四章：故障排除與 API
- [4.1 常見問題排查](#41-常見問題排查)
- [4.2 API 文檔](#42-api-文檔)

---

# 第一章：整體運作流程

## 1.1 核心架構原則

AIVA 採用模組化、AI 驅動的自動化安全測試架構，具備以下核心特性：

### ✅ 並行執行能力

**Core 模組可以同時發送命令給多個模組：**
- Core → Scan（Phase 0/1 掃描）
- Core → Integration（歷史數據比對）
- Core → Features（Phase 2 攻擊測試）

**並行執行，無需等待，大幅提升效率！**

### ✅ 暫停/恢復機制

- 每個階段都有 `task_id` 追蹤
- 可隨時 Ctrl+C 優雅中斷
- 保存執行狀態，支持恢復執行
- 關鍵決策點自動暫停等待確認

### ✅ AI 智能決策

**AI 在 3 個關鍵點進行決策：**
1. **Phase 0 → Phase 1**：選擇掃描引擎和策略
2. **Phase 1 → Phase 2**：評估攻擊價值，決定是否繼續
3. **Phase 2 → 深入測試**：判斷是否有更深層漏洞

### ✅ 提前終止機制

當 AI 判斷無攻擊價值時，自動跳過後續階段，節省時間。

---

## 1.2 完整自動化操作流程

### 步驟 0: 用戶輸入（結構化參數）

用戶提供**結構化輸入**（網址 + 限制），有 3 種方式：

#### **方式 1: Python 調用（最靈活）**

```python
from services.core.aiva_core.task_planning.ai_commander_v2 import AICommanderV2

# 初始化 AI Commander
commander = AICommanderV2()
await commander.initialize()

# 執行掃描任務
result = await commander.execute_task(
    task_description="掃描多個靶場",
    parameters={
        "targets": [
            "http://localhost:3000",  # Juice Shop
            "http://localhost:3001",  # BWAPP
            "http://localhost:3003",  # Shockle
            "http://localhost:8080"   # WebGoat
        ],
        "max_depth": 3,
        "timeout": 300,
        "scan_profile": "fast",  # fast/balanced/deep
        "engines": ["rust", "python", "typescript", "go"],
        
        # ⭐ 暫停控制選項
        "pause_after_phase0": False,  # Phase 0 後是否暫停等待確認
        "pause_after_phase1": False,  # Phase 1 後是否暫停
        "pause_before_attack": True,  # Phase 2 攻擊前暫停確認
        
        # ⭐ 並行控制
        "max_concurrent": 4,  # 最多同時掃描 4 個目標
        "parallel_scan": True,  # 啟用並行掃描
    }
)
```

#### **方式 2: CLI 調用（簡化版）**

```bash
# 基本掃描
python scripts/ui/aiva_cli.py --attack "掃描 http://localhost:3000"

# 多目標掃描
python scripts/ui/aiva_cli.py --attack "掃描 http://localhost:3000, http://localhost:3001"

# 指定引擎
python scripts/ui/aiva_cli.py --attack "用 Rust 引擎快速掃描 http://localhost:3000"
```

#### **方式 3: API 調用（HTTP REST）**

```bash
curl -X POST http://localhost:8000/api/scan \
  -H "Content-Type: application/json" \
  -d '{
    "targets": ["http://localhost:3000"],
    "max_depth": 3,
    "timeout": 300,
    "scan_profile": "fast",
    "pause_before_attack": true
  }'
```

---

### 步驟 1: Core 模組接收並分析

**位置**: `services/core/aiva_core/task_planning/ai_commander_v2.py:270`

```python
# AICommanderV2.execute_task()

1. 生成 task_id = "task_1701234567890"
2. 識別任務領域 = TaskDomain.ANALYSIS  # 自動識別為分析掃描
3. 獲取 AnalysisCoordinator
4. 創建 CoordinatorTask
```

**輸出：**
```json
{
  "task_id": "task_1701234567890",
  "domain": "analysis",
  "status": "started",
  "timestamp": "2025-12-01T10:00:00Z"
}
```

---

### 步驟 2: AnalysisCoordinator 分解任務

**位置**: `services/core/aiva_core/task_planning/coordinators/analysis_coordinator.py`

```python
# AnalysisCoordinator.decompose_task()

subtasks = [
    {
        "module_id": "scanner",  # ← 調用 ScannerPlugin
        "parameters": {
            "targets": ["http://localhost:3000", ...],
            "phase": "phase0",
            "engines": ["rust"]
        }
    }
]
```

---

### 步驟 3: ScannerPlugin 創建掃描命令

**位置**: `services/core/aiva_core/plugins/scanner_plugin.py:165`

```python
# ScannerPlugin.execute_task()

# 創建 AICommand
command = AICommand(
    command_id="scan_task_1701234567890_1701234567891",  # ← 自動生成
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={
        "scan_id": "scan_task_1701234567890_1701234567891",  # ← scan_id
        "targets": ["http://localhost:3000", ...],
        "max_depth": 3,
        "timeout": 300,
        "scan_profile": "fast"
    },
    priority=5,
    timeout=300
)

# ✅ 通過 CommandCenter 發送到 Scan 模組
result = await self.command_center.execute(command)
```

**關鍵：ScannerPlugin 自己創建 scan_id，不需要 Integration！**

---

### 步驟 4: Phase 0 執行（Rust 引擎快速偵察）

**位置**: `services/scan/engines/rust_engine/`  
**時間**: 5-10 分鐘（並行掃描 4 個目標）

#### ✅ 並行執行 4 個目標：

```
目標 1: http://localhost:3000 (Juice Shop)
目標 2: http://localhost:3001 (BWAPP)
目標 3: http://localhost:3003 (Shockle)
目標 4: http://localhost:8080 (WebGoat)
```

#### 每個目標執行：

```
A. 端點發現
   └─ 爬取常見路徑 40+ 個

B. JS 文件分析
   └─ 提取 API 端點

C. 技術棧識別
   └─ Express.js, React, MySQL 等

D. 初步風險評估
   └─ High: 9, Medium: 18
```

#### 輸出：Phase0CompletedPayload

```json
{
  "scan_id": "scan_task_1701234567890_1701234567891",
  "targets_completed": 4,
  "endpoints_discovered": 167,
  "tech_stacks": ["Express.js", "React", "MySQL"],
  "initial_risks": {
    "high": 36,
    "medium": 72
  },
  "execution_time": "8m 23s"
}
```

#### ⏸️ 暫停點 1（可選）：

```
Phase 0 完成！發現 167 個端點。

是否繼續進入 Phase 1？[Y/n/detail]
  Y - 繼續
  n - 停止（生成 Phase 0 報告）
  detail - 查看詳細結果
```

---

### 步驟 5a: Core → Integration 並行請求歷史數據

**✅ 這一步與 Phase 0 同時進行，不等待！**

**位置**: `services/integration/`

```python
# Core 發送命令給 Integration（並行）
command = AICommand(
    command_type=CommandType.INTEGRATION_QUERY_HISTORY,
    target_module="integration",
    payload={
        "targets": ["http://localhost:3000", ...],
        "query_type": "similar_targets",
        "include_success_rate": True
    }
)

result = await command_center.execute(command)
```

#### Integration 返回：

```json
{
  "similar_targets_found": 12,
  "avg_success_rate": 0.78,
  "recommended_payloads": [
    "<script>alert(1)</script>",
    "' OR 1=1--"
  ],
  "known_vulnerabilities": [
    {"type": "XSS", "confidence": 0.92},
    {"type": "SQLi", "confidence": 0.85}
  ]
}
```

---

### 步驟 6: AI 決策階段 1（Core 模組）

**位置**: `services/core/aiva_core/plugins/bio_neuron_plugin.py`

#### AI 接收兩份數據：

```python
inputs = {
    "phase0_data": phase0_result,      # Phase 0 掃描結果
    "integration_data": integration_result  # 歷史數據（可能為 None）
}

# AI 分析
decision = await bio_neuron.analyze(inputs)
```

#### 決策輸出：

```json
{
  "continue_phase1": true,
  "selected_engines": ["python", "typescript"],
  "strategy": "balanced",
  "estimated_time": "10-30 minutes",
  "reasoning": "發現 React SPA，需要 TypeScript 引擎動態渲染",
  "confidence": 0.89
}
```

#### 🤔 未知情況處理（RAG 網路搜索）：

```python
if decision["confidence"] < 0.7:
    # RAG 網路搜索外部建議
    external_advice = await rag.web_search(
        query=f"如何測試 {tech_stack} 應用的安全漏洞"
    )
    # 重新分析
    decision = await bio_neuron.analyze_with_external(
        inputs, 
        external_advice
    )
```

---

### 步驟 7: Phase 1 執行（多引擎並行掃描）

**位置**: `services/scan/engines/`  
**時間**: 10-30 分鐘

#### ✅ 並行執行多個引擎 × 多個目標：

```
並行組 1（目標 1 + 目標 2）：
  ├─ Python 引擎 → 目標 1: 靜態爬取 → 120 Assets
  ├─ Python 引擎 → 目標 2: 靜態爬取 → 95 Assets
  ├─ TypeScript 引擎 → 目標 1: 動態渲染 → 55 Assets
  └─ TypeScript 引擎 → 目標 2: 動態渲染 → 42 Assets

並行組 2（目標 3 + 目標 4）：
  ├─ Python 引擎 → 目標 3: 靜態爬取 → 78 Assets
  ├─ Python 引擎 → 目標 4: 靜態爬取 → 110 Assets
  ├─ TypeScript 引擎 → 目標 3: 動態渲染 → 38 Assets
  └─ TypeScript 引擎 → 目標 4: 動態渲染 → 61 Assets

總計：599 Assets → 去重 → 478 唯一資產
```

#### 實時進度顯示：

```
Phase 1 Progress:
  [████████████████████░░░░] 80% (4/4 targets, 2/2 engines)
  
  Target 1 (Juice Shop): ✅ Python (120) ✅ TypeScript (55)
  Target 2 (BWAPP):       ✅ Python (95)  ✅ TypeScript (42)
  Target 3 (Shockle):     ✅ Python (78)  ⏳ TypeScript (processing...)
  Target 4 (WebGoat):     ⏳ Python (110)  ⏳ TypeScript (queued)
  
  Estimated remaining time: 12 minutes
```

#### 輸出：Phase1CompletedPayload

```json
{
  "scan_id": "scan_task_1701234567890_1701234567891",
  "assets_discovered": 478,
  "engines_used": ["python", "typescript"],
  "execution_time": "18m 45s",
  "asset_types": {
    "forms": 120,
    "apis": 180,
    "parameters": 178
  }
}
```

#### ⏸️ 暫停點 2（可選）：

```
Phase 1 完成！發現 478 個資產。

資產分佈：
  - 表單: 120 個
  - API 端點: 180 個
  - URL 參數: 178 個

是否繼續進入 Phase 2 攻擊測試？[Y/n/detail]
  Y - 繼續
  n - 停止（生成 Phase 1 報告）
  detail - 查看資產詳情
```

---

### 步驟 8a: Core → Integration 再次並行比對

**✅ 與 Phase 1 結果分析同時進行！**

```python
# Core 發送資產分析請求給 Integration
command = AICommand(
    command_type=CommandType.INTEGRATION_ANALYZE_ASSETS,
    target_module="integration",
    payload={
        "assets": phase1_assets,
        "query_known_vulnerabilities": True
    }
)

result = await command_center.execute(command)
```

#### Integration 返回：

```json
{
  "assets_matched": 89,
  "known_vulnerable_assets": [
    {
      "asset_id": "form_login",
      "vulnerability": "SQLi",
      "success_rate": 0.91,
      "recommended_payloads": ["' OR 1=1--", "admin'--"]
    },
    {
      "asset_id": "param_search",
      "vulnerability": "XSS",
      "success_rate": 0.87,
      "recommended_payloads": ["<script>alert(1)</script>"]
    }
  ]
}
```

---

### 步驟 9: AI 決策階段 2（Core 模組）

#### AI 評估資產質量和攻擊價值：

```python
decision = {
    "phase1_confirmed_vulnerabilities": 2,  # SQLi, XSS 高置信度
    "has_deep_layer_potential": True,       # 可能有更深層漏洞
    "attack_value": "HIGH",                 # 有攻擊價值
    "continue_phase2": True,
    "selected_features": [
        {"module": "function_sqli", "priority": "HIGH"},
        {"module": "function_xss", "priority": "HIGH"},
        {"module": "function_idor", "priority": "MEDIUM"},
        {"module": "function_ssrf", "priority": "LOW"}
    ],
    "estimated_time": "15-25 minutes"
}
```

#### 🛑 提前終止機制：

```python
if decision["attack_value"] == "NONE":
    return {
        "status": "early_stop",
        "reason": "Phase 1 未發現有價值的攻擊資產",
        "assets_found": 478,
        "vulnerabilities_potential": "VERY_LOW",
        "report": generate_phase1_report()
    }
```

**AI 會自動跳過 Phase 2，節省時間！**

---

### 步驟 10: Phase 2 執行（Features 模組攻擊測試）

#### ⚠️ 攻擊前確認（預設暫停）：

```
========================================
  ⚠️  即將開始實際攻擊測試！
========================================

目標數量: 4 個靶場
測試模組: function_sqli, function_xss, function_idor, function_ssrf
預計時間: 15-25 分鐘
風險等級: MEDIUM（測試環境）

確認開始攻擊？[Y/n/config]
  Y - 開始
  n - 取消
  config - 調整攻擊配置（延遲、重試次數等）
```

#### 並行執行攻擊測試：

**位置**: `services/features/`  
**時間**: 15-25 分鐘

```
並行組 1（高優先級）：
  ├─ function_sqli → 50 個表單 → 3 Findings (SQLi)
  │   ├─ Boolean-based SQLi: 1 finding
  │   ├─ Error-based SQLi: 1 finding
  │   └─ Time-based SQLi: 1 finding
  │
  └─ function_xss  → 50 個表單 → 2 Findings (XSS)
      ├─ Reflected XSS: 1 finding
      └─ Stored XSS: 1 finding

並行組 2（中優先級）：
  ├─ function_idor → 80 個 API → 1 Finding (IDOR)
  └─ function_ssrf → 120 個參數 → 0 Findings

總計：6 個真實漏洞確認
```

#### 實時進度顯示：

```
Phase 2 Attack Progress:
  [████████████████░░░░░░░░] 65% (3/4 modules completed)
  
  function_sqli: ✅ Completed (3 findings - HIGH severity)
    └─ Tested 50 forms in 8m 32s
  
  function_xss:  ✅ Completed (2 findings - MEDIUM severity)
    └─ Tested 50 forms in 6m 18s
  
  function_idor: ✅ Completed (1 finding - HIGH severity)
    └─ Tested 80 APIs in 4m 27s
  
  function_ssrf: ⏳ Testing... (120/180 parameters, ETA: 5 min)
  
  Current Findings: 6 vulnerabilities
  Estimated remaining time: 5 minutes
```

#### 輸出：List[FindingPayload]

```json
{
  "findings_count": 6,
  "execution_time": "15m 12s",
  "vulnerabilities": [
    {
      "id": "VULN-001",
      "type": "SQL Injection",
      "severity": "CRITICAL",
      "confidence": "CONFIRMED",
      "affected_url": "http://localhost:3000/rest/user/login",
      "payload": "' OR 1=1--",
      "evidence": "Database error: syntax error near..."
    },
    // ... 其他 5 個漏洞
  ]
}
```

---

### 步驟 11: AI 決策階段 3（最終評估）

#### AI 最終評估漏洞真實性：

```python
decision = {
    "confirmed_real_vulnerabilities": 6,
    "false_positives": 0,
    "deeper_layer_potential": False,  # 無更深層漏洞
    "continue_deeper": False,
    "final_action": "generate_report",
    "risk_assessment": {
        "overall_risk": "HIGH",
        "business_impact": "CRITICAL",
        "exploitability": "EASY"
    }
}
```

#### 🔍 深層測試（可選）：

```python
if decision["deeper_layer_potential"]:
    # 啟用高級功能模組
    advanced_modules = [
        "function_bizlogic",  # 業務邏輯漏洞
        "function_crypto",    # 密碼學漏洞
        "function_authn_go"   # 認證漏洞（Go 引擎）
    ]
    
    # 繼續深入測試
    deeper_results = await execute_advanced_tests(advanced_modules)
```

**在本例中，AI 判斷無需深入，直接生成報告。**

---

### 步驟 12: Integration 整合階段（生成報告）

**位置**: `services/integration/`  
**時間**: 2-5 分鐘

#### A. 數據關聯

```
資產與漏洞映射：
  ├─ 表單 "login" → SQLi (VULN-001)
  ├─ 表單 "search" → XSS (VULN-002, VULN-003)
  └─ API "/api/users/:id" → IDOR (VULN-004)

攻擊路徑構建：
  User Input → Parameter "username" → SQL Query → Database
```

#### B. 風險評估

```
CVSS 評分計算：
  ├─ VULN-001 (SQLi):  CVSS 9.1 - CRITICAL
  │   └─ AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
  │
  ├─ VULN-002 (XSS):   CVSS 6.8 - MEDIUM
  │   └─ AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N
  │
  └─ VULN-004 (IDOR):  CVSS 8.2 - HIGH
      └─ AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N

業務影響分析：
  ├─ 數據洩露風險: CRITICAL
  ├─ 權限提升風險: HIGH
  └─ 服務可用性: LOW
```

#### C. 報告生成

```
生成多種格式報告：
  ├─ 執行摘要（管理層）: report_executive_summary.pdf
  ├─ 技術詳情（開發團隊）: report_technical_details.pdf
  ├─ 修復建議（優先級排序）: report_remediation_guide.pdf
  ├─ 證據附件（PoC 截圖）: evidence/
  └─ SARIF 格式（CI/CD 整合）: report.sarif
```

---

### 步驟 13: 完成（返回結果）

#### 最終輸出：

```json
{
  "task_id": "task_1701234567890",
  "status": "completed",
  "execution_time": "35 minutes 42 seconds",
  
  "summary": {
    "targets_scanned": 4,
    "assets_discovered": 478,
    "vulnerabilities_found": 6,
    "risk_level": "HIGH",
    "false_positives": 0
  },
  
  "phases": {
    "phase0": {
      "status": "completed",
      "time": "8m 23s",
      "endpoints": 167,
      "tech_stacks": ["Express.js", "React", "MySQL"]
    },
    "phase1": {
      "status": "completed",
      "time": "18m 45s",
      "assets": 478,
      "engines_used": ["rust", "python", "typescript"]
    },
    "phase2": {
      "status": "completed",
      "time": "15m 12s",
      "tests_executed": 250,
      "findings": 6
    }
  },
  
  "vulnerabilities": [
    {
      "id": "VULN-001",
      "type": "SQL Injection",
      "severity": "CRITICAL",
      "cvss_score": 9.1,
      "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
      "target": "http://localhost:3000",
      "affected_url": "/rest/user/login",
      "affected_parameter": "username",
      "payload": "' OR 1=1--",
      "evidence": {
        "request": "POST /rest/user/login HTTP/1.1...",
        "response": "500 Internal Server Error\nDatabase error...",
        "screenshot": "evidence/vuln-001-screenshot.png"
      },
      "remediation": "使用參數化查詢（Prepared Statements）防止 SQL 注入",
      "references": [
        "https://owasp.org/www-community/attacks/SQL_Injection",
        "https://cwe.mitre.org/data/definitions/89.html"
      ]
    },
    {
      "id": "VULN-002",
      "type": "Cross-Site Scripting (XSS)",
      "severity": "MEDIUM",
      "cvss_score": 6.8,
      "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N",
      "target": "http://localhost:3000",
      "affected_url": "/search",
      "affected_parameter": "q",
      "payload": "<script>alert(document.cookie)</script>",
      "evidence": {
        "request": "GET /search?q=<script>alert(1)</script>",
        "response": "200 OK\n<div>Results for: <script>alert(1)</script></div>",
        "screenshot": "evidence/vuln-002-screenshot.png"
      },
      "remediation": "對用戶輸入進行 HTML 編碼，使用 Content Security Policy (CSP)",
      "references": [
        "https://owasp.org/www-community/attacks/xss/",
        "https://cwe.mitre.org/data/definitions/79.html"
      ]
    }
    // ... 其他 4 個漏洞
  ],
  
  "reports": {
    "executive_summary": "reports/scan_task_1701234567890_executive.pdf",
    "technical_details": "reports/scan_task_1701234567890_technical.pdf",
    "remediation_guide": "reports/scan_task_1701234567890_remediation.pdf",
    "sarif": "reports/scan_task_1701234567890.sarif",
    "evidence_folder": "reports/evidence/"
  },
  
  "next_steps": [
    "1. 立即修復 CRITICAL 漏洞 (VULN-001)",
    "2. 在 2 週內修復 HIGH 漏洞 (VULN-004)",
    "3. 在 1 個月內修復 MEDIUM 漏洞 (VULN-002, VULN-003)",
    "4. 修復後重新掃描驗證"
  ]
}
```

#### 控制台輸出：

```
========================================
  ✅ 掃描完成！
========================================

執行時間: 35 分鐘 42 秒

掃描摘要:
  - 目標數量: 4 個靶場
  - 發現資產: 478 個
  - 發現漏洞: 6 個
  - 風險等級: HIGH

漏洞分佈:
  - CRITICAL: 1 個 (SQL Injection)
  - HIGH:     2 個 (IDOR, ...)
  - MEDIUM:   2 個 (XSS, ...)
  - LOW:      1 個

報告已生成:
  📄 執行摘要: reports/scan_task_1701234567890_executive.pdf
  📄 技術詳情: reports/scan_task_1701234567890_technical.pdf
  📄 修復建議: reports/scan_task_1701234567890_remediation.pdf
  📁 證據附件: reports/evidence/

建議優先修復:
  1. 🔴 SQL Injection (CVSS 9.1) - 立即修復
  2. 🟠 IDOR (CVSS 8.2) - 2 週內修復
  3. 🟡 XSS (CVSS 6.8) - 1 個月內修復

========================================
```

---

## 1.3 暫停/恢復機制

### 暫停方式

#### 1. **自動暫停點**（預設）：

- ⏸️ Phase 0 → Phase 1 之間（可選）
- ⏸️ Phase 1 → Phase 2 之間（攻擊前，預設啟用）
- ⏸️ 發現 CRITICAL 漏洞時（可選）

#### 2. **手動暫停**：

```bash
# 在任何時候按下
Ctrl+C  # 優雅中斷，保存當前狀態
```

**系統會：**
- ✅ 保存已完成的階段結果
- ✅ 記錄當前執行位置
- ✅ 生成中間報告
- ✅ 允許後續恢復執行

#### 3. **API 暫停**：

```bash
# 暫停正在執行的任務
curl -X POST http://localhost:8000/api/scan/task_1701234567890/pause

# 取消任務
curl -X POST http://localhost:8000/api/scan/task_1701234567890/cancel
```

### 恢復執行

#### Python 調用：

```python
# 從中斷點恢復
result = await commander.resume_task(
    task_id="task_1701234567890",
    from_phase="phase1"  # 從 Phase 1 繼續
)
```

#### CLI 調用：

```bash
# 恢復任務
python scripts/ui/aiva_cli.py --resume task_1701234567890

# 查看任務狀態
python scripts/ui/aiva_cli.py --status task_1701234567890
```

#### API 調用：

```bash
# 恢復任務
curl -X POST http://localhost:8000/api/scan/task_1701234567890/resume \
  -H "Content-Type: application/json" \
  -d '{"from_phase": "phase1"}'
```

---

## 1.4 流程總結

### 核心特性

✅ **Core 模組是真正的入口**
- AICommanderV2 接收結構化輸入
- 不依賴 Integration 模組
- 可同時並行調用多個模組

✅ **並行執行能力**
- 多目標並行掃描
- 多引擎並行執行
- Core ↔ Integration 並行通信

✅ **AI 全程智能決策**
- 3 個關鍵決策點
- RAG 知識庫支持
- 網路搜索處理未知情況

✅ **支持暫停/恢復**
- 隨時 Ctrl+C 中斷
- 保存執行狀態
- 支持恢復執行

✅ **提前終止機制**
- 無攻擊價值自動停止
- 節省時間和資源

### 執行時間估算

| 階段 | 時間範圍 | 說明 |
|------|---------|------|
| Phase 0 | 5-10 分鐘 | Rust 引擎快速偵察 |
| AI 決策 1 | 10-30 秒 | 選擇 Phase 1 策略 |
| Phase 1 | 10-30 分鐘 | 多引擎深度掃描 |
| AI 決策 2 | 10-30 秒 | 評估攻擊價值 |
| Phase 2 | 15-25 分鐘 | 功能模組攻擊測試 |
| AI 決策 3 | 10-30 秒 | 最終評估 |
| Integration | 2-5 分鐘 | 報告生成 |
| **總計** | **35-75 分鐘** | **視目標複雜度而定** |

### 模組間通信架構

```
用戶輸入
    ↓
AICommanderV2 (Core)
    ↓
┌───┴───┬────────┬─────────┐
↓       ↓        ↓         ↓
Scan  Integration  Features  ...
(並行執行，無需等待)
```

**這就是 AIVA 的完整自動化執行流程！**

---

# 第二章：系統安裝與配置

## 2.1 系統概述

AIVA（Autonomous Intelligence Virtual Assistant）是一個企業級的 AI 驅動安全測試平台，具備：

- **真實 AI 大腦**：5M 參數神經網路（BioNeuron）
- **多語言支持**：Python (495)、Rust (123)、TypeScript (84)、Go (80) 共 782 個能力
- **自主決策**：AI 接收結構化輸入並自動執行完整流程
- **企業級架構**：基於 Strangler Fig 模式的漸進式遷移

### 當前系統狀態

```python
# 實際程式碼驗證（services/core/aiva_core/__init__.py:53）
self.current_phase = MigrationPhase.TRANSITION  # 過渡期
```

**遷移階段說明**：
- ❌ LEGACY（純舊系統）
- ✅ **TRANSITION（當前狀態）** - 新舊系統共存，功能開關控制
- ⏳ MODERN（新系統主導）
- ⏳ COMPLETE（遷移完成）

**功能開關狀態**（全部已啟用）：
```python
self.feature_flags = {
    V1_CAPABILITY_REGISTRY: True,      # 能力註冊系統
    AI_MODULE_ORCHESTRATION: True,     # AI 模組編排
    ENHANCED_MESSAGE_BROKER: True,     # 增強消息系統
    RISK_CONTROL_SYSTEM: True,         # 風險控制
    TOPOLOGICAL_SORTING: True,         # 拓撲排序
}
```

---

## 2.2 環境需求

### 必需軟體

- **Python**: 3.10+ （推薦 3.11）
- **PostgreSQL**: 14+ （用於能力元數據）
- **ChromaDB**: 最新版（用於 RAG 知識庫）
- **Node.js**: 18+ （如需使用 TypeScript 能力）
- **Rust**: 1.70+ （如需使用 Rust 能力）
- **Go**: 1.20+ （如需使用 Go 能力）

### Python 依賴

主要套件：
```
fastapi >= 0.104.0
uvicorn[standard] >= 0.24.0
torch >= 2.0.0
transformers >= 4.35.0
chromadb >= 0.4.15
psycopg2-binary >= 2.9.9
PyJWT >= 2.8.0
httpx >= 0.25.0
rich >= 13.0.0
```

安裝方式：
```bash
pip install -r requirements.txt
```

### 環境配置

創建 `.env` 文件：
```env
# 資料庫連接
DATABASE_URL=postgresql://user:password@localhost:5432/aiva_db
CHROMA_HOST=localhost
CHROMA_PORT=8000

# JWT 密鑰
JWT_SECRET=your-secret-key-change-in-production

# API 設置
API_HOST=0.0.0.0
API_PORT=8000

# 日誌級別
LOG_LEVEL=INFO
```

---

## 2.3 啟動方式

### 方式1: 快速啟動（推薦）

#### Windows 用戶

**雙擊批處理文件**：
```
啟動AI服務.bat
```

這會自動：
1. 設置 PYTHONPATH
2. 啟動 API 服務（端口 8000）
3. 顯示即時日誌

**啟動訊息**：
```
========================================
  AIVA AI Service - API Mode
========================================

Starting AIVA AI Service...
API will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

Press Ctrl+C to stop the service
```

#### 訪問點

- **API 根路徑**: http://localhost:8000/
- **Swagger 文檔**: http://localhost:8000/docs
- **ReDoc 文檔**: http://localhost:8000/redoc

#### 預設帳號

```
管理員帳號:
  用戶名: admin
  密碼: aiva-admin-2025

一般用戶:
  用戶名: user
  密碼: aiva-user-2025
```

---

### 方式2: 命令行啟動

#### 啟動 API 服務

```bash
# 基本啟動
python scripts/startup/start_ai_service.py --mode api

# 自定義端口
python scripts/startup/start_ai_service.py --mode api --port 9000

# 自定義綁定地址
python scripts/startup/start_ai_service.py --mode api --host 127.0.0.1 --port 8080
```

#### 啟動後台監控模式

持續監控目標並自動掃描：

```bash
# 基本監控（預設1小時掃描一次）
python scripts/startup/start_ai_service.py --mode monitor \
    --targets http://localhost:3000

# 自定義間隔（30分鐘）
python scripts/startup/start_ai_service.py --mode monitor \
    --targets http://example.com http://test.local \
    --interval 1800

# 監控多個目標
python scripts/startup/start_ai_service.py --mode monitor \
    --targets http://app1.com http://app2.com http://app3.com \
    --interval 3600
```

**監控模式輸出**：
```
🔍 監控模式啟動
📋 監控目標: http://localhost:3000
⏱️  掃描間隔: 3600秒 (60.0分鐘)

============================================================
🚀 開始第 1 次掃描 [22:30:00]
============================================================
✅ 掃描完成: success
📊 發現資產: 5
🎯 資產摘要:
   [1] url: http://localhost:3000
   [2] endpoint: /api/users
   [3] endpoint: /api/posts
   ... 還有 2 個資產

💤 等待 3600秒後進行下一次掃描...
```

#### 啟動交互式模式

命令行交互操作：

```bash
python scripts/startup/start_ai_service.py --mode interactive
```

**可用命令**：
```
AIVA> scan http://localhost:8080           # 掃描目標
AIVA> scan-fast http://example.com         # 快速掃描
AIVA> status                               # 查看狀態
AIVA> engines                              # 查看引擎狀態
AIVA> help                                 # 顯示幫助
AIVA> quit                                 # 退出
```

#### 啟動守護進程模式

結合 API + 監控：

```bash
python scripts/startup/start_ai_service.py --mode daemon \
    --port 8000 \
    --targets http://localhost:3000 \
    --interval 3600
```

這會同時運行：
- API 服務（http://localhost:8000）
- 後台監控（自動掃描目標）

---

### 方式3: CLI 交互模式

AIVA CLI 提供直接的 AI 交互接口。

#### 啟動交互式選單

```bash
python aiva_cli.py
```

會顯示 ASCII 藝術橫幅和主選單：
```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ██╗██╗   ██╗ █████╗     ██████╗██╗     ██╗      ║
║    ██╔══██╗██║██║   ██║██╔══██╗   ██╔════╝██║     ██║      ║
║    ███████║██║██║   ██║███████║   ██║     ██║     ██║      ║
║    ██╔══██║██║╚██╗ ██╔╝██╔══██║   ██║     ██║     ██║      ║
║    ██║  ██║██║ ╚████╔╝ ██║  ██║   ╚██████╗███████╗██║      ║
║    ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝    ╚═════╝╚══════╝╚═╝      ║
║                                                              ║
║        AI-Driven Vulnerability Assessment System            ║
║              高級人工智慧漏洞評估平台                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

#### AI 執行攻擊（重點功能）

**自然語言下令**：
```bash
# 掃描網站
python aiva_cli.py --attack "幫我跑 http://localhost:8080/WebGoat 的掃描"

# SQL 注入測試
python aiva_cli.py --attack "對 http://example.com 執行 SQL 注入測試"

# XSS 檢測
python aiva_cli.py --attack "掃描 http://test.local 找出 XSS 漏洞"

# 完整滲透測試
python aiva_cli.py --attack "對 http://target.com 進行完整的 web 應用滲透測試"
```

**AI 處理流程**：
1. 分析自然語言指令
2. 從資料庫查詢相關能力（782 個能力中篩選）
3. 讀取 invocation_metadata 確定調用方式
4. 執行實際攻擊（跨語言調用）
5. 返回結果並記錄經驗

#### 查詢能力

```bash
# 查詢攻擊相關能力
python aiva_cli.py --query "攻擊路徑分析"

# 查詢掃描能力
python aiva_cli.py --query "網路掃描"

# 查詢特定技術
python aiva_cli.py --query "SQL 注入檢測"

# 返回更多結果
python aiva_cli.py --query "漏洞利用" --top-k 10
```

#### 獲取工作流推薦

```bash
# Web 應用滲透測試流程
python aiva_cli.py --workflow "web 應用滲透測試"

# API 安全測試流程
python aiva_cli.py --workflow "API 安全評估"

# 網路安全測試流程
python aiva_cli.py --workflow "網路滲透測試"
```

#### 系統管理命令

```bash
# 查看統計資訊
python aiva_cli.py --stats

# 同步能力到 RAG 知識庫
python aiva_cli.py --sync

# 運行 AI 分析測試
python aiva_cli.py --test
```

---

# 第三章：核心功能使用

## 3.1 四種運作模式詳解

### 1. API 服務模式（推薦用於生產環境）

**適用場景**：
- 持續運行的服務
- 多用戶訪問
- CI/CD 整合
- 企業級部署

**特點**：
- REST API 接口
- JWT 認證
- 完整的 Swagger 文檔
- CORS 支持

**端點列表**：
```
POST   /api/v1/auth/login              # 用戶登入
POST   /api/v1/auth/register           # 用戶註冊
GET    /api/v1/capabilities/list       # 列出所有能力
POST   /api/v1/capabilities/query      # 查詢能力
POST   /api/v1/capabilities/execute    # 執行能力
GET    /api/v1/capabilities/stats      # 統計資訊
POST   /api/v1/security/scan           # 快速掃描
POST   /api/v1/security/pentest        # 滲透測試
GET    /api/v1/admin/system-info       # 系統資訊
```

### 2. 監控模式（適用於持續監控）

**適用場景**：
- 持續監控生產環境
- 定期安全掃描
- 自動化檢測

**工作流程**：
```
啟動 → 初始化掃描器 → 執行掃描 → 記錄結果 → 等待間隔 → 重複
```

**配置選項**：
- `--targets`: 監控目標列表
- `--interval`: 掃描間隔（秒）
- `--log-level`: 日誌詳細程度

### 3. 交互式模式（適用於手動操作）

**適用場景**：
- 手動測試
- 學習系統操作
- 即時互動控制

**命令系統**：
- `scan <url>`: 標準掃描（使用平衡策略）
- `scan-fast <url>`: 快速掃描（僅 Python 引擎）
- `status`: 顯示系統狀態和運行時間
- `engines`: 查看引擎可用性
- `help`: 顯示命令說明
- `quit`: 優雅退出

### 4. 守護進程模式（適用於混合需求）

**適用場景**：
- 需要 API 訪問
- 同時需要後台監控
- 企業級自動化

**特點**：
- 雙重運行
- 資源優化
---

## 3.2 核心功能使用

## 核心功能使用

### 1. 能力查詢系統

AIVA 擁有 782 個能力，分布在 4 種語言中：

```python
# 語言分布
Python:     495 個能力
Rust:       123 個能力
TypeScript:  84 個能力
Go:          80 個能力
```

**查詢方式**：

1. **通過 CLI 查詢**：
```bash
python aiva_cli.py --query "SQL 注入"
```

2. **通過 API 查詢**：
```bash
curl -X POST http://localhost:8000/api/v1/capabilities/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "SQL 注入", "top_k": 5}'
```

3. **通過 Python 代碼**：
```python
from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery

query = AICapabilityQuery()
results = await query.query_capabilities("SQL 注入", top_k=5)

for cap in results:
    print(f"能力: {cap.capability_name}")
    print(f"語言: {cap.language}")
    print(f"調用方式: {cap.invocation_protocol}")
```

### 2. AI 執行攻擊

**端到端流程**：

```
用戶指令 → AI 解析 → 能力查詢 → 讀取 invocation_metadata → 跨語言調用 → 返回結果
```

**示例**：

```bash
# 1. 自然語言下令
python aiva_cli.py --attack "掃描 http://localhost:8080"

# 2. AI 自動處理
# - 分析「掃描」關鍵字
# - 查詢相關能力（如 network_scanner, vulnerability_scanner）
# - 選擇最佳能力
# - 讀取 invocation_metadata
# - 構造調用參數

# 3. 執行並返回
✅ 掃描完成
📊 發現 15 個資產
📈 檢測到 3 個潛在漏洞
```

### 3. 掃描策略

AIVA 提供三種掃描策略：

#### 快速掃描（Fast Strategy）

```python
# 特點
- 僅使用 Python 引擎
- 速度最快
- 基礎資產發現

# 使用場景
- 快速檢查
- CI/CD 流程
- 初步評估
```

#### 平衡掃描（Balanced Strategy）

```python
# 特點
- 使用 Python + TypeScript 引擎
- 速度與深度平衡
- 較完整的資產發現

# 使用場景
- 一般安全測試
- 定期掃描
- 標準評估
```

#### 全面掃描（Comprehensive Strategy）

```python
# 特點
- 使用所有引擎（Python/TypeScript/Rust/Go）
- 最深入的檢測
- 完整的資產發現

# 使用場景
- 詳細滲透測試
- 合規審計
- 全面安全評估
```

### 4. 多語言能力調用

AIVA 支援三種調用協議：

#### Unified Caller（統一調用器）

```python
# 適用語言: Python
# 調用方式: 直接函數調用

invocation_metadata = {
    "protocol": "unified_caller",
    "import_path": "services.scan.engines.python_engine.network_scanner",
    "class_name": "NetworkScanner",
    "method_name": "scan"
}
```

#### HTTP 協議

```python
# 適用語言: TypeScript, Go
# 調用方式: HTTP API

invocation_metadata = {
    "protocol": "http",
    "endpoint": "http://localhost:3001/scan",
    "method": "POST",
    "headers": {"Content-Type": "application/json"}
}
```

#### gRPC 協議

```python
# 適用語言: Rust, Go
# 調用方式: gRPC

invocation_metadata = {
    "protocol": "grpc",
    "server": "localhost:50051",
    "service": "ScannerService",
    "method": "PerformScan"
}
```

---

## 3.3 系統架構狀態

### 核心組件狀態

| 組件 | 狀態 | 版本 | 說明 |
|------|------|------|------|
| **MigrationController** | ✅ TRANSITION | v3.0.0-alpha | 遷移控制器 |
| **Scanner Plugin** | ✅ 已實作 | v2.1.0 | 掃描器插件 |
| **Data Hub Plugin** | ✅ 已實作 | v2.0.0 | 數據中心插件 |
| **Exploiter Plugin** | ✅ 已實作 | v1.5.0 | 漏洞利用插件 |
| **BioNeuron Plugin** | ✅ 已實作 | v1.0.0 | 5M 參數神經網絡 |
| **AI Commander V2** | ✅ 已實作 | v2.0.0 | AI 指揮中心 |
| **Capability Registry** | ✅ 運行中 | v2.1.2 | 能力註冊系統 |
| **RAG Knowledge Base** | ✅ 運行中 | v2.1.2 | 知識庫系統 |

### 插件整合狀態

**Scanner Plugin**（`services/core/aiva_core/plugins/scanner_plugin.py`）:
```python
✅ 被動掃描器: NetworkScanner
✅ 主動掃描器: VulnerabilityScanner
✅ 功能清單:
   - passive_scan（被動掃描）
   - active_scan（主動掃描）
   - port_scan（端口掃描）
   - service_detection（服務識別）
   - fingerprint（指紋識別）
   - vulnerability_scan（漏洞掃描）
   - network_mapping（網路映射）
```

**Data Hub Plugin**（`services/core/aiva_core/plugins/data_hub_plugin.py`）:
```python
✅ AI Operation Recorder V2
✅ Experience Repository
✅ Unified Data Manager
✅ 功能清單:
   - record_operation（記錄操作）
   - save_experience（保存經驗）
   - query_experiences（查詢經驗）
   - manage_attack_paths（管理攻擊路徑）
   - prepare_training_dataset（準備訓練數據）
   - export_data（導出數據）
   - import_data（導入數據）
```

**AI Commander V2**（`services/core/aiva_core/task_planning/ai_commander_v2.py`）:
```python
✅ ModuleRegistry（模組註冊）
✅ WeightManager（權重管理）
✅ 協調器系統:
   - AttackCoordinator（攻擊協調器）
   - DefenseCoordinator（防禦協調器）
   - AnalysisCoordinator（分析協調器）
   - TrainingCoordinator（訓練協調器）
```

### 遷移階段詳情

**當前階段**: TRANSITION（過渡期）

**特徵**：
- 新舊系統共存
- 功能開關控制路由
- 降級策略確保穩定性

**路由規則**：
```python
routing_rules = {
    'capability_registry': {
        'legacy_path': 'aiva_common.plugins',
        'modern_path': 'aiva_core.plugins.ai_summary_plugin.global_capability_registry',
        'fallback_strategy': 'legacy_first'  # 優先使用舊系統
    },
    'message_broker': {
        'legacy_path': 'aiva_core.messaging.message_router',
        'modern_path': 'aiva_core.messaging.message_broker.enhanced_broker',
        'fallback_strategy': 'modern_first'  # 優先使用新系統
    },
    'risk_control': {
        'legacy_path': 'aiva_core.authz.base_authz',
        'modern_path': 'aiva_core.authz.permission_matrix.RiskGuard',
        'fallback_strategy': 'modern_first'  # 優先使用新系統
    }
}
```

**推進遷移**（管理員操作）：
```python
from services.core.aiva_core import migration_controller

# 推進到下一階段（MODERN）
migration_controller.advance_migration_phase()

# 查看當前狀態
status = migration_controller.get_migration_status()
print(f"當前階段: {status['current_phase']}")
print(f"功能開關: {status['feature_flags']}")
print(f"統計資訊: {status['stats']}")
```

---

# 第四章：故障排除與 API

## 4.1 常見問題排查

### 1. API 服務無法啟動

**症狀**：
```
ERROR: Could not import module 'main'
```

**原因**：缺少依賴或路徑問題

**解決方案**：
```bash
# 1. 檢查依賴
pip install -r requirements.txt

# 2. 確認在正確目錄
cd c:\D\fold7\AIVA-git

# 3. 使用腳本啟動
python scripts/startup/start_ai_service.py --mode api
```

### 2. 資料庫連接失敗

**症狀**：
```
ERROR: could not connect to server: Connection refused
```

**原因**：PostgreSQL 未啟動或配置錯誤

**解決方案**：
```bash
# 1. 檢查 PostgreSQL 服務
# Windows:
services.msc → PostgreSQL

# 2. 檢查 .env 配置
DATABASE_URL=postgresql://user:password@localhost:5432/aiva_db

# 3. 測試連接
psql -U user -d aiva_db
```

### 3. CLI 無法執行攻擊

**症狀**：
```
[Error] Core modules not available
```

**原因**：核心模組導入失敗

**解決方案**：
```bash
# 1. 確認 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:c:\D\fold7\AIVA-git"

# 2. 檢查模組
python -c "from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery; print('OK')"

# 3. 重新安裝核心依賴
pip install -e .
```

### 4. 權重文件未找到

**症狀**：
```
WARNING: BioNeuron model not available
```

**原因**：5M 參數權重文件未配置

**解決方案**：
```bash
# 1. 確認權重目錄
mkdir -p data/weights/bio_neuron

# 2. 配置路徑
# 在 .env 或配置文件中設置
WEIGHTS_BASE_DIR=data/weights

# 3. 檢查權重管理器
python -c "
from services.core.aiva_core.plugin_system.weight_manager import WeightManager
wm = WeightManager()
print(wm.get_weight_status('bio_neuron'))
"
```

### 5. 掃描無結果

**症狀**：
```
✅ 掃描完成
📊 發現 0 個資產
```

**原因**：目標不可達或引擎未啟動

**解決方案**：
```bash
# 1. 測試目標可達性
curl http://localhost:8080

# 2. 檢查引擎狀態（交互模式）
python scripts/startup/start_ai_service.py --mode interactive
AIVA> engines

# 3. 使用快速掃描測試
python scripts/startup/start_ai_service.py --mode interactive
AIVA> scan-fast http://localhost:8080
```

### 6. JWT 認證失敗

**症狀**：
```
401 Unauthorized: Invalid token
```

**原因**：Token 過期或密鑰不匹配

**解決方案**：
```bash
# 1. 重新登入獲取新 Token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "aiva-admin-2025"}'

# 2. 檢查 JWT_SECRET
# 確保 .env 中的 JWT_SECRET 與 API 使用的一致

# 3. 使用新 Token
export TOKEN="eyJ..."
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/capabilities/list
---

## 4.2 API 文檔

## API 文檔

### 認證端點

#### POST /api/v1/auth/login
登入並獲取 JWT Token

**請求**：
```json
{
  "username": "admin",
  "password": "aiva-admin-2025"
}
```

**響應**：
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### POST /api/v1/auth/register
註冊新用戶（需要管理員權限）

**請求**：
```json
{
  "username": "newuser",
  "password": "secure-password",
  "role": "user"
}
```

### 能力查詢端點

#### GET /api/v1/capabilities/list
列出所有能力

**參數**：
- `language` (optional): 過濾語言（python/rust/typescript/go）
- `category` (optional): 過濾類別
- `limit` (optional): 限制數量（預設 100）

**響應**：
```json
{
  "capabilities": [
    {
      "capability_id": "cap_001",
      "capability_name": "network_scanner",
      "language": "python",
      "description": "網路掃描器",
      "invocation_protocol": "unified_caller"
    }
  ],
  "total": 782
}
```

#### POST /api/v1/capabilities/query
查詢相關能力

**請求**：
```json
{
  "query": "SQL 注入",
  "top_k": 5
}
```

**響應**：
```json
{
  "results": [
    {
      "capability_name": "sql_injection_detector",
      "language": "python",
      "confidence": 0.95,
      "description": "SQL 注入檢測器"
    }
  ],
  "query_time": 0.123
}
```

#### POST /api/v1/capabilities/execute
執行指定能力

**請求**：
```json
{
  "capability_id": "cap_001",
  "parameters": {
    "target": "http://localhost:8080",
    "scan_type": "quick"
  }
}
```

**響應**：
```json
{
  "execution_id": "exec_12345",
  "status": "success",
  "result": {
    "assets": 15,
    "vulnerabilities": 3
  },
  "execution_time": 5.67
}
```

#### GET /api/v1/capabilities/stats
獲取能力統計

**響應**：
```json
{
  "total_capabilities": 782,
  "by_language": {
    "python": 495,
    "rust": 123,
    "typescript": 84,
    "go": 80
  },
  "most_used": [
    {"name": "network_scanner", "count": 1234},
    {"name": "vulnerability_scanner", "count": 987}
  ]
}
```

### 安全掃描端點

#### POST /api/v1/security/scan
快速掃描

**請求**：
```json
{
  "target": "http://localhost:8080",
  "strategy": "fast"
}
```

**響應**：
```json
{
  "scan_id": "scan_12345",
  "status": "completed",
  "assets": 15,
  "vulnerabilities": [
    {
      "type": "XSS",
      "severity": "high",
      "location": "/search?q="
    }
  ]
}
```

#### POST /api/v1/security/pentest
完整滲透測試

**請求**：
```json
{
  "target": "http://example.com",
  "scope": ["http://example.com/*"],
  "strategy": "comprehensive"
}
```

### 管理端點

#### GET /api/v1/admin/system-info
獲取系統資訊（需要管理員權限）

**響應**：
```json
{
  "version": "v2.1.2",
  "migration_phase": "TRANSITION",
  "feature_flags": {
    "V1_CAPABILITY_REGISTRY": true,
    "AI_MODULE_ORCHESTRATION": true
  },
  "uptime": "5 days, 3:24:15"
}
```

---

## 開發者指南

### 添加新能力

#### 1. 定義能力

```python
# services/scan/engines/python_engine/my_scanner.py

class MyScanner:
    """我的自訂掃描器"""
    
    def scan(self, target: str, options: dict) -> dict:
        """
        執行掃描
        
        Args:
            target: 目標 URL
            options: 掃描選項
            
        Returns:
            掃描結果字典
        """
        # 實作掃描邏輯
        results = {
            "status": "success",
            "assets": [],
            "vulnerabilities": []
        }
        return results
```

#### 2. 註冊能力

```python
# 自動發現（推薦）
# 能力會在啟動時自動掃描並註冊

# 或手動註冊
from services.core.aiva_core.internal_exploration.capability_registry import CapabilityRegistry

registry = CapabilityRegistry()
await registry.register_capability(
    capability_id="my_scanner",
    capability_name="My Custom Scanner",
    language="python",
    module_path="services.scan.engines.python_engine.my_scanner",
    invocation_metadata={
        "protocol": "unified_caller",
        "import_path": "services.scan.engines.python_engine.my_scanner",
        "class_name": "MyScanner",
        "method_name": "scan",
        "required_params": ["target"],
        "optional_params": ["options"]
    }
)
```

#### 3. 同步到 RAG

```bash
# CLI 方式
python aiva_cli.py --sync

# 或程式化
from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector

connector = InternalLoopConnector()
await connector.sync_to_rag()
```

#### 4. 測試能力

```bash
# 查詢測試
python aiva_cli.py --query "my scanner"

# 執行測試
python aiva_cli.py --attack "使用 my scanner 掃描 http://localhost"
```

### 創建自定義插件

參考文檔：[AI_MODULE_INTEGRATION_QUICKSTART.md](AI_MODULE_INTEGRATION_QUICKSTART.md)

---

## 安全建議

### 1. 生產環境配置

```env
# .env (生產環境)

# 使用強隨機密鑰
JWT_SECRET=$(openssl rand -base64 32)

# 限制 CORS
CORS_ORIGINS=["https://yourdomain.com"]

# 使用 HTTPS
API_HOST=0.0.0.0
API_PORT=443

# 資料庫使用 SSL
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### 2. 更改預設密碼

```python
# 首次啟動後立即更改
POST /api/v1/auth/change-password
{
  "old_password": "aiva-admin-2025",
  "new_password": "your-strong-password"
}
```

### 3. 定期更新

```bash
# 更新依賴
pip install --upgrade -r requirements.txt

# 檢查安全公告
python scripts/check_security.py
```

### 4. 審計日誌

```python
# 啟用詳細日誌
LOG_LEVEL=DEBUG

# 查看審計日誌
tail -f logs/audit.log
```

---

## 效能優化

### 1. 資料庫優化

```sql
-- 創建索引
CREATE INDEX idx_capabilities_language ON capabilities(language);
CREATE INDEX idx_capabilities_name ON capabilities(capability_name);

-- 定期維護
VACUUM ANALYZE capabilities;
```

### 2. 快取配置

```python
# 啟用 RAG 快取
CHROMA_CACHE_ENABLED=true
CHROMA_CACHE_SIZE=1000

# 啟用能力查詢快取
CAPABILITY_CACHE_TTL=3600
```

### 3. 並發設置

```bash
# 增加 Uvicorn workers
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000

# 或在啟動腳本中
python scripts/startup/start_ai_service.py --mode api --workers 4
```

---

## 監控與維護

### 1. 健康檢查

```bash
# API 健康檢查
curl http://localhost:8000/health

# 系統狀態
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/admin/system-info
```

### 2. 日誌監控

```bash
# 即時日誌
tail -f logs/aiva.log

# 錯誤日誌
grep ERROR logs/aiva.log

# 效能日誌
grep "execution_time" logs/aiva.log
```

### 3. 定期備份

```bash
# 備份資料庫
pg_dump aiva_db > backup_$(date +%Y%m%d).sql

# 備份 ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/chroma/

# 備份權重
tar -czf weights_backup_$(date +%Y%m%d).tar.gz data/weights/
```

---

## 總結

本操作手冊經過實際測試驗證，所有啟動方式和功能都已確認可用。

### 核心要點

✅ **系統狀態**：TRANSITION 階段（非 MODERN）  
✅ **核心組件**：17 個組件全部驗證通過  
✅ **能力數量**：782 個（跨 4 種語言）  
✅ **啟動方式**：3 種主要方式，4 種運作模式  
✅ **AI 能力**：支援自然語言下令執行攻擊  

### 推薦使用流程

1. **快速開始**：雙擊 `啟動AI服務.bat`
2. **訪問文檔**：http://localhost:8000/docs
3. **登入系統**：admin / aiva-admin-2025
4. **執行測試**：`python aiva_cli.py --attack "掃描 http://localhost:8080"`
5. **查看結果**：API 響應或 CLI 輸出

### 下一步

- 📖 閱讀 [AI_MODULE_INTEGRATION_QUICKSTART.md](AI_MODULE_INTEGRATION_QUICKSTART.md)
- 🔧 查看 [開發者文檔](docs/)
- 🐛 報告問題到 [GitHub Issues](https://github.com/kyle0527/AIVA/issues)

---

**版本歷史**：
- v1.0 (2025-11-29): 初始版本，經完整測試驗證
- v1.1 (待定): 計劃添加 Docker 部署指南

**維護者**：AIVA 開發團隊  
**最後測試**：2025年11月29日 22:38
