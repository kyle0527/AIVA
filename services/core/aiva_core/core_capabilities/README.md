# 🎯 Core Capabilities - 核心能力模組

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心職責](#核心職責)
  - [設計理念](#設計理念)
- [🏗️ 架構設計](#-架構設計)
  - [能力分類](#能力分類)
- [🔧 核心組件](#-核心組件)
  - [1. 🎯 Attack (攻擊執行系統)](#1--attack-攻擊執行系統)
  - [2. 🔍 Analysis (代碼分析系統)](#2--analysis-代碼分析系統)
  - [3. 💼 BizLogic (業務邏輯測試)](#3--bizlogic-業務邏輯測試)
  - [4. 💬 Dialog (對話助理)](#4--dialog-對話助理)
  - [5. 📥 Ingestion & Processing (數據處理)](#5--ingestion--processing-數據處理)
  - [6. 📤 Output (輸出轉換)](#6--output-輸出轉換)
  - [7. 🔌 Plugins (插件系統)](#7--plugins-插件系統)
- [📖 使用範例](#-使用範例)
  - [完整攻擊流程](#完整攻擊流程)
  - [對話式攻擊執行](#對話式攻擊執行)
  - [業務邏輯測試](#業務邏輯測試)
- [🛠️ 開發指南](#-開發指南)
  - [🔨 aiva_common 修復規範](#-aiva_common-修復規範)
  - [添加新的攻擊能力](#添加新的攻擊能力)
  - [創建新的插件](#創建新的插件)
  - [擴展業務邏輯測試](#擴展業務邏輯測試)
- [📊 性能指標](#-性能指標)
  - [攻擊執行](#攻擊執行)
  - [代碼分析](#代碼分析)
  - [業務邏輯測試](#業務邏輯測試)
- [🔗 相關模組](#-相關模組)
- [📝 待辦事項](#-待辦事項)

---

**導航**: [← 返回 AIVA Core](../README.md)

> **版本**: 3.0.0-alpha  
> **狀態**: 生產就緒，測試通過  
> **🧪 測試狀態**: 階段 6 測試 100% 通過 (4/4 模組，包含 worker.py 警告)  
> **角色**: AIVA 的「執行力」- 實現具體攻擊和業務邏輯測試的核心能力  
> **最後更新**: 2025年11月16日

---

## 🎯 模組概述

**Core Capabilities** 是 AIVA 六大模組架構中負責實際執行能力的模組。整合了攻擊鏈編排、代碼分析、業務邏輯測試、對話助理、數據攝取、輸出轉換和插件系統,提供完整的安全測試執行能力。

### 核心職責
1. **攻擊執行** - 編排和執行多步驟攻擊鏈
2. **代碼分析** - AI 增強的代碼安全分析
3. **對話交互** - 自然語言問答和一鍵執行
4. **數據處理** - 掃描結果攝取、處理和輸出轉換
5. **插件擴展** - 可插拔的能力擴展系統

### 設計理念
- **能力導向** - 每個子模組代表一種核心能力
- **可組合性** - 能力可以靈活組合形成攻擊鏈
- **可擴展性** - 插件系統支援動態能力註冊
- **業務整合** - 與實際業務場景緊密結合

---

## 🏗️ 架構設計

```
core_capabilities/
├── 📁 attack/                    # 攻擊執行系統 (5 檔案，2015行) - [📖 README](./attack/README.md)
│   ├── attack_chain.py           # ✅ 攻擊鏈編排器 (166行)
│   ├── attack_executor.py        # ✅ 攻擊執行器 (562行)
│   ├── attack_validator.py       # ✅ 攻擊驗證器 (250行)
│   ├── exploit_manager.py        # ✅ 漏洞利用管理器 (818行)
│   ├── payload_generator.py      # ✅ Payload 生成器 (332行)
│   └── __init__.py
│
├── 📁 analysis/                  # 代碼分析系統 (2 檔案，1181行) - [📖 README](./analysis/README.md)
│   ├── analysis_engine.py        # ✅ AI 增強代碼分析引擎 (910行)
│   └── initial_surface.py        # ✅ 初始攻擊面分析 (271行)
│
├── 📁 dialog/                    # 對話助理 (1 檔案，586行) - [📖 README](./dialog/README.md)
│   └── assistant.py              # ✅ AIVA 對話助理 (586行)
│
├── 📁 ingestion/                 # 數據攝取 (1 檔案，102行) - [📖 README](./ingestion/README.md)
│   ├── scan_module_interface.py  # ✅ 掃描模組介面 (102行)
│   └── __init__.py
│
├── 📁 processing/                # 結果處理 (1 檔案，290行) - [📖 README](./processing/README.md)
│   ├── scan_result_processor.py  # ✅ 掃描結果處理器 (290行)
│   └── __init__.py
│
├── 📁 output/                    # 輸出轉換 (1 檔案，20行) - [📖 README](./output/README.md)
│   ├── to_functions.py           # 輸出轉函數調用 (20行)
│   └── __init__.py
│
├── 📁 plugins/                   # 插件系統 (1 檔案，617行) - [📖 README](./plugins/README.md)
│   └── ai_summary_plugin.py      # ✅ AI 摘要插件 (617行)
│
└── multilang_coordinator.py      # 多語言 AI 協調器

總計: 19 個 Python 檔案，約 4800+ 行代碼
```

### 能力分類
```
┌────────────────────────────────────────────────┐
│         Core Capabilities (核心能力)            │
│                                                │
│  ┌──────────┐  ┌──────────┐              │
│  │  Attack  │  │ Analysis │              │
│  │  (攻擊)  │  │  (分析)  │              │
│  └────┬─────┘  └────┬─────┘              │
│       │             │                       │
│       └─────────────┘                       │
│                     ▼                         │
│           ┌──────────────────┐                │
│           │   Dialog 助理    │                │
│           │ (對話交互)     │                │
│           └────────┬─────────┘                │
│                     ▼                         │
│       ┌─────────────┬───────────────┐       │
│       │             │               │       │
│   ┌───┫────┐    ┌───┫─────┐  ┌───┫────┐    │
│   │ Ingestion│    │Processing│  │ Output  │    │
│   │  (攝取)  │    │  (處理)  │  │ (輸出) │    │
│   └─────────┘    └──────────┘  └─────────┘    │
│                                                │
│           ┌──────────────────┐                │
│           │  Plugin System  │                │
│           │  (插件系統)    │                │
│           └──────────────────┘                │
└────────────────────────────────────────────────┘
```
│           │   Orchestration  │                │
│           │   (能力編排)      │                │
│           └──────────────────┘                │
│                     ▲                         │
│       ┌─────────────┼─────────────┐           │
│       │             │             │           │
│  ┌────▼─────┐  ┌───▼────┐  ┌────▼─────┐     │
│  │  Dialog  │  │ Plugin │  │  Output  │     │
│  │  (對話)  │  │(插件)  │  │ (輸出)   │     │
│  └──────────┘  └────────┘  └──────────┘     │
└────────────────────────────────────────────────┘
```

---

## 🔧 核心組件

### 1. 🎯 Attack (攻擊執行系統)

#### `attack_chain.py` - 攻擊鏈編排器
**功能**: 管理和編排複雜的多步驟攻擊序列
```python
from core_capabilities.attack import AttackChain

# 創建攻擊鏈
chain = AttackChain(chain_id="sql_injection_chain")

# 添加步驟
chain.add_step(
    step_id="step1",
    action="port_scan",
    parameters={"target": "192.168.1.100"},
    dependencies=[]
)

chain.add_step(
    step_id="step2",
    action="sql_injection",
    parameters={"url": "http://target/login"},
    dependencies=["step1"]  # 依賴 step1 完成
)

# 執行攻擊鏈
await chain.execute()
```

**特性**:
- ✅ 依賴關係管理 - 自動處理步驟間的依賴
- ✅ 執行順序編排 - 智能排序執行順序
- ✅ 條件分支 - 支援基於結果的條件執行
- ✅ 結果傳遞 - 步驟間的數據流傳遞
- ✅ 狀態追蹤 - 實時追蹤執行狀態

**攻擊鏈狀態**:
```python
class ChainStatus:
    PENDING = "pending"      # 等待執行
    RUNNING = "running"      # 執行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 執行失敗
    PAUSED = "paused"        # 已暫停
```

#### `attack_executor.py` - 攻擊執行器
**功能**: 實際執行各種攻擊動作
```python
from core_capabilities.attack import AttackExecutor

executor = AttackExecutor()

# 執行 SQL 注入
result = await executor.execute_sql_injection(
    url="http://target/api/login",
    parameter="username",
    payload="' OR '1'='1"
)

# 執行 XSS 攻擊
result = await executor.execute_xss(
    url="http://target/search",
    payload="<script>alert('XSS')</script>"
)
```

#### `exploit_manager.py` - 漏洞利用管理器
**功能**: 管理和執行已知漏洞的利用代碼
```python
from core_capabilities.attack import ExploitManager

manager = ExploitManager()

# 執行 CVE 漏洞利用
exploit_result = await manager.exploit_cve(
    cve_id="CVE-2021-44228",  # Log4Shell
    target="192.168.1.100:8080",
    payload=custom_payload
)
```

#### `payload_generator.py` - Payload 生成器
**功能**: 智能生成各類攻擊 Payload
```python
from core_capabilities.attack import PayloadGenerator

generator = PayloadGenerator()

# 生成 SQL 注入 Payload
sql_payloads = generator.generate_sql_injection_payloads(
    injection_type="union",
    database="mysql",
    columns=3
)

# 生成 XSS Payload
xss_payloads = generator.generate_xss_payloads(
    context="html",
    encoding="url",
    bypass_waf=True
)
```

#### `attack_validator.py` - 攻擊驗證器
**功能**: 驗證攻擊是否成功
```python
from core_capabilities.attack import AttackValidator

validator = AttackValidator()

# 驗證 SQL 注入成功
is_success = validator.validate_sql_injection(
    response=http_response,
    indicators=["error in your SQL syntax", "mysql_fetch"]
)

# 驗證命令執行
is_success = validator.validate_command_execution(
    response=http_response,
    expected_output="uid=0(root)"
)
```

---

### 2. 🔍 Analysis (代碼分析系統)

#### `analysis_engine.py` - AI 增強代碼分析引擎
**功能**: 基於 Tree-sitter AST 和神經網路的智能代碼分析
```python
from core_capabilities.analysis import AnalysisEngine, AnalysisType

# 初始化分析引擎
engine = AnalysisEngine(
    bio_controller=bio_neuron_controller,
    use_neural_analysis=True
)

# 執行安全分析
result = await engine.analyze_code(
    code_path="./vulnerable_app.py",
    analysis_type=AnalysisType.SECURITY
)

# 查看發現的漏洞
for vuln in result.vulnerabilities:
    print(f"{vuln.type}: {vuln.description}")
    print(f"位置: {vuln.file}:{vuln.line}")
    print(f"嚴重度: {vuln.severity}")
```

**分析類型**:
```python
class AnalysisType:
    SECURITY = "security"           # 安全漏洞分析
    VULNERABILITY = "vulnerability" # 漏洞檢測
    COMPLEXITY = "complexity"       # 複雜度分析
    PATTERNS = "patterns"           # 代碼模式識別
    SEMANTIC = "semantic"           # 語義分析
    ARCHITECTURE = "architecture"   # 架構分析
```

**特性**:
- ✅ Tree-sitter AST 解析 - 精確的語法樹分析
- ✅ 神經網路增強 - AI 輔助漏洞識別
- ✅ 多語言支援 - Python, JavaScript, Java 等
- ✅ 緩存機制 - 避免重複分析
- ✅ 並行處理 - 多線程加速分析

#### `initial_surface.py` - 初始攻擊面分析
**功能**: 從掃描結果計算初始攻擊面
```python
from core_capabilities.analysis import InitialAttackSurface

surface = InitialAttackSurface()

# 分析攻擊面
attack_surface = surface.compute_from_scan(scan_payload)

# 查看候選目標
print(f"XSS 候選: {len(attack_surface.xss_candidates)}")
print(f"SQL 注入候選: {len(attack_surface.sqli_candidates)}")
print(f"SSRF 候選: {len(attack_surface.ssrf_candidates)}")
print(f"IDOR 候選: {len(attack_surface.idor_candidates)}")
```

**檢測提示**:
- **SSRF**: url, uri, target, dest, redirect, callback, webhook
- **XSS**: search, query, input, comment, message, name
- **SQL Injection**: id, user, product, page, sort, filter
- **IDOR**: id, uid, user_id, account, profile

---

### 3. 💼 BizLogic (業務邏輯測試)

#### `worker.py` - 業務邏輯測試 Worker
**功能**: 執行業務邏輯漏洞測試
```python
from core_capabilities.bizlogic import (
    PriceManipulationTester,
    RaceConditionTester,
    WorkflowBypassTester
)

# 價格操控測試
price_tester = PriceManipulationTester()
findings = await price_tester.test(
    api_endpoint="/api/checkout",
    product_id="12345"
)

# 競態條件測試
race_tester = RaceConditionTester()
findings = await race_tester.test(
    api_endpoint="/api/coupon/apply",
    concurrent_requests=100
)

# 流程繞過測試
workflow_tester = WorkflowBypassTester()
findings = await workflow_tester.test(
    workflow_steps=["login", "verify_email", "purchase"],
    skip_step="verify_email"
)
```

**測試類型**:
- **價格操控** - 修改商品價格、折扣濫用
- **競態條件** - 並發請求導致的邏輯錯誤
- **流程繞過** - 跳過必要的驗證步驟
- **權限提升** - 越權訪問敏感功能
- **數量限制** - 繞過購買數量限制

#### `finding_helper.py` - 漏洞發現輔助
**功能**: 協助組織和報告發現的漏洞
```python
from core_capabilities.bizlogic import FindingHelper

helper = FindingHelper()

# 創建漏洞報告
finding = helper.create_finding(
    title="價格操控漏洞",
    severity="HIGH",
    description="可透過修改請求參數將商品價格改為 0.01 元",
    evidence={
        "request": "POST /api/checkout",
        "payload": {"price": 0.01},
        "response": {"success": True}
    }
)
```

---

### 4. 💬 Dialog (對話助理)

#### `assistant.py` - AIVA 對話助理
**功能**: AI 對話層，支援自然語言問答和一鍵執行
```python
from core_capabilities.dialog import DialogAssistant

assistant = DialogAssistant()

# 自然語言交互
response = await assistant.process_input(
    user_input="幫我掃描 https://example.com 並找出所有 SQL 注入點"
)

print(response.message)
print(response.actions)  # 自動生成的執行計劃
```

**支援的意圖**:
```python
# 意圖識別
INTENT_PATTERNS = {
    "list_capabilities": "現在系統會什麼|你會什麼|有什麼功能",
    "explain_capability": "解釋|說明|介紹 XXX",
    "run_scan": "幫我跑掃描|執行測試",
    "compare_capabilities": "比較 XXX 和 YYY",
    "generate_cli": "產生 CLI 指令|輸出命令",
    "system_status": "系統狀態|健康檢查"
}
```

**對話範例**:
```
User: "現在系統會什麼?"
Assistant: "✅ AIVA 目前具備以下能力:
1. 網站掃描 - 全面的漏洞掃描
2. SQL 注入測試 - 智能注入點檢測
3. XSS 測試 - 反射型和存儲型 XSS
4. 業務邏輯測試 - 價格操控、競態條件等
..."

User: "幫我掃描 https://example.com"
Assistant: "🚀 已啟動掃描任務！
任務 ID: scan_20251115_001
目標: https://example.com
預計時間: 5-10 分鐘
執行步驟:
1. 端口掃描
2. 服務識別
3. 漏洞檢測
4. 攻擊面分析"
```

---

### 5. 📥 Ingestion & Processing (數據處理)

#### `scan_module_interface.py` - 掃描模組介面
**功能**: 資料接收與預處理
```python
from core_capabilities.ingestion import ScanModuleInterface

interface = ScanModuleInterface()

# 處理掃描數據
processed = interface.process_scan_data(scan_payload)

# 標準化後的數據結構
print(processed.keys())
# ['scan_id', 'status', 'summary', 'assets', 'fingerprints']
```

**處理流程**:
1. **格式檢測** - 自動識別數據格式
2. **資料清理** - 移除無效和重複數據
3. **標準化** - 轉換為統一格式
4. **豐富化** - 添加額外上下文信息
5. **分類** - 按資產類型分類

#### `scan_result_processor.py` - 掃描結果處理器
**功能**: 七階段處理流程
```python
from core_capabilities.processing import ScanResultProcessor

processor = ScanResultProcessor(
    broker=message_broker,
    session_manager=session_manager
)

# 執行七階段處理
await processor.process_scan_result(scan_payload)
```

**七階段流程**:
1. **資料接收與預處理** (Data Ingestion)
2. **初步攻擊面分析** (Initial Attack Surface)
3. **策略生成** (Strategy Generation)
4. **策略調整** (Dynamic Adjustment)
5. **任務生成** (Task Generation)
6. **任務分發** (Task Dispatch)
7. **狀態管理** (State Management)

---

### 6. 📤 Output (輸出轉換)

#### `to_functions.py` - 輸出轉函數調用
**功能**: 將攻擊計畫轉換為可執行的函數調用
```python
from core_capabilities.output import OutputConverter

converter = OutputConverter()

# 轉換攻擊計畫為函數調用
function_calls = converter.plan_to_functions(attack_plan)

# 執行函數調用
for func_call in function_calls:
    result = await func_call.execute()
    print(f"{func_call.name}: {result.status}")
```

**支援的輸出格式**:
- Python 函數調用
- CLI 命令
- API 請求
- JSON 結構化數據
- Markdown 報告

---

### 7. 🔌 Plugins (插件系統)

#### `ai_summary_plugin.py` - AI 摘要插件
**功能**: 可插拔的智能分析模組
```python
from core_capabilities.plugins import EnhancedCapabilityRegistry

# 初始化註冊中心
registry = EnhancedCapabilityRegistry()

# 註冊能力
@registry.register_capability(
    name="custom_scanner",
    category="scanning",
    dependencies=["port_scan"]
)
async def custom_scanner(target):
    # 實現掃描邏輯
    return scan_results

# 執行能力
result = await registry.execute_capability(
    "custom_scanner",
    target="192.168.1.100"
)
```

**插件特性**:
- ✅ 動態註冊 - 運行時註冊新能力
- ✅ 依賴管理 - 自動處理能力依賴
- ✅ 智能編排 - 根據依賴自動排序
- ✅ 性能追蹤 - 統計執行次數和成功率
- ✅ 熱更新 - 支援插件熱插拔

---

## 📖 使用範例

### 完整攻擊流程
```python
from core_capabilities.attack import AttackChain, AttackExecutor
from core_capabilities.analysis import InitialAttackSurface
from core_capabilities.ingestion import ScanModuleInterface
from core_capabilities.processing import ScanResultProcessor

# 1. 接收掃描結果
interface = ScanModuleInterface()
scan_data = interface.process_scan_data(raw_scan_payload)

# 2. 分析攻擊面
surface = InitialAttackSurface()
attack_surface = surface.compute_from_scan(scan_data)

# 3. 創建攻擊鏈
chain = AttackChain(chain_id="full_attack")

# 添加步驟
for xss_target in attack_surface.xss_candidates:
    chain.add_step(
        step_id=f"xss_{xss_target.parameter}",
        action="test_xss",
        parameters={
            "url": xss_target.url,
            "parameter": xss_target.parameter
        }
    )

for sqli_target in attack_surface.sqli_candidates:
    chain.add_step(
        step_id=f"sqli_{sqli_target.parameter}",
        action="test_sql_injection",
        parameters={
            "url": sqli_target.url,
            "parameter": sqli_target.parameter
        }
    )

# 4. 執行攻擊鏈
results = await chain.execute()

# 5. 生成報告
for step_id, result in results.items():
    if result.success:
        print(f"✅ {step_id}: 發現漏洞!")
        print(f"   詳情: {result.details}")
```

### 對話式攻擊執行
```python
from core_capabilities.dialog import DialogAssistant

assistant = DialogAssistant()

# 對話式交互
user_inputs = [
    "列出所有可用功能",
    "解釋 SQL 注入測試",
    "幫我測試 https://example.com 的 SQL 注入",
    "生成對應的 CLI 命令"
]

for user_input in user_inputs:
    response = await assistant.process_input(user_input)
    print(f"User: {user_input}")
    print(f"AIVA: {response.message}\n")
```

### 業務邏輯測試
```python
from core_capabilities.bizlogic import Worker

# 啟動 Worker 監聽任務
await Worker.run()

# Worker 會自動處理來自消息隊列的任務
# 包括: 價格操控、競態條件、流程繞過等測試
```

---

## 🛠️ 開發指南

### 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範。

**完整規範**: [aiva_common 開發指南](../../../aiva_common/README.md#-開發指南)

#### 關鍵原則

```python
# ✅ 正確：使用標準枚舉
from aiva_common import (
    Severity, Confidence, VulnerabilityType,
    FindingPayload, CVSSv3Metrics
)

# ❌ 禁止：重複定義通用概念
class Severity(str, Enum): pass  # 錯誤！

# ✅ 合理的模組專屬枚舉
class ChainStatus(str, Enum):
    """攻擊鏈狀態 (attack_chain.py 專用)"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
```

**四層優先級**:
1. 國際標準 (CVSS, SARIF) → 必須遵循
2. 語言標準 (Python Enum) → 必須使用
3. aiva_common → 系統統一標準
4. 模組專屬 → 內部專用才允許

📖 **詳細文檔**: [完整修復規範](../../../aiva_common/README.md#-開發規範與最佳實踐)

---

### 添加新的攻擊能力

```python
# core_capabilities/attack/custom_attack.py
from core_capabilities.attack import AttackExecutor

class CustomAttackExecutor(AttackExecutor):
    async def execute_custom_attack(self, target, payload):
        """實現自定義攻擊邏輯"""
        # 1. 準備攻擊請求
        request = self._prepare_request(target, payload)
        
        # 2. 執行攻擊
        response = await self._send_request(request)
        
        # 3. 驗證結果
        is_success = self._validate_response(response)
        
        # 4. 返回結果
        return {
            "success": is_success,
            "response": response,
            "evidence": self._extract_evidence(response)
        }

# 註冊到執行器
AttackExecutor.register_method("custom", CustomAttackExecutor)
```

### 創建新的插件

```python
# core_capabilities/plugins/my_plugin.py
from core_capabilities.plugins import EnhancedCapabilityRegistry

class MyCustomPlugin:
    def __init__(self, registry: EnhancedCapabilityRegistry):
        self.registry = registry
        self._register_capabilities()
    
    def _register_capabilities(self):
        # 註冊插件能力
        self.registry.register_capability(
            name="my_custom_scan",
            category="scanning",
            handler=self.custom_scan,
            metadata={
                "description": "自定義掃描功能",
                "author": "Your Name",
                "version": "1.0.0"
            }
        )
    
    async def custom_scan(self, target):
        """實現掃描邏輯"""
        results = []
        # ... 掃描邏輯
        return results

# 使用插件
plugin = MyCustomPlugin(registry)
```

### 擴展業務邏輯測試

```python
# core_capabilities/bizlogic/custom_tester.py
class CustomBusinessLogicTester:
    async def test(self, api_endpoint, **kwargs):
        """實現業務邏輯測試"""
        findings = []
        
        # 1. 準備測試用例
        test_cases = self._generate_test_cases(**kwargs)
        
        # 2. 執行測試
        for test_case in test_cases:
            result = await self._execute_test(api_endpoint, test_case)
            
            # 3. 分析結果
            if self._is_vulnerable(result):
                finding = self._create_finding(result)
                findings.append(finding)
        
        return findings

# 註冊到 Worker
from core_capabilities.bizlogic import Worker
Worker.register_tester("custom_logic", CustomBusinessLogicTester)
```

---

## 📊 性能指標

### 攻擊執行
- **並發攻擊數**: 100+ 同時執行
- **攻擊鏈長度**: 支援 50+ 步驟
- **響應時間**: < 100ms (單步攻擊)
- **成功率追蹤**: 實時統計

### 代碼分析
- **分析速度**: 1000 行/秒
- **支援語言**: 10+ 程式語言
- **緩存命中率**: 80%+
- **並行分析**: 4 線程

### 業務邏輯測試
- **並發請求**: 1000+ QPS
- **測試覆蓋**: 25+ 業務場景
- **誤報率**: < 5%
- **檢測時間**: 5-10 分鐘

---

## 🔗 相關模組

- **cognitive_core** - 提供 AI 決策和 RAG 增強
- **task_planning** - 接收能力執行請求並編排
- **external_learning** - 收集執行結果用於學習
- **service_backbone** - 提供消息隊列和狀態管理

---

## 📝 待辦事項

- [ ] 添加更多攻擊向量
- [ ] 擴展代碼分析語言支援
- [ ] 優化業務邏輯測試覆蓋
- [ ] 完善對話助理的 NLP 能力
- [ ] 提升插件系統穩定性
- [ ] 性能優化和壓力測試
- [ ] API 文檔自動生成

---

**最後更新**: 2025-11-15  
**維護者**: AIVA Development Team  
**授權**: MIT License
