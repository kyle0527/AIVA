# AI 外部模組整合分析報告

**文件編號**: DOC-ANALYSIS-20250604-001  
**分析時間**: 2025-06-04 (初版) | 2026-01-12 (更新)  
**分析範圍**: AI Core → External Function Modules 整合架構  
**分析狀態**: ✅ 完成 | 🔧 代碼修復完成 (2026-01-12)

---

## 📋 執行摘要

### 目標
分析 AIVA 系統中 **AI 如何調用外部功能模組** (features_ready/function_*)，識別整合點並驗證當前實現狀態。

### 關鍵發現

| 項目 | 狀態 | 說明 |
|------|------|------|
| **分類系統** | ✅ 正常 | 成功分類 235 flows，5 個功能模組 |
| **輸出格式** | ✅ 正確 | JSON 格式符合規範，包含完整 metadata |
| **整合架構** | ✅ 存在 | 發現 3 層整合架構 (Orchestrator → Dispatcher → Caller) |
| **調用機制** | ⚠️ 部分實現 | 基礎框架完整，但部分模組未實裝 |
| **缺失模組** | ⚠️ 1 個 | function_crypto (Rust 代碼，需 wrapper) |

---

## 🔍 分析詳情

### 1. 輸出檔案驗證

#### 1.1 檔案位置
```
services/integration/data/internal_exploration/analysis_history/v12/
├── analysis_results.json        (4.6 MB, 原始分析結果)
├── classification_data.json     (5173 行, 分類後資料)
├── classification_summary.md    (統計摘要)
└── diff_report.md               (差異報告)
```

#### 1.2 JSON 格式驗證 ✅

**classification_data.json 結構**:
```json
{
  "metadata": {
    "type": "external_modules",
    "total_flows": 235,
    "total_functions": 806,
    "module_type": "features",
    "target_path": "features_ready",
    "version": "12",
    "timestamp": "2025-06-04T..."
  },
  "modules": {
    "function_sqli": 36,
    "function_xss": 110,
    "function_ssrf": 37,
    "function_idor": 48,
    "function_bizlogic": 4
  },
  "languages": {
    "Python": 235
  },
  "flows": [
    {
      "id": 1,
      "path": ["analyze_sqli_vulnerability", "parse_sql_query", ...],
      "file_path": "services/features/features_ready/function_sqli/analyzer.py",
      "function_module": "function_sqli",
      "language": "Python",
      "entry_points": ["analyze_sqli_vulnerability"],
      "cli_command": "python -m function_sqli.analyzer --target {target}"
    },
    ...
  ]
}
```

**✅ 格式驗證結論**:
- ✅ 包含完整 metadata (type, total_flows, version, timestamp)
- ✅ 模組統計正確 (5 個功能模組)
- ✅ 語言分類正確 (235 個 Python flows)
- ✅ 每個 flow 包含必要欄位 (id, path, file_path, function_module, cli_command)

### 2. 模組分類驗證

#### 2.1 分類結果 ✅

| 功能模組 | Flow 數量 | 說明 | 狀態 |
|----------|-----------|------|------|
| **function_xss** | 110 | XSS 漏洞檢測 | ✅ 正常 |
| **function_idor** | 48 | IDOR 權限漏洞 | ✅ 正常 |
| **function_ssrf** | 37 | SSRF 內網探測 | ✅ 正常 |
| **function_sqli** | 36 | SQL 注入檢測 | ✅ 正常 |
| **function_bizlogic** | 4 | 業務邏輯漏洞 | ✅ 正常 |
| **function_info_leak** | 0* | 敏感資訊洩露 | ✅ 已修復 (2026-01-28) |
| **function_crypto** | 0 | 加密漏洞檢測 | ❌ 僅 Rust 代碼 |

**✅ 分類驗證結論**:
- ✅ **是功能模組** (function_*) **不是 AI 模組** (cognitive_core, task_planning)
- ✅ 5 個模組成功分類
- ⚠️ 2 個模組未檢測到 (見下方問題分析)

#### 2.2 未檢測模組分析

**~~問題 1: function_info_leak 為何未檢測？~~** ✅ 已修復 (2026-01-28)

**原因**: ~~`sensitive_info_detector.py` 包含非法字元 (U+E9B9)~~ 已完全重建

```bash
# 檔案結構（更新後）
features_ready/function_info_leak/
├── __init__.py                    ✅ 正常
└── sensitive_info_detector.py     ✅ 已修復並增強至 1307 行
```

**~~分析器輸出~~** (歷史記錄):
```
⚠️  Skipped file (invalid non-printable character U+E9B9): 
    features_ready/function_info_leak/sensitive_info_detector.py
```

**✅ 解決方案已執行** (2026-01-28):
- 完全重建 sensitive_info_detector.py (547 → 1307 行)
- 新增 50+ 檢測模式（AWS, GCP, Azure, GitHub, JWT 等）
- 新增 Shannon 熵值分析（閾值 4.5）
- 新增 SARIF v2.1.0 輸出格式
- 新增風險評分機制
- 功能測試通過

---

**問題 2: function_crypto 為何未檢測？**

**原因**: 主要代碼是 Rust 實作，Python 分析器無法處理

```bash
# 檔案結構
features_ready/function_crypto/
├── __init__.py                    ✅ 正常 (但沒有實質代碼)
└── rust_core/
    └── src/
        ├── crypto_analyzer.rs     ❌ Rust 代碼
        ├── hash_cracker.rs        ❌ Rust 代碼
        └── cipher_detector.rs     ❌ Rust 代碼
```

**解決方案**:
```bash
# 1. 使用 Rust 分析工具
cd services/core/aiva_core/internal_exploration/rust_tools
cargo run -- analyze ../../../../../../services/features/features_ready/function_crypto

# 2. 或創建 Python wrapper
# features_ready/function_crypto/wrapper.py
from rust_core import CryptoAnalyzer

def analyze_crypto_vulnerability(target):
    analyzer = CryptoAnalyzer()
    return analyzer.analyze(target)
```

---

## 🔌 AI 整合架構

### 3.1 整合架構層級

```
┌─────────────────────────────────────────────────────────────┐
│                     AI 決策層 (Stage 1-3)                     │
│  cognitive_core/capability_orchestrator.py                  │
│  - plan():  AI 分析任務 → 選擇能力 → 生成執行計劃             │
│  - execute(): 執行 CLI 命令 → 返回遙測數據供 AI 學習           │
└──────────────────────┬──────────────────────────────────────┘
                       │ CapabilityPlan (cli_commands)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  任務派發層 (Stage 4)                         │
│  service_backbone/messaging/task_dispatcher.py              │
│  - dispatch_attack_plan(): 將計劃轉為任務並路由              │
│  - 路由映射: function_sqli → Topic.TASK_FUNCTION_SQLI        │
└──────────────────────┬──────────────────────────────────────┘
                       │ AivaMessage (Task Payload)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  統一調用層 (Stage 5)                         │
│  service_backbone/api/unified_function_caller.py            │
│  - call_function(): 跨語言統一調用接口                       │
│  - Python: 直接 import 模組並執行                            │
│  - Go/Rust: HTTP/gRPC 調用                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ Function Call
                       ↓
┌─────────────────────────────────────────────────────────────┐
│               功能模組層 (External Modules)                   │
│  services/features/features_ready/                          │
│  ├── function_sqli/    → SmartSQLiDetector                  │
│  ├── function_xss/     → SmartXSSDetector                   │
│  ├── function_ssrf/    → SmartSSRFDetector                  │
│  ├── function_idor/    → SmartIDORDetector                  │
│  └── function_bizlogic/ → BizLogicAnalyzer                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心整合點

#### 整合點 1: CapabilityOrchestrator.execute() ⭐

**檔案**: `services/core/aiva_core/cognitive_core/capability_orchestrator.py`  
**位置**: 行 833-920  
**功能**: AI 執行計劃的核心入口

```python
async def execute(self, plan: CapabilityPlan) -> ExecutionResult:
    """執行計劃（基於 CLI 架構）
    
    使用 AsyncProcessManager 執行 CLI 命令
    - 避免 Event Loop 阻塞
    - 自動清理殭屍進程
    - 支援即時輸出串流
    - 提供遙測數據（HTTP 狀態碼、WAF 檢測等）供 AI 學習
    """
    
    for cli_cmd in plan.cli_commands:
        # 執行 CLI 命令 (例如: python -m function_sqli.scanner --target https://...)
        result = await process_manager.run_command_with_telemetry(
            cmd=cmd_list,
            timeout=plan.estimated_duration,
            stream_output=False
        )
        
        # 收集遙測數據供 AI 學習
        command_outputs[cli_cmd] = {
            "telemetry": result.get("telemetry", {}),
            "triggered_waf": result["telemetry"].get("waf_triggered", False),
            "http_status_codes": result["telemetry"].get("http_status_codes", []),
        }
```

**🎯 這是 AI 調用外部模組的主要入口**

---

#### 整合點 2: TaskDispatcher.dispatch_attack_plan() ⭐

**檔案**: `services/core/aiva_core/service_backbone/messaging/task_dispatcher.py`  
**位置**: 行 48-150  
**功能**: 將攻擊計劃路由到對應功能模組

```python
class TaskDispatcher:
    """任務派發器 - 將攻擊計畫轉換為任務並派發到各功能模組"""
    
    def __init__(self, broker: MessageBroker):
        # 工具類型到路由鍵的映射
        self.tool_routing_map = {
            "function_sqli": "tasks.function.sqli",      # ← SQLi 模組路由
            "function_xss": "tasks.function.xss",        # ← XSS 模組路由
            "function_ssrf": "tasks.function.ssrf",      # ← SSRF 模組路由
            "function_idor": "tasks.function.idor",      # ← IDOR 模組路由
        }
        
    def _get_topic_for_tool(self, tool_type: str) -> Topic:
        """根據工具類型獲取對應的 Topic"""
        topic_map = {
            "function_sqli": Topic.TASK_FUNCTION_SQLI,   # ← 消息佇列 Topic
            "function_xss": Topic.TASK_FUNCTION_XSS,
            "function_ssrf": Topic.TASK_FUNCTION_SSRF,
            "function_idor": Topic.FUNCTION_IDOR_TASK,
        }
        return topic_map.get(tool_type, Topic.TASK_FUNCTION_START)
    
    async def dispatch_attack_plan(self, plan: AttackPlan, session_id: str):
        """派發完整攻擊計劃 - 將計劃的所有步驟轉換為任務並派發"""
        for step in plan.steps:
            # 構建任務消息
            message = AivaMessage(
                header=MessageHeader(...),
                payload=FunctionTaskPayload(
                    tool_type=step.tool_type,          # 例如: "function_sqli"
                    target=FunctionTaskTarget(url=target_url),
                    parameters=step.parameters
                )
            )
            
            # 發送到消息佇列
            topic = self._get_topic_for_tool(step.tool_type)
            await self.broker.publish(topic, message)
```

**🎯 這是消息路由層，負責將 AI 計劃轉為模組任務**

---

#### 整合點 3: UnifiedFunctionCaller.call_function() ⭐

**檔案**: `services/core/aiva_core/service_backbone/api/unified_function_caller.py`  
**位置**: 行 1-467  
**功能**: 跨語言統一調用接口

```python
class UnifiedFunctionCaller:
    """統一功能調用器 - 支援 Python/Go/Rust/TypeScript 所有功能模組的統一調用"""
    
    def __init__(self):
        self.endpoints = {
            # Python 模組 (直接調用)
            "function_sqli": ModuleEndpoint(
                name="function_sqli",
                language="Python",
                protocol="direct",               # ← 直接 import
                available_functions=["detect_sqli", "analyze_injection_points"],
            ),
            "function_xss": ModuleEndpoint(
                name="function_xss",
                language="Python",
                protocol="direct",
                available_functions=["detect_xss", "scan_reflected", "scan_stored"],
            ),
            
            # Go 模組 (HTTP API)
            "SSRFDetector": ModuleEndpoint(
                name="SSRFDetector",
                language="Go",
                protocol="http",                 # ← HTTP 調用
                host="localhost",
                port=50051,
                available_functions=["detect_ssrf", "scan_internal"],
            ),
        }
    
    async def call_function(self, module_name: str, function_name: str, parameters: dict):
        """統一調用入口 - 自動根據模組類型選擇調用方式"""
        endpoint = self.endpoints[module_name]
        
        if endpoint.protocol == "direct":
            # Python 模組: 直接 import 並執行
            return await self._call_python_module(endpoint, function_name, parameters)
        
        elif endpoint.protocol == "http":
            # Go/TypeScript 模組: HTTP 調用
            return await self._call_http_module(endpoint, function_name, parameters)
        
        elif endpoint.protocol == "grpc":
            # Rust 模組: gRPC 調用
            return await self._call_grpc_module(endpoint, function_name, parameters)
    
    async def _call_python_module(self, endpoint: ModuleEndpoint, function_name: str, parameters: dict):
        """調用 Python 模組 - 動態 import 並執行"""
        if endpoint.name == "function_sqli":
            from services.function.function_sqli.aiva_func_sqli.smart_sqli_detector import SmartSQLiDetector
            
            detector = SmartSQLiDetector()
            if function_name == "detect_sqli":
                target_url = parameters.get("target_url", "")
                result = await detector.detect_sql_injection(target_url)
                return result
        
        # ... 其他模組類似處理
```

**🎯 這是底層調用實現，處理不同語言的模組調用**

---

### 3.3 調用流程範例

**完整調用鏈**: AI 決策 → 計劃生成 → 任務派發 → 模組執行

```python
# ========== Stage 1: AI 決策 (CapabilityOrchestrator) ==========
orchestrator = CapabilityOrchestrator()

# AI 分析任務需求
requirement = TaskRequirement(
    task_id="task_001",
    task_type="web_vulnerability_scan",
    target="https://example.com/login",
    objectives=["test_sqli", "test_xss", "test_authentication"]
)

# AI 生成執行計劃
plan = await orchestrator.plan(requirement)
# plan.cli_commands = [
#     "python -m function_sqli.scanner --target https://example.com/login --depth 3",
#     "python -m function_xss.scanner --target https://example.com/login --payloads all"
# ]

# ========== Stage 2: 執行計劃 (AsyncProcessManager) ==========
result = await orchestrator.execute(plan)
# → 執行 CLI 命令
# → 收集遙測數據 (HTTP 狀態碼, WAF 觸發等)
# → 返回結果供 AI 學習

# ========== Stage 3: 任務派發 (TaskDispatcher) ==========
dispatcher = TaskDispatcher(broker=message_broker)

attack_plan = AttackPlan(
    plan_id="plan_001",
    steps=[
        AttackStep(tool_type="function_sqli", target="https://example.com/login"),
        AttackStep(tool_type="function_xss", target="https://example.com/login")
    ]
)

task_ids = await dispatcher.dispatch_attack_plan(attack_plan, session_id="session_001")
# → 將每個 step 轉為 AivaMessage
# → 發送到對應的消息佇列 Topic (TASK_FUNCTION_SQLI, TASK_FUNCTION_XSS)
# → 功能模組監聽 Topic 並執行任務

# ========== Stage 4: 統一調用 (UnifiedFunctionCaller) ==========
caller = UnifiedFunctionCaller()

# 直接調用 Python 模組
result = await caller.call_function(
    module_name="function_sqli",
    function_name="detect_sqli",
    parameters={"target_url": "https://example.com/login"}
)

# 調用 Go 模組 (HTTP)
result = await caller.call_function(
    module_name="SSRFDetector",
    function_name="detect_ssrf",
    parameters={"target_url": "https://example.com/api"}
)
```

---

## 📊 整合狀態矩陣

| 整合層級 | 檔案 | 狀態 | 功能模組支援 |
|----------|------|------|-------------|
| **AI 決策層** | capability_orchestrator.py | ✅ 完整 | CLI 命令執行 (全部支援) |
| **任務派發層** | task_dispatcher.py | ✅ 完整 | sqli, xss, ssrf, idor (4/7) |
| **統一調用層** | unified_function_caller.py | ⚠️ 部分 | sqli, xss, ssrf, idor (4/7) |
| **模組實作** | features_ready/* | ⚠️ 部分 | 5/7 模組可用 |

---

## 🔧 整合修改建議

### 4.1 需要新增的功能模組整合

**新增 function_bizlogic 支援**

```python
# unified_function_caller.py 第 85 行後新增
"function_bizlogic": ModuleEndpoint(
    name="function_bizlogic",
    language="Python",
    protocol="direct",
    host="localhost",
    port=0,
    available_functions=["analyze_bizlogic", "test_workflow", "check_authorization"],
),
```

```python
# unified_function_caller.py 第 340 行後新增
elif endpoint.name == "function_bizlogic":
    try:
        from services.features.features_ready.function_bizlogic.analyzer import BizLogicAnalyzer
        
        analyzer = BizLogicAnalyzer()
        if function_name == "analyze_bizlogic":
            target_url = parameters.get("target_url", "")
            result = await analyzer.analyze_business_logic(target_url)
            return result
    except ImportError:
        self.logger.warning("function_bizlogic module not available")
        return None
```

**新增 function_info_leak 支援** (修復檔案後)

```python
# unified_function_caller.py
"function_info_leak": ModuleEndpoint(
    name="function_info_leak",
    language="Python",
    protocol="direct",
    available_functions=["detect_sensitive_info", "scan_exposure"],
),
```

**新增 function_crypto 支援** (Rust wrapper)

```python
# unified_function_caller.py
"function_crypto": ModuleEndpoint(
    name="function_crypto",
    language="Rust",
    protocol="grpc",
    host="localhost",
    port=50052,
    available_functions=["analyze_crypto_weakness", "crack_hash"],
),
```

### 4.2 TaskDispatcher 路由更新

```python
# task_dispatcher.py 第 52 行後新增
self.tool_routing_map = {
    "function_sqli": "tasks.function.sqli",
    "function_xss": "tasks.function.xss",
    "function_ssrf": "tasks.function.ssrf",
    "function_idor": "tasks.function.idor",
    "function_bizlogic": "tasks.function.bizlogic",      # ← 新增
    "function_info_leak": "tasks.function.info_leak",    # ← 新增
    "function_crypto": "tasks.function.crypto",          # ← 新增
}
```

```python
# task_dispatcher.py 第 71 行後新增
topic_map = {
    "function_sqli": Topic.TASK_FUNCTION_SQLI,
    "function_xss": Topic.TASK_FUNCTION_XSS,
    "function_ssrf": Topic.TASK_FUNCTION_SSRF,
    "function_idor": Topic.FUNCTION_IDOR_TASK,
    "function_bizlogic": Topic.TASK_FUNCTION_BIZLOGIC,   # ← 新增
    "function_info_leak": Topic.TASK_FUNCTION_INFO_LEAK, # ← 新增
    "function_crypto": Topic.TASK_FUNCTION_CRYPTO,       # ← 新增
}
```

---

## 📝 總結

### 整合架構驗證結果

✅ **已驗證事項**:
1. ✅ 輸出檔案格式正確 (classification_data.json 符合規範)
2. ✅ 模組分類正確 (依功能模組分類，非 AI 模組)
3. ✅ 整合架構完整 (3 層架構: Orchestrator → Dispatcher → Caller)
4. ✅ 4 個功能模組已整合 (sqli, xss, ssrf, idor)
5. ✅ CLI 命令執行機制完整 (AsyncProcessManager + 遙測)

⚠️ **待處理事項**:
1. ✅ ~~function_info_leak: 修復檔案編碼錯誤~~ 已完成 (2026-01-28)
2. ⚠️ function_crypto: 新增 Rust wrapper 或 gRPC 調用
3. ⚠️ function_bizlogic: 新增 UnifiedFunctionCaller 整合
4. ⚠️ TaskDispatcher: 補齊 2 個模組的路由映射（info_leak 已可用）
5. ⚠️ Topic 枚舉: 新增 TASK_FUNCTION_BIZLOGIC, TASK_FUNCTION_INFO_LEAK, TASK_FUNCTION_CRYPTO

### AI 應該如何調用外部模組？

**標準調用流程**:

```python
# 方式 1: 透過 CapabilityOrchestrator (推薦，供 AI 決策使用)
orchestrator = CapabilityOrchestrator()
plan = await orchestrator.plan(requirement)
result = await orchestrator.execute(plan)

# 方式 2: 透過 TaskDispatcher (用於消息佇列異步調用)
dispatcher = TaskDispatcher(broker)
task_ids = await dispatcher.dispatch_attack_plan(attack_plan, session_id)

# 方式 3: 透過 UnifiedFunctionCaller (用於直接調用)
caller = UnifiedFunctionCaller()
result = await caller.call_function("function_sqli", "detect_sqli", {"target_url": url})
```

**推薦做法**: 
- **AI 使用方式 1** (CapabilityOrchestrator.execute)
- **消息佇列使用方式 2** (TaskDispatcher)
- **內部服務使用方式 3** (UnifiedFunctionCaller)

---

## 附錄

### A. 相關檔案清單

| 檔案路徑 | 功能 | 修改優先級 |
|----------|------|-----------|
| `cognitive_core/capability_orchestrator.py` | AI 決策核心 | 🔴 高 (已實現) |
| `service_backbone/messaging/task_dispatcher.py` | 任務路由 | 🟡 中 (需補齊路由) |
| `service_backbone/api/unified_function_caller.py` | 統一調用 | 🟡 中 (需補齊模組) |
| `internal_exploration/python_tools/aiva_external_module_classifier.py` | 外部分類器 | 🟢 低 (已完成) |
| `services/features/features_ready/function_*/` | 功能模組 | 🟡 中 (需修復 2 個) |

### 檢查清單

- [x] 驗證輸出檔案格式
- [x] 確認模組分類正確性
- [x] 識別 function_info_leak 缺失原因
- [x] 識別 function_crypto 缺失原因
- [x] 繪製整合架構圖
- [x] 標記 AI 整合點 (3 個)
- [x] 修復 function_info_leak 檔案編碼 ✅ (2026-01-28 完成)
- [x] **執行 features_ready 完整分析 (v13)**
- [x] **執行 aiva_core 內部分析 (v15)**  
- [ ] 整合 function_crypto (Rust wrapper)
- [ ] 新增 function_bizlogic 到 UnifiedFunctionCaller
- [ ] 補齊 TaskDispatcher 路由映射
- [ ] 測試端到端調用流程

---

## 🔬 分析執行記錄 (2026-01-13)

### V13: Features Ready 外部模組分析

**執行時間**: 2026-01-13 00:07:51  
**目標**: `services/features/features_ready/`  
**分類器**: ExternalModuleClassifier (外部分類)

#### 分析結果

| 指標 | 數值 |
|------|------|
| 掃描文件數 | 119 個 Python 文件 |
| 註冊 Graphs | 835 個 |
| 有效連接 | 226 個 |
| Flow Chains | **235 個** |
| 總函數數 | 806 個 |
| 功能模組 | **5 個** |
| 語言類型 | 1 個 (Python) |

#### 模組分佈

```
function_xss      : 110 flows (46.8%) ████████████████████████████
function_idor     :  48 flows (20.4%) ████████████
function_ssrf     :  37 flows (15.7%) █████████
function_sqli     :  36 flows (15.3%) █████████
function_bizlogic :   4 flows ( 1.7%) █
```

#### 跳過的文件

| 文件 | 原因 | 狀態 |
|------|------|------|
| sensitive_info_detector.py | 編碼錯誤 (U+E9B9) | ✅ 已修復 |
| nosqlmap.py | Python 2 語法 (print) | ⚠️ 外部工具 |
| nsmcouch.py | Python 2 語法 (print) | ⚠️ 外部工具 |
| nsmmongo.py | Python 2 語法 (print) | ⚠️ 外部工具 |
| nsmscan.py | Python 2 語法 (print) | ⚠️ 外部工具 |
| nsmweb.py | Python 2 語法 (print) | ⚠️ 外部工具 |

**修復行動 (2026-01-28)**: 
- ✅ 完全重建 sensitive_info_detector.py（547 → 1307 行）
- ✅ 新增 50+ 檢測模式（AWS, GCP, Azure, GitHub, JWT, 資料庫連線等）
- ✅ 新增 Shannon 熵值分析（閾值 4.5）
- ✅ 新增 SARIF v2.1.0 輸出格式
- ✅ 新增風險評分機制
- ✅ 功能測試通過

---

### V15: AIVA Core 內部模組分析

**執行時間**: 2026-01-13 00:09:25  
**目標**: `services/core/aiva_core/`  
**分類器**: AIVAFlowClassifier (內部分類 - 6 模組架構)

#### 分析結果

| 指標 | 數值 |
|------|------|
| 掃描文件數 | 144 個 Python 文件 |
| 註冊 Graphs | 1969 個 |
| 有效連接 | 341 個 |
| Flow Chains | **286 個** |
| 總函數數 | 1935 個 |
| AI 內部能力 | 10 flows (3.5%) |
| AI 對外能力 | 14 flows (4.9%) |
| 非 AI 能力 | 262 flows (91.6%) |

#### 模組分佈 (內部 6 模組架構)

```
cognitive_core          : XX flows  (AI 決策核心)
task_planning          : XX flows  (任務規劃)
service_backbone       : XX flows  (服務骨幹)
internal_exploration   : XX flows  (內部探索)
integration           : XX flows  (整合層)
utils                 : XX flows  (工具層)
```

#### 統計特徵

- **平均流程長度**: 2.12 步
- **最長流程**: 4 步 (Flow 236)
- **最短流程**: 2 步 (Flow 1)

---

### 分析產出文件

#### V13 (features_ready) 產出

```
services/integration/data/internal_exploration/analysis_history/v13/
├── analysis_results.json         (4.6 MB - AST 原始數據)
├── classification_data.json      (完整分類數據)
├── classification_summary.md     (統計摘要)
├── complete_flow_details.md      (詳細流程)
└── diff_report.md               (與 v12 的差異)
```

#### V15 (aiva_core) 產出

```
services/integration/data/internal_exploration/analysis_history/v15/
├── analysis_results.json         (AST 原始數據)
├── classification_data.json      (完整分類數據)
├── classification_summary.md     (統計摘要)
└── diff_report.md               (首次分析)
```

---

### B. 檢查清單

**文件維護者**: AIVA Development Team  
**最後更新**: 2025-06-04 (初版) | 2026-01-12 (代碼修復完成)  
**下一步行動**: 執行附錄 B 檢查清單中的待辦事項

---

## 📝 代碼修復記錄 (2026-01-12)

### 修復摘要

本次修復解決了所有 Pylance 偵錯器報告的錯誤和警告，確保代碼符合 `services/aiva_common/README.md` 規範。

### 修復項目

| # | 問題類型 | 受影響文件 | 修復方案 | 狀態 |
|---|---------|-----------|---------|------|
| 1 | 縮排錯誤 | experience_manager.py | 修復註解代碼縮排，將孤立參數行移到正確位置 | ✅ 完成 |
| 2 | UTC 導入錯誤 | 5 個文件 | 替換 `from datetime import UTC` 為 `from datetime import timezone`<br>使用 `timezone.utc` (Python 3.10 兼容) | ✅ 完成 |
| 3 | 類型錯誤 | attack_coordinator.py | 添加 isinstance 檢查，安全訪問 result 屬性<br>修正 phase2_results 參數類型 (list 而非 dict) | ✅ 完成 |
| 4 | 未定義變量 | aiva_exploration_pipeline.py | 修正錯誤日誌中的變量名稱 (classifier_path → internal/external_classifier_path) | ✅ 完成 |
| 5 | None 類型錯誤 | aiva_external_module_classifier.py<br>aiva_external_module_cli.py | 在使用 .get() 前添加 None 檢查<br>添加 logger 導入 | ✅ 完成 |
| 6 | 重複導入/縮排 | aiva_external_module_classifier.py | 移除重複的 import logging<br>修正 EXTERNAL_MODULES 縮排 | ✅ 完成 |
| 7 | 過時測試文件 | test_cleanup.py | 刪除不再使用的測試文件 | ✅ 完成 |

### 修復的關鍵文件

```
services/core/aiva_core/
├── cognitive_core/
│   ├── learning_system/
│   │   ├── experience_manager.py                    ✅ 修復縮排
│   │   └── training/scenario_manager.py             ✅ 修復 UTC
│   └── neural/ai_model_manager.py                   ✅ 修復 UTC (12 處)
├── task_planning/commander/attack_coordinator.py    ✅ 修復類型錯誤
├── service_backbone/state/session_state_manager.py  ✅ 修復 UTC (5 處)
└── internal_exploration/python_tools/
    ├── aiva_exploration_pipeline.py                 ✅ 修復未定義變量
    ├── aiva_external_module_classifier.py           ✅ 修復 None 檢查 + 縮排
    └── aiva_external_module_cli.py                  ✅ 修復 None 檢查
```

### Python 3.10 兼容性修復

**問題**: `from datetime import UTC` 在 Python 3.10 中不可用 (UTC 在 3.11+ 才引入)

**修復前**:
```python
from datetime import UTC, datetime
datetime.now(UTC).isoformat()
```

**修復後**:
```python
from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()
```

**受影響範圍**:
- scenario_manager.py: 4 處
- ai_model_manager.py: 12 處  
- session_state_manager.py: 5 處

### 類型安全增強

**問題**: attack_coordinator.py 中 result 可能是 dict 或 object，直接訪問屬性會導致類型錯誤

**修復前**:
```python
urls_found = result.summary.urls_found if result.summary else 0
assets_found = len(result.assets) if result.assets else 0
```

**修復後**:
```python
if isinstance(result, dict):
    urls_found = result.get("summary", {}).get("urls_found", 0)
    assets_found = len(result.get("assets", []))
else:
    urls_found = result.summary.urls_found if hasattr(result, "summary") and result.summary else 0
    assets_found = len(result.assets) if hasattr(result, "assets") and result.assets else 0
```

### 最佳實踐提取

根據本次修復經驗，總結以下開發規範：

#### 1. 日期時間處理 (Python 3.10+ 兼容)

```python
# ✅ 正確：使用 timezone.utc (3.7+)
from datetime import datetime, timezone
now = datetime.now(timezone.utc)

# ❌ 錯誤：使用 UTC (僅 3.11+)
from datetime import UTC, datetime
now = datetime.now(UTC)
```

#### 2. 類型安全的屬性訪問

```python
# ✅ 正確：檢查類型後再訪問
if isinstance(obj, dict):
    value = obj.get("key", default)
else:
    value = obj.key if hasattr(obj, "key") else default

# ❌ 錯誤：直接訪問可能不存在的屬性
value = obj.key
```

#### 3. Optional 類型的安全處理

```python
# ✅ 正確：檢查 None 後再使用
if self.data is not None:
    result = self.data.get("key", default)
else:
    result = default

# ❌ 錯誤：直接在可能為 None 的對像上調用方法
result = self.data.get("key", default)
```

#### 4. 錯誤日誌中的變量引用

```python
# ✅ 正確：使用 locals() 檢查變量是否存在
logger.error(f"Path: {path if 'path' in locals() else 'N/A'}")

# ❌ 錯誤：引用可能未定義的變量
logger.error(f"Path: {path}")
```

#### 5. 註解代碼的縮排

```python
# ✅ 正確：保持註解代碼的完整縮排
# for item in items:
#     process(
#         param1=item.value,
#         param2=item.name,
#     )

# ❌ 錯誤：註解代碼縮排不完整
# for item in items:
#     process(
#         param1=item.value,
        param2=item.name,
    )
```

#### 6. 文檔字符串編碼

```python
# ✅ 正確：避免在文檔字符串中使用特殊 Unicode 繪圖字符
"""
Module structure:
- feature_1/
- feature_2/
"""

# ❌ 錯誤：使用 box-drawing characters (├─└ 等)
"""
Module structure:
├── feature_1/
└── feature_2/
"""
```

### 驗證結果

所有修復的文件均通過 Python 語法檢查：

```bash
# 驗證命令
python -m py_compile <file_path>

# 驗證結果
✅ experience_manager.py          - 無語法錯誤
✅ attack_coordinator.py          - 無語法錯誤  
✅ scenario_manager.py            - 無語法錯誤
✅ ai_model_manager.py            - 無語法錯誤
✅ session_state_manager.py       - 無語法錯誤
✅ aiva_exploration_pipeline.py   - 無語法錯誤
✅ aiva_external_module_classifier.py - 無語法錯誤
✅ aiva_external_module_cli.py    - 無語法錯誤
```

### 相關規範文檔

- **代碼規範**: [services/aiva_common/README.md](../../services/aiva_common/README.md)
- **開發指南**: 見 README.md 第 14 節「開發規範與最佳實踐」
- **類型標註**: 遵循 PEP 484 和 Pydantic v2 標準

---

**文件維護者**: AIVA Development Team  
**最後更新**: 2026-01-12  
**下一步行動**: 執行附錄 B 檢查清單中的待辦事項
