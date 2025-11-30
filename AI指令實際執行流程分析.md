# AIVA AI 指令實際執行流程完整分析

## 📋 目錄

- [數據合約版本說明](#數據合約版本說明)
- [場景 1: AI 下令使用 Rust 掃描模組](#場景-1-ai-下令使用-rust-掃描模組)
- [場景 2: AI 下令使用 XSS 功能模組](#場景-2-ai-下令使用-xss-功能模組)
- [參數調整與重複執行](#參數調整與重複執行)
- [當前架構的關鍵問題](#當前架構的關鍵問題)
- [關鍵概念解釋](#關鍵概念解釋)

---

## 數據合約版本說明

AIVA 系統中存在多個版本的數據合約，分別用於不同的模組和場景：

### 📦 統一命令層 (跨模組通用)

#### 1. AICommand - 統一指令格式
**位置**: `services/aiva_common/schemas/commands.py`
**用途**: AI Core → 各模組的統一命令格式
**支持模組**: Scan, Features, Integration, Core

```python
class AICommand(BaseModel):
    command_id: str              # 唯一命令 ID
    command_type: CommandType    # SCAN_PHASE0, FEATURE_XSS_TEST, etc.
    target_module: str           # "scan", "features", "integration", "core"
    payload: Dict[str, Any]      # 模組特定的 payload（下面詳述）
    priority: CommandPriority    # LOW=1, NORMAL=5, HIGH=8, URGENT=10
    timeout: int = 300           # 超時時間（秒）
    trace_id: Optional[str]      # 追蹤 ID
    session_id: Optional[str]    # 會話 ID
    metadata: Dict[str, Any]     # 額外元數據
```

**關鍵特性**:
- ✅ **payload 是萬用字典**: 可以裝載任何模組特定的數據
- ✅ **支持優先級控制**: priority 參數控制執行順序
- ✅ **支持超時控制**: timeout 參數限制執行時間
- ✅ **支持追蹤**: trace_id 用於關聯多個命令

#### 2. AICommandResult - 統一結果格式
**位置**: `services/aiva_common/schemas/commands.py`
**用途**: 各模組 → AI Core 的統一返回格式

```python
class AICommandResult(BaseModel):
    command_id: str              # 對應的命令 ID
    status: CommandStatus        # COMPLETED, FAILED, TIMEOUT, etc.
    success: bool                # 是否成功
    result: Dict[str, Any]       # 模組特定的結果（下面詳述）
    execution_time: float        # 執行時間（秒）
    error: Optional[str]         # 錯誤訊息
    metrics: Dict[str, Any]      # 性能指標
```

### 📦 Scan 模組專用數據合約

#### 3. Phase0StartPayload - Phase 0 掃描請求
**位置**: `services/aiva_common/schemas/testing/tasks.py`
**用途**: 包裝在 `AICommand.payload` 中，用於 Phase 0 掃描

```python
class Phase0StartPayload(BaseModel):
    scan_id: str                     # 掃描 ID（必須 scan_ 前綴）
    targets: List[HttpUrl]           # 目標 URL 列表
    max_depth: int = 1               # 掃描深度
    timeout: int = 300               # 超時時間
    scan_profile: str = "fast"       # 掃描配置（fast/balanced/deep）
    exclude_patterns: List[str] = [] # 排除模式
```

**實際使用**:
```python
AICommand(
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload=Phase0StartPayload(
        scan_id="scan_001",
        targets=["http://target.com"],
        max_depth=3
    ).model_dump()  # ← 轉換為字典裝入 payload
)
```

#### 4. Phase0CompletedPayload - Phase 0 掃描結果
**位置**: `services/aiva_common/schemas/testing/tasks.py`
**用途**: 包裝在 `AICommandResult.result` 中

```python
class Phase0CompletedPayload(BaseModel):
    scan_id: str
    assets: List[Asset]              # 發現的資產
    summary: ScanSummary             # 掃描摘要
    recommendations: Dict[str, Any]  # AI 建議
    metadata: Dict[str, Any]         # 元數據
```

#### 5. Phase1StartPayload - Phase 1 掃描請求
**位置**: `services/aiva_common/schemas/testing/tasks.py`

```python
class Phase1StartPayload(BaseModel):
    scan_id: str
    targets: List[HttpUrl]
    selected_engines: List[str]      # ["python", "typescript", "rust", "go"]
    max_depth: int = 5
    max_urls: int = 1000
    timeout: int = 1800
    phase0_data: Optional[Dict] = None  # Phase 0 的結果（可選）
```

### 📦 Features 模組專用數據合約

#### 6. FunctionTaskPayload - 功能測試任務
**位置**: `services/aiva_common/schemas/testing/tasks.py`
**用途**: 包裝在 `AICommand.payload` 中，用於 XSS/SQLi/SSRF 等測試

```python
class FunctionTaskPayload(BaseModel):
    task_id: str                      # 任務 ID（必須 task_ 前綴）
    scan_id: str                      # 關聯的掃描 ID
    priority: int = 5                 # 優先級 1-10
    target: FunctionTaskTarget        # 目標詳情（下面詳述）
    context: FunctionTaskContext      # 上下文信息
    strategy: str = "full"            # 測試策略（fast/full/deep）
    custom_payloads: List[str] = []   # 自定義 payloads
    test_config: FunctionTaskTestConfig  # 測試配置（下面詳述）
```

#### 7. FunctionTaskTarget - 測試目標
**位置**: `services/aiva_common/schemas/testing/tasks.py`

```python
class FunctionTaskTarget(BaseModel):
    url: HttpUrl                      # 目標 URL
    method: str = "GET"               # HTTP 方法
    parameter: str                    # 測試參數名稱
    parameter_location: str = "query" # 參數位置（query/body/header/cookie）
    headers: Dict[str, str] = {}      # 自定義 headers
    cookies: Dict[str, str] = {}      # 自定義 cookies
    body: Optional[str] = None        # POST body
    auth: Optional[Dict] = None       # 認證信息
```

#### 8. FunctionTaskTestConfig - 測試配置
**位置**: `services/aiva_common/schemas/testing/tasks.py`

```python
class FunctionTaskTestConfig(BaseModel):
    payloads: List[str] = ["basic"]   # Payload 類型
    custom_payloads: List[str] = []   # 自定義 payloads
    blind_xss: bool = False           # 是否測試 Blind XSS
    dom_testing: bool = False         # 是否測試 DOM XSS
    timeout: Optional[float] = None   # 單個請求超時
    max_retries: int = 3              # ⭐ 重試次數（控制執行次數）
    delay_between_requests: float = 0 # ⭐ 請求間延遲（秒）
```

**關鍵特性**:
- ✅ **支持自定義 payloads**: `custom_payloads` 參數
- ✅ **支持重試控制**: `max_retries` 控制執行次數
- ✅ **支持延遲控制**: `delay_between_requests` 控制請求速度
- ✅ **支持多種測試模式**: blind_xss, dom_testing

### 📦 Features 模組專用結果合約

#### 9. FeatureResult - 功能測試結果
**位置**: `services/features/base/result_schema.py`
**用途**: 包裝在 `AICommandResult.result` 中

```python
class FeatureResult(BaseModel):
    feature_name: str                 # 功能模組名稱（"xss_detector"）
    task_id: str                      # 任務 ID
    status: FeatureExecutionStatus    # SUCCESS/FAILURE/TIMEOUT
    execution_time: float             # 執行時間（秒）
    findings: List[Finding]           # 發現的漏洞列表（下面詳述）
    statistics: Dict[str, Any]        # 統計信息
    error_message: Optional[str]      # 錯誤訊息
```

#### 10. Finding - 漏洞發現
**位置**: `services/features/base/result_schema.py`

```python
class Finding(BaseModel):
    finding_id: str                   # 發現的唯一 ID
    vulnerability_type: str           # 漏洞類型（"xss", "sqli"）
    severity: FindingSeverity         # CRITICAL/HIGH/MEDIUM/LOW/INFO
    confidence: FindingConfidence     # CONFIRMED/HIGH/MEDIUM/LOW
    title: str                        # 標題
    description: str                  # 詳細描述
    affected_url: str                 # 受影響的 URL
    affected_parameter: Optional[str] # 受影響的參數
    payload: Optional[str]            # 使用的 payload
    evidence: Dict[str, Any]          # 證據（響應內容等）
    remediation: Optional[str]        # 修復建議
    references: List[str] = []        # 參考鏈接
    timestamp: datetime               # 發現時間
```

### 📊 數據合約使用流程圖

```
AI Core 發起命令
    ↓
AICommand {
    command_type: SCAN_PHASE0 或 FEATURE_XSS_TEST
    payload: {
        # Scan 模組: Phase0StartPayload 的字典形式
        # Features 模組: FunctionTaskPayload 的字典形式
    }
}
    ↓
Command Center 路由
    ↓
目標模組解析 payload
    ↓ (Scan 模組)
Phase0StartPayload.model_validate(command.payload)
    ↓ (Features 模組)
FunctionTaskPayload.model_validate(command.payload)
    ↓
模組執行任務
    ↓
封裝結果
    ↓ (Scan 模組)
Phase0CompletedPayload → AICommandResult.result
    ↓ (Features 模組)
FeatureResult → AICommandResult.result
    ↓
AICommandResult 返回給 AI Core
```

### 🔄 版本總結

| 層級 | 數據合約 | 用途 | 支持的控制參數 |
|------|---------|------|--------------|
| **統一層** | AICommand | 跨模組命令格式 | priority, timeout, trace_id |
| **統一層** | AICommandResult | 跨模組結果格式 | status, success, metrics |
| **Scan 層** | Phase0StartPayload | Phase 0 掃描請求 | max_depth, timeout, scan_profile |
| **Scan 層** | Phase1StartPayload | Phase 1 掃描請求 | selected_engines, max_depth, max_urls |
| **Features 層** | FunctionTaskPayload | 功能測試請求 | strategy, custom_payloads, test_config |
| **Features 層** | FunctionTaskTestConfig | 測試配置 | ⭐ max_retries, delay_between_requests, timeout |
| **Features 層** | FeatureResult | 功能測試結果 | findings, statistics |

**重點**: 
- ✅ **AICommand.payload 是萬用容器**，可裝載任何模組特定的 Payload
- ✅ **支持參數調整**: 通過 `FunctionTaskTestConfig` 控制重試次數、延遲等
- ✅ **支持自定義 payloads**: 通過 `custom_payloads` 參數傳遞

---

## 場景 1: AI 下令使用 Rust 掃描模組

### 🎯 目標
對靶場 `http://target.com` 執行 Rust 掃描引擎的快速偵察

### 📊 完整執行流程

#### 步驟 1: AI Core 發起命令

```python
# 位置: services/core/aiva_core/task_planning/ai_commander_v2.py

# AI Commander V2 接收到外部請求
await commander.execute_task(
    task_description="掃描 target.com",
    parameters={
        "targets": ["http://target.com"],
        "use_engine": "rust"
    },
    domain=TaskDomain.ANALYSIS  # 自動識別為分析領域
)
```

**數據結構**:
```python
{
    "task_id": "task_1701234567890",
    "task_description": "掃描 target.com",
    "parameters": {
        "targets": ["http://target.com"],
        "use_engine": "rust"
    },
    "domain": "analysis"
}
```

#### 步驟 2: 協調器分發

```python
# 位置: services/core/aiva_core/task_planning/coordinators/analysis_coordinator.py

# AnalysisCoordinator 接收任務
coordinator = self.coordinators[TaskDomain.ANALYSIS]

coordinator_task = CoordinatorTask(
    task_id="task_1701234567890",
    task_type="analysis",
    description="掃描 target.com",
    parameters={...}
)

result = await coordinator.execute_task(coordinator_task)
```

#### 步驟 3: 協調器調用 ScannerPlugin

```python
# 位置: services/core/aiva_core/plugins/scanner_plugin.py

# ⚠️ 當前問題: ScannerPlugin 實現的是舊接口
# 它無法直接下達 AICommand，而是使用 AITask

plugin = self.module_registry.get_plugin("scanner")

ai_task = AITask(
    task_id="task_1701234567890",
    task_type=AITaskType.SCAN,
    description="掃描 target.com",
    parameters={
        "target": "http://target.com",
        "use_engine": "rust"
    }
)

# ❌ 問題: 這裡無法直接調用 Scan 模組的 command_handler
plugin_result = await plugin.execute_task(ai_task)
```

**當前實現的問題**:
```python
# services/core/aiva_core/plugins/scanner_plugin.py (Line 134-170)

async def execute_task(self, task: AITask) -> AIResult:
    """執行掃描任務"""
    
    # ❌ 問題 1: 沒有使用 AICommand 數據合約
    # ❌ 問題 2: 沒有調用 services/scan/command_handler.py
    # ❌ 問題 3: 使用舊的 NetworkScanner/VulnerabilityScanner
    
    if "passive" in task_lower:
        result_data = await self._passive_scan(task.parameters)
    elif "active" in task_lower:
        result_data = await self._active_scan(task.parameters)  # ← 這裡
    
    # ❌ 問題 4: _active_scan 是備用實現，不會真正掃描
    return AIResult(success=True, data=result_data)
```

#### 步驟 4: ❌ **實際執行失敗點**

**當前代碼執行到這裡**:
```python
# services/core/aiva_core/plugins/scanner_plugin.py (Line 220-250)

async def _active_scan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """主動掃描"""
    
    if self.active_scanner:
        try:
            result = await self.active_scanner.scan(
                parameters.get("target", ""),
                parameters.get("options", {})
            )
            return result
        except Exception as e:
            logger.error(f"Active scan error: {e}")
    
    # ❌ 降級到備用實現（返回假數據）
    return {
        "scan_type": "active",
        "target": parameters.get("target", "unknown"),
        "vulnerabilities": [
            {
                "type": "sql_injection",
                "location": "/api/login",
                "severity": "high",
                "confidence": 0.9
            }
        ],
        "total_vulnerabilities": 1,
        "scan_timestamp": time.time()
    }
```

**❌ 問題根源**: 
- ScannerPlugin 沒有導入 `ScanCommandHandler`
- 沒有使用 `AICommandCenter` 來下達命令
- 直接返回假數據，沒有真正調用 Rust 引擎

---

### ✅ 正確的執行流程（應該如何實現）

#### 修正後的步驟 3-4: 使用 Command Center

```python
# services/core/aiva_core/plugins/scanner_plugin.py (修正版)

from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import AICommand, CommandType

class ScannerPlugin(AIModulePlugin):
    def __init__(self):
        self.command_center = None  # 新增
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        # 初始化命令中心連接
        self.command_center = get_command_center()
        
        # 確保 Scan 模組已註冊
        from services.scan.command_handler import ScanCommandHandler
        if "scan" not in self.command_center._handlers:
            scan_handler = ScanCommandHandler()
            self.command_center.register_module("scan", scan_handler)
        
        self.initialized = True
        return True
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行掃描任務"""
        
        # ✅ 構建 AICommand
        command = AICommand(
            command_id=f"scan_{task.task_id}",
            command_type=CommandType.SCAN_PHASE0,  # 或根據任務決定
            target_module="scan",
            payload={
                "scan_id": f"scan_{task.task_id}",
                "targets": [task.parameters.get("target")],
                "max_depth": task.parameters.get("max_depth", 3),
                "timeout": task.parameters.get("timeout", 300)
            }
        )
        
        # ✅ 通過命令中心下達命令
        command_result = await self.command_center.execute(command)
        
        # ✅ 轉換為 AIResult
        return AIResult(
            success=command_result.success,
            data=command_result.result,
            execution_time=command_result.execution_time,
            error=command_result.error
        )
```

#### 步驟 4: Command Center 路由

```python
# services/aiva_common/command_center.py (Line 147-220)

async def execute(self, command: AICommand) -> AICommandResult:
    """執行命令"""
    
    self.logger.info(
        f"🎯 執行命令: {command.command_id} "
        f"[{command.command_type.value}] → {command.target_module}"
    )
    
    # 1. 獲取處理器
    handler = self._handlers.get(command.target_module)  # "scan"
    
    # 2. 調用處理器
    result = await handler.handle_command(command, context)
    
    return result
```

**傳輸的數據合約**:
```python
AICommand(
    command_id="scan_task_1701234567890",
    command_type=CommandType.SCAN_PHASE0,  # "scan_phase0"
    target_module="scan",
    payload={
        "scan_id": "scan_task_1701234567890",
        "targets": ["http://target.com"],
        "max_depth": 3,
        "timeout": 300
    },
    priority=CommandPriority.NORMAL,  # 5
    timeout=600,
    trace_id=None,
    session_id=None,
    parent_command_id=None,
    callback_url=None
)
```

#### 步驟 5: Scan 模組處理命令

```python
# services/scan/command_handler.py (Line 90-140)

class ScanCommandHandler:
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令"""
        
        self.logger.info(
            f"📥 Scan 模組收到命令: {command.command_id} "
            f"[{command.command_type.value}]"
        )
        
        # 根據命令類型路由
        if command.command_type == CommandType.SCAN_PHASE0:
            return await self._handle_phase0(command, context)
        elif command.command_type == CommandType.SCAN_PHASE1:
            return await self._handle_phase1(command, context)
        # ...
```

**接收到的 payload 解析**:
```python
# services/scan/command_handler.py (Line 150-180)

async def _handle_phase0(
    self,
    command: AICommand,
    context: Optional[CommandContext] = None
) -> AICommandResult:
    """處理 Phase 0 快速偵察"""
    
    # 1. 解析命令負載為數據合約
    phase0_payload = Phase0StartPayload(**command.payload)
    
    # 解析後的結構:
    # Phase0StartPayload(
    #     scan_id="scan_task_1701234567890",
    #     targets=[HttpUrl("http://target.com")],
    #     max_depth=3,
    #     timeout=300
    # )
    
    self.logger.info(
        f"🦀 開始 Phase 0 快速偵察: {phase0_payload.scan_id} "
        f"(目標: {len(phase0_payload.targets)}個)"
    )
    
    # 2. 調用 Rust 引擎
    phase0_result = await self.coordinator.execute_phase0(
        scan_id=phase0_payload.scan_id,
        targets=[str(url) for url in phase0_payload.targets],
        max_depth=phase0_payload.max_depth,
        timeout=phase0_payload.timeout
    )
    
    return AICommandResult(...)
```

#### 步驟 6: MultiEngineCoordinator 調用 Rust Adapter

```python
# services/scan/coordinators/multi_engine_coordinator.py

async def execute_phase0(
    self,
    scan_id: str,
    targets: List[str],
    max_depth: int = 1,
    timeout: int = 300
) -> Phase0CompletedPayload:
    """執行 Phase 0"""
    
    # 調用 Rust 引擎
    rust_result = await self.rust_adapter.scan(
        targets=targets,
        options={
            "mode": "fast",
            "timeout": timeout,
            "max_depth": max_depth
        }
    )
    
    return Phase0CompletedPayload(...)
```

#### 步驟 7: Rust Adapter 執行實際掃描

```python
# services/scan/coordinators/engines/rust_adapter.py (Line 55-150)

async def scan(
    self,
    targets: List[str],
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """執行 Rust 引擎掃描"""
    
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        def _sync_scan_all_targets():
            all_rust_assets = []
            
            for target in targets:  # ["http://target.com"]
                try:
                    # ✅ 實際調用 Rust FFI
                    result = self.rust_scanner.scan_target(target, {
                        "mode": options.get("mode", "fast"),
                        "timeout": options.get("timeout", 10)
                    })
                    
                    # 提取資產
                    if result and result.get("success"):
                        results_data = result["results"]
                        raw_assets = results_data.get("assets", [])
                        all_rust_assets.extend(raw_assets)
                        
                except Exception as e:
                    self.logger.warning(f"Rust 掃描失敗 {target}: {e}")
            
            return all_rust_assets
        
        # ✅ 在線程池中執行（避免阻塞事件循環）
        raw_assets = await loop.run_in_executor(pool, _sync_scan_all_targets)
    
    return {
        "assets": standardized_assets,
        "metadata": {...}
    }
```

**實際發送到 Rust 的數據**:
```python
# Rust FFI 調用
self.rust_scanner.scan_target(
    "http://target.com",  # target: str
    {
        "mode": "fast",    # options: Dict
        "timeout": 10
    }
)
```

**Rust 引擎返回的數據**:
```python
{
    "success": true,
    "results": {
        "assets": [
            {
                "asset_id": "rust_http://target.com_0",
                "type": "url",
                "value": "http://target.com/api/login",
                "parameters": ["username", "password"],
                "has_form": true
            },
            {
                "asset_id": "rust_http://target.com_1",
                "type": "api",
                "value": "http://target.com/api/users",
                "parameters": [],
                "has_form": false
            }
        ],
        "scan_time": 2.5,
        "targets_scanned": 1
    }
}
```

#### 步驟 8: 結果返回鏈

```
Rust Adapter 
  → MultiEngineCoordinator (Phase0CompletedPayload)
    → ScanCommandHandler (AICommandResult)
      → Command Center
        → ScannerPlugin (AIResult)
          → AnalysisCoordinator (CoordinatorResult)
            → AICommanderV2 (Task Result Dict)
```

**最終返回給 AI Core 的數據**:
```python
{
    "success": true,
    "task_id": "task_1701234567890",
    "domain": "analysis",
    "result": {
        "assets": [
            {
                "asset_id": "asset_001",
                "asset_type": "url",
                "url": "http://target.com/api/login",
                "parameters": ["username", "password"],
                "has_form": true,
                "http_methods": ["GET", "POST"],
                "discovered_by": "rust",
                "confidence_score": 1.0
            },
            {
                "asset_id": "asset_002",
                "asset_type": "api_endpoint",
                "url": "http://target.com/api/users",
                "parameters": [],
                "has_form": false,
                "http_methods": ["GET"],
                "discovered_by": "rust",
                "confidence_score": 1.0
            }
        ],
        "summary": {
            "urls_found": 2,
            "forms_found": 1,
            "apis_found": 1,
            "total_assets": 2
        }
    },
    "execution_time": 3.2,
    "subtasks_executed": 1
}
```

---

## 場景 2: AI 下令使用 XSS 功能模組

### 🎯 目標
對靶場 `http://target.com/search?q=test` 執行 XSS 漏洞檢測

### 📊 完整執行流程

#### 步驟 1: AI Core 發起命令

```python
# 位置: services/core/aiva_core/task_planning/ai_commander_v2.py

await commander.execute_task(
    task_description="測試 XSS 漏洞",
    parameters={
        "target": "http://target.com/search?q=test",
        "vulnerability_type": "xss",
        "parameter": "q"
    },
    domain=TaskDomain.ATTACK
)
```

#### 步驟 2: ❌ **當前架構缺失：Features 模組沒有 CommandHandler**

**問題 1: Features 模組沒有統一的命令處理器**

當前狀態:
```bash
services/features/
├── function_xss/
│   ├── worker.py          # ❌ 只處理 RabbitMQ 消息
│   ├── traditional_detector.py  # ✅ 實際檢測邏輯
│   └── dom_xss_detector.py      # ✅ DOM XSS 檢測
├── function_sqli/
│   └── worker.py          # ❌ 只處理 RabbitMQ 消息
└── base/
    └── result_schema.py   # ✅ FeatureResult 數據合約
```

**缺失的組件**:
```python
# ❌ 不存在: services/features/command_handler.py
# ❌ 不存在: services/features/features_invoker.py
```

#### ❌ 當前實際執行路徑（錯誤）

```python
# 1. AttackCoordinator 嘗試調用 ExploiterPlugin
coordinator = self.coordinators[TaskDomain.ATTACK]

# 2. ExploiterPlugin 嘗試直接實例化檢測器
# services/core/aiva_core/plugins/exploiter_plugin.py (Line 113-117)

from services.features.function_xss.traditional_detector import TraditionalXssDetector
from services.aiva_common.schemas.task_schema import FunctionTaskPayload

temp_task = FunctionTaskPayload(target="temp", task_type="xss")
detector = TraditionalXssDetector(task=temp_task, timeout=30.0)

# ❌ 問題: 
# - 沒有使用 AICommand 數據合約
# - 沒有通過 Command Center
# - 直接實例化，無法處理複雜場景
```

---

### ✅ 正確的執行流程（應該如何實現）

#### 第一步：創建 Features Command Handler

```python
# 新建: services/features/command_handler.py

from services.aiva_common.schemas import AICommand, AICommandResult, CommandType
from services.features.base import FeatureResult, FeatureExecutionStatus

class FeaturesCommandHandler:
    """Features 模組命令處理器"""
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理功能測試命令"""
        
        self.logger.info(
            f"📥 Features 模組收到命令: {command.command_id} "
            f"[{command.command_type.value}]"
        )
        
        # 根據命令類型路由
        if command.command_type == CommandType.FEATURE_XSS_TEST:
            return await self._handle_xss_test(command)
        elif command.command_type == CommandType.FEATURE_SQLI_TEST:
            return await self._handle_sqli_test(command)
        # ...
    
    async def _handle_xss_test(
        self,
        command: AICommand
    ) -> AICommandResult:
        """處理 XSS 測試命令"""
        
        # 1. 解析 payload
        xss_payload = XSSTestPayload(**command.payload)
        
        # 2. 構建 FunctionTaskPayload
        task = FunctionTaskPayload(
            task_id=command.command_id,
            target=Target(
                url=xss_payload.target_url,
                parameter=xss_payload.parameter,
                parameter_location=xss_payload.parameter_location,
                method=xss_payload.method,
                headers=xss_payload.headers,
                cookies=xss_payload.cookies
            ),
            task_type="xss",
            options=xss_payload.options
        )
        
        # 3. 調用 XSS 檢測器
        from services.features.function_xss.traditional_detector import TraditionalXssDetector
        from services.features.function_xss.payload_generator import XssPayloadGenerator
        
        # 生成 payloads
        payload_gen = XssPayloadGenerator()
        payloads = payload_gen.generate(
            context=xss_payload.parameter_location,
            advanced=xss_payload.options.get("advanced", False)
        )
        
        # 執行檢測
        detector = TraditionalXssDetector(
            task=task,
            timeout=xss_payload.timeout
        )
        
        xss_results = await detector.execute(payloads)
        
        # 4. 轉換為 FeatureResult
        findings = []
        for xss_result in xss_results:
            finding = Finding(
                finding_id=f"xss_{len(findings)}",
                vulnerability_type="xss",
                severity=FindingSeverity.HIGH,
                confidence=FindingConfidence.HIGH,
                title=f"Reflected XSS in {xss_payload.parameter}",
                description=f"Payload {xss_result.payload} was reflected",
                affected_url=str(xss_result.request.url),
                affected_parameter=xss_payload.parameter,
                payload=xss_result.payload,
                evidence={
                    "response_status": xss_result.response_status,
                    "response_headers": xss_result.response_headers,
                    "response_excerpt": xss_result.response_text[:500]
                }
            )
            findings.append(finding)
        
        feature_result = FeatureResult(
            feature_name="xss_detector",
            task_id=command.command_id,
            status=FeatureExecutionStatus.SUCCESS,
            execution_time=xss_payload.timeout,
            findings=findings,
            statistics={
                "payloads_tested": len(payloads),
                "vulnerabilities_found": len(findings)
            }
        )
        
        # 5. 封裝為 AICommandResult
        return AICommandResult(
            command_id=command.command_id,
            status=CommandStatus.COMPLETED,
            success=True,
            result=feature_result.model_dump(),
            execution_time=xss_payload.timeout
        )
```

#### 修正後的步驟 2-3: 註冊並調用

```python
# 1. 系統啟動時註冊 Features 模組
from services.aiva_common.command_center import get_command_center
from services.features.command_handler import FeaturesCommandHandler

command_center = get_command_center()
features_handler = FeaturesCommandHandler()
command_center.register_module("features", features_handler)

# 2. ExploiterPlugin 通過 Command Center 調用
class ExploiterPlugin(AIModulePlugin):
    async def execute_task(self, task: AITask) -> AIResult:
        """執行攻擊任務"""
        
        # 構建 AICommand
        command = AICommand(
            command_id=f"xss_{task.task_id}",
            command_type=CommandType.FEATURE_XSS_TEST,
            target_module="features",
            payload={
                "target_url": task.parameters.get("target"),
                "parameter": task.parameters.get("parameter", "q"),
                "parameter_location": "query",
                "method": "GET",
                "timeout": 30.0,
                "options": {
                    "advanced": True
                }
            }
        )
        
        # 通過命令中心下達
        command_result = await self.command_center.execute(command)
        
        return AIResult(
            success=command_result.success,
            data=command_result.result
        )
```

#### 步驟 4: XSS 檢測器實際執行

**傳輸到 XSS 檢測器的數據**:
```python
# FunctionTaskPayload 結構
FunctionTaskPayload(
    task_id="xss_task_1701234567890",
    target=Target(
        url="http://target.com/search",
        parameter="q",
        parameter_location="query",
        method="GET",
        headers={},
        cookies={},
        body=None
    ),
    task_type="xss",
    options={
        "advanced": True
    }
)
```

**XSS Payload 生成**:
```python
# services/features/function_xss/payload_generator.py

payloads = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(1)",
    "<svg onload=alert(1)>",
    # ... 更多 payloads
]
```

**實際 HTTP 請求**:
```python
# services/features/function_xss/traditional_detector.py (Line 60-95)

async def execute(self, payloads: Sequence[str]) -> list[XssDetectionResult]:
    """執行 XSS 檢測"""
    
    results = []
    client = httpx.AsyncClient(follow_redirects=True, timeout=self._timeout)
    
    for payload in payloads:
        # 構建請求
        method = self._task.target.method or "GET"
        url = self._inject_payload_to_url(
            str(self._task.target.url), 
            self._task.target.parameter, 
            payload
        )
        
        # ✅ 實際發送 HTTP 請求
        response = await client.request(
            method=method,
            url=url,  # http://target.com/search?q=<script>alert(1)</script>
            headers=self._task.target.headers,
            cookies=self._task.target.cookies
        )
        
        # 檢查 payload 是否反射在響應中
        body_text = response.text
        if _payload_in_response(payload, body_text):
            results.append(XssDetectionResult(
                payload=payload,
                request=response.request,
                response_status=response.status_code,
                response_headers=dict(response.headers),
                response_text=body_text
            ))
    
    return results
```

**實際發送的 HTTP 請求示例**:
```http
GET /search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E HTTP/1.1
Host: target.com
User-Agent: python-httpx/0.24.0
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive
```

**檢測邏輯**:
```python
def _payload_in_response(payload: str, response_text: str) -> bool:
    """檢查 payload 是否反射"""
    
    # 解碼各種編碼形式
    decoded_payload = unquote_plus(payload)
    unescaped_payload = unescape(decoded_payload)
    
    # 檢查原始和解碼形式
    return (
        payload in response_text or
        decoded_payload in response_text or
        unescaped_payload in response_text
    )
```

**靶場響應示例（存在 XSS）**:
```html
HTTP/1.1 200 OK
Content-Type: text/html

<html>
<body>
    <h1>搜尋結果</h1>
    <p>您搜尋的關鍵字: <script>alert(1)</script></p>
    <!-- ↑ payload 未經過濾，直接反射 -->
</body>
</html>
```

#### 步驟 5: 結果返回

**XSS 檢測結果**:
```python
XssDetectionResult(
    payload="<script>alert(1)</script>",
    request=<Request('GET', 'http://target.com/search?q=...')>,
    response_status=200,
    response_headers={
        "Content-Type": "text/html",
        "Content-Length": "1234"
    },
    response_text="<html>...<script>alert(1)</script>...</html>"
)
```

**轉換為 FeatureResult**:
```python
FeatureResult(
    feature_name="xss_detector",
    task_id="xss_task_1701234567890",
    status=FeatureExecutionStatus.SUCCESS,
    execution_time=15.3,
    findings=[
        Finding(
            finding_id="xss_0",
            vulnerability_type="xss",
            severity=FindingSeverity.HIGH,
            confidence=FindingConfidence.HIGH,
            title="Reflected XSS in parameter 'q'",
            description="The payload '<script>alert(1)</script>' was reflected without sanitization",
            affected_url="http://target.com/search?q=<script>alert(1)</script>",
            affected_parameter="q",
            payload="<script>alert(1)</script>",
            evidence={
                "response_status": 200,
                "response_headers": {...},
                "response_excerpt": "<html>...<script>alert(1)</script>..."
            },
            remediation="Implement proper output encoding and CSP headers",
            timestamp="2024-11-30T12:34:56.789Z"
        )
    ],
    statistics={
        "payloads_tested": 50,
        "vulnerabilities_found": 1,
        "execution_time_per_payload": 0.306
    }
)
```

**最終返回給 AI Core**:
```python
{
    "success": true,
    "task_id": "task_1701234567890",
    "domain": "attack",
    "result": {
        "feature_name": "xss_detector",
        "status": "success",
        "findings": [
            {
                "finding_id": "xss_0",
                "vulnerability_type": "xss",
                "severity": "high",
                "confidence": "high",
                "title": "Reflected XSS in parameter 'q'",
                "affected_url": "http://target.com/search?q=<script>alert(1)</script>",
                "payload": "<script>alert(1)</script>"
            }
        ],
        "statistics": {
            "payloads_tested": 50,
            "vulnerabilities_found": 1
        }
    },
    "execution_time": 15.3
}
```

---

## 當前架構的關鍵問題

### ❌ 問題 1: ScannerPlugin 沒有使用 AICommand

**位置**: `services/core/aiva_core/plugins/scanner_plugin.py`

**現狀**:
```python
# ❌ 使用舊接口
async def execute_task(self, task: AITask) -> AIResult:
    result_data = await self._active_scan(task.parameters)
    return AIResult(success=True, data=result_data)
```

**應改為**:
```python
# ✅ 使用 Command Center
async def execute_task(self, task: AITask) -> AIResult:
    command = AICommand(
        command_id=f"scan_{task.task_id}",
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={...}
    )
    result = await self.command_center.execute(command)
    return AIResult(success=result.success, data=result.result)
```

### ❌ 問題 2: Features 模組沒有 CommandHandler

**缺失文件**: 
- `services/features/command_handler.py`
- `services/features/features_invoker.py`

**當前狀態**:
- XSS/SQLi/SSRF 等模組只有 `worker.py`（RabbitMQ 消費者）
- 無法通過 AICommand 調用
- ExploiterPlugin 直接實例化檢測器（繞過架構）

**需要創建**:
```python
# services/features/command_handler.py
class FeaturesCommandHandler:
    async def handle_command(self, command: AICommand) -> AICommandResult:
        if command.command_type == CommandType.FEATURE_XSS_TEST:
            return await self._handle_xss_test(command)
        # ...
```

### ❌ 問題 3: ExploiterPlugin 繞過架構

**位置**: `services/core/aiva_core/plugins/exploiter_plugin.py` (Line 113-117)

**現狀**:
```python
# ❌ 直接實例化，繞過 Command Center
from services.features.function_xss.traditional_detector import TraditionalXssDetector
detector = TraditionalXssDetector(task=temp_task, timeout=30.0)
```

**應改為**:
```python
# ✅ 通過 Command Center 調用
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features",
    payload={...}
)
result = await self.command_center.execute(command)
```

### ❌ 問題 4: 數據合約不統一

**當前狀態**:
- Scan 模組使用: `Phase0StartPayload`, `Phase1StartPayload`
- Features 模組使用: `FunctionTaskPayload`
- Core 模組使用: `AITask`
- 三者之間沒有統一轉換邏輯

**需要統一**:
- 所有模組都通過 `AICommand` 接收命令
- 各模組內部的 payload 格式可以不同，但都封裝在 `AICommand.payload` 中
- 返回都使用 `AICommandResult`

### ✅ 正確的數據流

```
AI Core (AITask)
  ↓ 轉換為 AICommand
Command Center
  ↓ 路由到對應模組
Scan/Features CommandHandler
  ↓ 解析 payload
模組內部執行 (Phase0StartPayload / FunctionTaskPayload)
  ↓ 實際操作（HTTP請求、FFI調用）
返回結果 (FeatureResult / Phase0CompletedPayload)
  ↓ 封裝為 AICommandResult
Command Center
  ↓ 返回
AI Core (AIResult)
```

---

## 總結

### 場景 1 (Rust 掃描) 的實際傳輸內容

```python
# AI Core → Command Center
AICommand(
    command_id="scan_task_xxx",
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={
        "scan_id": "scan_task_xxx",
        "targets": ["http://target.com"],
        "max_depth": 3,
        "timeout": 300
    }
)

# Scan CommandHandler → Rust Adapter
{
    "targets": ["http://target.com"],
    "options": {
        "mode": "fast",
        "timeout": 10,
        "max_depth": 3
    }
}

# Rust FFI 實際調用
rust_scanner.scan_target("http://target.com", {"mode": "fast", "timeout": 10})

# Rust 返回
{
    "success": true,
    "results": {
        "assets": [
            {"asset_id": "...", "type": "url", "value": "http://target.com/api/login", ...}
        ]
    }
}

# 最終返回 AI Core
{
    "success": true,
    "task_id": "task_xxx",
    "result": {
        "assets": [...],
        "summary": {"urls_found": 2, "forms_found": 1}
    }
}
```

### 場景 2 (XSS 測試) 的實際傳輸內容

```python
# AI Core → Command Center
AICommand(
    command_id="xss_task_xxx",
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features",
    payload={
        "target_url": "http://target.com/search",
        "parameter": "q",
        "parameter_location": "query",
        "method": "GET",
        "timeout": 30.0
    }
)

# Features CommandHandler → XSS Detector
FunctionTaskPayload(
    task_id="xss_task_xxx",
    target=Target(
        url="http://target.com/search",
        parameter="q",
        parameter_location="query",
        method="GET"
    ),
    task_type="xss"
)

# XSS Detector → 靶場（實際 HTTP 請求）
GET /search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E HTTP/1.1
Host: target.com

# 靶場響應
HTTP/1.1 200 OK
<html>...<script>alert(1)</script>...</html>

# 最終返回 AI Core
{
    "success": true,
    "result": {
        "feature_name": "xss_detector",
        "findings": [
            {
                "vulnerability_type": "xss",
                "severity": "high",
                "payload": "<script>alert(1)</script>",
                "affected_url": "http://target.com/search?q=..."
            }
        ]
    }
}
```

### 核心問題

1. **ScannerPlugin 沒有使用 Command Center**
2. **Features 模組缺少 CommandHandler**
3. **ExploiterPlugin 繞過架構直接實例化**
4. **數據合約轉換邏輯缺失**

這些問題導致當前架構無法按照設計的方式運行，需要補充缺失的組件才能實現完整的指令傳遞鏈路。

---

## 參數調整與重複執行

### 🎯 問題：如何調整能力執行次數和參數？

目前的數據合約 **完全支持** 參數調整和重複執行控制：

### 1. XSS 測試的重複執行控制

```python
# AI Core 下達命令時，可以精確控制測試行為
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features",
    payload={
        "task_id": "task_xss_001",
        "scan_id": "scan_001",
        "target": {
            "url": "http://target.com/search",
            "parameter": "q",
            "parameter_location": "query",
            "method": "GET"
        },
        "test_config": {
            # ⭐ 控制執行次數
            "max_retries": 5,               # 每個 payload 失敗後重試 5 次
            "delay_between_requests": 0.5,  # 每個請求間延遲 0.5 秒
            "timeout": 10.0,                # 單個請求超時 10 秒
            
            # ⭐ 控制測試範圍
            "payloads": ["basic", "advanced", "obfuscated"],  # 使用多種 payload 集
            "custom_payloads": [            # 自定義 payloads
                "<script>alert(document.domain)</script>",
                "<img src=x onerror=alert(1)>",
                "javascript:alert('custom')"
            ],
            
            # ⭐ 控制測試類型
            "blind_xss": True,              # 啟用 Blind XSS 測試
            "dom_testing": True             # 啟用 DOM XSS 測試
        },
        "strategy": "deep"  # fast/full/deep（影響 payload 數量）
    }
)
```

**實際效果**:
```python
# FunctionTaskTestConfig 控制的行為：

# 1. 基礎 payloads: ~20 個
# 2. advanced payloads: +30 個
# 3. obfuscated payloads: +25 個
# 4. custom_payloads: +3 個
# 總計: 78 個 payloads

# 每個 payload:
#   - 最多重試 5 次（如果失敗）
#   - 請求間延遲 0.5 秒
#   - 單次超時 10 秒

# 總執行時間（最壞情況）:
# 78 payloads × (1 + 5 retries) × 10 seconds = 4680 秒 ≈ 78 分鐘
# 實際時間: 78 × 0.5 延遲 + 成功請求時間 ≈ 40-60 秒
```

### 2. SQLi 測試的參數調整

```python
command = AICommand(
    command_type=CommandType.FEATURE_SQLI_TEST,
    target_module="features",
    payload={
        "task_id": "task_sqli_001",
        "scan_id": "scan_001",
        "target": {
            "url": "http://target.com/api/user",
            "parameter": "id",
            "parameter_location": "query",
            "method": "GET"
        },
        "context": {
            # ⭐ 提供上下文幫助測試
            "db_type_hint": "mysql",  # 指定數據庫類型
            "waf_detected": False     # 是否檢測到 WAF
        },
        "test_config": {
            "payloads": ["union", "boolean", "time", "error"],  # 測試多種注入類型
            "custom_payloads": [
                "1' OR '1'='1",
                "1' UNION SELECT NULL--",
                "1' AND SLEEP(5)--"
            ],
            "max_retries": 3,
            "delay_between_requests": 1.0,  # WAF 繞過可能需要更長延遲
            "timeout": 15.0                 # Time-based SQLi 需要更長超時
        }
    }
)
```

### 3. Rust 掃描的參數調整

```python
command = AICommand(
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={
        "scan_id": "scan_001",
        "targets": ["http://target.com"],
        
        # ⭐ 控制掃描深度和範圍
        "max_depth": 5,           # 爬取深度（默認 1）
        "timeout": 600,           # 總超時時間（秒）
        "scan_profile": "deep",   # fast/balanced/deep
        
        # ⭐ 控制掃描行為
        "exclude_patterns": [     # 排除特定路徑
            "/admin/*",
            "/logout",
            "*.pdf"
        ],
        "rate_limit": {           # 速率限制
            "requests_per_second": 10,
            "concurrent_requests": 5
        }
    }
)
```

### 4. 批量執行相同測試

```python
# 如果需要對同一個目標執行多次測試（例如測試穩定性）
commands = []
for i in range(5):  # 執行 5 次
    command = AICommand(
        command_id=f"xss_test_{i}",
        command_type=CommandType.FEATURE_XSS_TEST,
        target_module="features",
        payload={...}  # 相同的測試配置
    )
    commands.append(command)

# 使用批量執行
batch = AICommandBatch(
    batch_id="batch_stability_test",
    commands=commands,
    execution_mode="parallel"  # 或 "sequential"
)

result = await command_center.execute_batch(batch)
```

### 5. 動態調整參數（基於結果）

```python
# 第一次測試：快速掃描
phase0_result = await command_center.execute(AICommand(
    command_type=CommandType.SCAN_PHASE0,
    payload={"max_depth": 1, "scan_profile": "fast"}
))

# 根據結果調整第二次測試
if phase0_result.result.get("summary", {}).get("forms_found", 0) > 10:
    # 發現大量表單，增加深度和超時
    phase1_command = AICommand(
        command_type=CommandType.SCAN_PHASE1,
        payload={
            "max_depth": 10,          # ⭐ 增加深度
            "timeout": 1800,          # ⭐ 增加超時
            "selected_engines": ["python", "typescript"],  # ⭐ 使用多引擎
            "max_urls": 5000          # ⭐ 增加 URL 限制
        }
    )
    phase1_result = await command_center.execute(phase1_command)
```

### 📊 參數傳遞完整性總結

| 控制需求 | 支持的參數 | 位置 | 範例值 |
|---------|-----------|------|--------|
| **執行次數** | `max_retries` | FunctionTaskTestConfig | 0-10 |
| **請求延遲** | `delay_between_requests` | FunctionTaskTestConfig | 0.1-5.0 秒 |
| **單次超時** | `timeout` | FunctionTaskTestConfig | 5-60 秒 |
| **總超時** | `timeout` | AICommand | 60-3600 秒 |
| **Payload 數量** | `payloads`, `custom_payloads` | FunctionTaskTestConfig | 數組 |
| **測試深度** | `strategy` | FunctionTaskPayload | fast/full/deep |
| **掃描深度** | `max_depth` | Phase0StartPayload | 1-20 |
| **並發控制** | `concurrent_requests` | RateLimit | 1-100 |
| **速率限制** | `requests_per_second` | RateLimit | 1-1000 |
| **優先級** | `priority` | AICommand | 1-10 |

### ✅ 結論

**目前的數據合約完全支持**:
1. ✅ 調整能力執行次數（max_retries）
2. ✅ 調整能力參數（timeout, delay, payloads）
3. ✅ 自定義 payloads
4. ✅ 動態調整策略
5. ✅ 批量執行
6. ✅ 條件執行

**唯一需要的是**：確保各模組的 CommandHandler 正確解析這些參數並傳遞給實際執行邏輯。

---

## 關鍵概念解釋

### 🔍 問題 1: "沒有使用 AICommandCenter 來下達命令" 是什麼意思？

**背景**: AIVA v2.0 架構設計了統一的命令中心模式：

```
正確流程:
AI Core → AICommandCenter → 模組 CommandHandler → 實際執行邏輯

錯誤流程（當前）:
AI Core → Plugin → 直接實例化執行邏輯（繞過架構）
```

#### 錯誤示例（當前 ScannerPlugin）

**位置**: `services/core/aiva_core/plugins/scanner_plugin.py` (Line 134-170)

```python
class ScannerPlugin(AIModulePlugin):
    async def execute_task(self, task: AITask) -> AIResult:
        """執行掃描任務"""
        
        # ❌ 問題: 直接調用內部方法，沒有使用 Command Center
        if "active" in task.description.lower():
            result_data = await self._active_scan(task.parameters)
            # ↑ 直接調用，繞過了 AICommandCenter
            # ↑ 沒有經過 ScanCommandHandler
            # ↑ 沒有使用 AICommand 數據合約
        
        return AIResult(success=True, data=result_data)
    
    async def _active_scan(self, parameters: Dict) -> Dict:
        """❌ 這是備用實現，返回假數據"""
        return {
            "scan_type": "active",
            "vulnerabilities": [
                {"type": "sql_injection", "severity": "high"}  # ← 假數據
            ]
        }
```

#### 正確示例（應該如何實現）

```python
class ScannerPlugin(AIModulePlugin):
    def __init__(self):
        self.command_center = None  # 新增
    
    async def initialize(self, config: Dict) -> bool:
        """初始化時獲取 Command Center 引用"""
        from services.aiva_common.command_center import get_command_center
        from services.scan.command_handler import ScanCommandHandler
        
        # ✅ 獲取命令中心
        self.command_center = get_command_center()
        
        # ✅ 確保 Scan 模組已註冊
        if "scan" not in self.command_center._handlers:
            scan_handler = ScanCommandHandler()
            self.command_center.register_module("scan", scan_handler)
        
        self.initialized = True
        return True
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行掃描任務"""
        
        # ✅ 構建標準的 AICommand
        command = AICommand(
            command_id=f"scan_{task.task_id}",
            command_type=CommandType.SCAN_PHASE0,
            target_module="scan",  # ← 告訴 Command Center 路由到 scan 模組
            payload={
                "scan_id": f"scan_{task.task_id}",
                "targets": [task.parameters.get("target")],
                "max_depth": task.parameters.get("max_depth", 3)
            }
        )
        
        # ✅ 通過 Command Center 下達命令
        command_result = await self.command_center.execute(command)
        # ↑ Command Center 會:
        #   1. 查找 "scan" 模組的 handler
        #   2. 調用 ScanCommandHandler.handle_command(command)
        #   3. ScanCommandHandler 解析 payload
        #   4. 調用 Rust/Python/TypeScript 引擎
        #   5. 返回真實的掃描結果
        
        # ✅ 轉換為 AIResult 返回
        return AIResult(
            success=command_result.success,
            data=command_result.result,  # ← 真實的掃描結果
            execution_time=command_result.execution_time
        )
```

**關鍵差異對比**:

| 方面 | 錯誤方式（當前） | 正確方式（應該） |
|------|----------------|----------------|
| **命令格式** | `AITask` (舊接口) | `AICommand` (新接口) |
| **調用路徑** | Plugin → 內部方法 | Plugin → Command Center → Handler |
| **數據合約** | 無標準格式 | `Phase0StartPayload` |
| **結果來源** | 假數據 | 真實引擎執行結果 |
| **可追蹤性** | 無 | 完整的 command_id, trace_id |
| **超時控制** | 無 | AICommand.timeout |
| **優先級** | 無 | AICommand.priority |

### 🔍 問題 2: "直接返回假數據，沒有真正調用 Rust 引擎" 是什麼意思？

**位置**: `services/core/aiva_core/plugins/scanner_plugin.py` (Line 245-280)

```python
async def _active_scan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """主動掃描"""
    
    target = parameters.get("target", "")
    logger.info(f"Executing active scan on: {target}")
    
    if self.active_scanner:
        try:
            # ⚠️ 這裡嘗試使用 active_scanner
            result = await self.active_scanner.scan(target, parameters.get("options", {}))
            return result
        except Exception as e:
            logger.error(f"Active scan error: {e}")
    
    # ❌ 問題: 降級到備用實現（返回假數據）
    vulnerabilities = [
        {
            "type": "XSS",
            "location": "/search?q=",
            "severity": "high",
            "confidence": 0.87,
            "description": "Reflected XSS vulnerability",
            "payload": "<script>alert('XSS')</script>"
        },
        {
            "type": "sql_injection",
            "location": "/api/user?id=",
            "severity": "critical",
            "confidence": 0.92,
            "description": "SQL Injection vulnerability",
            "payload": "1' OR '1'='1"
        },
        {
            "type": "open_redirect",
            "location": "/redirect?url=",
            "severity": "medium",
            "confidence": 0.75,
            "description": "Open redirect vulnerability",
            "payload": "http://evil.com"
        }
    ]
    
    # ❌ 返回的是硬編碼的假數據，不是真實掃描結果
    return {
        "scan_type": "active",
        "target": target,
        "vulnerabilities": vulnerabilities,  # ← 假的！
        "total_vulnerabilities": len(vulnerabilities),
        "scan_timestamp": time.time(),
        "scan_duration": 0.001  # ← 假的執行時間
    }
```

**為什麼會返回假數據？**

1. **active_scanner 沒有正確初始化**:
   ```python
   # Line 90-105
   if config.get("active_enabled", True):
       try:
           from services.scan.engines.python_engine.vulnerability_scanner import VulnerabilityScanner
           self.active_scanner = VulnerabilityScanner()
       except ImportError as e:
           logger.warning(f"Active scanner not available: {e}")
           self.active_scanner = None  # ← 導入失敗，設為 None
   ```

2. **VulnerabilityScanner 可能不存在或路徑錯誤**:
   ```bash
   # 實際上這個文件可能不存在或已改名
   services/scan/engines/python_engine/vulnerability_scanner.py
   ```

3. **即使存在，也不會調用 Rust 引擎**:
   - `VulnerabilityScanner` 是 Python 引擎的掃描器
   - Rust 引擎在 `services/scan/coordinators/engines/rust_adapter.py`
   - 兩者完全不同，沒有關聯

**正確的調用鏈（通過 Command Center）**:

```
ScannerPlugin
    ↓ (構建 AICommand)
Command Center
    ↓ (路由到 scan 模組)
ScanCommandHandler
    ↓ (解析 Phase0StartPayload)
MultiEngineCoordinator
    ↓ (選擇引擎)
RustAdapter
    ↓ (調用 FFI)
Rust 引擎（真實掃描）
    ↓ (HTTP 請求到靶場)
靶場響應
    ↓ (解析結果)
返回真實的資產和漏洞
```

**實際的 Rust 引擎調用**:

```python
# services/scan/coordinators/engines/rust_adapter.py (Line 94-104)

result = self.rust_scanner.scan_target(target, {
    "mode": options.get("mode", "fast"),
    "timeout": options.get("timeout", 10)
})
# ↑ 這會真正調用 Rust FFI
# ↑ Rust 代碼會發送 HTTP 請求到 target
# ↑ 解析響應並返回真實的端點、表單、API 等
```

**真實 vs 假數據對比**:

| 數據來源 | 假數據（當前） | 真實數據（正確流程） |
|---------|--------------|-------------------|
| **漏洞來源** | 硬編碼數組 | 實際掃描發現 |
| **URL** | `/search?q=` (假的) | `http://target.com/api/login` (真實) |
| **參數** | 無 | `["username", "password"]` |
| **HTTP 請求** | 0 個 | 可能數百個 |
| **執行時間** | 0.001 秒 (假的) | 5-60 秒 (真實) |
| **資產數量** | 固定 3 個 | 動態（0-1000+） |
| **證據** | 無 | HTTP 響應、Headers、Body |

### 💡 總結這兩個問題的意義

#### 問題 1: "沒有使用 AICommandCenter"
- **意義**: 架構設計被繞過，無法享受統一命令系統的優勢
- **影響**: 
  - ❌ 無法統一追蹤和監控
  - ❌ 無法使用優先級和超時控制
  - ❌ 無法與其他模組協同工作
  - ❌ 無法使用標準的數據合約

#### 問題 2: "直接返回假數據"
- **意義**: 系統看起來在工作，實際上沒有真正執行任何掃描
- **影響**:
  - ❌ 無法發現真實漏洞
  - ❌ 浪費用戶時間（以為在掃描）
  - ❌ 無法評估系統真實能力
  - ❌ 無法用於實際的滲透測試

**為什麼會這樣設計？**
- 這是 **備用機制（Fallback）**，當無法加載真實掃描器時避免崩潰
- 但應該 **明確告警**，而不是悄悄返回假數據
- 正確做法：返回錯誤，而不是假裝成功

**如何修復？**
1. 修改 ScannerPlugin 使用 Command Center
2. 移除假數據的備用實現，改為返回錯誤
3. 確保 ScanCommandHandler 正確註冊
4. 確保 Rust/Python/TypeScript 引擎可用

這兩個問題是當前架構最核心的缺陷，必須修復才能讓系統真正工作。

---

## 當前架構的關鍵問題
