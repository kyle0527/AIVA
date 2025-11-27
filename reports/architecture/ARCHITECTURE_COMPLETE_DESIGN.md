# 🏛️ AIVA 完整架構設計與理念

> **建立日期**: 2025-01-21
> **目的**: 為 AI 協作和未來開發建立正確的架構理解基準

---

## 📑 目錄

- [🎯 核心架構理念](#核心架構理念)
  - [1. **職責單一原則** (Single Responsibility)](#1-職責單一原則-single-responsibility)
  - [2. **高度解耦設計** (Loose Coupling)](#2-高度解耦設計-loose-coupling)
  - [3. **並行優先策略** (Concurrency First)](#3-並行優先策略-concurrency-first)
  - [4. **標準化數據流** (Standardized Data Flow)](#4-標準化數據流-standardized-data-flow)
- [🏗️ 六大核心服務](#六大核心服務)
  - [1. **aiva_common** - 共享基礎設施](#1-aivacommon-共享基礎設施)
  - [2. **scan** - 資產發現模組 (Phase 0-1)](#2-scan-資產發現模組-phase-01)
    - [Phase 0 - Rust 快速偵察 (5-10分鐘)](#phase-0-rust-快速偵察-510分鐘)
    - [Phase 1 - 多引擎深度掃描 (10-30分鐘)](#phase-1-多引擎深度掃描-1030分鐘)
  - [3. **features** - 漏洞攻擊測試模組 (Phase 2)](#3-features-漏洞攻擊測試模組-phase-2)
    - [Phase 2 - 功能模組攻擊測試 (5-20分鐘)](#phase-2-功能模組攻擊測試-520分鐘)
  - [4. **integration** - 企業級整合中樞](#4-integration-企業級整合中樞)
  - [5. **core** - AI 驅動核心引擎](#5-core-ai-驅動核心引擎)
- [🔄 三階段掃描流程](#三階段掃描流程)
  - [完整數據流程圖](#完整數據流程圖)
- [📊 數據流與合約](#數據流與合約)
  - [數據合約詳解](#數據合約詳解)
    - [1. Phase 0 → Phase 1](#1-phase-0-phase-1)
    - [2. Phase 1 → Phase 2](#2-phase-1-phase-2)
    - [3. Phase 2 內部](#3-phase-2-內部)
- [🎯 模組職責劃分](#模組職責劃分)
  - [Scan 模組 vs Features 模組](#scan-模組-vs-features-模組)
  - [引擎分類](#引擎分類)
    - [Scan 模組的引擎](#scan-模組的引擎)
    - [Features 模組的 Workers](#features-模組的-workers)
- [🔑 關鍵設計決策](#關鍵設計決策)
  - [1. 為什麼 Python 引擎只提取表單參數?](#1-為什麼-python-引擎只提取表單參數)
  - [2. 為什麼 SSRF 檢測不依賴 Python 引擎?](#2-為什麼-ssrf-檢測不依賴-python-引擎)
  - [3. 為什麼 Phase 1 引擎並行而非串行?](#3-為什麼-phase-1-引擎並行而非串行)
  - [4. 為什麼 Go 引擎在 Phase 1 而非 Phase 2?](#4-為什麼-go-引擎在-phase-1-而非-phase-2)
- [⚠️ 常見誤解澄清](#常見誤解澄清)
  - [誤解 1: "Scan 模組負責漏洞檢測"](#誤解-1-scan-模組負責漏洞檢測)
  - [誤解 2: "Python → Go 串行依賴"](#誤解-2-python-go-串行依賴)
  - [誤解 3: "SSRF 檢測依賴 Python 參數提取"](#誤解-3-ssrf-檢測依賴-python-參數提取)
  - [誤解 4: "Go 引擎在 Phase 2"](#誤解-4-go-引擎在-phase-2)
- [📋 實際運作流程範例](#實際運作流程範例)
  - [場景: 掃描 Juice Shop](#場景-掃描-juice-shop)
- [🎓 設計理念總結](#設計理念總結)
  - [核心原則](#核心原則)
  - [數據流向](#數據流向)
  - [模組協作](#模組協作)
- [📚 延伸閱讀](#延伸閱讀)

---

## 🎯 核心架構理念

AIVA 的設計遵循以下核心原則:

### 1. **職責單一原則** (Single Responsibility)
- **Scan 模組**: 只負責資產發現，不進行漏洞攻擊測試
- **Features 模組**: 只負責漏洞檢測，不做資產爬取
- **Integration 模組**: 負責調度協調、資料庫比對、歷史分析、最終報告產出

**重要**: Integration 模組在整個流程中的關鍵職責：
- ✅ 整合 Phase 0/1/2 的所有結果
- ✅ 與歷史資料庫比對，識別新增/修復的漏洞
- ✅ AI 第 3 次決策（風險評估、優先級排序）
- ✅ 產出最終報告（無論是否找到漏洞都會產出）

### 2. **高度解耦設計** (Loose Coupling)
- 模組間通過**標準數據合約**通信 (aiva_common.schemas)
- 功能模組獨立運作,不依賴其他模組的內部實現
- 引擎間並行執行,無串行依賴關係

### 3. **並行優先策略** (Concurrency First)
- Phase 1 多引擎**真正並行**執行 (asyncio.gather)
- 功能模組在 Phase 2 並行攻擊測試
- 能並行就不串行,最大化效率

### 4. **標準化數據流** (Standardized Data Flow)
```
Phase 0 → Phase0CompletedPayload (端點發現)
Phase 1 → Phase1CompletedPayload (資產列表: Asset[])
Phase 2 → FindingPayload[] (漏洞報告)
```

---

## 🏗️ 六大核心服務

```
services/
├── aiva_common/       # 共享基礎設施
├── core/              # AI 驅動核心引擎
├── scan/              # Phase 0-1 資產發現
├── features/          # Phase 2 漏洞攻擊測試
├── integration/       # 企業級整合中樞
└── (root service)     # 服務管理層
```

### 1. **aiva_common** - 共享基礎設施

**職責**: 提供所有服務共用的基礎組件

**關鍵內容**:
- `schemas.py`: 統一數據格式 (Asset, FunctionTaskPayload, FindingPayload)
- `enums.py`: 枚舉類型 (VulnerabilityType, Severity, ScanStrategy)
- `utils/`: 通用工具函數
- `config.py`: 配置管理

**設計理念**: 
> 所有模組必須使用 aiva_common 定義的標準數據格式,確保系統一致性

---

### 2. **scan** - 資產發現模組 (Phase 0-1)

**職責**: **只負責資產發現**,不進行漏洞攻擊測試

**架構**:
```
scan/
├── coordinators/
│   ├── multi_engine_coordinator.py  # 多引擎協調器
│   └── README.md                    # 協調器說明
├── engines/
│   ├── rust_engine/      # Phase 0 快速偵察
│   ├── python_engine/    # Phase 1 靜態爬取
│   ├── typescript_engine/# Phase 1 動態渲染
│   └── go_engine/        # Phase 1 高並發掃描器
└── README.md             # Scan 模組總覽
```

**工作流程**:

#### Phase 0 - Rust 快速偵察 (5-10分鐘)
```python
async def execute_phase0(self, target: str):
    """Rust 引擎必須執行"""
    result = await rust_engine.scan(target)
    return Phase0CompletedPayload(
        endpoints=result.endpoints,  # 40+ 路徑
        technologies=result.tech_stack,  # Express.js, React
        js_findings=result.js_findings  # JS 中的 API
    )
```

**輸出**: 端點列表、技術棧識別、初步評估

#### Phase 1 - 多引擎深度掃描 (10-30分鐘)

**引擎組合** (並行執行):

| 引擎 | 執行時間 | 職責 | 輸出 |
|------|---------|------|------|
| **Python** | 20-30s | 靜態爬取、表單參數提取 | 150+ 資產 (有表單參數) |
| **TypeScript** | 30-45s | 動態渲染、監聽 AJAX | 50+ 資產 (動態路由、API) |
| **Rust** | 10-20s | 敏感資訊深度掃描 | 密鑰、令牌 |
| **Go** | 5-15s | SSRF/CSPM/SCA 掃描器 | 高並發資產掃描 |

**協調器聚合**:
```python
async def execute_phase1(self, phase0_result):
    # 並行執行選定引擎
    results = await asyncio.gather(
        self._run_python_engine(phase0_result),
        self._run_typescript_engine(phase0_result),
        self._run_rust_engine(phase0_result),
        self._run_go_engine(phase0_result) if strategy == "comprehensive" else None
    )
    
    # 資產聚合去重
    all_assets = self._merge_and_deduplicate(results)
    
    return Phase1CompletedPayload(
        assets=all_assets,  # List[Asset]
        summary=self._generate_summary(all_assets)
    )
```

**輸出**: `Phase1CompletedPayload` 包含完整資產列表

**關鍵數據結構**:
```python
@dataclass
class Asset:
    asset_id: str
    asset_type: str  # "url" | "form" | "api" | "endpoint"
    value: str  # http://example.com/login
    parameters: Optional[List[str]]  # ["username", "password"]
    has_form: bool  # True if 表單資產
    method: str  # "GET" | "POST"
    metadata: Dict  # 額外資訊
```

---

### 3. **features** - 漏洞攻擊測試模組 (Phase 2)

**職責**: **只負責漏洞攻擊測試**,接收 Phase 1 的資產進行攻擊驗證

**架構**:
```
features/
├── function_ssrf/         # SSRF 檢測 (90% 完成)
├── function_sqli/         # SQL 注入檢測 (95% 完成)
├── function_xss/          # XSS 檢測 (90% 完成)
├── function_idor/         # IDOR 檢測 (85% 完成)
├── function_authn_go/     # 認證繞過 (Go)
├── function_crypto/       # 加密漏洞 (Rust)
├── function_postex/       # 後滲透模組
├── function_bizlogic/     # 業務邏輯檢測
├── ... (10+ 功能模組)
└── README.md
```

**工作流程**:

#### Phase 2 - 功能模組攻擊測試 (5-20分鐘)

```python
# 1. Integration 模組分配任務
for asset in phase1_result.assets:
    if asset.has_form and any(param in ["url", "uri", "redirect"] 
                               for param in asset.parameters):
        # 分配給 SSRF 模組
        task = FunctionTaskPayload(
            task_id=uuid4(),
            scan_id=scan_id,
            target=FindingTarget(
                url=asset.value,
                parameter="url",  # 明確指定測試參數
                method=asset.method
            ),
            strategy="aggressive"
        )
        await send_to_worker("function_ssrf", task)

# 2. SSRF Worker 處理任務
async def process_task(task: FunctionTaskPayload) -> TaskExecutionResult:
    """function_ssrf/worker.py"""
    
    # 自己分析參數,不依賴 Scan 模組
    analyzer = ParamSemanticsAnalyzer()
    plan: AnalysisPlan = analyzer.analyze(task)
    
    # 生成 SSRF 測試向量
    for vector in plan.vectors:
        payload = await _resolve_payload(vector, dispatcher, task)
        # 如: http://169.254.169.254/latest/meta-data/
        
        # 發送測試請求
        response = await _issue_request(client, task, vector, payload)
        
        # 檢測內網訪問
        detector = InternalAddressDetector()
        detection = detector.analyze(response)
        
        if detection.matched:
            finding = FindingPayload(
                finding_id=uuid4(),
                vulnerability=Vulnerability(
                    name=VulnerabilityType.SSRF,
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH
                ),
                target=task.target,
                evidence={
                    "payload": payload,
                    "response": response.text,
                    "detection": detection.reason
                },
                recommendation="實施白名單驗證,禁止內網地址訪問"
            )
            findings.append(finding)
    
    return TaskExecutionResult(
        findings=findings,
        telemetry=telemetry
    )
```

**關鍵組件**:
- `worker.py`: 任務處理主邏輯
- `engine/`: 引擎核心 (如 SSRFEngine, SQLiEngine)
- `analyzers/`: 參數分析器、模式識別器
- `detectors/`: 漏洞檢測器、響應分析器

**輸出**: `FindingPayload[]` 漏洞報告列表

---

### 4. **integration** - 企業級整合中樞

**職責**: 調度協調 Scan 和 Features 模組,管理任務分配

**關鍵功能**:
- 接收 Phase 1 的資產列表
- 分析資產類型,決定調用哪些功能模組
- 創建 `FunctionTaskPayload` 分配給 Workers
- 收集 `FindingPayload` 並整合報告

**工作流程**:
```python
async def orchestrate_scan(target: str):
    # 1. Phase 0-1: Scan 模組執行
    phase0_result = await scan_service.execute_phase0(target)
    phase1_result = await scan_service.execute_phase1(phase0_result)
    
    # 2. 資產分析與任務分配
    tasks = []
    for asset in phase1_result.assets:
        # 根據資產類型決定功能模組
        if asset.has_form:
            tasks.append(("function_sqli", asset))
            tasks.append(("function_xss", asset))
        
        if any(param in ["url", "redirect"] for param in asset.parameters):
            tasks.append(("function_ssrf", asset))
        
        if asset.asset_type == "api":
            tasks.append(("function_idor", asset))
    
    # 3. Phase 2: 並行執行功能模組
    findings = await asyncio.gather(*[
        features_service.execute(module, asset) 
        for module, asset in tasks
    ])
    
    # 4. 整合報告
    report = ReportGenerator().generate(
        assets=phase1_result.assets,
        findings=flatten(findings)
    )
    
    return report
```

---

### 5. **core** - AI 驅動核心引擎

**職責**: AI 決策、自我優化、命令生成

**關鍵功能**:
- 決策引擎: 根據 Phase 0 結果選擇引擎組合
- RAG 系統: 查詢歷史數據和知識庫
- 自我優化: 分析掃描結果並改進策略
- AI 命令生成: 自動生成攻擊指令

---

## 🔄 三階段掃描流程

### 完整數據流程圖

```mermaid
graph TB
    Start([用戶發起掃描]) --> Phase0
    
    subgraph Phase0 [Phase 0 - Rust 快速偵察 Scan模組]
        R1[Rust Engine 必須執行]
        R2[技術棧識別: Express.js, React]
        R3[端點發現: 40+ 路徑]
        R4[JS文件分析]
        ROut[輸出: Phase0CompletedPayload<br/>- endpoints List<br/>- technologies List<br/>- js_findings List]
    end
    
    Phase0 --> AI1[AI 核心決策<br/>分析技術棧 + 歷史數據 + RAG]
    
    AI1 --> Phase1Decision{選擇 Phase 1 引擎組合}
    
    Phase1Decision -->|靜態內容優先| PythonE
    Phase1Decision -->|動態渲染需求| TypeScriptE
    Phase1Decision -->|敏感掃描| RustE
    Phase1Decision -->|高並發需求| GoE
    
    subgraph Phase1 [Phase 1 - 深度資產掃描 Scan模組]
        direction TB
        PythonE[Python Engine 20-30s<br/>靜態爬取<br/>表單參數提取]
        TypeScriptE[TypeScript Engine 30-45s<br/>Playwright 動態渲染<br/>監聽 AJAX]
        RustE[Rust Engine 10-20s<br/>敏感資訊深度掃描]
        GoE[Go Engine 5-15s<br/>SSRF/CSPM/SCA 掃描器<br/>高並發資產掃描]
        
        PythonE --> Coord[MultiEngineCoordinator<br/>asyncio.gather 並行執行<br/>資產聚合去重]
        TypeScriptE --> Coord
        RustE --> Coord
        GoE --> Coord
        
        Coord --> P1Out[輸出: Phase1CompletedPayload<br/>- assets List[Asset]<br/>  - URL with 參數<br/>  - Form 結構<br/>  - API 端點<br/>- summary Summary]
    end
    
    P1Out --> AI2[AI 二次決策<br/>評估資產質量<br/>決定是否進入 Phase 2]
    
    AI2 -->|有可攻擊資產| Phase2
    AI2 -->|無有效資產| StopScan[結束掃描<br/>生成基礎報告]
    
    subgraph Phase2 [Phase 2 - 漏洞攻擊測試 Features模組]
        direction TB
        Assets[接收資產列表<br/>List[Asset]]
        
        Assets --> IntegrationLayer[Integration 任務分配<br/>根據資產類型選擇功能模組]
        
        IntegrationLayer --> TaskGen[生成 FunctionTaskPayload<br/>為每個功能模組]
        
        TaskGen --> SSRF[function_ssrf Worker<br/>SSRF 漏洞檢測<br/>- ParamSemanticsAnalyzer<br/>- InternalAddressDetector<br/>- OastDispatcher]
        TaskGen --> SQLi[function_sqli Worker<br/>SQL 注入檢測<br/>6個引擎並行]
        TaskGen --> XSS[function_xss Worker<br/>XSS 檢測]
        TaskGen --> IDOR[function_idor Worker<br/>IDOR 權限檢測]
        TaskGen --> Others[其他功能模組<br/>10+ Workers]
        
        SSRF --> Findings[收集 FindingPayload<br/>漏洞報告]
        SQLi --> Findings
        XSS --> Findings
        IDOR --> Findings
        Others --> Findings
        
        Findings --> FinalOut[輸出: List[FindingPayload]<br/>- vulnerability<br/>- evidence<br/>- severity<br/>- recommendation]
    end
    
    FinalOut --> Integration[Integration 模組<br/>整合資產與漏洞<br/>生成攻擊路徑]
    
    Integration --> Report[最終報告<br/>- 資產清單 175個<br/>- 漏洞列表 12個<br/>- 攻擊路徑圖<br/>- 風險評估<br/>- 修復建議]
    
    StopScan --> Report
    
    style Phase0 fill:#e1f5ff
    style Phase1 fill:#fff3e0
    style Phase2 fill:#ffe0e0
    style AI1 fill:#e8f5e9
    style AI2 fill:#e8f5e9
    style Integration fill:#f3e5f5
    style Report fill:#c8e6c9
```

---

## 📊 數據流與合約

### 數據合約詳解

#### 1. Phase 0 → Phase 1

```python
@dataclass
class Phase0CompletedPayload:
    """Rust 快速偵察輸出"""
    endpoints: List[str]  # ["/api/users", "/api/products"]
    technologies: List[str]  # ["Express.js", "Node.js", "Angular"]
    js_findings: List[str]  # ["ApiEndpoint: /api/SecurityQuestions"]
    risk_summary: Dict  # {"high_risk": 9, "medium_risk": 18}
    execution_time: float  # 0.2s
```

#### 2. Phase 1 → Phase 2

```python
@dataclass
class Asset:
    """資產標準格式"""
    asset_id: str  # UUID
    asset_type: str  # "url" | "form" | "api" | "endpoint"
    value: str  # "http://localhost:3000/login"
    parameters: Optional[List[str]]  # ["username", "password"]
    has_form: bool  # True
    method: str  # "POST"
    headers: Dict  # 請求頭
    metadata: Dict  # 額外資訊

@dataclass
class Phase1CompletedPayload:
    """多引擎聚合輸出"""
    assets: List[Asset]  # 完整資產列表
    total_assets: int  # 175
    by_type: Dict  # {"form": 50, "api": 80, "url": 45}
    summary: Summary  # 統計資訊
    execution_time: float  # 35s
```

**Python 引擎資產示例**:
```python
Asset(
    asset_id="asset_001",
    asset_type="form",
    value="http://localhost:3000/login",
    parameters=["username", "password"],  # ✅ 表單參數
    has_form=True,
    method="POST"
)
```

**TypeScript 引擎資產示例**:
```python
Asset(
    asset_id="asset_002",
    asset_type="api",
    value="http://localhost:3000/api/users",
    parameters=["id", "limit", "offset"],  # ✅ 從 AJAX 提取
    has_form=False,
    method="GET"
)
```

#### 3. Phase 2 內部

```python
@dataclass
class FunctionTaskPayload:
    """功能模組統一任務格式"""
    task_id: str  # UUID
    scan_id: str  # 掃描會話 ID
    target: FindingTarget  # 目標資產
    strategy: str  # "fast" | "normal" | "aggressive"
    timeout: int  # 超時設置

@dataclass
class FindingTarget:
    """攻擊目標"""
    url: str  # "http://localhost:3000/login"
    parameter: str  # "username"
    method: str  # "POST"
    headers: Dict
    body: Optional[Dict]

@dataclass
class FindingPayload:
    """漏洞報告標準格式"""
    finding_id: str  # UUID
    vulnerability: Vulnerability
    target: FindingTarget
    evidence: Dict  # 證據
    timestamp: datetime
    recommendation: str  # 修復建議
    
@dataclass
class Vulnerability:
    name: VulnerabilityType  # SSRF | SQLI | XSS | IDOR
    severity: Severity  # LOW | MEDIUM | HIGH | CRITICAL
    confidence: Confidence  # LOW | MEDIUM | HIGH
    description: str
```

---

## 🎯 模組職責劃分

### Scan 模組 vs Features 模組

| 維度 | Scan 模組 (Phase 0-1) | Features 模組 (Phase 2) |
|------|----------------------|------------------------|
| **職責** | 資產發現 | 漏洞攻擊測試 |
| **輸入** | 目標 URL | Asset 列表 |
| **輸出** | Asset 列表 | FindingPayload 列表 |
| **引擎** | Rust, Python, TypeScript, Go (掃描器) | SSRF, SQLi, XSS, IDOR (攻擊器) |
| **執行時間** | 5-45s | 5-20s |
| **並行方式** | 多引擎並行 | 多功能模組並行 |
| **依賴關係** | 獨立並行 | 依賴 Scan 輸出 |

**關鍵區別**:
- ✅ **Scan**: 爬取、發現、提取、識別
- ✅ **Features**: 攻擊、驗證、檢測、確認

### 引擎分類

#### Scan 模組的引擎

| 引擎 | 位置 | 職責 | 階段 |
|------|------|------|------|
| **Rust** | scan/engines/rust_engine/ | 快速偵察、端點發現 | Phase 0 |
| **Python** | scan/engines/python_engine/ | 靜態爬取、表單參數提取 | Phase 1 |
| **TypeScript** | scan/engines/typescript_engine/ | 動態渲染、AJAX 監聽 | Phase 1 |
| **Go** | scan/engines/go_engine/ | SSRF/CSPM/SCA 掃描器 | Phase 1 |

#### Features 模組的 Workers

| Worker | 位置 | 職責 | 語言 |
|--------|------|------|------|
| **SSRF** | features/function_ssrf/ | SSRF 漏洞檢測 | Python |
| **SQLi** | features/function_sqli/ | SQL 注入檢測 | Python |
| **XSS** | features/function_xss/ | XSS 檢測 | Python |
| **IDOR** | features/function_idor/ | IDOR 檢測 | Python |
| **Authn** | features/function_authn_go/ | 認證繞過 | Go |
| **Crypto** | features/function_crypto/ | 加密漏洞 | Rust |

---

## 🔑 關鍵設計決策

### 1. 為什麼 Python 引擎只提取表單參數?

**決策**: Python 引擎**只提取表單字段**,不提取 URL 查詢參數

**原因**:
1. **職責單一**: Python 引擎專注於靜態 HTML 爬取和表單提取
2. **性能考量**: 表單參數提取簡單高效,URL 參數解析成本較高
3. **互補設計**: TypeScript 引擎監聽 AJAX 可補充 API 參數
4. **實際需求**: 大部分漏洞在表單提交 (登錄、註冊、搜索)

**示例對比**:

```html
<!-- ✅ Python 引擎可處理 -->
<form action="/login" method="POST">
    <input name="username" type="text">
    <input name="password" type="password">
    <button type="submit">Login</button>
</form>

<!-- ❌ Python 引擎無法提取參數 -->
<a href="/search?q=test&sort=asc">Search</a>
<a href="/api/users?id=123&limit=10">API</a>
```

**提取結果**:
```python
# ✅ 表單資產
Asset(
    value="http://localhost:3000/login",
    parameters=["username", "password"],  # ✅ 成功提取
    has_form=True
)

# ❌ URL 資產
Asset(
    value="http://localhost:3000/search?q=test&sort=asc",
    parameters=None,  # ❌ 無參數
    has_form=False
)
```

**影響分析**:
- ✅ **表單攻擊場景**: SQL 注入、XSS、認證繞過 → **不受影響**
- ❌ **URL 參數場景**: IDOR (如 `/api/users?id=123`) → **需要 TypeScript 補充**
- ✅ **SSRF 檢測**: Features 模組有自己的參數分析器 → **不受影響**

---

### 2. 為什麼 SSRF 檢測不依賴 Python 引擎?

**決策**: `function_ssrf` 模組完全獨立,有自己的參數分析器

**原因**:
1. **高度解耦**: 功能模組不應依賴 Scan 模組的內部實現
2. **獨立完整**: SSRF 檢測需要複雜的語義分析,不只是簡單參數提取
3. **靈活性**: SSRF 可以分析任何類型的資產,不局限於表單

**SSRF Worker 完整流程**:
```python
# function_ssrf/worker.py

async def process_task(task: FunctionTaskPayload) -> TaskExecutionResult:
    # 1. 自己分析參數 (不依賴 Python 引擎)
    analyzer = ParamSemanticsAnalyzer()
    plan: AnalysisPlan = analyzer.analyze(task)
    # plan.vectors = [
    #     Vector(param="url", payload="http://169.254.169.254/..."),
    #     Vector(param="redirect", payload="http://localhost:8080/"),
    #     ...
    # ]
    
    # 2. 生成 SSRF 測試向量
    for vector in plan.vectors:
        # 解析 payload (可能需要 DNS 回調)
        payload = await _resolve_payload(vector, dispatcher, task)
        
        # 3. 發送測試請求
        response = await _issue_request(client, task, vector, payload)
        
        # 4. 檢測內網訪問
        detector = InternalAddressDetector()
        detection = detector.analyze(response)
        
        if detection.matched:
            # 5. 確認 SSRF 漏洞
            finding = _build_internal_finding(
                task, vector, payload, response, detection
            )
            findings.append(finding)
    
    return TaskExecutionResult(findings=findings, telemetry=telemetry)
```

**關鍵組件**:
- `ParamSemanticsAnalyzer`: 語義分析器,識別可能的 SSRF 參數
- `OastDispatcher`: 帶外檢測調度器 (DNS/HTTP 回調)
- `InternalAddressDetector`: 內網訪問檢測器

**為什麼不用 Python 提取的參數?**
```python
# ❌ 錯誤理解
python_asset = Asset(
    value="http://localhost:3000/api",
    parameters=["url", "redirect"]  # 假設 Python 提取
)
# Go 引擎接收參數進行 SSRF 測試

# ✅ 正確設計
function_task = FunctionTaskPayload(
    scan_id="scan_001",
    target=FindingTarget(
        url="http://localhost:3000/api",
        parameter="url",  # 由 Integration 或任務生成
        method="GET"
    )
)
# SSRF Worker 自己分析參數並測試
```

---

### 3. 為什麼 Phase 1 引擎並行而非串行?

**決策**: Python、TypeScript、Rust、Go 引擎**真正並行**執行

**錯誤理解**:
```
Phase 1: Python 提取參數
Phase 2: Go 接收參數測試 SSRF
依賴: Python → Go (串行)
```

**正確設計**:
```python
# services/scan/coordinators/multi_engine_coordinator.py

async def execute_phase1(self, phase0_result, strategy="balanced"):
    # 選擇引擎組合
    engines = self._select_engines(strategy)
    
    # ✅ 並行執行
    results = await asyncio.gather(
        self._run_python_engine(phase0_result) if "python" in engines else None,
        self._run_typescript_engine(phase0_result) if "typescript" in engines else None,
        self._run_rust_engine(phase0_result) if "rust" in engines else None,
        self._run_go_engine(phase0_result) if "go" in engines else None,
        return_exceptions=True  # 容錯處理
    )
    
    # 資產聚合去重
    all_assets = self._merge_and_deduplicate([r for r in results if r])
    
    return Phase1CompletedPayload(assets=all_assets, ...)
```

**並行優勢**:
- ⚡ **性能**: 30s (並行) vs 120s (串行 30+45+20+25)
- 🎯 **獨立性**: 引擎故障不影響其他引擎
- 🔄 **可擴展**: 新增引擎不需要修改其他引擎

---

### 4. 為什麼 Go 引擎在 Phase 1 而非 Phase 2?

**位置**: `services/scan/engines/go_engine/` (Scan 模組)

**職責**: SSRF/CSPM/SCA **掃描器**,不是漏洞攻擊器

**理解誤區**:
- ❌ Go 引擎 = SSRF 攻擊測試器
- ❌ Go 引擎在 Phase 2 接收 Python 參數

**正確理解**:
- ✅ Go 引擎 = Phase 1 的高並發資產掃描器
- ✅ Go 引擎與 Python/TypeScript/Rust 並行執行
- ✅ SSRF **攻擊測試**在 `features/function_ssrf/` (Phase 2)

**兩種 SSRF 的區別**:

| 項目 | Scan 的 Go 引擎 | Features 的 SSRF Worker |
|------|----------------|------------------------|
| **位置** | scan/engines/go_engine/ | features/function_ssrf/ |
| **階段** | Phase 1 (資產發現) | Phase 2 (漏洞攻擊) |
| **職責** | SSRF 掃描器,資產發現 | SSRF 攻擊測試,漏洞確認 |
| **輸入** | Phase 0 端點列表 | Phase 1 資產列表 |
| **輸出** | Asset 列表 | FindingPayload 列表 |
| **執行方式** | 與其他引擎並行 | 與其他功能模組並行 |

---

## ⚠️ 常見誤解澄清

### 誤解 1: "Scan 模組負責漏洞檢測"

**錯誤**:
```
Scan 模組 → 發現資產 → 同時測試 SSRF/SQLi/XSS
```

**正確**:
```
Scan 模組 (Phase 0-1) → 只發現資產
Features 模組 (Phase 2) → 漏洞攻擊測試
```

**職責劃分**:
- Scan: 爬取、發現、提取、識別
- Features: 攻擊、驗證、檢測、確認

---

### 誤解 2: "Python → Go 串行依賴"

**錯誤流程**:
```
Phase 1: Python 提取參數
  └─ 輸出: ["username", "password", "url"]

Phase 2: Go 接收 "url" 參數
  └─ 測試 SSRF: http://169.254.169.254/...
```

**正確流程**:
```
Phase 1 (並行):
  ├─ Python: 表單參數
  ├─ TypeScript: 動態路由
  ├─ Rust: 敏感掃描
  └─ Go: SSRF/CSPM/SCA 掃描器
  └─ 輸出: 聚合後的 Asset 列表

Phase 2 (並行):
  ├─ function_ssrf: 自己分析參數並測試
  ├─ function_sqli: SQL 注入測試
  └─ 其他功能模組
```

---

### 誤解 3: "SSRF 檢測依賴 Python 參數提取"

**錯誤理解**:
```
Python 提取不到 URL 參數 → SSRF 檢測無法工作
```

**正確理解**:
```
function_ssrf 完全獨立:
  ├─ 自己的 ParamSemanticsAnalyzer
  ├─ 自己的測試向量生成
  ├─ 自己的帶外檢測
  └─ 不依賴 Python 引擎
```

**SSRF Worker 獨立性**:
```python
# function_ssrf 不需要 Python 提取的參數
analyzer = ParamSemanticsAnalyzer()
plan = analyzer.analyze(task)  # 自己分析目標

for vector in plan.vectors:
    # 自己生成 payload
    payload = await _resolve_payload(vector, dispatcher, task)
    # 自己發送請求
    response = await _issue_request(client, task, vector, payload)
    # 自己檢測漏洞
    detection = detector.analyze(response)
```

---

### 誤解 4: "Go 引擎在 Phase 2"

**錯誤**:
```
Phase 2: Go 引擎執行 SSRF 測試
```

**正確**:
```
Phase 1: scan/engines/go_engine/ (SSRF 掃描器)
Phase 2: features/function_ssrf/ (SSRF 攻擊測試)
```

**Go 引擎的真實位置**:
- ✅ Phase 1: 作為資產發現工具
- ❌ Phase 2: 不在這裡

---

## 📋 實際運作流程範例

### 場景: 掃描 Juice Shop

```
1. 用戶發起掃描
   └─ Target: http://localhost:3000

2. Phase 0 - Rust 快速偵察 (0.2s)
   ✅ 發現 45 個端點
   ✅ 識別技術棧: Express.js + Angular
   ✅ JS 分析: /api/Users, /api/Products, /api/SecurityQuestions
   └─ 建議: Python (靜態) + TypeScript (動態)

3. AI 核心決策
   ✅ 分析: 有表單 + SPA + API
   ✅ 策略: balanced
   ✅ 選擇: Python + TypeScript 並行

4. Phase 1 - 多引擎並行 (35s)
   
   Python Engine (30s):
   ┌────────────────────────────────────┐
   │ 爬取 150+ 頁面                      │
   │ 發現表單:                           │
   │  - /login (email, password)        │
   │  - /register (email, password, ...) │
   │  - /#/search (q)                   │
   │ 輸出: 120 資產 (帶表單參數)        │
   └────────────────────────────────────┘
   
   TypeScript Engine (35s):
   ┌────────────────────────────────────┐
   │ Playwright 渲染 SPA                │
   │ 監聽 AJAX 請求:                    │
   │  - GET /api/Users                  │
   │  - POST /api/Products/search       │
   │  - GET /api/SecurityQuestions      │
   │ 輸出: 55 資產 (API 端點)           │
   └────────────────────────────────────┘
   
   協調器聚合:
   ┌────────────────────────────────────┐
   │ 合併: 175 資產                      │
   │ 去重: 160 唯一資產                  │
   │ 分類:                               │
   │  - Form: 50                        │
   │  - API: 80                         │
   │  - URL: 30                         │
   └────────────────────────────────────┘

5. AI 二次決策
   ✅ 發現大量表單和 API
   ✅ 檢測到潛在攻擊點
   ✅ 決定: 進入 Phase 2

6. Phase 2 - 功能模組並行攻擊 (18s)
   
   Integration 任務分配:
   ┌────────────────────────────────────┐
   │ 分析 160 資產                       │
   │ 分配任務:                           │
   │  - SQL 注入: 50 個表單資產         │
   │  - XSS: 50 個表單資產              │
   │  - IDOR: 80 個 API 資產            │
   │  - SSRF: 15 個特定參數資產         │
   └────────────────────────────────────┘
   
   function_sqli Worker (12s):
   ┌────────────────────────────────────┐
   │ 測試 /login 的 email 參數          │
   │ 6 個引擎並行:                      │
   │  ✓ Boolean-based: ' OR '1'='1      │
   │  ✓ Error-based: ' AND 1=CAST...    │
   │  ✓ Time-based: ' AND SLEEP(5)--    │
   │ 發現: SQL 注入漏洞 (HIGH)          │
   └────────────────────────────────────┘
   
   function_xss Worker (10s):
   ┌────────────────────────────────────┐
   │ 測試 /#/search 的 q 參數           │
   │ Payload:                           │
   │  - <script>alert(1)</script>       │
   │  - <img src=x onerror=alert(1)>    │
   │ 發現: 反射型 XSS (MEDIUM)          │
   └────────────────────────────────────┘
   
   function_idor Worker (15s):
   ┌────────────────────────────────────┐
   │ 測試 /api/Users?id=X               │
   │ 嘗試訪問其他用戶:                   │
   │  - GET /api/Users?id=2 (403)       │
   │  - GET /api/Users?id=3 (200) ✓     │
   │ 發現: IDOR 漏洞 (HIGH)             │
   └────────────────────────────────────┘
   
   function_ssrf Worker (8s):
   ┌────────────────────────────────────┐
   │ 測試 API 端點 (如有 url 參數)      │
   │ Payload:                           │
   │  - http://169.254.169.254/...      │
   │  - http://localhost:8080/admin     │
   │ 本例中: 未發現 SSRF (無此類參數)   │
   └────────────────────────────────────┘

7. Integration 整合報告
   ┌────────────────────────────────────┐
   │ 收集 Findings:                     │
   │  - SQL 注入: 1 個 (HIGH)           │
   │  - XSS: 1 個 (MEDIUM)              │
   │  - IDOR: 1 個 (HIGH)               │
   │                                    │
   │ 攻擊路徑分析:                       │
   │  1. IDOR 繞過認證獲取用戶資訊      │
   │  2. SQL 注入提取數據庫             │
   │  3. XSS 竊取 Session               │
   │                                    │
   │ 風險評分: 8.5/10 (HIGH)            │
   └────────────────────────────────────┘

8. 最終報告
   ┌────────────────────────────────────┐
   │ 📊 資產統計:                        │
   │   - 總資產: 160 個                  │
   │   - 表單: 50 個                     │
   │   - API: 80 個                      │
   │   - URL: 30 個                      │
   │                                    │
   │ 🐛 漏洞列表:                        │
   │   1. SQL 注入 (HIGH)                │
   │      位置: /login (email 參數)      │
   │      證據: ' OR '1'='1 繞過登錄     │
   │      修復: 使用參數化查詢           │
   │                                    │
   │   2. XSS (MEDIUM)                   │
   │      位置: /#/search (q 參數)       │
   │      證據: <script> 成功執行        │
   │      修復: 輸出編碼 + CSP           │
   │                                    │
   │   3. IDOR (HIGH)                    │
   │      位置: /api/Users (id 參數)     │
   │      證據: 可訪問其他用戶資料       │
   │      修復: 實施授權檢查             │
   │                                    │
   │ 🔧 修復優先級:                      │
   │   1. SQL 注入 (立即修復)            │
   │   2. IDOR (立即修復)                │
   │   3. XSS (近期修復)                 │
   │                                    │
   │ 總執行時間: 53s                     │
   └────────────────────────────────────┘
```

---

## 🎓 設計理念總結

### 核心原則

1. **職責單一**: 每個模組專注自己的職責,不越界
2. **高度解耦**: 模組間只通過標準數據格式通信
3. **並行優先**: 能並行就不串行,最大化效率
4. **獨立完整**: 每個功能模組都是完整的檢測單元
5. **標準化**: 統一使用 aiva_common 的 Schema 和枚舉

### 數據流向

```
Scan (資產發現) → Integration (任務調度) → Features (漏洞攻擊) → Report (報告)
```

### 模組協作

```
┌─────────────────────────────────────────────────────────┐
│                      AIVA 架構                           │
├─────────────────────────────────────────────────────────┤
│  aiva_common: 共享基礎設施 (Schema, Enum, Utils)        │
├─────────────────────────────────────────────────────────┤
│  core: AI 核心 (決策, RAG, 自我優化)                    │
├─────────────────────────────────────────────────────────┤
│  Scan 模組 (Phase 0-1): 資產發現                        │
│    ├─ Phase 0: Rust 快速偵察                            │
│    └─ Phase 1: Python, TypeScript, Rust, Go 並行       │
│        └─ 輸出: Asset[]                                 │
├─────────────────────────────────────────────────────────┤
│  Integration 模組: 任務調度                              │
│    ├─ 接收 Asset[]                                      │
│    ├─ 分析資產類型                                       │
│    └─ 分配 FunctionTaskPayload 給功能模組               │
├─────────────────────────────────────────────────────────┤
│  Features 模組 (Phase 2): 漏洞攻擊測試                  │
│    ├─ function_ssrf: SSRF 檢測                          │
│    ├─ function_sqli: SQL 注入                           │
│    ├─ function_xss: XSS 檢測                            │
│    ├─ function_idor: IDOR 檢測                          │
│    └─ 其他 10+ 功能模組                                 │
│        └─ 輸出: FindingPayload[]                        │
├─────────────────────────────────────────────────────────┤
│  Integration 模組: 整合報告                              │
│    ├─ 收集 Asset[] + FindingPayload[]                   │
│    ├─ 生成攻擊路徑                                       │
│    └─ 輸出: 最終報告                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 延伸閱讀

- [services/README.md](../services/README.md) - 六大核心服務總覽
- [services/scan/README.md](../services/scan/README.md) - Scan 模組詳細說明
- [services/features/README.md](../services/features/README.md) - Features 模組詳細說明
- [services/scan/coordinators/README.md](../services/scan/coordinators/README.md) - 協調器架構
- [services/integration/README.md](../services/integration/README.md) - Integration 調度機制

---

**版本**: 1.0.0  
**最後更新**: 2025-01-21  
**維護者**: AIVA 開發團隊  
**目的**: 為 AI 協作和未來開發提供正確的架構基準
