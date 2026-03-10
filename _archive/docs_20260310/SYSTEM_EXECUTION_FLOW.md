# AIVA 系統完整執行流程說明

## 📋 目錄

- [概述](#概述)
- [系統架構圖](#系統架構圖)
- [完整執行步驟](#完整執行步驟)
  - [步驟 1：用戶提供網址](#步驟-1用戶提供網址)
  - [步驟 2：API 接收請求](#步驟-2api-接收請求)
  - [步驟 3：認知核心 AI 分析](#步驟-3認知核心-ai-分析)
  - [步驟 4：CognitiveDispatcher 調用 CLI](#步驟-4cognitivedispatcher-調用-cli)
  - [步驟 5：多語言 CLI 執行掃描](#步驟-5多語言-cli-執行掃描)
  - [步驟 6：結果處理與 AI 決策](#步驟-6結果處理與-ai-決策)
  - [步驟 7：Phase1 深度掃描](#步驟-7phase1-深度掃描)
  - [步驟 8：最終報告生成](#步驟-8最終報告生成)
- [關鍵文件職責說明](#關鍵文件職責說明)
- [數據流轉詳解](#數據流轉詳解)
- [實際測試示例](#實際測試示例)

---

## 概述

AIVA 是一個 AI 驅動的安全掃描系統。系統採用 **subprocess 直接調用 CLI** 的架構，而非分布式 MQ Workers。

**核心特點：**
- 🤖 **AI 驅動決策**：使用 5M 參數神經網絡 + 知識圖譜
- 🔄 **兩階段掃描**：Phase0（快速探測）→ Phase1（深度分析）
- 🔧 **CLI 直接調用**：通過 subprocess 調用多語言 CLI（Rust/Python/TypeScript）
- 🧠 **認知核心**：整合 Dual-CLI + embedded_knowledge 的智能決策引擎

**架構選擇說明：**
- 系統使用 `subprocess.run()` 直接調用 CLI 工具
- MQ（RabbitMQ）用於事件通知和日誌，而非任務分發
- 這種設計更簡單、更直接，適合單機部署

---

## 系統架構圖

```
用戶
  ↓ (提供 URL)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 1. app.py (FastAPI 入口)                                │
│    - POST /scan endpoint                                 │
│    - 端口: 8000                                          │
└─────────────────────────────────────────────────────────┘
  ↓ (調用)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. EnhancedDecisionAgent (認知核心)                     │
│    - 文件: cognitive_core/decision/enhanced_decision_agent.py │
│    - 整合 InternalLoopConnector + ExternalLoopConnector │
│    - 整合 4 個 embedded_knowledge 引擎                  │
└─────────────────────────────────────────────────────────┘
  ↓ (AI 決策：選擇掃描策略)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. CognitiveDispatcher (調度器)                         │
│    - 文件: cognitive_core/dispatcher.py                  │
│    - 使用 subprocess.run() 調用 CLI                     │
└─────────────────────────────────────────────────────────┘
  ↓ (subprocess 調用)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 多語言 CLI 引擎                                       │
│    - Rust CLI: services/scan/rust_engine/               │
│    - Python CLI: services/scan/python_engine/           │
│    - TypeScript CLI: services/scan/typescript_engine/   │
└─────────────────────────────────────────────────────────┘
  ↓ (返回結果 - stdout)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 5. ScanResultProcessor (結果處理器)                     │
│    - 文件: core_capabilities/processing/scan_result_processor.py │
│    - 7 階段處理流程                                      │
│    - 調用 EnhancedDecisionAgent 決策 Phase1             │
└─────────────────────────────────────────────────────────┘
  ↓ (AI 決策：選擇 Phase1 引擎)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Phase1 深度掃描 (再次 subprocess 調用)               │
│    - 根據 AI 選擇的引擎執行                              │
│    - VulnerabilityDetector                              │
│    - CVEIdentifier                                      │
│    - WAFBypassEngine                                    │
│    - WebArchitectureAnalyzer                            │
└─────────────────────────────────────────────────────────┘
  ↓ (最終結果)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 7. 報告生成 & 存檔                                       │
│    - data/scan_results/                                 │
│    - 完整的掃描報告 JSON                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 完整執行步驟

### 步驟 1：用戶提供網址

**操作：**
```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"target": "http://localhost:3000"}'
```

**數據結構：**
```json
{
  "target": "http://localhost:3000",
  "scan_type": "comprehensive"
}
```

---

### 步驟 2：API 接收請求

**文件位置：** `services/core/aiva_core/service_backbone/api/app.py`

**執行內容：**
```python
@app.post("/scan", response_model=ScanResponse)
async def start_scan(request: ScanRequest) -> ScanResponse:
    # 1. 生成 scan_id
    scan_id = f"scan_{uuid4().hex[:8]}"
    
    # 2. 構建決策上下文
    context = DecisionContext()
    context.target_info = {
        "type": "web",
        "value": request.target,
        "id": scan_id,
    }
    
    # 3. 調用認知核心進行 AI 分析
    initial_decision = await decision_agent.make_enhanced_decision(
        context=context,
        use_embedded_knowledge=True
    )
    
    # 4. 返回 scan_id 供後續查詢
    return ScanResponse(scan_id=scan_id, status="started", ...)
```

---

### 步驟 3：認知核心 AI 分析

**文件位置：** `services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py`

**執行內容：** `make_enhanced_decision()` 方法

```python
async def make_enhanced_decision(self, context, use_embedded_knowledge=True):
    # 階段 1: Internal CLI 查詢 (RAG 向量數據庫)
    internal_result = await self.internal_cli_connector.query_with_rag(...)
    # 讀取: data/internal_exploration/latest_classification.json
    # 查詢: data/vector_db/chroma/
    
    # 階段 2: Embedded Knowledge 分析
    if use_embedded_knowledge:
        vuln_info = self.vulnerability_detector.detect(target)
        cve_info = self.cve_identifier.identify(target)
        waf_info = self.waf_bypass_engine.analyze(target)
        arch_info = self.web_architecture_analyzer.analyze(target)
    
    # 階段 3: 神經網絡融合決策
    final_decision = self.real_decision_engine.decide(
        internal_knowledge=internal_result,
        embedded_knowledge=combined_knowledge
    )
    # 使用: ai_models/bio_inspired_5m.pth (5M 參數神經網絡)
    
    return Decision(action=..., confidence=..., reasoning=...)
```

**調用的 4 個 embedded_knowledge 引擎：**
- `cognitive_core/embedded_knowledge/vulnerability_detector.py`
- `cognitive_core/embedded_knowledge/cve_identifier.py`
- `cognitive_core/embedded_knowledge/waf_bypass_engine.py`
- `cognitive_core/embedded_knowledge/web_architecture_analyzer.py`

---

### 步驟 4：CognitiveDispatcher 調用 CLI

**文件位置：** `services/core/aiva_core/cognitive_core/dispatcher.py`

**關鍵點：系統使用 subprocess 直接調用 CLI，而非 MQ Workers**

```python
import subprocess

class CognitiveDispatcher:
    def call_rust_scanner_sync(
        self, 
        targets: list[str],
        mode: str = "fast",
        timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """
        同步調用 Rust 掃描引擎
        """
        cmd = [
            "rust_scanner", "scan",
            "--url", *targets,
            "--mode", mode,
            "--format", "json"
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    def call_python_scanner_sync(
        self, 
        capability: str,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        同步調用 Python 掃描模塊
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.cli.aiva_cli",
            capability,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True
        )
```

---

### 步驟 5：多語言 CLI 執行掃描

#### 5.1 Rust CLI (Phase0 快速偵察)

**文件位置：** `services/scan/rust_engine/src/main.rs`

**CLI 命令：**
```bash
rust_scanner scan --url https://example.com --mode fast --format json
```

**執行內容：**
- 端口掃描
- HTTP 指紋識別
- TLS/SSL 證書檢測
- 技術棧識別

**輸出格式：** JSON 到 stdout
```json
{
  "mode": "fast",
  "targets": [{
    "url": "https://example.com",
    "status_code": 200,
    "technologies": ["nginx", "Express.js"],
    "endpoints": ["/api/users", "/login"]
  }],
  "summary": {
    "urls_found": 15,
    "forms_found": 3
  }
}
```

#### 5.2 Python CLI (漏洞檢測)

**文件位置：** `services/core/aiva_core/core_capabilities/cli/aiva_cli.py`

**CLI 命令：**
```bash
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli sqli_scan \
  --target "https://example.com/search?q=test"
```

**執行內容：**
- SQL 注入檢測
- XSS 漏洞掃描
- CSRF 檢測
- 敏感文件探測

#### 5.3 TypeScript CLI (Web 分析)

**文件位置：** `services/scan/typescript_engine/`

**CLI 命令：**
```bash
ts-node scanner.ts analyze --url https://example.com
```

**執行內容：**
- DOM 結構分析
- JavaScript 框架識別
- 前端資源映射

---

### 步驟 6：結果處理與 AI 決策

**文件位置：** `services/core/aiva_core/core_capabilities/processing/scan_result_processor.py`

**7 階段處理流程：**

```python
async def process_phase0(self, payload, broker, trace_id):
    # 階段 1: Raw Data Validation
    # 階段 2: Data Normalization
    # 階段 3: Threat Assessment
    # 階段 4: Context Enrichment
    
    # 階段 5: AI Decision - 是否需要 Phase1
    need_phase1, reason = await self._analyze_phase0_and_decide(
        scan_id, payload, processed_data
    )
    # 調用 EnhancedDecisionAgent.make_enhanced_decision()
    
    if not need_phase1:
        return False, reason, []  # 直接生成報告
    
    # 階段 6: 選擇 Phase1 引擎
    selected_engines = await self._select_engines_for_phase1(scan_id, payload)
    
    return True, reason, selected_engines
```

**AI 決策邏輯 (`_analyze_phase0_and_decide`)：**
```python
# 使用 AI 增強決策
decision = await self.decision_agent.make_enhanced_decision(
    context=context,
    use_embedded_knowledge=True
)

# 根據 AI 決策結果
if decision.confidence >= 0.7:
    need_phase1 = True
    reason = f"AI Decision (High Confidence): {decision.reasoning}"
elif decision.confidence >= 0.4:
    need_phase1 = True
    reason = f"AI Decision (Medium Confidence): Recommend Phase1"
else:
    # 低置信度：回退到規則引擎
    need_phase1, reason = self._fallback_rule_decision(processed_data)
```

---

### 步驟 7：Phase1 深度掃描

**觸發條件：** AI 決定需要深度掃描

**執行方式：** 再次使用 subprocess 調用 CLI

```python
# 根據 AI 選擇的引擎執行
for engine in selected_engines:
    if engine == "VulnerabilityDetector":
        result = dispatcher.call_python_scanner_sync(
            capability="deep_vuln_scan",
            target=target
        )
    elif engine == "WAFBypassEngine":
        result = dispatcher.call_python_scanner_sync(
            capability="waf_bypass_test",
            target=target
        )
    # ...
```

**Phase1 引擎選擇邏輯：**
- 高風險目標 → VulnerabilityDetector + CVEIdentifier
- 檢測到 WAF → WAFBypassEngine
- 複雜技術棧 → WebArchitectureAnalyzer

---

### 步驟 8：最終報告生成

**執行位置：** `scan_result_processor.py` 階段 7

```python
def _generate_report(self, scan_results):
    report = {
        "scan_id": scan_results["session_id"],
        "target": scan_results["target"],
        "executive_summary": self._generate_summary(scan_results),
        "ai_decision_trail": scan_results.get("ai_decisions", []),
        "phase0_results": scan_results.get("phase0", {}),
        "phase1_results": scan_results.get("phase1", {}),
        "detailed_findings": self._format_findings(scan_results),
        "recommendations": self._generate_recommendations(scan_results),
    }
    return report

async def _persist_results(self, report):
    # 存儲到文件系統
    output_dir = Path("data/scan_results") / datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"scan_{report['scan_id']}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
```

**存儲位置：** `data/scan_results/YYYY-MM-DD/scan_{session_id}.json`

---

## 關鍵文件職責說明

| 文件 | 職責 | 調用方式 |
|------|------|----------|
| `app.py` | 系統入口，REST API | FastAPI 直接運行 |
| `enhanced_decision_agent.py` | AI 決策核心 | Python 內部調用 |
| `dispatcher.py` | CLI 調度器 | subprocess.run() |
| `scan_result_processor.py` | 結果處理 | Python 內部調用 |
| `rust_engine/main.rs` | Rust 掃描引擎 | CLI 被 subprocess 調用 |
| `python_engine/*.py` | Python 掃描模塊 | CLI 被 subprocess 調用 |
| `aiva_cli.py` | Python CLI 入口 | CLI 被 subprocess 調用 |

---

## 數據流轉詳解

```
1. HTTP Request (用戶 → app.py)
   數據: {"target": "http://localhost:3000"}
   協議: HTTP POST

2. AI Decision (app.py → EnhancedDecisionAgent)
   數據: DecisionContext object
   協議: Python 內部調用

3. CLI Dispatch (CognitiveDispatcher → CLI)
   數據: 命令行參數
   協議: subprocess.run()

4. CLI Execution (Rust/Python/TS CLI)
   數據: JSON stdout
   協議: 進程間通信 (stdout/stderr)

5. Result Processing (stdout → ScanResultProcessor)
   數據: JSON 字符串
   協議: Python 內部調用

6. Report Storage (ScanResultProcessor → File System)
   數據: JSON 文件
   協議: 文件 I/O
```

---

## 實際測試示例

### 1. 啟動系統
```bash
cd C:\D\fold7\AIVA-git
python services\core\aiva_core\service_backbone\api\app.py
```

### 2. 發送掃描請求
```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"target": "http://localhost:3000", "scan_type": "comprehensive"}'
```

### 3. 預期響應
```json
{
  "scan_id": "scan_abc12345",
  "status": "started",
  "message": "Scan initiated with AI decision: PROCEED_SCAN",
  "target": "http://localhost:3000",
  "estimated_time": 600
}
```

### 4. 查詢狀態
```bash
curl http://localhost:8000/status/scan_abc12345
```

### 5. 查看報告
```bash
cat data/scan_results/2026-01-19/scan_abc12345.json
```

---

## 總結

### 完整流程回顧：

1. **用戶發送請求** → `app.py` POST /scan
2. **AI 初步分析** → `EnhancedDecisionAgent.make_enhanced_decision()`
3. **CLI 調度** → `CognitiveDispatcher` 使用 `subprocess.run()` 調用
4. **Phase0 執行** → Rust CLI 快速偵察
5. **結果處理** → `ScanResultProcessor` 7 階段處理
6. **AI 決策 Phase1** → `EnhancedDecisionAgent` 決定是否深度掃描
7. **Phase1 執行** → 再次 subprocess 調用深度掃描引擎
8. **報告生成** → JSON 存儲到 `data/scan_results/`

### 關鍵設計特點：

- ✅ **CLI 直接調用**：使用 subprocess，不依賴 MQ Workers
- ✅ **同進程執行**：更簡單、更直接
- ✅ **AI 驅動**：3 次 AI 決策（初步分析 + Phase0→Phase1 + 引擎選擇）
- ✅ **多語言支持**：Rust/Python/TypeScript CLI 統一調用方式

---

**文檔版本：** 2.0 (修正版)  
**最後更新：** 2026-01-19  
**作者：** AIVA Development Team
