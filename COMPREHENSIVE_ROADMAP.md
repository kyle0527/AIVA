# AIVA 系統完整發展藍圖

## Comprehensive Development Roadmap 2025-2026

**文件版本**: 2.0  
**建立日期**: 2025-10-13  
**整合文件**: 架構報告、數據合約、Core 分析、掃描引擎報告、多語言架構方案  
**維護者**: AIVA 技術團隊

---

## 目錄

1. 專案現況總覽
2. 技術債務清單
3. 短期計畫 (Q4 2025)
4. 中期計畫 (Q1-Q2 2026)
5. 長期計畫 (Q3-Q4 2026)
6. [多語言架構遷移路線圖](#多語言架構遷移路線圖)
7. [資源需求與團隊配置](#資源需求與團隊配置)
8. [風險管理與應變計畫](#風險管理與應變計畫)
9. 關鍵績效指標 (KPIs)
10. 技術決策記錄 (ADR)

---

## 專案現況總覽

### 系統架構概覽

AIVA 智慧漏洞掃描系統
├── 四大核心模組 (Python)
│   ├── Core - 智慧分析與協調 ✅ 良好
│   ├── Scan - 爬蟲與資產發現 ✅ 已重構
│   ├── Function - 漏洞檢測 ⚠️ 部分完善
│   └── Integration - 資料整合與報告 ✅ 良好
│
├── 共用基礎設施
│   ├── aiva_common (數據合約) ✅ 已統一
│   ├── RabbitMQ (消息隊列) ✅ 運作中
│   └── PostgreSQL (數據庫) ✅ 運作中
│
└── 計畫中的多語言服務
    ├── Node.js - 動態掃描引擎 (Playwright) 📋 規劃中
    ├── Go - 高併發探測器 (SSRF/SQLi) 📋 規劃中
    └── Rust - 敏感資訊掃描器 📋 規劃中

```plaintext
AIVA 智慧漏洞掃描系統
├── 四大核心模組 (Python)
│   ├── Core - 智慧分析與協調 ✅ 良好
│   ├── Scan - 爬蟲與資產發現 ✅ 已重構
│   ├── Function - 漏洞檢測 ⚠️ 部分完善
│   └── Integration - 資料整合與報告 ✅ 良好
│
├── 共用基礎設施
│   ├── aiva_common (數據合約) ✅ 已統一
│   ├── RabbitMQ (消息隊列) ✅ 運作中
│   └── PostgreSQL (數據庫) ✅ 運作中
│
└── 計畫中的多語言服務
    ├── Node.js - 動態掃描引擎 (Playwright) 📋 規劃中
    ├── Go - 高併發探測器 (SSRF/SQLi) 📋 規劃中
    └── Rust - 敏感資訊掃描器 📋 規劃中
```

AIVA 智慧漏洞掃描系統
├── 四大核心模組 (Python)
│   ├── Core - 智慧分析與協調 ✅ 良好
│   ├── Scan - 爬蟲與資產發現 ✅ 已重構
│   ├── Function - 漏洞檢測 ⚠️ 部分完善
│   └── Integration - 資料整合與報告 ✅ 良好
│
├── 共用基礎設施
│   ├── aiva_common (數據合約) ✅ 已統一
│   ├── RabbitMQ (消息隊列) ✅ 運作中
│   └── PostgreSQL (數據庫) ✅ 運作中
│
└── 計畫中的多語言服務
    ├── Node.js - 動態掃描引擎 (Playwright) 📋 規劃中
    ├── Go - 高併發探測器 (SSRF/SQLi) 📋 規劃中
    └── Rust - 敏感資訊掃描器 📋 規劃中
```

### 完成度評估

| 模組/功能 | 完成度 | 狀態 | 優先級 |
|-----------|--------|------|--------|
| **基礎架構** | 95% | ✅ 穩定 | - |
| - 數據合約 (Schemas) | 100% | ✅ 完成 | - |
| - 枚舉定義 (Enums) | 100% | ✅ 完成 | - |
| - 消息隊列基礎設施 | 90% | ✅ 良好 | P2 |
| **Core 模組** | 75% | ⚠️ 需改進 | P0 |
| - 攻擊面分析 | 70% | ⚠️ 不完整 | P0 |
| - 策略生成器 | 0% | ❌ 已移除 | P1 |
| - 任務生成器 | 60% | ⚠️ 簡化 | P0 |
| - 狀態管理 | 90% | ✅ 良好 | - |
| **Scan 模組** | 85% | ✅ 已重構 | - |
| - 核心爬蟲引擎 | 95% | ✅ 優秀 | - |
| - 動態內容引擎 | 60% | ⚠️ 需整合 | P1 |
| - 資訊收集器 | 90% | ✅ 良好 | - |
| **Function 模組** | 70% | ⚠️ 不均 | P1 |
| - XSS 檢測 | 65% | ⚠️ 需增強 | P1 |
| - SQLi 檢測 | 70% | ⚠️ 需增強 | P1 |
| - SSRF 檢測 | 60% | ⚠️ 基礎 | P1 |
| - IDOR 檢測 | 80% | ✅ 較完善 | - |
| **Integration 模組** | 80% | ✅ 良好 | P2 |
| **測試覆蓋** | 15% | ❌ 嚴重不足 | P0 |
| **文檔完整性** | 85% | ✅ 良好 | - |

### 關鍵成就 (已完成)

#### ✅ 2025 Q3 完成項目

1. **架構統一**
   - 所有 dataclass 遷移至 Pydantic v2 BaseModel
   - 統一使用現代類型提示 (`X | None`)
   - 四大模組架構明確定義

2. **數據合約完善**
   - 26 個核心 Schema 完整定義
   - 7 個 Enum 類別統一管理
   - 消除所有重複定義

3. **掃描引擎重構**
   - URL 隊列管理器升級 (deque + set)
   - ScanContext 集中狀態管理
   - HTTP 客戶端安全增強

4. **代碼品質提升**
   - 通過 Ruff 格式化檢查
   - 日誌系統統一 (logger 取代 print)
   - 環境設置標準化

---

## 技術債務清單

### P0 - 關鍵級 (必須立即處理)

| ID | 項目 | 影響 | 預估工時 |
|----|------|------|----------|
| TD-001 | **測試覆蓋率嚴重不足** | 系統穩定性風險高 | 40 小時 |
| TD-002 | **Core 策略生成器被移除** | 智慧分析能力缺失 | 60 小時 |
| TD-003 | **任務生成邏輯簡化** | 漏洞檢測不完整 | 30 小時 |
| TD-004 | **攻擊面分析不完整** | IDOR 候選檢測缺失 | 20 小時 |

### P1 - 高優先級 (1 個月內處理)

| ID | 項目 | 影響 | 預估工時 |
|----|------|------|----------|
| TD-005 | **動態掃描引擎未整合** | Playwright 功能未啟用 | 80 小時 |
| TD-006 | **XSS 檢測能力不足** | DOM XSS 無法偵測 | 40 小時 |
| TD-007 | **SQLi 檢測單一** | 僅支援基本注入 | 50 小時 |
| TD-008 | **SSRF 防護不完整** | 雲端 Metadata 未阻擋 | 30 小時 |
| TD-009 | **配置管理硬編碼** | 缺少統一配置中心 | 25 小時 |

### P2 - 中優先級 (3 個月內處理)

| ID | 項目 | 影響 | 預估工時 |
|----|------|------|----------|
| TD-010 | **效能監控缺失** | 無法追蹤系統瓶頸 | 35 小時 |
| TD-011 | **CI/CD 流程不完整** | 部署風險高 | 40 小時 |
| TD-012 | **API 文檔過時** | 開發者體驗差 | 20 小時 |
| TD-013 | **錯誤處理不統一** | 調試困難 | 30 小時 |

---

## 短期計畫 (Q4 2025)

**時程**: 2025-10-13 ~ 2025-12-31  
**目標**: 補足關鍵功能,建立測試基礎

### Sprint 1: 測試基礎建設 (2週)

**時間**: Week 1-2 (10/13 - 10/26)

#### 任務清單

- [ ] **TD-001: 建立單元測試框架**
  - 選擇測試工具: pytest + pytest-asyncio + pytest-cov
  - 設置測試目錄結構
  - 撰寫測試輔助工具 (fixtures, mocks)
  - 目標: Core 模組測試覆蓋率 >60%

```python
# tests/core/test_task_generator.py
import pytest
from services.core.aiva_core.execution.task_generator import TaskGenerator
from services.aiva_common.schemas import Asset, Vulnerability

@pytest.fixture
def sample_assets():
    return [
        Asset(type="url", value="https://example.com/admin", metadata={}),
        Asset(type="form", value="login_form", metadata={"method": "POST"}),
    ]

@pytest.mark.asyncio
async def test_generate_idor_tasks(sample_assets):
    generator = TaskGenerator()
    tasks = await generator.generate_tasks(sample_assets, vulnerabilities=[])
    
    assert len(tasks) > 0
    assert any(t.module == "function_idor" for t in tasks)
```

- [ ] **建立集成測試環境**
  - Docker Compose 測試環境
  - 模擬 RabbitMQ 與 PostgreSQL
  - 端到端測試腳本

**交付成果**:

- ✅ `tests/` 目錄結構完整
- ✅ pytest 配置檔 (`pytest.ini`)
- ✅ Core 模組測試覆蓋率報告 (>60%)
- ✅ CI 集成 (GitHub Actions)

---

### Sprint 2: 策略生成器重建 (3週)

**時間**: Week 3-5 (10/27 - 11/16)

#### 任務清單

- [ ] **TD-002: 重建策略生成器**

##### Phase 1: 規則引擎基礎 (1週)

```python
# services/core/aiva_core/analysis/strategy_generator.py
from typing import List
from services.aiva_common.schemas import Asset, Vulnerability
from services.aiva_common.enums import VulnerabilityType

class StrategyGenerator:
    """基於攻擊面與已知漏洞生成測試策略"""
    
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.priority_calculator = PriorityCalculator()
    
    async def generate_strategy(
        self, 
        assets: List[Asset], 
        vulnerabilities: List[Vulnerability]
    ) -> TestStrategy:
        """
        生成測試策略
        
        Args:
            assets: 已發現的資產列表
            vulnerabilities: 已知漏洞列表
        
        Returns:
            TestStrategy: 包含優先級排序的測試任務
        """
        # 1. 分析資產特徵
        asset_features = self._extract_features(assets)
        
        # 2. 應用規則引擎
        candidate_tests = self.rule_engine.match_rules(asset_features)
        
        # 3. 計算優先級
        prioritized_tests = self.priority_calculator.rank(
            candidate_tests, 
            vulnerabilities
        )
        
        # 4. 生成策略
        return TestStrategy(
            tests=prioritized_tests,
            rationale=self._explain_strategy(prioritized_tests)
        )
```

##### Phase 2: AI 增強 (2週)

```python
class AIEnhancedStrategyGenerator(StrategyGenerator):
    """使用 AI 模型增強策略生成"""
    
    def __init__(self, model_path: str):
        super().__init__()
        self.ai_model = load_model(model_path)  # 載入訓練好的模型
    
    async def generate_strategy(
        self, 
        assets: List[Asset], 
        vulnerabilities: List[Vulnerability]
    ) -> TestStrategy:
        # 基礎規則引擎策略
        base_strategy = await super().generate_strategy(assets, vulnerabilities)
        
        # AI 模型優化
        optimized_strategy = await self._ai_optimize(base_strategy, assets)
        
        return optimized_strategy
```

**交付成果**:

- ✅ 規則引擎實作 (15+ 規則)
- ✅ 優先級計算器 (多維度評分)
- ✅ AI 模型整合介面
- ✅ 策略解釋器 (可解釋性)

---

### Sprint 3: 任務生成器增強 (2週)

**時間**: Week 6-7 (11/17 - 11/30)

#### Sprint 3 任務清單

- [ ] **TD-003: 增強任務生成邏輯**

**改進點**:

1. **IDOR 候選自動檢測**

```python
# services/core/aiva_core/execution/task_generator.py
class TaskGenerator:
    async def _detect_idor_candidates(self, assets: List[Asset]) -> List[Asset]:
        """
        檢測潛在的 IDOR 漏洞候選
        
        啟發式規則:
        1. URL 包含數字 ID: /user/123, /order/456
        2. 參數名稱包含 id, uid, user_id 等
        3. API 路徑符合 RESTful 模式
        """
        candidates = []
        
        for asset in assets:
            if asset.type == "url":
                # 規則 1: 路徑包含數字
                if re.search(r'/\d+(?:/|$)', asset.value):
                    candidates.append(asset)
                
                # 規則 2: 查詢參數包含 ID
                parsed = urlparse(asset.value)
                params = parse_qs(parsed.query)
                if any('id' in k.lower() for k in params.keys()):
                    candidates.append(asset)
        
        return candidates
```

1. **基於上下文的任務參數配置**

```python
async def generate_tasks(
    self, 
    assets: List[Asset], 
    vulnerabilities: List[Vulnerability],
    scan_context: dict
) -> List[FunctionTaskPayload]:
    """
    根據掃描上下文生成任務
    
    Args:
        scan_context: {
            "authentication": {...},  # 認證信息
            "rate_limit": {...},      # 速率限制
            "scope": {...}            # 掃描範圍
        }
    """
    tasks = []
    
    # XSS 任務
    for form_asset in self._filter_assets(assets, type="form"):
        tasks.append(FunctionTaskPayload(
            module="function_xss",
            target=form_asset,
            context={
                "authentication": scan_context.get("authentication"),
                "test_level": "thorough" if form_asset.metadata.get("critical") else "basic"
            }
        ))
    
    return tasks
```

**交付成果**:

- ✅ IDOR 候選檢測邏輯
- ✅ 上下文感知任務生成
- ✅ 任務參數優化器
- ✅ 單元測試覆蓋率 >80%

---

### Sprint 4: 攻擊面分析完善 (2週)

**時間**: Week 8-9 (12/01 - 12/14)

#### Sprint 4 任務清單

- [ ] **TD-004: 完善攻擊面分析**

**新增分析器**:

```python
# services/core/aiva_core/analysis/attack_surface_analyzer.py
class AttackSurfaceAnalyzer:
    """完整的攻擊面分析器"""
    
    async def analyze(self, scan_data: dict) -> AttackSurface:
        """
        執行全面的攻擊面分析
        
        分析維度:
        1. 認證端點 (登入、註冊、密碼重置)
        2. 授權檢查點 (角色、權限)
        3. 數據輸入點 (表單、API)
        4. 文件上傳點
        5. API 端點 (RESTful, GraphQL)
        6. WebSocket 連接
        7. 第三方整合點
        """
        return AttackSurface(
            authentication_points=self._find_auth_points(scan_data),
            authorization_checks=self._find_authz_checks(scan_data),
            data_input_points=self._find_input_points(scan_data),
            file_upload_points=self._find_upload_points(scan_data),
            api_endpoints=self._find_api_endpoints(scan_data),
            websocket_connections=self._find_websockets(scan_data),
            third_party_integrations=self._find_integrations(scan_data)
        )
```

**交付成果**:

- ✅ 7 種攻擊面分析器
- ✅ 攻擊面視覺化報告
- ✅ 風險評分模型
- ✅ 集成測試

---

### Sprint 5: CI/CD 與文檔 (2週)

**時間**: Week 10-11 (12/15 - 12/31)

#### Sprint 5 任務清單

- [ ] **TD-011: 完善 CI/CD 流程**

```yaml
# .github/workflows/ci.yml
name: AIVA CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install pytest pytest-asyncio pytest-cov
      
      - name: Run tests
        run: pytest --cov=services --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: chartboost/ruff-action@v1
  
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r services/ -f json -o bandit-report.json
```

- [ ] **TD-012: 更新 API 文檔**
  - 使用 FastAPI 自動生成 OpenAPI 文檔
  - 添加 Swagger UI
  - 撰寫開發者指南

**交付成果**:

- ✅ GitHub Actions CI/CD
- ✅ 自動化測試報告
- ✅ OpenAPI 文檔 (Swagger UI)
- ✅ 開發者指南 (README 更新)

---

## 中期計畫 (Q1-Q2 2026)

**時程**: 2026-01-01 ~ 2026-06-30  
**目標**: 多語言架構遷移,性能大幅提升

### Phase 1: 多語言架構基礎 (M1: 契約先行)

**時間**: 2026-01-01 ~ 2026-01-31 (1 個月)

#### 目標

建立跨語言通訊契約與基礎設施

#### Phase 1 任務清單

- [ ] **Proto 契約設計**

```protobuf
// proto/aiva/v1/scan.proto
syntax = "proto3";
package aiva.v1;

import "google/protobuf/timestamp.proto";

service ScanService {
  rpc SubmitTask(ScanTask) returns (TaskAck);
  rpc StreamFindings(ScanTask) returns (stream Finding);
  rpc CancelTask(TaskCancelRequest) returns (TaskCancelResponse);
}

message ScanTask {
  string task_id = 1;
  string module = 2;
  string target = 3;
  map<string,string> meta = 4;
  google.protobuf.Timestamp created_at = 5;
}

message Finding {
  string task_id = 1;
  string module = 2;
  Severity severity = 3;
  string title = 4;
  string summary = 5;
  bytes evidence = 6;
  repeated string cwe_ids = 7;
}

enum Severity {
  SEVERITY_UNSPECIFIED = 0;
  SEVERITY_INFO = 1;
  SEVERITY_LOW = 2;
  SEVERITY_MEDIUM = 3;
  SEVERITY_HIGH = 4;
  SEVERITY_CRITICAL = 5;
}
```

- [ ] **Buf 工作流程設置**
  - buf.yaml 配置
  - buf.gen.yaml 代碼生成配置
  - GitHub Actions 集成

- [ ] **多語言 SDK 生成**
  - Python SDK (使用 grpcio-tools)
  - Go SDK (使用 protoc-gen-go)
  - Node.js SDK (使用 @grpc/grpc-js)
  - Rust SDK (使用 tonic)

- [ ] **互通性測試**
  - Python 客戶端 → Go 服務端
  - Go 客戶端 → Node 服務端
  - 跨語言端到端測試

**交付成果**:

- ✅ Proto 契約倉庫 (proto/)
- ✅ Buf CI/CD 流程
- ✅ 四語言 SDK (Python, Go, Node, Rust)
- ✅ 互通性測試報告

---

### Phase 2: Node.js 掃描服務 (M2: 瀏覽器與觀測)

**時間**: 2026-02-01 ~ 2026-03-31 (2 個月)

#### Phase 2 目標

落地 Node.js + Playwright 動態掃描服務,建立全鏈路追蹤

#### Phase 2 任務清單

- [ ] **TD-005: 實作 aiva-scan-node 微服務**

**架構**:

```text
aiva-scan-node/
├── src/
│   ├── server.ts              # gRPC 服務主入口
│   ├── browser-pool.ts        # 瀏覽器池管理
│   ├── scanners/
│   │   ├── xss-scanner.ts     # XSS 掃描器
│   │   ├── dom-scanner.ts     # DOM 分析器
│   │   └── har-recorder.ts    # HAR 記錄器
│   ├── utils/
│   │   ├── logger.ts          # 日誌工具
│   │   └── tracer.ts          # OpenTelemetry 追蹤
│   └── generated/             # Proto 生成代碼
├── tests/
├── Dockerfile
└── package.json
```

**核心實作**:

```typescript
// src/scanners/xss-scanner.ts
import { chromium, Page } from 'playwright';
import { Finding, Severity } from '../generated/aiva/v1/scan_pb';
import { tracer } from '../utils/tracer';

export class XSSScanner {
  async scan(target: string, context: any): Promise<Finding[]> {
    const span = tracer.startSpan('xss_scan');
    const findings: Finding[] = [];

    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();

    try {
      // 啟用 HAR 記錄
      await page.route('**/*', route => {
        console.log(`${route.request().method()} ${route.request().url()}`);
        route.continue();
      });

      await page.goto(target, { waitUntil: 'networkidle' });

      // 注入污染追蹤腳本
      await page.addInitScript(() => {
        const originalSetAttribute = Element.prototype.setAttribute;
        Element.prototype.setAttribute = function(name: string, value: string) {
          if (name === 'src' && value.includes('<script>')) {
            (window as any).__AIVA_XSS_DETECTED__ = true;
          }
          return originalSetAttribute.call(this, name, value);
        };
      });

      // 測試所有輸入點
      const inputs = await page.locator('input[type="text"], textarea').all();
      for (const input of inputs) {
        const payload = '<img src=x onerror=alert(1)>';
        await input.fill(payload);
        await page.keyboard.press('Enter');

        const xssDetected = await page.evaluate(() => 
          (window as any).__AIVA_XSS_DETECTED__
        );

        if (xssDetected) {
          findings.push(new Finding({
            taskId: context.task_id,
            module: 'xss',
            severity: Severity.SEVERITY_HIGH,
            title: 'Reflected XSS Detected',
            summary: `Input field vulnerable: ${await input.getAttribute('name')}`,
            evidence: Buffer.from(await page.content())
          }));
        }
      }
    } finally {
      await browser.close();
      span.end();
    }

    return findings;
  }
}
```

- [ ] **OpenTelemetry 全鏈路追蹤**

**Python Core 整合**:

```python
# services/core/aiva_core/app.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# 設置 OTel
provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# 自動追蹤 FastAPI
FastAPIInstrumentor.instrument_app(app)

tracer = trace.get_tracer(__name__)

@app.post("/scan")
async def create_scan(task: ScanTask):
    with tracer.start_as_current_span("create_scan") as span:
        span.set_attribute("scan.target", task.target)
        span.set_attribute("scan.module", task.module)
        
        # 呼叫 Node.js gRPC 服務
        async with grpc.aio.insecure_channel('aiva-scan-node:50051') as channel:
            stub = ScanServiceStub(channel)
            async for finding in stub.StreamFindings(task):
                span.add_event("finding_received", {
                    "severity": finding.severity,
                    "title": finding.title
                })
```

**部署配置**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"
      - "4318:4318"
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml

  aiva-scan-node:
    build: ./services/scan/aiva_scan_node
    ports:
      - "50051:50051"
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
    depends_on:
      - otel-collector

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

**交付成果**:

- ✅ aiva-scan-node 微服務
- ✅ Playwright 整合 (Chromium, Firefox, WebKit)
- ✅ OpenTelemetry Collector 部署
- ✅ Jaeger UI 全鏈路追蹤視覺化
- ✅ Prometheus + Grafana 儀表板

---

### Phase 3: Go 高併發探測器 (M3: 性能提升)

**時間**: 2026-04-01 ~ 2026-06-30 (3 個月)

#### Phase 3 目標

以 Go 重構 SSRF/SQLi 探測器,實現 >30% 性能提升

#### Phase 3 任務清單

- [ ] **TD-008: Go SSRF 檢測器**

```go
// services/function/function_ssrf_go/detector.go
package main

import (
    "context"
    "net"
    "net/http"
    "time"
    
    pb "aiva/proto/v1"
    "google.golang.org/grpc"
)

type SSRFDetector struct {
    pb.UnimplementedScanServiceServer
    blockedRanges []*net.IPNet
    client        *http.Client
}

func NewSSRFDetector() *SSRFDetector {
    blockedCIDRs := []string{
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.169.254/32",  // AWS IMDS
        "metadata.google.internal/32",
    }
    
    var ranges []*net.IPNet
    for _, cidr := range blockedCIDRs {
        _, ipNet, _ := net.ParseCIDR(cidr)
        ranges = append(ranges, ipNet)
    }
    
    client := &http.Client{
        Timeout: 5 * time.Second,
        Transport: &http.Transport{
            DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
                host, _, _ := net.SplitHostPort(addr)
                ip := net.ParseIP(host)
                
                for _, blocked := range ranges {
                    if blocked.Contains(ip) {
                        return nil, fmt.Errorf("blocked IP: %s", ip)
                    }
                }
                
                return (&net.Dialer{
                    Timeout: 3 * time.Second,
                }).DialContext(ctx, network, addr)
            },
        },
    }
    
    return &SSRFDetector{
        blockedRanges: ranges,
        client:        client,
    }
}

func (s *SSRFDetector) StreamFindings(
    req *pb.ScanTask,
    stream pb.ScanService_StreamFindingsServer,
) error {
    payloads := []string{
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://127.0.0.1:8080/admin",
    }
    
    // 並發測試 (使用 Goroutine)
    results := make(chan *pb.Finding, len(payloads))
    sem := make(chan struct{}, 10) // 限制並發數
    
    for _, payload := range payloads {
        sem <- struct{}{}
        go func(p string) {
            defer func() { <-sem }()
            
            testURL := req.Target + "?url=" + p
            resp, err := s.client.Get(testURL)
            
            if err == nil && resp.StatusCode == 200 {
                results <- &pb.Finding{
                    TaskId:   req.TaskId,
                    Severity: pb.Severity_SEVERITY_HIGH,
                    Title:    "SSRF Vulnerability Confirmed",
                    Summary:  fmt.Sprintf("Payload %s succeeded", p),
                }
            }
        }(payload)
    }
    
    // 等待所有 Goroutine 完成
    for i := 0; i < len(payloads); i++ {
        sem <- struct{}{}
    }
    close(results)
    
    // 串流回報結果
    for finding := range results {
        if err := stream.Send(finding); err != nil {
            return err
        }
    }
    
    return nil
}
```

- [ ] **TD-007: Go SQLi 檢測器**

```go
// services/function/function_sqli_go/detector.go
type SQLiDetector struct {
    pb.UnimplementedScanServiceServer
    payloads []SQLiPayload
}

type SQLiPayload struct {
    Value       string
    Type        string  // "union", "boolean", "time-based"
    Description string
}

func (s *SQLiDetector) StreamFindings(
    req *pb.ScanTask,
    stream pb.ScanService_StreamFindingsServer,
) error {
    // 時間盲注測試
    timeBasedPayloads := []string{
        "1' AND SLEEP(5)--",
        "1' OR SLEEP(5)--",
        "1'; WAITFOR DELAY '00:00:05'--",
    }
    
    for _, payload := range timeBasedPayloads {
        start := time.Now()
        resp, _ := s.testPayload(req.Target, payload)
        elapsed := time.Since(start)
        
        if elapsed > 4*time.Second && elapsed < 6*time.Second {
            stream.Send(&pb.Finding{
                TaskId:   req.TaskId,
                Severity: pb.Severity_SEVERITY_CRITICAL,
                Title:    "Time-Based SQL Injection",
                Summary:  fmt.Sprintf("Response time: %.2fs", elapsed.Seconds()),
                Evidence: []byte(payload),
            })
        }
    }
    
    return nil
}
```

- [ ] **性能基準測試**

```go
// benchmarks/ssrf_benchmark_test.go
func BenchmarkSSRFDetection(b *testing.B) {
    detector := NewSSRFDetector()
    task := &pb.ScanTask{
        TaskId: "bench-001",
        Target: "http://testsite.local",
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        stream := &mockStream{}
        detector.StreamFindings(task, stream)
    }
}
```

**交付成果**:

- ✅ Go SSRF 檢測器 (含 IP 黑名單)
- ✅ Go SQLi 檢測器 (Union/Boolean/Time-based)
- ✅ 性能基準報告 (Go vs Python)
- ✅ A/B 測試 (20% 流量切換)
- ✅ 金絲雀部署文檔

---

## 長期計畫 (Q3-Q4 2026)

**時程**: 2026-07-01 ~ 2026-12-31  
**目標**: AI 增強,企業級功能

### Phase 1: Rust 敏感資訊掃描器 (Q3)

**時間**: 2026-07-01 ~ 2026-09-30

#### 長期 Phase 1 任務清單

- [ ] **Rust 正則引擎實作**

```rust
// services/info_gatherer_rust/src/detector.rs
use regex::RegexSet;
use rayon::prelude::*;
use tonic::{Request, Response, Status};

pub struct InfoGatherer {
    sensitive_patterns: RegexSet,
}

impl InfoGatherer {
    pub fn new() -> Self {
        let patterns = vec![
            r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",
            r"(?i)api[_-]?key\s*[:=]\s*[\w-]{20,}",
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  // Credit card
            r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
            r"eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*",  // JWT
        ];
        Self {
            sensitive_patterns: RegexSet::new(patterns).unwrap(),
        }
    }
}

#[tonic::async_trait]
impl ScanService for InfoGatherer {
    type StreamFindingsStream = ReceiverStream<Result<Finding, Status>>;

    async fn stream_findings(
        &self,
        request: Request<ScanTask>,
    ) -> Result<Response<Self::StreamFindingsStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let task = request.into_inner();
        let patterns = self.sensitive_patterns.clone();

        tokio::spawn(async move {
            // 並行掃描 (Rayon 資料並行)
            let findings: Vec<_> = task.content
                .par_lines()
                .enumerate()
                .filter_map(|(line_num, line)| {
                    if patterns.is_match(line) {
                        Some(Finding {
                            task_id: task.task_id.clone(),
                            severity: Severity::SevHigh as i32,
                            title: "Sensitive Information Leak".into(),
                            summary: format!("Line {}: {}", line_num, line),
                            ..Default::default()
                        })
                    } else {
                        None
                    }
                })
                .collect();

            for finding in findings {
                tx.send(Ok(finding)).await.ok();
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

**交付成果**:

- ✅ Rust 敏感資訊掃描器
- ✅ 性能提升 >10x (vs Python)
- ✅ WASM 編譯版本 (選配)

---

### Phase 2: AI 策略優化 (Q4)

**時間**: 2026-10-01 ~ 2026-12-31

#### 長期 Phase 2 任務清單

- [ ] **訓練資料收集**
  - 歷史掃描數據 (>10,000 次掃描)
  - 漏洞發現記錄
  - 攻擊面特徵標註

- [ ] **ML 模型訓練**
  - 特徵工程 (TF-IDF, Word2Vec)
  - 模型選擇 (XGBoost, Random Forest)
  - 超參數調優

- [ ] **模型整合**

```python
# services/core/aiva_core/ai_engine/ml_strategy_optimizer.py
import joblib
from sklearn.ensemble import RandomForestClassifier

class MLStrategyOptimizer:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict_vulnerability_likelihood(
        self, 
        asset_features: dict
    ) -> dict[str, float]:
        """
        預測各漏洞類型的可能性
        
        Returns:
            {'xss': 0.75, 'sqli': 0.45, 'ssrf': 0.20, ...}
        """
        X = self._extract_features(asset_features)
        probabilities = self.model.predict_proba([X])[0]
        
        return {
            vuln_type: prob
            for vuln_type, prob in zip(self.model.classes_, probabilities)
        }
```

**交付成果**:

- ✅ ML 模型訓練流程
- ✅ 模型準確率 >85%
- ✅ 策略優化效果評估

---

## 多語言架構遷移路線圖

### 整體時程

```text
2025 Q4          2026 Q1          2026 Q2          2026 Q3          2026 Q4
   |                |                |                |                |
   |-- Python 為主 --|                |                |                |
   |   (穩定現有)   |                |                |                |
   |                |-- M1: 契約 --|                |                |
   |                |    (Proto)    |                |                |
   |                |                |-- M2: Node.js--|                |
   |                |                |   (Playwright)|                |
   |                |                |                |-- M3: Go/Rust--|
   |                |                |                |   (高併發)    |
   |                |                |                |                |-- AI 增強 --|
```

### 語言選型總結

| 模組 | 當前語言 | 目標語言 | 遷移時程 | 預期提升 |
|------|----------|----------|----------|----------|
| Core | Python ✅ | Python | - | - |
| Scan (靜態) | Python ✅ | Python | - | - |
| Scan (動態) | Python ⚠️ | Node.js | 2026 Q1-Q2 | 效率 +50% |
| Info Gatherer | Python ⚠️ | Rust | 2026 Q3 | 速度 +10x |
| SSRF | Python ⚠️ | Go | 2026 Q2 | 吞吐 +3x |
| SQLi | Python ⚠️ | Go | 2026 Q2 | 吞吐 +3x |
| IDOR | Python ✅ | Python | - | - |
| XSS | Python ⚠️ | Node.js | 2026 Q1-Q2 | DOM 分析 +100% |

---

## 資源需求與團隊配置

### 團隊規模

| 角色 | 人數 | 技能需求 |
|------|------|----------|
| **後端工程師** | 2 | Python, FastAPI, 異步編程 |
| **全端工程師** | 1 | Python + Node.js + Go |
| **安全研究員** | 1 | 漏洞研究, 滲透測試 |
| **DevOps 工程師** | 1 | Docker, K8s, CI/CD |
| **QA 工程師** | 1 | 自動化測試, 性能測試 |

### 技能培訓計畫

#### Week 1-2: Go 基礎

- Go Tour 完成
- 併發模型 (Goroutine/Channel)
- gRPC 實作練習

#### Week 3-4: Protobuf & gRPC

- Protocol Buffers 設計
- gRPC 四種通訊模式
- 跨語言互通性測試

#### Week 5-6: Rust 基礎

- The Rust Book Ch 1-10
- 所有權系統理解
- Async Rust (Tokio)

#### Week 7-8: 觀測性工具

- OpenTelemetry SDK
- Prometheus + Grafana
- Jaeger 分散式追蹤

---

## 風險管理與應變計畫

### 風險矩陣

| 風險 | 可能性 | 影響 | 嚴重度 | 應變措施 |
|------|--------|------|--------|----------|
| **多語言整合失敗** | 中 | 高 | 🔴 高 | 保留 Python 備份,分階段切換 |
| **性能未達預期** | 低 | 中 | 🟡 中 | 基準測試驗證,A/B 測試 |
| **團隊技能不足** | 中 | 中 | 🟡 中 | 8 週培訓計畫,外部顧問 |
| **時程延誤** | 高 | 中 | 🟡 中 | 優先級調整,砍次要功能 |
| **依賴套件安全漏洞** | 中 | 高 | 🔴 高 | Dependabot,定期掃描 |

### 應變計畫

#### Plan A: 正常執行

按照路線圖逐步推進

#### Plan B: 時程延誤 (>20%)

- 砍掉 Rust 敏感資訊掃描器 (改用 Python)
- AI 策略優化推遲到 2027 Q1
- 保留核心功能 (Node.js + Go)

#### Plan C: 技術障礙 (多語言整合失敗)

- 全部保持 Python
- 優化現有代碼 (異步改進, 多進程)
- 採用 Cython 加速關鍵路徑

---

## 關鍵績效指標 (KPIs)

### 技術指標

| 指標 | 基線 (2025 Q3) | 目標 (2026 Q4) | 測量方式 |
|------|----------------|----------------|----------|
| **測試覆蓋率** | 15% | >80% | Codecov |
| **掃描吞吐量** | 50 tasks/min | >150 tasks/min | Prometheus Counter |
| **P95 延遲** | 15s | <5s | OTel Histogram |
| **錯誤率** | 2.5% | <0.5% | Error Rate Monitor |
| **服務可用性** | 95% | >99.9% | Uptime Monitor |
| **內存使用** | 2GB | <1.5GB | cAdvisor |

### 業務指標

| 指標 | 基線 | 目標 | 測量方式 |
|------|------|------|----------|
| **漏洞檢出率** | 60% | >90% | 人工驗證 |
| **誤報率** | 15% | <5% | 人工驗證 |
| **掃描時間** | 30 min | <10 min | 端到端計時 |
| **支援網站類型** | 10 | >30 | 測試用例 |

---

## 技術決策記錄 (ADR)

### ADR-001: 選擇 Pydantic v2 作為數據驗證框架

**日期**: 2025-10-13  
**狀態**: ✅ 已採納

**背景**:
需要統一的數據驗證框架,替換 dataclass

**決策**:
採用 Pydantic v2.12.0

**理由**:

1. 自動驗證 (型別檢查)
2. JSON 序列化/反序列化
3. FastAPI 原生支援
4. 性能優異 (Rust 核心)

**後果**:

- ✅ 代碼一致性提升
- ⚠️ 需遷移現有 dataclass (已完成)

---

### ADR-002: 選擇 gRPC 作為跨語言通訊協議

**日期**: 2025-10-13  
**狀態**: 📋 提議中

**背景**:
需要高性能的跨語言通訊機制

**決策**:
採用 gRPC + Protocol Buffers

**理由**:

1. 官方支援 12+ 語言
2. HTTP/2 性能優勢
3. 原生串流支援
4. 型別安全

**替代方案**:

- REST API (JSON) - 被拒絕 (性能較差)
- Thrift - 被拒絕 (社群較小)

---

### ADR-003: 選擇 Node.js + Playwright 作為動態掃描引擎

**日期**: 2025-10-13  
**狀態**: 📋 提議中

**背景**:
需要強大的瀏覽器自動化能力

**決策**:
採用 Node.js + Playwright

**理由**:

1. Playwright JS 生態最成熟
2. Event Loop 適合瀏覽器 I/O
3. 官方優先支援 TypeScript
4. 社群資源豐富

**替代方案**:

- Python + Playwright - 被拒絕 (性能較差)
- Selenium - 被拒絕 (過時)

---

## 執行建議

### 立即行動 (本週)

1. **團隊評審會議**
   - 討論本藍圖
   - 分配 Sprint 1 任務
   - 確認資源與時程

2. **設置開發環境**
   - 安裝 pytest, pytest-cov
   - 設置 GitHub Actions
   - 建立測試目錄結構

3. **啟動 Sprint 1**
   - 撰寫第一個單元測試
   - 設置 Codecov 整合
   - 建立每日站會機制

### 下個月目標

- ✅ 測試覆蓋率達 60%
- ✅ 策略生成器重建完成
- ✅ 任務生成器增強
- ✅ CI/CD 流程上線

### 季度里程碑

- **2025 Q4**: 補足關鍵功能,測試覆蓋率 >80%
- **2026 Q1**: Proto 契約完成,多語言 SDK 就緒
- **2026 Q2**: Node.js 掃描服務上線,性能提升 50%
- **2026 Q3**: Go 探測器上線,吞吐提升 3x
- **2026 Q4**: AI 增強策略,漏洞檢出率 >90%

---

## 附錄

### A. 參考文檔

- [ARCHITECTURE_REPORT.md](./ARCHITECTURE_REPORT.md) - 四大模組架構
- [DATA_CONTRACT.md](./DATA_CONTRACT.md) - 數據合約文檔
- [CORE_MODULE_ANALYSIS.md](./CORE_MODULE_ANALYSIS.md) - Core 模組分析
- [SCAN_ENGINE_IMPROVEMENT_REPORT.md](./SCAN_ENGINE_IMPROVEMENT_REPORT.md) - 掃描引擎報告
- [MULTI_LANGUAGE_ARCHITECTURE_PROPOSAL.md](./MULTI_LANGUAGE_ARCHITECTURE_PROPOSAL.md) - 多語言架構方案
- [QUICK_START.md](./QUICK_START.md) - 快速開始指南

### B. 外部資源

- [gRPC 官方文檔](https://grpc.io/docs/)
- [Playwright 文檔](https://playwright.dev/docs/intro)
- [OpenTelemetry 文檔](https://opentelemetry.io/docs/)
- [Go 併發模型](https://go.dev/blog/pipelines)
- [Rust 所有權系統](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)

### C. 工具清單

| 分類 | 工具 | 用途 |
|------|------|------|
| **測試** | pytest, pytest-asyncio, pytest-cov | 單元測試 |
| **代碼品質** | Ruff, Mypy, Bandit | 格式化, 型別檢查, 安全掃描 |
| **CI/CD** | GitHub Actions, Docker, K8s | 自動化部署 |
| **監控** | Prometheus, Grafana, Jaeger | 性能監控, 追蹤 |
| **文檔** | Sphinx, MkDocs | API 文檔生成 |

---

**文件結束**  
**維護者**: AIVA 技術團隊  
**下次更新**: 2026-01-13 (每季度更新)
