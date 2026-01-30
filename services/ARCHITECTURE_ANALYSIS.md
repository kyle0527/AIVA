# 📊 AIVA Services 架構深度分析報告
生成時間: 2026-01-10
分析目標: C:\D\fold7\AIVA-git\services

> ⚠️ **重要聲明**：本文檔記錄的是**已實現但部分從未被使用**的架構組件。  
> ✅ 實際運行架構請參考：[../AIVA_CLI_ARCHITECTURE_REFACTOR_PLAN.md](../AIVA_CLI_ARCHITECTURE_REFACTOR_PLAN.md)  
> 📌 本文檔價值：了解完整架構設計，為未來功能擴展提供參考  
> ❌ **Coordinator/Dispatcher 已驗證從未被 AI 調用** - AI 直接使用 subprocess + JSON

---

## 🎯 執行摘要

**AIVA Services** 是一個**企業級 Bug Bounty 微服務平台**，採用 **Python/TypeScript/Rust/Go** 四語言協同架構，專精於動態漏洞檢測、黑盒滲透測試和智能攻擊策略規劃。

### 核心數據
- **總代碼量**: 913+ Python 文件，2000+ 多語言組件
- **架構版本**: v7.1-stable
- **最後更新**: 2026-01-09
- **技術棧**: Python (主導) + TypeScript (動態掃描) + Rust (性能) + Go (並發)
- **標準支援**: CVSS v3.1、MITRE ATT&CK、SARIF v2.1.0、CVE/CWE/CAPEC

---

## 🏗️ 五大核心模組架構

### 1️⃣ **aiva_common - 共享基礎設施庫** (100+ 模組)

**定位**: 整個系統的底層基礎設施，提供統一的數據契約和跨語言適配

#### 核心組件
```
aiva_common/
├── ai/                        # 🤖 AI 基礎設施
│   ├── interfaces.py          # AI 抽象介面
│   ├── performance_config.py  # AI 性能配置
│   └── registry.py            # AI 模型註冊
├── enums/ (13個領域)          # 📋 標準枚舉定義
│   ├── academic.py            # 學術研究枚舉
│   ├── ai.py                  # AI 相關枚舉
│   ├── security.py            # 安全枚舉
│   ├── pentest.py             # 滲透測試枚舉
│   └── web_api_standards.py   # Web API 標準
├── schemas/ (200+ 模型)       # 📐 數據 Schema 定義
│   ├── _base/                 # 基礎 Schema
│   ├── analysis/              # 分析相關 Schema
│   ├── security/              # 安全相關 Schema
│   └── testing/               # 測試相關 Schema
├── cross_language/            # 🌐 跨語言適配器
│   ├── adapters/
│   │   ├── go_adapter.py      # Go 語言適配
│   │   └── rust_adapter.py    # Rust 語言適配
│   └── core.py                # 跨語言核心
├── messaging/                 # 📨 消息傳遞系統
│   ├── unified_topic_manager.py
│   └── retry_handler.py
├── protocols/                 # 🔗 gRPC 協議定義
│   ├── aiva_enums_pb2*.py
│   ├── aiva_services_pb2*.py
│   └── generate_proto.py
└── utils/                     # 🔧 通用工具
    ├── network/               # 網路工具
    ├── logging.py             # 日誌工具
    └── retry.py               # 重試工具
```

#### 關鍵特性
- ✅ **統一數據契約**: 200+ Pydantic 模型，確保類型安全
- ✅ **跨語言適配**: Go/Rust 適配器，實現多語言協同
- ✅ **標準化枚舉**: 13 個領域枚舉，避免魔法值
- ✅ **gRPC 協議**: 高性能跨服務通信
- ✅ **Schema 代碼生成**: 自動生成 TypeScript 類型定義

#### 技術亮點
```python
# 統一日誌系統
from aiva_common.utils.logging import get_logger
logger = get_logger(__name__)

# 統一錯誤處理
from aiva_common.error_handling import AIVAError, ErrorType
raise AIVAError(ErrorType.VALIDATION_ERROR, "Invalid input")

# 統一數據模型
from aiva_common.schemas import HighLevelIntent, DecisionConstraints
intent = HighLevelIntent(intent_type=IntentType.SCAN, target_info=...)
```

---

### 2️⃣ **core - AI 驅動核心引擎** (2000+ 行 AI 代碼)

**定位**: 系統的智能大腦，負責 AI 決策、學習和任務規劃

#### 核心子系統
```
core/aiva_core/
├── cognitive_core/            # 🧠 認知核心
│   ├── anti_hallucination/    # 🛡️ 反幻覺模組
│   ├── decision/              # 🎯 決策引擎
│   │   └── enhanced_decision_agent.py (2231 行)
│   ├── neural/                # 🧠 神經網路
│   │   └── real_neural_core.py
│   ├── rag/                   # 📚 檢索增強生成
│   ├── internal_loop_connector.py (2036 行)
│   ├── external_loop_connector.py
│   └── nlg_system.py          # 自然語言生成
├── core_capabilities/         # 💪 核心能力
│   ├── analysis/              # 🔍 分析引擎
│   ├── attack/                # ⚔️ 攻擊能力
│   ├── dialog/                # 💬 對話系統
│   │   └── assistant.py       # AI 助手
│   ├── ingestion/             # 📥 數據攝取
│   ├── processing/            # ⚙️ 處理引擎
│   └── multilang_coordinator.py
├── external_learning/         # 🎓 外部學習（外閉環）
│   ├── ai_model/              # 🤖 AI 模型
│   ├── analysis/              # 📊 分析學習
│   ├── learning/              # 📖 機器學習
│   ├── training/              # 🏋️ 訓練管理
│   └── experience_manager.py
├── internal_exploration/      # 🔬 內部探索（內閉環）
│   ├── capability_analyzer.py
│   ├── language_extractors.py
│   └── module_explorer.py
├── task_planning/             # 📋 任務規劃
│   ├── executor/              # 🚀 執行器
│   ├── planner/               # 📅 規劃器
│   ├── ai_commander.py
│   └── command_router.py
├── service_backbone/          # 🦴 服務骨幹
│   ├── coordination/          # 🤝 協調系統
│   ├── messaging/             # 📨 消息系統
│   ├── monitoring/            # 📊 監控系統
│   ├── storage/               # 💽 存儲管理
│   └── context_manager.py
└── ui_panel/                  # 🖥️ UI 面板
```

#### 關鍵 AI 組件

##### 1. **EnhancedDecisionAgent** (2231 行)
```python
class EnhancedDecisionAgent:
    """整合 5M 神經網路 + RAG 檢索的 AI 決策代理"""
    
    def __init__(self, knowledge_base=None, experience_manager=None):
        self.knowledge_base = knowledge_base
        self.experience_manager = experience_manager
        # 初始化真實神經網路引擎
        self.neural_engine = RealDecisionEngine()
    
    # 四大決策方法
    async def decide_scan_strategy(self, context) -> Decision:
        """智慧掃描工具選擇"""
        # 整合 RAG 檢索 + 神經網路決策
        
    async def decide_phase1_strategy(self, context) -> Decision:
        """Phase1 深度掃描決策"""
        
    async def decide_phase2_targets(self, context) -> Decision:
        """Phase2 攻擊目標優先級排序"""
        
    async def evaluate_phase2_results(self, results) -> Decision:
        """Phase2 結果評估"""
```

**特性**:
- ✅ 整合 5M 神經網路（RealDecisionEngine）
- ✅ RAG 向量檢索（去語意化反射引擎）
- ✅ 風險評估與經驗驅動決策
- ✅ 四階段完整決策流程

##### 2. **InternalLoopConnector** (2036 行)
```python
class InternalLoopConnector:
    """內部閉環連接器 - 實現 AI 自我認知"""
    
    def __init__(self, rag_knowledge_base):
        self.rag_kb = rag_knowledge_base
        self.classifier = CapabilityScopeClassifier()
    
    async def sync_to_rag(self, capabilities: List[ModuleCapability]):
        """將內部探索結果注入 RAG 知識庫"""
        for cap in capabilities:
            # 自動分類能力範圍
            scope, visibility = self.classifier.classify_scope(cap.file_path)
            # 注入 RAG
            await self.rag_kb.add_capability(cap)

class CapabilityScopeClassifier:
    """能力範圍分類器 - 基於文件路徑自動分類"""
    
    def classify_scope(self, file_path: str) -> tuple[CapabilityScope, CapabilityVisibility]:
        """
        分類規則:
        - services/features     → FEATURE (功能層能力)
        - services/scan         → INFRASTRUCTURE (掃描基礎)
        - services/integration  → INFRASTRUCTURE (整合基礎)
        - services/core         → INTERNAL (核心內部)
        """
```

**特性**:
- ✅ 三階段分析管道整合（aiva_flow_analyzer → classifier → implementation）
- ✅ 自動能力範圍分類（基於文件路徑）
- ✅ RAG 知識庫注入
- ✅ 實現 AI 對自身能力的認知

##### 3. **雙閉環學習架構**

**內閉環（Internal Loop）**:
```
internal_exploration/ → InternalLoopConnector → RAG Knowledge Base
                     ↓
              AI 自我認知能力
```

**外閉環（External Loop）**:
```
執行結果 → ExternalLoopConnector → Experience Manager → Training Pipeline
        ↓
   持續學習與優化
```

---

### 3️⃣ **features - 多語言安全功能** (2692 個組件)

**定位**: 實際的攻擊功能實現，支援多種漏洞檢測

#### 功能模組清單
```
features/
├── features_ready/ (生產就緒)
│   ├── function_sqli/         # 💉 SQL 注入檢測
│   │   ├── engines/ (6個引擎)
│   │   │   ├── boolean_detection_engine.py    # 布林盲注
│   │   │   ├── error_detection_engine.py      # 錯誤注入
│   │   │   ├── time_detection_engine.py       # 時間盲注
│   │   │   ├── union_detection_engine.py      # 聯合查詢
│   │   │   ├── oob_detection_engine.py        # 帶外檢測
│   │   │   └── hackingtool_engine.py          # Hackingtool 引擎
│   │   ├── detector/          # 檢測器
│   │   ├── config/            # 配置
│   │   ├── integration_tools/ # 整合工具
│   │   └── worker.py          # 工作器
│   ├── function_xss/          # 🚨 XSS 檢測
│   │   ├── traditional_detector.py    # 反射型 XSS
│   │   ├── stored_detector.py         # 存儲型 XSS
│   │   ├── payload_generator.py       # Payload 生成器
│   │   └── worker.py
│   ├── function_ssrf/         # 🔗 SSRF 檢測
│   │   ├── internal_address_detector.py
│   │   ├── oast_dispatcher.py         # 帶外檢測
│   │   ├── param_semantics_analyzer.py
│   │   └── worker.py
│   ├── function_idor/         # 🔓 IDOR 檢測
│   │   ├── smart_idor_detector.py
│   │   ├── resource_id_extractor.py
│   │   └── enhanced_worker.py
│   └── function_info_leak/    # 📄 信息洩漏檢測
│       └── sensitive_info_detector.py
├── features_in_development/ (開發中)
│   ├── function_postex/       # 🎯 後滲透
│   │   ├── engines/
│   │   │   ├── lateral_engine.py      # 橫向移動
│   │   │   ├── persistence_engine.py  # 持久化
│   │   │   └── privilege_engine.py    # 權限提升
│   │   └── worker/
│   ├── function_authn_go/     # 🔐 Go 認證功能
│   │   ├── cmd/worker/
│   │   └── internal/
│   ├── function_bizlogic/     # 🏢 業務邏輯漏洞
│   │   └── worker.py
│   └── function_crypto/       # 🔒 加密功能
│       ├── rust_core/         # Rust 核心
│       └── python_wrapper/    # Python 包裝器
├── common/                    # 🔗 通用功能
│   ├── go/aiva_common_go/     # Go 共享庫
│   │   ├── config/
│   │   ├── logger/
│   │   ├── metrics/
│   │   └── mq/
│   └── testers/
│       ├── cross_user_tester.py
│       └── vertical_escalation_tester.py
└── base/                      # 📦 基礎設施
    └── feature_registry.py
```

#### SQL 注入檢測引擎架構

**6 大檢測引擎協同工作**:
```python
# 1. 布林盲注引擎
class BooleanDetectionEngine:
    """基於布林邏輯的盲注檢測"""
    async def detect(self, target_url, payload):
        # 發送 True/False 條件 Payload
        # 比較響應差異
        
# 2. 時間盲注引擎
class TimeDetectionEngine:
    """基於時間延遲的盲注檢測"""
    async def detect(self, target_url, payload):
        # 注入 SLEEP() 函數
        # 測量響應時間
        
# 3. 錯誤注入引擎
class ErrorDetectionEngine:
    """基於錯誤信息的注入檢測"""
    async def detect(self, target_url, payload):
        # 觸發數據庫錯誤
        # 解析錯誤信息

# 4. 聯合查詢引擎
class UnionDetectionEngine:
    """基於 UNION 查詢的注入檢測"""
    
# 5. 帶外檢測引擎
class OOBDetectionEngine:
    """基於帶外通道的檢測"""
    
# 6. Hackingtool 引擎
class HackingtoolEngine:
    """整合專業滲透測試工具"""
```

**多引擎協同策略**:
1. 並行執行 6 個引擎
2. 結果去重和驗證
3. 風險評分聚合
4. 生成統一報告

#### XSS 檢測架構

```python
# 反射型 XSS 檢測
class TraditionalDetector:
    async def detect_reflected_xss(self, url, param):
        # 生成多種編碼的 Payload
        # 檢測響應中的 Payload 回顯
        # 驗證 XSS 是否可執行

# 存儲型 XSS 檢測
class StoredDetector:
    async def detect_stored_xss(self, url, storage_endpoint):
        # 提交 Payload 到存儲點
        # 訪問不同頁面觸發
        # 驗證持久化 XSS

# Payload 生成器
class PayloadGenerator:
    def generate_xss_payloads(self, context: str) -> List[str]:
        """
        根據上下文生成針對性 Payload:
        - HTML 標籤內
        - JavaScript 字符串內
        - 事件處理器內
        - URL 參數內
        """
```

---

### 4️⃣ **integration - 企業級整合中樞** (協調器架構)

**定位**: 協調各服務間的通信，提供統一的 API 閘道

#### 核心架構
```
integration/aiva_integration/
├── coordinators/              # 🎯 功能協調器
│   ├── base_coordinator.py    # 基礎協調器類
│   └── xss_coordinator.py     # XSS 協調器示例
├── api_gateway/               # 🌐 API 閘道
│   └── app.py                 # FastAPI 應用
├── capability/                # 🔧 能力管理
│   └── command_handler.py     # 命令處理器
├── tools/                     # 🛠️ 整合工具
├── scripts/                   # 📜 腳本工具
└── data/                      # 💾 數據存儲
    └── internal_exploration/
```

#### 協調器架構設計

##### 基礎協調器（BaseCoordinator）
```python
class BaseCoordinator(ABC):
    """統一協調器基類"""
    
    def __init__(self, feature_module: str):
        self.feature_module = feature_module
        self.logger = get_logger(f"{__name__}.{feature_module}")
        
    @abstractmethod
    async def execute(
        self, 
        target: str, 
        config: Optional[dict] = None
    ) -> FeatureResult:
        """執行攻擊檢測"""
        pass
    
    async def publish_result(self, result: FeatureResult):
        """發布結果到消息隊列"""
        
    async def verify(self, finding: CoordinatorFinding) -> VerificationResult:
        """驗證漏洞真實性"""
```

##### XSS 協調器實現
```python
class XSSCoordinator(BaseCoordinator):
    """XSS 功能協調器"""
    
    async def execute(self, target: str, config: Optional[dict] = None) -> FeatureResult:
        """
        執行 XSS 檢測流程:
        1. 調用 TraditionalDetector (反射型)
        2. 調用 StoredDetector (存儲型)
        3. 結果聚合和去重
        4. 風險評分
        5. 發布到消息隊列
        """
        findings = []
        
        # 反射型檢測
        reflected = await self._detect_reflected(target)
        findings.extend(reflected)
        
        # 存儲型檢測
        stored = await self._detect_stored(target)
        findings.extend(stored)
        
        # 聚合結果
        return self._aggregate_results(findings)
    
    def _classify_xss_payload(self, payload: str) -> str:
        """分類 XSS Payload 類型"""
        if "<script>" in payload:
            return "script_tag"
        elif "onerror=" in payload:
            return "event_handler"
        # ...
```

#### API 閘道架構

```python
# app.py - FastAPI 應用
from fastapi import FastAPI
from aiva_integration import VulnerabilityCorrelationAnalyzer

app = FastAPI(title="AIVA Integration API")

@app.post("/scan/xss")
async def scan_xss(target: str, config: dict = None):
    """XSS 掃描端點"""
    coordinator = XSSCoordinator("xss")
    result = await coordinator.execute(target, config)
    return result

@app.post("/scan/sqli")
async def scan_sqli(target: str, config: dict = None):
    """SQL 注入掃描端點"""
    # 類似實現

@app.get("/findings/{finding_id}")
async def get_finding(finding_id: str):
    """獲取漏洞詳情"""
    # 從數據庫查詢
```

---

### 5️⃣ **scan - 多語言統一掃描引擎** (289 個組件)

**定位**: 高性能多語言掃描引擎，支援主動掃描和被動監聽

#### 多語言協同架構
```
scan/
├── aiva_scan/                 # 🐍 Python 掃描核心
│   ├── core_crawling_engine/  # 🕷️ 核心爬蟲引擎
│   │   ├── crawler.py         # 智能爬蟲
│   │   ├── url_queue.py       # URL 隊列管理
│   │   └── session_manager.py # 會話管理
│   ├── dynamic_engine/        # ⚡ 動態掃描引擎
│   │   ├── browser_pool.py    # 瀏覽器池管理
│   │   └── ajax_handler.py    # AJAX 處理
│   └── intelligence/          # 🔍 情報收集
│       ├── js_analyzer.py     # JavaScript 分析
│       └── passive_scanner.py # 被動掃描
├── aiva_scan_node/            # 📘 TypeScript 動態掃描
│   ├── src/
│   │   ├── services/
│   │   │   ├── browser.service.ts       # 瀏覽器服務
│   │   │   ├── crawler.service.ts       # 爬蟲服務
│   │   │   ├── network-interceptor.service.ts  # 網路攔截
│   │   │   └── scan-service.ts          # 掃描服務
│   │   └── index.ts
│   └── package.json
├── go_scanners/               # 🐹 Go 高性能掃描器
│   ├── cloud_security/        # ☁️ 雲安全掃描
│   │   ├── aws_scanner.go
│   │   ├── azure_scanner.go
│   │   └── gcp_scanner.go
│   ├── sca_scanner/           # 📦 軟體組成分析
│   ├── secrets_scanner/       # 🔐 機密掃描
│   └── vulndb_scanner/        # 🗃️ 漏洞資料庫掃描
├── info_gatherer_rust/        # 🦀 Rust 情報收集器
│   ├── src/
│   │   ├── modules/
│   │   │   ├── port_scanner.rs      # 端口掃描
│   │   │   ├── fingerprint.rs       # 指紋識別
│   │   │   └── subdomain_enum.rs    # 子域名枚舉
│   │   └── utils/
│   └── Cargo.toml
└── python_engine/             # 🐍 Python 掃描引擎
    ├── deserialization_detector.py   # 反序列化檢測
    ├── xxe_detector.py               # XXE 檢測
    └── passive_analyzer.py           # 被動分析
```

#### Python 爬蟲引擎

```python
class CoreCrawlingEngine:
    """核心爬蟲引擎 - 反反爬蟲策略"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.url_queue = URLQueue()
        self.seen_urls = set()
    
    async def crawl(self, start_url: str, depth: int = 3):
        """
        智能爬蟲功能:
        - 反反爬蟲: User-Agent 輪換、延迟控制
        - 會話管理: Cookie 持久化、認證保持
        - 智能隊列: 優先級排序、去重
        - 深度控制: 避免無限循環
        """
        self.url_queue.add(start_url, priority=0)
        
        while not self.url_queue.empty():
            url, current_depth = self.url_queue.get()
            
            if current_depth >= depth:
                continue
            
            # 發送請求（帶反爬蟲策略）
            response = await self.session_manager.get(
                url,
                headers=self._get_random_headers()
            )
            
            # 提取新 URL
            new_urls = self._extract_urls(response)
            for new_url in new_urls:
                if new_url not in self.seen_urls:
                    self.url_queue.add(new_url, priority=current_depth+1)
                    self.seen_urls.add(new_url)
```

#### TypeScript 動態掃描引擎

```typescript
// browser.service.ts - Playwright 瀏覽器自動化
export class BrowserService {
    private browserPool: Browser[] = [];
    
    async initPool(size: number = 3) {
        // 初始化瀏覽器池（無頭模式）
        for (let i = 0; i < size; i++) {
            const browser = await chromium.launch({ headless: true });
            this.browserPool.push(browser);
        }
    }
    
    async scanSPA(url: string): Promise<ScanResult> {
        // 單頁應用掃描
        const page = await this.browserPool[0].newPage();
        
        // 網路攔截（捕獲 AJAX 請求）
        await page.route('**/*', async (route) => {
            const request = route.request();
            // 記錄 API 請求
            this.logAPIRequest(request);
            await route.continue();
        });
        
        // 訪問頁面
        await page.goto(url);
        
        // 等待 JavaScript 執行
        await page.waitForLoadState('networkidle');
        
        // 提取動態生成的內容
        const content = await page.content();
        
        return { urls: this.extractURLs(content), apis: this.apiRequests };
    }
}

// network-interceptor.service.ts - 網路攔截
export class NetworkInterceptorService {
    async interceptTraffic(page: Page): Promise<HttpRequest[]> {
        const requests: HttpRequest[] = [];
        
        page.on('request', (request) => {
            requests.push({
                url: request.url(),
                method: request.method(),
                headers: request.headers(),
                postData: request.postData()
            });
        });
        
        return requests;
    }
}
```

#### Rust 高性能情報收集

```rust
// port_scanner.rs - 高性能端口掃描
use tokio::net::TcpStream;
use tokio::time::{timeout, Duration};

pub struct PortScanner {
    target: String,
    timeout_ms: u64,
}

impl PortScanner {
    pub async fn scan_range(&self, start: u16, end: u16) -> Vec<u16> {
        let mut open_ports = Vec::new();
        let mut tasks = Vec::new();
        
        // 並發掃描（Tokio 異步運行時）
        for port in start..=end {
            let target = self.target.clone();
            let timeout_duration = Duration::from_millis(self.timeout_ms);
            
            let task = tokio::spawn(async move {
                let addr = format!("{}:{}", target, port);
                match timeout(timeout_duration, TcpStream::connect(&addr)).await {
                    Ok(Ok(_)) => Some(port),
                    _ => None,
                }
            });
            
            tasks.push(task);
        }
        
        // 收集結果
        for task in tasks {
            if let Ok(Some(port)) = task.await {
                open_ports.push(port);
            }
        }
        
        open_ports
    }
}

// fingerprint.rs - 服務指紋識別
pub struct FingerprintDetector {
    signatures: HashMap<String, ServiceSignature>,
}

impl FingerprintDetector {
    pub async fn identify_service(&self, host: &str, port: u16) -> Option<ServiceInfo> {
        // 發送探測包
        let banner = self.grab_banner(host, port).await?;
        
        // 匹配簽名
        for (service_name, signature) in &self.signatures {
            if signature.matches(&banner) {
                return Some(ServiceInfo {
                    name: service_name.clone(),
                    version: signature.extract_version(&banner),
                    cpe: signature.cpe.clone(),
                });
            }
        }
        
        None
    }
}
```

#### Go 雲安全掃描

```go
// aws_scanner.go - AWS 安全配置檢查
package cloud_security

import (
    "context"
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/s3"
)

type AWSScanner struct {
    s3Client  *s3.Client
    findings  []SecurityFinding
}

func (s *AWSScanner) ScanS3Buckets(ctx context.Context) error {
    // 列出所有 S3 Bucket
    result, err := s.s3Client.ListBuckets(ctx, &s3.ListBucketsInput{})
    if err != nil {
        return err
    }
    
    // 並發檢查每個 Bucket
    for _, bucket := range result.Buckets {
        go s.checkBucketSecurity(ctx, *bucket.Name)
    }
    
    return nil
}

func (s *AWSScanner) checkBucketSecurity(ctx context.Context, bucketName string) {
    // 檢查公開訪問
    acl, _ := s.s3Client.GetBucketAcl(ctx, &s3.GetBucketAclInput{
        Bucket: &bucketName,
    })
    
    // 檢查加密
    encryption, _ := s.s3Client.GetBucketEncryption(ctx, &s3.GetBucketEncryptionInput{
        Bucket: &bucketName,
    })
    
    // 生成發現
    if isPublic(acl) {
        s.findings = append(s.findings, SecurityFinding{
            Type:     "PUBLIC_BUCKET",
            Severity: "HIGH",
            Resource: bucketName,
            Message:  "S3 bucket is publicly accessible",
        })
    }
    
    if encryption == nil {
        s.findings = append(s.findings, SecurityFinding{
            Type:     "UNENCRYPTED_BUCKET",
            Severity: "MEDIUM",
            Resource: bucketName,
            Message:  "S3 bucket is not encrypted",
        })
    }
}

// secrets_scanner.go - 機密掃描
type SecretsScanner struct {
    patterns []SecretPattern
}

func (s *SecretsScanner) ScanFile(filePath string) []SecretFinding {
    content, _ := ioutil.ReadFile(filePath)
    var findings []SecretFinding
    
    for _, pattern := range s.patterns {
        matches := pattern.Regex.FindAllString(string(content), -1)
        for _, match := range matches {
            findings = append(findings, SecretFinding{
                Type:     pattern.Type,
                Value:    match,
                FilePath: filePath,
                LineNo:   findLineNumber(content, match),
            })
        }
    }
    
    return findings
}
```

---

## 🔄 服務間協作流程

### 完整攻擊流程示例

```
┌──────────────────────────────────────────────────────────┐
│           1. AI 決策階段 (Core)                          │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
            EnhancedDecisionAgent.decide()
                        │
                        ├─► 分析目標信息
                        ├─► RAG 檢索歷史數據
                        ├─► 神經網路決策
                        └─► 生成攻擊策略
                        
┌──────────────────────────────────────────────────────────┐
│           2. 掃描階段 (Scan)                             │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
           ┌────────────┴────────────┐
           │                         │
    Python 爬蟲引擎          TypeScript 動態掃描
           │                         │
           ├─► 智能爬取 URL         ├─► SPA 掃描
           ├─► 會話管理            ├─► AJAX 攔截
           └─► 指紋識別            └─► JS 分析
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
           ┌────────────┴────────────┐
           │                         │
    Rust 端口掃描           Go 雲安全掃描
           │                         │
           ├─► 並發掃描            ├─► AWS 配置檢查
           ├─► 服務識別            ├─► 機密掃描
           └─► 漏洞匹配            └─► SCA 分析
           
┌──────────────────────────────────────────────────────────┐
│           3. 攻擊執行階段 (Features)                     │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
           Integration Coordinator 路由
                        │
           ┌────────────┼────────────┐
           │            │            │
     SQL 注入     XSS 檢測     SSRF 檢測
           │            │            │
      6 個引擎    2 個檢測器   帶外檢測
           │            │            │
           └────────────┴────────────┘
                        │
┌──────────────────────────────────────────────────────────┐
│           4. 結果聚合與學習 (Integration)                │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
              VulnerabilityCorrelationAnalyzer
                        │
                        ├─► 結果去重
                        ├─► 風險評分
                        ├─► CVSS 計算
                        └─► 生成報告
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│           5. 經驗學習 (Core - External Loop)             │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
              ExternalLoopConnector
                        │
                        ├─► 記錄執行軌跡
                        ├─► 更新經驗庫
                        ├─► 訓練 AI 模型
                        └─► 優化策略
```

---

## 📊 技術統計

### 代碼規模
| 模組 | Python 文件 | TypeScript 文件 | Go 文件 | Rust 文件 | 總計 |
|------|------------|----------------|---------|-----------|------|
| aiva_common | 400+ | 50+ (生成) | 0 | 0 | 450+ |
| core | 200+ | 0 | 0 | 0 | 200+ |
| features | 150+ | 0 | 30+ | 20+ | 200+ |
| integration | 50+ | 0 | 0 | 0 | 50+ |
| scan | 100+ | 30+ | 40+ | 15+ | 185+ |
| **總計** | **900+** | **80+** | **70+** | **35+** | **1085+** |

### 關鍵組件統計
- **AI 決策引擎**: 2231 行（EnhancedDecisionAgent）
- **內閉環連接器**: 2036 行（InternalLoopConnector）
- **SQL 注入引擎**: 6 個專業引擎
- **掃描引擎**: 4 語言協同（Python/TS/Go/Rust）
- **Schema 定義**: 200+ Pydantic 模型
- **枚舉定義**: 13 個領域標準

---

## 🎯 架構設計原則

### 1. **微服務架構**
- 每個模組獨立部署
- 通過消息隊列（RabbitMQ）通信
- 支援水平擴展

### 2. **統一數據契約**
```python
# 所有模組使用 aiva_common 的 Schema
from aiva_common.schemas import (
    HighLevelIntent,      # AI 決策輸出
    FeatureResult,        # 功能執行結果
    CoordinatorFinding,   # 協調器發現
    ScanResult,           # 掃描結果
)
```

### 3. **多語言協同**
- **Python**: AI 決策、協調邏輯、快速開發
- **TypeScript**: 動態掃描、瀏覽器自動化、SPA 支援
- **Rust**: 高性能、內存安全、並發掃描
- **Go**: 雲安全、並發處理、系統編程

### 4. **雙閉環學習**
- **內閉環**: AI 自我認知（InternalLoopConnector）
- **外閉環**: 經驗學習（ExternalLoopConnector）

### 5. **標準化輸出**
- SARIF v2.1.0（靜態分析結果格式）
- CVSS v3.1（漏洞評分）
- MITRE ATT&CK（攻擊技術分類）
- CVE/CWE/CAPEC（漏洞分類）

---

## ⚠️ 當前狀態與待驗證項

### ✅ 已完成
1. **AI 決策核心**: EnhancedDecisionAgent 四大決策方法
2. **內閉環整合**: InternalLoopConnector RAG 注入
3. **Features 整合**: 協調器架構實現
4. **多語言掃描**: Python/TS/Go/Rust 引擎完成

### ⚠️ 待驗證
1. **靶場實戰測試**: HTTP 客戶端需實際目標驗證
2. **外閉環觸發**: 經驗學習自動化流程
3. **性能壓測**: 大規模並發掃描測試
4. **Go/Rust 整合**: 跨語言調用穩定性

### 🔴 已知問題
根據 [CLEANUP_SUMMARY.md](services/CLEANUP_SUMMARY.md):
- ✅ CLI Registry 已清理（移除 970+ 行未使用代碼）
- ⚠️ MultiEngineCoordinator 部分導入錯誤（FunctionTaskSchema 缺失）

---

## 🚀 快速開始

### 環境要求
```bash
# Python 環境
Python 3.11+
Poetry (依賴管理)

# Node.js 環境（TypeScript 掃描）
Node.js 16+
npm / yarn

# Go 環境（雲安全掃描）
Go 1.21+

# Rust 環境（高性能掃描）
Rust 1.75+
```

### 安裝流程
```bash
# 1. 安裝 Python 依賴
cd services
poetry install

# 2. 安裝 TypeScript 依賴
cd scan/aiva_scan_node
npm install

# 3. 編譯 Go 掃描器
cd scan/go_scanners
go build -o bin/scanner ./cmd/scanner

# 4. 編譯 Rust 情報收集器
cd scan/info_gatherer_rust
cargo build --release
```

### 啟動服務
```bash
# 1. 啟動 Integration API 閘道
cd integration/aiva_integration
poetry run uvicorn app:app --host 0.0.0.0 --port 8000

# 2. 啟動 Core AI 服務
cd core
poetry run python -m aiva_core.service_backbone.coordination.core_service_coordinator

# 3. 啟動 Features 工作器
cd features/features_ready/function_sqli
poetry run python worker.py

# 4. 啟動 Scan 引擎
cd scan
poetry run python -m aiva_scan.main
```

---

## 📚 相關文檔

### 核心文檔
- [services/README.md](services/README.md) - 服務架構總覽
- [services/aiva_common/README.md](services/aiva_common/README.md) - 共享庫文檔
- [services/core/README.md](services/core/README.md) - AI 核心引擎
- [services/features/README.md](services/features/README.md) - 功能模組
- [services/integration/README.md](services/integration/README.md) - 整合架構
- [services/scan/README.md](services/scan/README.md) - 掃描引擎

### 技術文檔
- [CLEANUP_SUMMARY.md](services/CLEANUP_SUMMARY.md) - 代碼清理報告
- [POST_CLEANUP_EVALUATION_REPORT.md](services/POST_CLEANUP_EVALUATION_REPORT.md) - 清理後評估

---

## 🔒 安全性

### 數據保護
- 敏感信息加密存儲
- API 密鑰安全管理
- 掃描結果隔離

### 權限控制
- 基於角色的訪問控制（RBAC）
- API 密鑰認證
- 操作審計日誌

### 合規性
- 遵循 OWASP Top 10
- 支援 GDPR 數據保護
- CVE/CWE/CAPEC 標準

---

## 📊 監控與可觀測性

### 日誌系統
```python
from aiva_common.utils.logging import get_logger
logger = get_logger(__name__)

logger.info("Scan started", extra={"target": target_url})
logger.error("Detection failed", extra={"error": str(e)})
```

### 指標收集
- 掃描性能指標
- AI 決策準確率
- 漏洞檢測成功率
- 服務健康狀態

### 分散式追蹤
- 端到端請求追蹤
- 跨服務調用鏈
- 性能瓶頸分析

---

## 🎯 未來規劃

### 短期目標（1-3 個月）
1. 完成靶場實戰測試
2. 優化 AI 決策準確率
3. 完善外閉環學習觸發
4. Go/Rust 引擎穩定性提升

### 中期目標（3-6 個月）
1. 新增更多漏洞檢測類型
2. 提升掃描性能（10x）
3. 完善 UI 面板
4. 支援更多雲平台

### 長期目標（6-12 個月）
1. 完全自動化的 Bug Bounty 平台
2. 業界領先的 AI 決策能力
3. 全球化部署
4. 商業化產品

---

## 📞 聯繫方式

- **項目**: AIVA Services
- **版本**: v7.1-stable
- **最後更新**: 2026-01-09
- **文檔生成**: 2026-01-10

---

**報告結束** ✅
