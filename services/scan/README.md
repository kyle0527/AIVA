# AIVA Scan - 多語言統一掃描引擎

AIVA Scan 是一個強大的多語言統一掃描引擎，整合了 Python、TypeScript 和 Rust 三種技術的優勢，提供全方位的網路安全掃描能力。

## 🏗️ 整體架構

```mermaid
graph TB
    subgraph "AIVA Scan 統一掃描引擎"
        direction TB
        
        subgraph "協調層 | Orchestration Layer"
            USG[統一掃描引擎<br/>UnifiedScanEngine]
            SO[掃描編排器<br/>ScanOrchestrator]
        end

        subgraph "Python 掃描引擎 | aiva_scan/"
            direction LR
            SC[掃描上下文<br/>ScanContext]
            CCE[核心爬取引擎<br/>CoreCrawlingEngine]
            VS[漏洞掃描器<br/>VulnerabilityScanner]
            AM[認證管理<br/>AuthManager]
            FM[指紋管理<br/>FingerprintManager]
            NSC[網路掃描器<br/>NetworkScanner]
            SDT[服務檢測<br/>ServiceDetector]
        end

        subgraph "TypeScript 動態引擎 | aiva_scan_node/"
            direction LR
            PIS[Phase-I 整合服務<br/>PhaseIService]
            PW[Playwright 引擎<br/>Browser Automation]
            CSA[客戶端授權檢測<br/>Client Auth Bypass]
            JS[JavaScript 分析器<br/>JS Analyzer]
        end

        subgraph "Rust 資訊收集器 | info_gatherer_rust/"
            direction LR
            SEC[敏感資訊檢測<br/>Secret Scanner]
            GH[Git 歷史掃描<br/>Git History]
            API[API 密鑰檢測<br/>API Key Scanner]
            REG[正則引擎<br/>Regex Engine]
        end

        subgraph "共享模組 | Shared Components"
            SCHEMA[統一 Schema<br/>Unified Models]
            CONFIG[配置管理<br/>Configuration]
            MQ[消息隊列<br/>Message Queue]
            LOGS[日誌系統<br/>Logging]
        end
    end

    %% 連接關係
    USG --> SO
    SO --> SC
    SC --> CCE
    SC --> VS
    CCE --> AM
    CCE --> FM
    
    USG --> PIS
    PIS --> PW
    PIS --> CSA
    PW --> JS
    
    USG --> SEC
    SEC --> GH
    SEC --> API
    SEC --> REG
    
    SO -.-> MQ
    PIS -.-> MQ
    SEC -.-> LOGS
    
    %% 樣式
    classDef pythonStyle fill:#3776ab,stroke:#2d5aa0,stroke-width:2px,color:#fff
    classDef typescriptStyle fill:#3178c6,stroke:#2b6cb0,stroke-width:2px,color:#fff
    classDef rustStyle fill:#ce422b,stroke:#a0351f,stroke-width:2px,color:#fff
    classDef sharedStyle fill:#6b7280,stroke:#4b5563,stroke-width:2px,color:#fff
    
    class SO,SC,CCE,VS,AM,FM,NSC,SDT pythonStyle
    class PIS,PW,CSA,JS typescriptStyle
    class SEC,GH,API,REG rustStyle
    class USG,SCHEMA,CONFIG,MQ,LOGS sharedStyle
```

## 📁 目錄結構

```
services/scan/
├── __init__.py                    # 模組入口點
├── README.md                     # 本文檔
├── unified_scan_engine.py        # 🎯 統一掃描引擎
├── schemas.py                    # 掃描相關 Schema
├── models.py                     # 資料模型定義
├── discovery_schemas.py          # 發現模組 Schema
│
├── aiva_scan/                    # 🐍 Python 核心掃描引擎
│   ├── __init__.py
│   ├── scan_orchestrator.py     # 掃描編排器 (核心)
│   ├── scan_context.py          # 掃描上下文管理
│   ├── strategy_controller.py   # 策略控制器
│   ├── worker.py                # 工作程序
│   │
│   ├── core_crawling_engine/    # 核心爬取引擎
│   │   ├── http_client_hi.py
│   │   ├── static_content_parser.py
│   │   └── url_queue_manager.py
│   │
│   ├── dynamic_engine/          # 動態掃描引擎
│   │   ├── dynamic_content_extractor.py
│   │   └── headless_browser_pool.py
│   │
│   ├── info_gatherer/          # 資訊收集器
│   │   ├── javascript_source_analyzer.py
│   │   └── sensitive_info_detector.py
│   │
│   ├── authentication_manager.py # 認證管理
│   ├── fingerprint_manager.py   # 指紋識別
│   ├── vulnerability_scanner.py # 漏洞掃描
│   ├── network_scanner.py       # 網路掃描
│   ├── service_detector.py      # 服務檢測
│   └── examples/               # 使用範例
│
├── aiva_scan_node/             # 🟨 TypeScript 動態掃描引擎
│   ├── package.json
│   ├── tsconfig.json
│   ├── README.md
│   ├── phase-i-integration.service.ts # Phase-I 整合服務
│   ├── src/                    # 源代碼
│   │   ├── services/
│   │   ├── interfaces/
│   │   └── utils/
│   └── types/                  # 型別定義
│       └── playwright.d.ts
│
└── info_gatherer_rust/        # 🦀 Rust 高性能資訊收集器
    ├── Cargo.toml
    ├── Cargo.lock
    ├── README.md
    ├── src/
    │   ├── main.rs
    │   ├── scanner.rs
    │   └── patterns.rs
    └── target/               # 編譯輸出
```

## 🚀 現有功能

### 1. Python 核心掃描引擎 (aiva_scan/)

#### 核心功能
- **ScanOrchestrator**: 統一掃描編排，支援多種掃描策略
- **策略控制**: 支援 FAST/COMPREHENSIVE/AGGRESSIVE 三種掃描模式
- **動態掃描**: 整合 Headless Browser 進行 JavaScript 渲染
- **認證管理**: 支援多種認證方式 (Basic, Bearer, Custom Headers)
- **指紋識別**: 自動識別技術棧和服務版本

#### 掃描能力
- **網路掃描**: 端口掃描、服務檢測、橫幅抓取
- **漏洞掃描**: SQL 注入、XSS、CSRF、目錄遍歷等
- **內容分析**: 靜態內容解析、JavaScript 源碼分析
- **敏感資訊**: API 密鑰、密碼、憑證等檢測

### 2. TypeScript 動態引擎 (aiva_scan_node/)

#### Phase-I 高價值功能
- **客戶端授權繞過檢測**: 檢測前端權限控制漏洞
- **Playwright 自動化**: 真實瀏覽器環境掃描
- **JavaScript 深度分析**: 代碼邏輯分析和漏洞挖掘
- **DOM 操作檢測**: 動態內容變化監控

#### 技術特性
- **TypeScript 型別安全**: 完整的型別定義
- **非同步處理**: 高效的並發掃描
- **模組化架構**: 易於擴展和維護

### 3. Rust 資訊收集器 (info_gatherer_rust/)

#### 性能優勢
- **極致性能**: 比 Python 快 10-100 倍
- **低記憶體消耗**: 單次掃描僅需 ~5 MB
- **並行處理**: Rayon 並行引擎
- **零拷貝**: 高效字串處理

#### 檢測類型
- AWS Access/Secret Keys
- GitHub Tokens  
- API Keys
- Private Keys (PEM/SSH)
- JWT Tokens
- Database Connection Strings
- Email 地址和 IP 位址

### 4. 統一掃描引擎 (unified_scan_engine.py)

#### 整合能力
- **多引擎協調**: 統一管理三種技術引擎
- **Phase-I 整合**: 高價值功能模組整合
- **策略配置**: 彈性的掃描策略配置
- **結果聚合**: 統一的結果格式和匯報

## 💻 如何使用

### 1. 快速開始

```python
# 使用統一掃描引擎
from services.scan.unified_scan_engine import UnifiedScanEngine

# 創建快速掃描
engine = UnifiedScanEngine.create_fast_scan(["https://example.com"])

# 執行掃描
results = await engine.run_comprehensive_scan()
print(f"掃描完成: {results['scan_id']}")
```

### 2. 使用 Python 掃描器

```python
from services.scan.aiva_scan import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

# 創建掃描請求
request = ScanStartPayload(
    scan_id="scan_001",
    targets=["https://example.com"],
    strategy="COMPREHENSIVE"
)

# 執行掃描
orchestrator = ScanOrchestrator()
result = await orchestrator.execute_scan(request)
```

### 3. 配置 TypeScript 引擎

```bash
# 安裝依賴
cd services/scan/aiva_scan_node
npm install

# 安裝瀏覽器
npm run install:browsers

# 啟動服務
npm run dev
```

### 4. 編譯 Rust 收集器

```bash
cd services/scan/info_gatherer_rust

# 開發模式
cargo run

# 釋出模式 (最佳化)
cargo build --release
.\target\release\info_gatherer_rust.exe
```

### 5. 工作程序模式

```python
# 啟動掃描工作程序 (透過 RabbitMQ)
from services.scan.aiva_scan.worker import run

# 監聽掃描任務
await run()
```

## 📋 新增/刪減功能 SOP

### 新增功能 SOP

#### 1. 分析需求階段
```markdown
📝 需求分析檢查清單:
□ 確定功能屬於哪個引擎 (Python/TypeScript/Rust)
□ 評估性能要求和技術適用性
□ 檢查是否需要跨引擎整合
□ 確認 Schema 定義需求
```

#### 2. 技術選型指引

| 需求類型 | 推薦技術 | 理由 |
|---------|---------|------|
| 高性能文本處理 | Rust | 正則引擎性能最佳 |
| 瀏覽器自動化 | TypeScript + Playwright | 生態最成熟 |
| 複雜業務邏輯 | Python | 開發效率高 |
| 跨語言整合 | unified_scan_engine.py | 統一入口 |

#### 3. 開發流程

**Step 1: Schema 定義**
```python
# 1. 在 models.py 中添加新的資料模型
class NewFeatureConfig(BaseModel):
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

# 2. 更新 __init__.py 的 __all__ 列表
__all__ = [
    # ... 現有項目
    "NewFeatureConfig",
]
```

**Step 2: 核心實作**
```python
# Python 範例: 新增掃描器
class NewScanner:
    def __init__(self, config: NewFeatureConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    async def scan(self, target: str) -> List[Finding]:
        # 實作掃描邏輯
        pass
```

**Step 3: 整合到編排器**
```python
# 在 ScanOrchestrator 中整合
class ScanOrchestrator:
    def __init__(self):
        # ... 現有初始化
        self.new_scanner = NewScanner()
    
    async def execute_scan(self, request: ScanStartPayload):
        # ... 現有邏輯
        
        # 添加新功能
        if strategy_params.enable_new_feature:
            new_results = await self.new_scanner.scan(target)
            context.add_findings(new_results)
```

**Step 4: 測試驗證**
```python
# 創建單元測試
class TestNewScanner:
    async def test_basic_scan(self):
        scanner = NewScanner(NewFeatureConfig())
        results = await scanner.scan("https://example.com")
        assert len(results) >= 0
```

**Step 5: 文檔更新**
```markdown
# 更新本 README.md
## 新功能: XXX
- 功能描述
- 使用方式  
- 配置選項
```

### 刪減功能 SOP

#### 1. 影響評估
```markdown
🔍 刪減前檢查清單:
□ 檢查功能使用情況 (logs, metrics)
□ 確認依賴關係 (grep -r "功能名稱" services/)
□ 評估向後兼容性影響
□ 準備遷移計畫 (如果需要)
```

#### 2. 安全刪除流程

**Step 1: 標記為廢棄**
```python
import warnings

@deprecated("此功能將在 v2.0 移除，請使用 NewFeature 替代")
class OldScanner:
    def __init__(self):
        warnings.warn(
            "OldScanner 已廢棄，請遷移至 NewScanner",
            DeprecationWarning,
            stacklevel=2
        )
```

**Step 2: 配置開關**
```python
# 添加功能開關
class StrategyParameters:
    enable_old_feature: bool = False  # 預設關閉
```

**Step 3: 逐步移除**
```python
# 第一階段：條件執行
if strategy_params.enable_old_feature:
    # 舊功能代碼
    pass
else:
    logger.info("舊功能已停用")

# 第二階段：完全移除代碼
# (在確認無影響後)
```

**Step 4: 清理工作**
```bash
# 移除相關文件
rm services/scan/old_feature.py

# 更新導入
# 從 __init__.py 和其他文件移除相關導入

# 更新測試
# 移除相關測試案例

# 更新文檔
# 從 README.md 移除相關描述
```

### 版本控制最佳實踐

#### Git 工作流程
```bash
# 1. 創建功能分支
git checkout -b feature/new-scanner

# 2. 開發並提交
git add .
git commit -m "feat(scan): 添加新掃描器功能

- 實作 NewScanner 類別
- 整合到 ScanOrchestrator
- 添加單元測試
- 更新文檔

Closes #123"

# 3. 合併到主分支
git checkout main
git merge feature/new-scanner

# 4. 標記版本
git tag -a v1.2.0 -m "Release v1.2.0: 新增 NewScanner 功能"
```

#### 變更紀錄格式
```markdown
## [1.2.0] - 2025-10-24

### Added
- 新增 NewScanner 掃描器
- 支援 XXX 類型漏洞檢測
- 添加 Phase-I 高價值功能整合

### Changed  
- 改進 ScanOrchestrator 性能
- 更新統一掃描引擎配置介面

### Deprecated
- OldScanner 標記為廢棄，將在 v2.0 移除

### Removed
- 移除已廢棄的 LegacyFeature

### Fixed
- 修復 TypeScript 引擎記憶體洩漏問題
- 解決 Rust 收集器 Unicode 處理錯誤
```

### 監控與維護

#### 性能監控
```python
# 添加性能指標
import time
from prometheus_client import Counter, Histogram

SCAN_DURATION = Histogram('scan_duration_seconds', 'Scan duration')
SCAN_ERRORS = Counter('scan_errors_total', 'Scan errors')

class ScanOrchestrator:
    async def execute_scan(self, request):
        start_time = time.time()
        try:
            result = await self._do_scan(request)
            return result
        except Exception as e:
            SCAN_ERRORS.inc()
            raise
        finally:
            SCAN_DURATION.observe(time.time() - start_time)
```

#### 日誌管理
```python
# 結構化日誌
logger.info(
    "掃描完成",
    extra={
        "scan_id": request.scan_id,
        "targets_count": len(request.targets),
        "duration": context.scan_duration,
        "findings_count": len(context.findings)
    }
)
```

## 🔧 故障排除

### 常見問題

#### 1. Python 引擎問題
```bash
# 依賴問題
pip install -r requirements.txt

# 導入錯誤
export PYTHONPATH="${PYTHONPATH}:/path/to/AIVA-git"

# 權限問題  
chmod +x scripts/launcher/aiva_launcher.py
```

#### 2. TypeScript 引擎問題
```bash
# Node.js 版本問題
nvm use 18

# 依賴安裝問題
rm -rf node_modules package-lock.json
npm install

# Playwright 瀏覽器問題
npx playwright install --with-deps
```

#### 3. Rust 引擎問題
```bash
# 編譯問題
cargo clean
cargo build --release

# 依賴問題
cargo update
```

#### 4. 整合問題
```bash
# RabbitMQ 連線問題
docker-compose up -d rabbitmq

# 權限問題
sudo chown -R $USER:$USER services/scan/

# 環境變數
export RABBITMQ_URL=amqp://aiva:dev_password@localhost:5672/
```

## 📊 性能基準

### 掃描性能對比

| 引擎類型 | 小型站點 (10 頁面) | 中型站點 (100 頁面) | 大型站點 (1000 頁面) |
|---------|-------------------|-------------------|-------------------|
| Python 核心 | ~30 秒 | ~5 分鐘 | ~45 分鐘 |
| TypeScript 動態 | ~45 秒 | ~8 分鐘 | ~60 分鐘 |
| Rust 資訊收集 | ~2 秒 | ~15 秒 | ~2 分鐘 |
| 統一引擎 | ~50 秒 | ~10 分鐘 | ~65 分鐘 |

### 資源消耗

| 引擎類型 | 記憶體消耗 | CPU 使用率 | 磁碟 I/O |
|---------|-----------|-----------|----------|
| Python | ~100 MB | ~20% | 中等 |
| TypeScript | ~200 MB | ~30% | 高 (瀏覽器快取) |
| Rust | ~5 MB | ~10% | 極低 |

## 🏆 最佳實踐

### 1. 掃描策略選擇
```python
# 快速掃描：適用於 CI/CD 流水線
UnifiedScanEngine.create_fast_scan(targets)

# 綜合掃描：適用於定期安全檢查  
UnifiedScanEngine.create_comprehensive_scan(targets)

# 自訂掃描：適用於特殊需求
config = UnifiedScanConfig(
    targets=targets,
    scan_type="comprehensive",
    max_depth=5,
    max_pages=500,
    enable_plugins=True
)
```

### 2. 效能優化
```python
# 平行掃描
strategy_params.requests_per_second = 10
strategy_params.connection_pool_size = 50

# 記憶體控制
strategy_params.max_pages = 100
strategy_params.enable_dynamic_scan = False  # 如不需要 JS 渲染
```

### 3. 安全配置
```python
# 認證配置
authentication = {
    "type": "bearer",
    "token": "your-api-token"
}

# 自訂標頭
custom_headers = {
    "User-Agent": "AIVA-Scanner/1.0",
    "X-Custom-Header": "value"
}
```

---

📝 **文檔版本**: v1.0.0  
🔄 **最後更新**: 2025-10-24  
👥 **維護者**: AIVA Development Team  

有任何問題或建議，請提交 Issue 或 Pull Request！